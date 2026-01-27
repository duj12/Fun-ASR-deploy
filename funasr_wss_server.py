# -*- encoding: utf-8 -*-
import asyncio
import json
import websockets
import copy
import logging
import argparse
import ssl
import os
import numpy as np
import torch
import traceback
import uuid
from typing import Optional, List, Dict, Any
# 需要引下这个，不然会报错AssertionError: FunASRNano is not registered
# issue见：https://github.com/modelscope/FunASR/issues/2741
from funasr.models.fun_asr_nano.model import FunASRNano
from funasr import AutoModel
from funasr.utils.load_utils import extract_fbank
from torch.nn.utils.rnn import pad_sequence

logging.root.handlers = []  # 清空modelscope修改后的handlers
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] %(process)d %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# vLLM 相关全局变量
global vllm_engine, vllm_sampling_params, model_asr_nano, asr_tokenizer, asr_frontend
global model_asr_streaming, model_vad, model_punc
vllm_engine = None
vllm_sampling_params = None
model_asr_nano = None  # FunASRNano 模型实例或带 llm 的 ASR 模型
asr_tokenizer = None
asr_frontend = None
model_asr_streaming = None
model_vad = None
model_punc = None

# 语言代码到可读文本的映射（用于构造自然语言 prompt）
LANG_TEXT_MAP = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
}


def build_prompt(hotwords, language_code, itn):
    """
    根据 itn、hotwords、language 构造文本提示，逻辑参考 Fun-ASR-vllm/model.py(553-568)。
    - hotwords: List[str]
    - language_code: "zh" / "en" / "ja" / 其它
    - itn: bool
    """
    hotwords = hotwords or []
    # 1. 热词上下文
    if len(hotwords) > 0:
        hotwords_str = ", ".join(hotwords)
        prompt = (
            "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
            "**上下文信息：**\n\n\n"
        )
        prompt += f"热词列表：[{hotwords_str}]\n"
    else:
        prompt = ""

    # 2. 语言
    lang_text = None
    if language_code is not None:
        # 客户端传 zh/en/ja，这里转成中文描述；若是其它值，则直接拼接
        lang_text = LANG_TEXT_MAP.get(language_code, language_code)

    if lang_text is None:
        prompt += "语音转写"
    else:
        prompt += f"语音转写成{lang_text}"

    # 3. 是否 ITN
    if itn is False:
        prompt += "，不进行文本规整"

    prompt += "："
    return prompt


def get_args():
    """
    解析命令行参数。
    包含了服务端监听配置、模型路径配置、硬件设备配置等。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", required=False, help="监听 IP，localhost 或 0.0.0.0"
    )
    parser.add_argument("--port", type=int, default=10095, required=False, help="服务端口")
    parser.add_argument(
        "--asr_model",
        type=str,
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        help="离线 ASR 模型名称 (从 ModelScope 下载)",
    )
    parser.add_argument("--asr_model_revision", type=str, default=None, help="模型版本")
    parser.add_argument(
        "--asr_model_online",
        type=str,
        default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        help="流式 ASR 模型名称 (从 ModelScope 下载)",
    )
    parser.add_argument("--asr_model_online_revision", type=str, default='v2.0.4', help="模型版本")
    parser.add_argument(
        "--vad_model",
        type=str,
        default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        help="VAD 模型名称",
    )
    parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="模型版本")
    parser.add_argument(
        "--punc_model",
        type=str,
        # default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        default="",
        help="标点恢复模型名称, Fun-ASR-Nano自回归带标点",
    )
    parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="模型版本")
    parser.add_argument("--ngpu", type=int, default=1, help="GPU 数量，0 为 CPU")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备: cuda 或 cpu")
    parser.add_argument("--ncpu", type=int, default=4, help="CPU 核心数")
    parser.add_argument(
        "--certfile",
        type=str,
        default="",
        required=False,
        help="SSL 证书文件",
    )
    parser.add_argument(
        "--keyfile",
        type=str,
        default="",
        required=False,
        help="SSL 密钥文件",
    )
    parser.add_argument("--fp16", action="store_true", help="使用 fp16 进行推理")
    parser.add_argument(
        "--vllm_model_dir",
        type=str,
        default="checkpoints/yuekai/Fun-ASR-Nano-2512-vllm",
        help="vLLM 模型目录路径，如果提供则使用 vLLM 进行异步推理",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.08,
        help="vLLM GPU 内存利用率 (0.0-1.0)",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=16,
        help="vLLM 最大并发序列数",
    )

    return parser.parse_args()

args = get_args()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints")
asr_model_dir = os.path.join(checkpoint_dir, args.asr_model)
if not os.path.exists(asr_model_dir):
    raise FileNotFoundError(f"{asr_model_dir} 模型不存在")
else:
    args.asr_model = asr_model_dir
asr_model_online_dir = os.path.join(checkpoint_dir, args.asr_model_online)
if not os.path.exists(asr_model_online_dir):
    raise FileNotFoundError(f"{asr_model_online_dir} 模型不存在")
else:
    args.asr_model_online = asr_model_online_dir
vad_model_dir = os.path.join(checkpoint_dir, args.vad_model)
if not os.path.exists(vad_model_dir):
    raise FileNotFoundError(f"{vad_model_dir} 模型不存在")
else:
    args.vad_model = vad_model_dir
if args.punc_model:
    punc_model_dir = os.path.join(checkpoint_dir, args.punc_model)
    if not os.path.exists(punc_model_dir):
        raise FileNotFoundError(f"{punc_model_dir} 模型不存在")
    else:
        args.punc_model = punc_model_dir


websocket_users = set()

# 连接级别状态（按 uuid 隔离，避免不同连接 cache/参数互相干扰）
# session_id -> {"asr": {}, "asr_online": {"cache": {}, "is_final": False}, "vad": {"cache": {}, "is_final": False}, "punc": {"cache": {}}}
session_states: Dict[str, Dict[str, Any]] = {}

def new_session_state() -> Dict[str, Any]:
    return {
        "asr": {},
        "asr_online": {"cache": {}, "is_final": False},
        "vad": {"cache": {}, "is_final": False},
        "punc": {"cache": {}},
    }

def get_state(sid) -> Dict[str, Any]:
    if sid not in session_states:
        state_dict = new_session_state()
        session_states[sid] = copy.deepcopy(state_dict)
    return session_states[sid]


# 异步模型推理辅助函数
async def run_model_inference(model, input, **kwargs):
    """
    
    Args:
        model: FunASR 模型实例
        input: 模型输入数据 (通常是 Tensor 列表或文本)
        **kwargs: 额外的推理参数 (如 status_dict)
    
    Returns:
        推理结果
    """   
    return model.generate(input=input, **kwargs)


async def async_asr_with_vllm(audio_tensor: torch.Tensor, websocket, sid, **kwargs):
    """
    使用 vLLM 进行异步 ASR 推理（适用于离线/2pass ASR）。
    结合 itn / hotwords / language 动态构造 prompt，并计算 embeddings 后再 generate。
    
    Args:
        audio_tensor: 音频张量
        websocket: WebSocket 连接对象
        **kwargs: 额外的推理参数
    
    Returns:
        List[{"text": str}]
    """
    global vllm_engine, vllm_sampling_params, model_asr_nano
    global asr_tokenizer, asr_frontend

    # 如果 vLLM 未初始化，则回退到传统 FunASR 推理
    if vllm_engine is None or model_asr_nano is None or asr_tokenizer is None or asr_frontend is None:
        return await run_model_inference(model_asr_nano, [audio_tensor], **kwargs)

    # 1. 从 websocket / kwargs 里拿到 itn / hotwords / language 参数
    state = get_state(sid)
    itn = kwargs.get("itn", state['asr'].get("itn", True))
    hotwords = kwargs.get("hotwords", state['asr'].get("hotwords", []))
    language_code = kwargs.get("language", state['asr'].get("language", None))

    prompt = build_prompt(hotwords, language_code, itn)

    # 2. 构造 LLM prompt，并计算文本 embeddings
    loop = asyncio.get_running_loop()
    device = next(model_asr_nano.parameters()).device

    async def encode_audio_and_prompt():
        # 2.1 特征提取
        speech, speech_lengths = extract_fbank(
            [audio_tensor],
            frontend=asr_frontend,
            is_final=True,
        )
        speech = speech.to(device)
        speech_lengths = speech_lengths.to(device)

        # 2.2 音频编码
        encoder_out, encoder_out_lens = model_asr_nano.audio_encoder(speech, speech_lengths)
        encoder_out, encoder_out_lens = model_asr_nano.audio_adaptor(encoder_out, encoder_out_lens)

        # 2.3 文本 prompt -> token ids -> embeddings
        instruction_prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}"
        )
        prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

        prefix_ids = asr_tokenizer.encode(instruction_prompt)
        suffix_ids = asr_tokenizer.encode(prompt_suffix)

        emb_layer = model_asr_nano.llm.model.get_input_embeddings()
        prefix_emb = emb_layer(torch.tensor(prefix_ids, dtype=torch.int64, device=device))
        suffix_emb = emb_layer(torch.tensor(suffix_ids, dtype=torch.int64, device=device))

        # 2.4 拼接：prefix + speech_emb + suffix
        speech_emb = encoder_out[0, :encoder_out_lens[0], :]
        input_embedding = torch.cat([prefix_emb, speech_emb, suffix_emb], dim=0)
        return input_embedding

    input_embedding = await encode_audio_and_prompt()

    # 3. 调用 vLLM generate
    async def vllm_generate():
        from vllm.sampling_params import RequestOutputKind
        # 设置为返回完整输出而非流式输出
        vllm_sampling_params.output_kind = RequestOutputKind.FINAL_ONLY         
        # 收集所有生成的文本
        full_text = ""
        async for outputs in vllm_engine.generate(
            {"prompt_embeds": input_embedding},
            vllm_sampling_params,
            request_id=sid,
        ):
            # 累积每次生成的文本
            if outputs.outputs:
                full_text = outputs.outputs[0].text

        return full_text

    text = await vllm_generate()
    return [{"text": text}]


def decode_audio_chunk(chunk_bytes):
    """
    将接收到的原始音频字节流解码为 PyTorch Tensor。
    输入格式默认假设为: PCM, 16000Hz, 16bit, Mono.
    
    Args:
        chunk_bytes (bytes): 原始音频二进制数据
    
    Returns:
        torch.Tensor:float32: 归一化到 [-1.0, 1.0] 的音频张量
    """
    # 1. Bytes -> Int16 Numpy
    data_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
    # 2. Int16 -> Float32 (Normalize to -1.0 ~ 1.0)
    data_float32 = data_int16.astype(np.float32) / 32768.0
    # 3. Numpy -> Torch Tensor
    return torch.from_numpy(data_float32)


async def ws_reset(websocket, sid=None):
    """
    重置 WebSocket 连接对应的状态缓存。
    当连接断开及为了安全起见清理内存时调用。
    """
    logger.info(f"ws reset now, total num is: {len(websocket_users)}, session_id={sid}")

    # 清理该连接的状态缓存
    if sid is not None and sid in session_states:
        try:
            session_states[sid]["asr_online"]["cache"] = {}
            session_states[sid]["asr_online"]["is_final"] = True
            session_states[sid]["vad"]["cache"] = {}
            session_states[sid]["vad"]["is_final"] = True
            session_states[sid]["punc"]["cache"] = {}
        except Exception:
            pass
        session_states.pop(sid, None)

    await websocket.close()


async def clear_websocket():
    """
    清理所有活跃的 WebSocket 连接。
    """
    for websocket in list(websocket_users):
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path=None):
    """
    WebSocket 服务端主处理逻辑。
    负责处理单个客户端的完整生命周期：握手 -> 音频流处理 -> 返回结果 -> 断开。
    """
    frames = [] 
    frames_asr = [] # 离线 ASR 缓冲区 (由 VAD 分割)
    frames_asr_online = [] # 在线流式 ASR 缓冲区
    
    websocket.send_lock = asyncio.Lock()
    global websocket_users
    websocket_users.add(websocket)

    # 为每个连接生成 uuid，并初始化/获取该连接的隔离状态
    sid = str(uuid.uuid4())
    state = get_state(sid)
    
    state["chunk_interval"] = 10
    state["vad_pre_idx"] = 0
    speech_start = False
    speech_end_i = -1
    state["wav_name"] = "microphone"
    state["mode"] = "2pass"
    state["is_speaking"] = True
    
    logger.info(f"new user connected, session_id={sid}. Current total num is: {len(websocket_users)}")

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    messagejson = json.loads(message)
                    logger.info({"session_id": sid, **messagejson})
                    # logger.info(f"state: {state}")
                    if "is_speaking" in messagejson:
                        state["is_speaking"] = messagejson["is_speaking"]
                        state["asr_online"]["is_final"] = not state["is_speaking"]
                    if "chunk_interval" in messagejson:
                        state["chunk_interval"] = messagejson["chunk_interval"]
                    if "wav_name" in messagejson:
                        state["wav_name"] = messagejson.get("wav_name")
                    if "chunk_size" in messagejson:
                        chunk_size = messagejson["chunk_size"]
                        if isinstance(chunk_size, str):
                            chunk_size = chunk_size.split(",")
                        state["asr_online"]["chunk_size"] = [int(x) for x in chunk_size]
                    if "encoder_chunk_look_back" in messagejson:
                        state["asr_online"]["encoder_chunk_look_back"] = messagejson["encoder_chunk_look_back"]
                    if "decoder_chunk_look_back" in messagejson:
                        state["asr_online"]["decoder_chunk_look_back"] = messagejson["decoder_chunk_look_back"]
                    if "hotwords" in messagejson and len(messagejson['hotwords'])>0:
                        hotword_dict = json.loads(messagejson["hotwords"])
                        hotword_list = list(hotword_dict.keys())
                        state["asr"]["hotwords"] = hotword_list
                        logger.info(f"hotword_list: {hotword_list}")
                    if "itn" in messagejson:
                        itn = messagejson["itn"]
                        state["asr"]["itn"] = itn
                    if "svs_lang" in messagejson:
                        # 客户端传 zh/en/ja 这样的代码，这里直接保存，后续在 build_prompt 里映射为中文描述
                        language = messagejson["svs_lang"]
                        state["asr"]["language"] = language
                    # VAD 相关参数映射：客户端字段 -> VAD status_dict 字段
                    # vad_tail_sil  -> max_end_silence_time
                    # vad_max_len   -> max_single_segment_time
                    # vad_energy    -> decibel_thres
                    if "vad_tail_sil" in messagejson:
                        state["vad"]["max_end_silence_time"] = messagejson["vad_tail_sil"]
                    if "vad_max_len" in messagejson:
                        state["vad"]["max_single_segment_time"] = messagejson["vad_max_len"]
                    if "vad_energy" in messagejson:
                        state["vad"]["decibel_thres"] = messagejson["vad_energy"]
                    if "mode" in messagejson:
                        state["mode"] = messagejson["mode"]
                        # if state["mode"] == "online":
                        #     state["mode"] = "2pass"
                except Exception as e:
                    print("JSON error:", e)

            # 确保 VAD 的分块大小正确计算
            if "chunk_size" in state["asr_online"]:
                 state["vad"]["chunk_size"] = int(
                    state["asr_online"]["chunk_size"][1] * 60 / state["chunk_interval"]
                )
            
            # 处理音频数据
            if len(frames_asr_online) > 0 or len(frames_asr) >= 0 or not isinstance(message, str):
                if not isinstance(message, str):
                    # 收到的是音频块
                    frames.append(message)
                    duration_ms = len(message) // 32 # 16k rate, 16bit = 2 bytes. 1ms = 16 samples = 32 bytes
                    state["vad_pre_idx"] += duration_ms

                    # 1. 送入在线流式 ASR (Online ASR)
                    frames_asr_online.append(message)
                    state["asr_online"]["is_final"] = speech_end_i != -1
                    
                    # 根据 chunk 间隔或语音结束信号触发在线推理
                    if (len(frames_asr_online) % state["chunk_interval"] == 0 
                        or state["asr_online"]["is_final"]):
                        
                        if state["mode"] == "2pass" or state["mode"] == "online":
                            audio_in = b"".join(frames_asr_online)
                            try:
                                await async_asr_online(websocket, audio_in, sid)
                            except Exception as e:
                                print(f"error in asr streaming: {e}")
                                traceback.print_exc()
                        
                        frames_asr_online = [] # 清空在线缓冲区

                    # 2. 送入 VAD 检测
                    if speech_start:
                        frames_asr.append(message) # 收集用于离线识别的音频
                    
                    try:
                        speech_start_i, speech_end_i = await async_vad(websocket, message, sid)
                    except Exception as e:
                        logger.error(f"error in vad: {e}")
                        speech_start_i, speech_end_i = -1, -1
                    
                    # 处理 VAD 的语音开始信号
                    if speech_start_i != -1:
                        speech_start = True
                        # 回溯音频池，捕获语音起始段
                        beg_bias = (state["vad_pre_idx"] - speech_start_i) // duration_ms
                        frames_pre = frames[-beg_bias:]
                        frames_asr = []
                        frames_asr.extend(frames_pre)
                
                # 3. 处理语音结束或流结束 -> 触发离线 ASR + 标点恢复
                if speech_end_i != -1 or not state["is_speaking"]:
                    if state["mode"] == "2pass" or state["mode"] == "offline":
                        audio_in = b"".join(frames_asr)
                        try:
                            await async_asr(websocket, audio_in, sid)
                        except Exception as e:
                            logger.error(f"error in asr offline: {e}")
                            traceback.print_exc()
                    else:  # online only
                        audio_in = b""
                        try:
                            await async_asr_online(websocket, audio_in, sid)
                        except Exception as e:
                            logger.error(f"error in asr offline: {e}")
                            traceback.print_exc()
                        
                    
                    # 重置状态
                    frames_asr = []
                    speech_start = False
                    frames_asr_online = []
                    state["asr_online"]["cache"] = {}
                    
                    if not state["is_speaking"]:
                        state["vad_pre_idx"] = 0
                        frames = []
                        state["vad"]["cache"] = {}
                    else:
                        # 保留少量上下文
                        frames = frames[-20:]

                    # 如果当前轮结束（is_speaking=False），等待客户端关闭后再清理 session cache
                    # 这里不主动 close，让客户端逻辑决定；连接关闭会触发 ws_reset 进行最终清理

        if websocket.closed:
            await ws_reset(websocket, sid)
            if websocket in websocket_users:
                websocket_users.remove(websocket)
            logger.info(f"sid: {sid} 连接已关闭。Current websocket_users: {len(websocket_users)}")
    except Exception as e:
        logger.error(f"Exception: {e}")
        traceback.print_exc()


async def async_vad(websocket, audio_in, sid):
    """
    异步 VAD (语音活动检测) 处理函数。
    检测输入音频中是否包含人声，并返回语音片段的起止时间。
    
    Args:
        websocket: WebSocket 连接对象，包含 VAD 模型的状态字典
        audio_in (bytes): 原始音频字节流
        
    Returns:
        tuple (int, int): (speech_start, speech_end)
              -1 表示未检测到开始或结束。
    """
    # 将 Bytes 转为 Tensor
    audio_tensor = decode_audio_chunk(audio_in)
    # 异步并发调用 VAD 模型
    # 注意：这里我们依旧要遵守 Nano 模型的规则（虽然是 VAD，但保持输入格式一致比较安全），传入 list
    state = get_state(sid)
    segments_result_list = await run_model_inference(
        model_vad, input=[audio_tensor], **state["vad"]
    )
    if not segments_result_list or len(segments_result_list) == 0:
        return -1, -1
    segments_result = segments_result_list[0]["value"]
    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


async def async_asr(websocket, audio_in, sid):
    """
    异步离线 ASR (2pass-offline) 处理函数。
    对完整的语音片段进行高精度识别，通常在 VAD 检测到语音结束时调用。
    包含：ASR 识别 -> 标点恢复 (Punctuation Restoration) -> 发送最终结果 (is_final=True)。
    
    优化：优先使用 vLLM 进行推理，提升并发性能。
    
    Args:
        websocket: WebSocket 连接对象
        audio_in (bytes): 完整的语音片段字节流
    """
    # 离线识别 (最终修正)
    state = get_state(sid)
    if len(audio_in) > 0:
        audio_tensor = decode_audio_chunk(audio_in)
        
        # 优先使用 vLLM 进行推理（如果可用）
        if vllm_engine is not None and model_asr_nano is not None:
            try:
                rec_result_list = await async_asr_with_vllm(
                    audio_tensor, websocket, sid, **state["asr"]
                )
            except Exception as e:
                logger.warning(f"vLLM 推理失败，回退到传统方式: {e}")
                # 回退到传统推理方式
                rec_result_list = await run_model_inference(
                    model_asr_nano, input=[audio_tensor], **state["asr"]
                )
        else:
            # 使用传统推理方式
            rec_result_list = await run_model_inference(
                model_asr_nano, input=[audio_tensor], **state["asr"]
            )
        
        if not rec_result_list or len(rec_result_list) == 0:
           # 如果为空，直接返回空文本
           rec_result = {"text": ""}
        else:
           rec_result = rec_result_list[0]
        
        text = rec_result['text']
        if len(text)>0:
            logger.info(f"2pass-offline, sid={sid}, name={state['wav_name']}: {text}" )
        # 标点恢复
        if model_punc is not None and len(rec_result["text"]) > 0:
            # 异步并发调用标点模型
            punc_result_list = await run_model_inference(
                model_punc, input=rec_result["text"], **state["punc"]
            )
            if punc_result_list and len(punc_result_list) > 0:
                rec_result = punc_result_list[0]
        
        # 始终发送结果，即使为空，否则客户端会一直等待直到超时
        mode = "2pass-offline" if "2pass" in state["mode"] else state["mode"]
        message = json.dumps(
            {
                "mode": mode,
                "text": rec_result["text"],
                "wav_name": state["wav_name"],
                "is_final": not state["is_speaking"],
            }
        )
        try:
            async with websocket.send_lock:
                await websocket.send(message)
        except Exception as e:
            # 客户端断开，安全忽略
            logger.error(f"Client disconnected during async_asr send: {e}")
    else:
        # Empty audio result
        mode = "2pass-offline" if "2pass" in state["mode"] else state["mode"]
        message = json.dumps(
            {
                "mode": mode,
                "text": "",
                "wav_name": state["wav_name"],
                "is_final": not state["is_speaking"],
            }
        )
        try:
            async with websocket.send_lock:
                await websocket.send(message)
        except Exception as e:
            logger.error(f"Client disconnected during async_asr empty send: {e}")


async def async_asr_online(websocket, audio_in, sid):
    """
    异步在线流式 ASR (Online Streaming) 处理函数。
    对实时到达的音频流进行增量识别，返回中间结果 (is_final=False)。
    
    优化：使用异步推理，减少阻塞，提升并发能力。
    
    Args:
        websocket: WebSocket 连接对象
        audio_in (bytes): 实时音频流片段
    """
    # 在线流式识别
    # 注意：流式 ASR 通常不使用 vLLM，因为需要实时性
    # vLLM 更适合离线/2pass ASR 的批量处理
    state = get_state(sid)
    if len(audio_in) > 0:
        audio_tensor = decode_audio_chunk(audio_in)
        # 异步并发调用流式 ASR 模型
        # 使用 run_model_inference 确保不阻塞事件循环
        try:
            rec_result_list = await run_model_inference(
                 model_asr_streaming, input=[audio_tensor], **state["asr_online"]
            )
        except Exception as e:
            logger.error(f"流式 ASR 推理错误: {e}")
            return
        
        if not rec_result_list or len(rec_result_list) == 0:
             return
        rec_result = rec_result_list[0]
        text = rec_result['text']
        if len(text)>0:
            logger.info(f"2pass-online: {text}" )
        if state["mode"] == "2pass" and state["asr_online"].get("is_final", False):
            return

        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in state["mode"] else state["mode"]
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": state["wav_name"],
                    "is_final": not state["is_speaking"],
                }
            )
            try:
                await websocket.send(message)
            except Exception as e:
                logger.error(f"Client disconnected during async_asr_online send: {e}")
    else:
        # Empty audio result
        mode = "2pass-online" if "2pass" in state["mode"] else state["mode"]
        message = json.dumps(
            {
                "mode": mode,
                "text": "",
                "wav_name": state["wav_name"],
                "is_final": not state["is_speaking"],
            }
        )
        try:
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Client disconnected during async_asr empty send: {e}")

async def main(): 
    global model_asr_nano, model_asr_streaming, model_vad, model_punc, asr_tokenizer, asr_frontend
    logger.info("Load model...")
    # ASR 模型 (离线/2pass + 在线/流式)
    logger.info(f"Load ASR Model: {args.asr_model} ...")
    if args.vllm_model_dir:
        model_asr_nano, kwargs_asr = AutoModel.build_model(
            model=args.asr_model, 
            trust_remote_code=True, 
            # remote_code="./models/fun_asr_nano.py",
            device=args.device
        )
        asr_tokenizer, asr_frontend = kwargs_asr["tokenizer"], kwargs_asr["frontend"]
    else:
        model_asr_nano = AutoModel(
            model=args.asr_model,
            model_revision=args.asr_model_revision,
            ngpu=args.ngpu,
            ncpu=args.ncpu,
            device=args.device,
            disable_pbar=True,
            disable_log=True,
            fp16=args.fp16,
        )

    # ASR Streaming 模型
    logger.info(f"Load ASR Online Model: {args.asr_model_online} ...")
    model_asr_streaming = AutoModel(
        model=args.asr_model_online, model_revision=args.asr_model_online_revision)

    # VAD 模型
    logger.info(f"Load VAD Model: {args.vad_model} ...")
    model_vad = AutoModel(
        model=args.vad_model,
        model_revision=args.vad_model_revision,
        trust_remote_code=True, 
        remote_code="./models/fsmn_vad_streaming.py",
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
        fp16=args.fp16,
    )

    # 标点模型
    if args.punc_model != "":
        logger.info(f"Load Punc Model: {args.punc_model} ...")
        model_punc = AutoModel(
            model=args.punc_model,
            model_revision=args.punc_model_revision,
            ngpu=args.ngpu,
            ncpu=args.ncpu,
            device=args.device,
            disable_pbar=True,
            disable_log=True,
            fp16=args.fp16,
        )
    else:
        model_punc = None

    logger.info("Load model finished.")  

    if args.vllm_model_dir:
        global vllm_engine, vllm_sampling_params, prompt_prefix_embeddings, prompt_suffix_embeddings
        logger.info(f"Initializing vLLM engine from {args.vllm_model_dir}...")
        try:            
            os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
            from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
            from vllm.config import CompilationConfig  
            
            if asr_tokenizer is None or asr_frontend is None:
                logger.warning("无法获取 tokenizer 或 frontend，vLLM 功能可能不可用")
            else:
                # 初始化 vLLM
                os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
                os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                cudagraph_sizes = [x for x in range(1, args.vllm_max_num_seqs + 1)]

                engine_args = AsyncEngineArgs(
                    model=args.vllm_model_dir,
                    enable_prompt_embeds=True,
                    disable_custom_all_reduce=True,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    dtype="bfloat16",
                    compilation_config=CompilationConfig(
                        cudagraph_capture_sizes=cudagraph_sizes),
                    tensor_parallel_size=1,
                    max_num_seqs=args.vllm_max_num_seqs,
                    max_model_len=2048,               # 限制单个序列长度
                    max_num_batched_tokens=1024,      # 减少批处理token数
                    trust_remote_code=True,
                )
                vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)      
                vllm_sampling_params = SamplingParams(
                    top_p=0.001,
                    max_tokens=500,
                )
                    
                # 准备 prompt embeddings
                instruction = "语音转写："
                prompt_prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}"
                prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
                prompt_prefix_ids = asr_tokenizer.encode(prompt_prefix)
                prompt_suffix_ids = asr_tokenizer.encode(prompt_suffix)
                prompt_prefix_ids = torch.tensor(prompt_prefix_ids, dtype=torch.int64).to(args.device)
                prompt_suffix_ids = torch.tensor(prompt_suffix_ids, dtype=torch.int64).to(args.device)
                
                prompt_prefix_embeddings = model_asr_nano.llm.model.get_input_embeddings()(prompt_prefix_ids)
                prompt_suffix_embeddings = model_asr_nano.llm.model.get_input_embeddings()(prompt_suffix_ids)
                
                logger.info("vLLM engine initialized successfully")
        except ImportError:
            logger.warning("vLLM 未安装，将使用传统推理方式。安装命令: pip install vllm")
        except Exception as e:
            logger.error(f"初始化 vLLM 失败: {e}")
            traceback.print_exc()
   
    if len(args.certfile) > 0:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_cert = args.certfile
        ssl_key = args.keyfile
        ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
        start_server = websockets.serve(
            ws_serve, args.host, args.port, subprotocols=None, ping_interval=None, ssl=ssl_context
        )
    else:
        start_server = websockets.serve(
            ws_serve, args.host, args.port, subprotocols=None, ping_interval=None
        )
    logger.info(f"服务已启动，监听地址: ws://{args.host}:{args.port}")
    await start_server
    await asyncio.get_event_loop().create_future() # 永久运行


if __name__ == "__main__":
    asyncio.run(main())