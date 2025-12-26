import os
import torch
from funasr.models.fun_asr_nano.model import FunASRNano

# === 参数设置 ===
# 是否使用 FP16 (半精度)
# 默认精度为 FP32 (float32)。设置为 True 可节省约 50% 显存。
# 如果想用默认的 FP32，将此处改为 False 即可，无需注释下面的代码
USE_FP16 = False 

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
    return "N/A"

print("=== Fun-ASR 推理测试 ===")
print(f"Initial VRAM: {get_vram_usage()}")

# 自动检测 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# 模型 ID (会自动去 modelscope cache 找，或者下载)
MODEL_ID = 'FunAudioLLM/Fun-ASR-Nano-2512'

# 指定本地缓存目录 (与 download_model.py 保持一致)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(CURRENT_DIR, 'models')

# 构建本地模型路径 (确保之前 download_model.py 已执行成功)
# 结构: models/FunAudioLLM/Fun-ASR-Nano-2512
local_model_path = os.path.join(MODELS_ROOT, MODEL_ID)

print(f"Loading model from local path: {local_model_path}")

print(f"FPS16: {USE_FP16}")

try:
    from funasr import AutoModel
    from funasr.models.fun_asr_nano.model import FunASRNano # Manual import to register model
    # 加载模型
    # 直接指定本地绝对路径，避免 FunASR 尝试去 modelscope 下载
    model = AutoModel(
        model=local_model_path,
        trust_remote_code=True,
        fp16=USE_FP16
    )
    
    print(f"Model loaded. VRAM: {get_vram_usage()}")

    # 测试音频 (使用阿里云的示例音频)
    audio_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
    print(f"\nTranscribing: {audio_file}")

    # 1. Download/Read as bytes to simulate server receiving bytes
    import requests
    audio_bytes = requests.get(audio_file).content
    print(f"Audio bytes len: {len(audio_bytes)}")

    print("\n=== Testing with raw bytes input (Should Fail) ===")
    try:
        res = model.generate(input=audio_bytes, batch_size_s=300)
        print(res)
    except Exception as e:
        print(f"Failed with bytes: {e}")

    print("\n=== Testing with decoded numpy array (Solution) ===")
    try:
        import numpy as np
        
        # Simplified load_bytes logic from official load_utils.py
        # Assuming input is valid PCM 16k 16bit mono for now (or wav bytes)
        def simple_load_bytes(input_bytes):
            # If it's a WAV file with header (like from URL), we must skip header or use soundfile
            # But here let's try assuming it's just raw PCM for the "streaming" simulation part
            # Or use soundfile for the file part.
            pass

        import io
        import soundfile as sf
        # 1. Decode properly to float32 numpy
        audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
        audio_data = audio_data.astype(np.float32)
        print(f"Decoded shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        
        # 2. Wrap in list!
        print("Input: [audio_data]")
        audio_data = torch.from_numpy(audio_data)
        res = model.generate(input=[audio_data], batch_size_s=300)
        print(res)
    except Exception as e:
        print(f"Failed with numpy list: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Failed with bytes: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Recognition Result (BYTES TEST) ===")
    print(f"Final VRAM: {get_vram_usage()}")

except Exception as e:
    print(f"[Error] An error occurred: {e}")
    # 打印更详细的错误可能有助于排查 OOM
    import traceback
    traceback.print_exc()
