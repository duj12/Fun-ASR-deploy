import sys
from funasr.models.fun_asr_nano.model import FunASRNano
from funasr import AutoModel

model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)
wav_path = sys.argv[1]
res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
for bid, res_item in enumerate(res):
    text = res[bid]["text"]
    print(f"{bid}: {text}")
