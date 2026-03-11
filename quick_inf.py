import torch
from qwen_asr import Qwen3ASRModel
import os

INPUT_DIR = "res_testSet"
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
print(f"发现 {len(files)} 个音频文件，开始预测...")
wav_files = []

for i, filename in enumerate(files):
    input_path = os.path.join(INPUT_DIR, filename)
    wav_files.append(input_path)

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    #"./qwen3-asr-lora-chinese-1.7B/checkpoint-114",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    #audio=["testSet/test3.wav","testSet/test2.wav","testSet/test4.wav"]
    audio = wav_files
)
for r in results:
    print(r.text)
