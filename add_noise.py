from math import log10  # <--- 加上这个
import os
import random
from pydub import AudioSegment
from pydub.utils import make_chunks

# ================= 配置区域 =================
INPUT_DIR = "./split_wavs"       # 你原始干净的 76 个音频所在文件夹
OUTPUT_DIR = "./augmented_data"  # 输出增强后音频的文件夹
NOISE_FILE = "tower_noise.wav"   # <--- 请修改为你的噪音文件名！

# 增强参数
MIN_SNR = 15                     # 最小信噪比 (dB)，数值越小噪音越大
MAX_SNR = 30                     # 最大信噪比 (dB)，数值越大越干净
MIN_SILENCE_MS = 200             # 开头最少加多少毫秒静音
MAX_SILENCE_MS = 800             # 开头最多加多少毫秒静音
VOL_CHANGE_MIN = -3              # 音量最小变化 dB
VOL_CHANGE_MAX = 2               # 音量最大变化 dB
# ===========================================

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载噪音文件
if not os.path.exists(NOISE_FILE):
    print(f"❌ 错误：找不到噪音文件 {NOISE_FILE}")
    exit(1)

print(f"正在加载噪音文件：{NOISE_FILE} ...")
noise = AudioSegment.from_wav(NOISE_FILE)
# 如果噪音太短，循环它直到足够长（可选，通常塔台噪音很长，不需要）
if len(noise) < 60000: # 小于 60 秒
    noise = noise * 10 

print(f"噪音文件长度：{len(noise)/1000:.2f} 秒")

# 获取所有 wav 文件
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')])
print(f"发现 {len(files)} 个音频文件，开始处理...")

for i, filename in enumerate(files):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        # 1. 加载原始音频
        audio = AudioSegment.from_wav(input_path)
        
        # 2. [可选] 随机调整音量 (模拟距离远近)
        gain = random.uniform(VOL_CHANGE_MIN, VOL_CHANGE_MAX)
        audio = audio.apply_gain(gain)
        
        # 3. 随机截取一段噪音 (长度与音频一致)
        start_idx = random.randint(0, max(0, len(noise) - len(audio) - 1))
        noise_clip = noise[start_idx : start_idx + len(audio)]
        
        # 4. 计算并应用随机信噪比 (SNR)
        # 公式：SNR = 10 * log10(P_signal / P_noise)
        # 我们需要调整 noise_clip 的音量来达到目标 SNR
        target_snr = random.uniform(MIN_SNR, MAX_SNR)
        
        # 计算信号和噪音的功率 (RMS)
        audio_rms = audio.rms
        noise_rms = noise_clip.rms
        
        if audio_rms == 0:
            continue # 跳过空音频
            
        # 计算需要的噪音 RMS
        # target_snr = 10 * log10(audio_rms^2 / target_noise_rms^2)
        # => target_noise_rms = audio_rms / (10 ** (target_snr / 20))
        target_noise_rms = audio_rms / (10 ** (target_snr / 20))
        
        # 计算增益因子
        if noise_rms > 0:
            gain_factor = target_noise_rms / noise_rms
            # 转换为 dB 增益
            noise_gain_db = 20 * (log10(gain_factor) if gain_factor > 0 else -100)
            noise_clip = noise_clip.apply_gain(noise_gain_db)
        else:
            noise_clip = noise_clip.apply_gain(-100) # 静音
            
        # 5. 混合音频
        mixed_audio = audio.overlay(noise_clip)
        
        # 6. [关键] 前后随机添加静音 (打破时间对齐规律)
        silence_front = AudioSegment.silent(duration=random.randint(MIN_SILENCE_MS, MAX_SILENCE_MS))
        silence_back = AudioSegment.silent(duration=random.randint(100, 300))
        final_audio = silence_front + mixed_audio + silence_back
        
        # 7. 导出
        final_audio.export(output_path, format="wav")
        
        if (i + 1) % 10 == 0:
            print(f"已处理 {i+1}/{len(files)} ...")
            
    except Exception as e:
        print(f"❌ 处理 {filename} 时出错：{e}")

print("✅ 所有音频增强完成！")
print(f"新数据位于：{OUTPUT_DIR}")

