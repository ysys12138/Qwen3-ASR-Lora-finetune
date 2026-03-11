import os
from pydub import AudioSegment

# ================= 配置区域 =================
INPUT_DIR = "./testSet"   # <--- 你当前 48k 音频所在的文件夹
OUTPUT_DIR = "./res_testSet"   # <--- 输出 16k 音频的新文件夹
TARGET_SR = 16000                # 目标采样率
# ===========================================

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取所有 wav 文件
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
print(f"发现 {len(files)} 个音频文件，开始降采样处理...")

success_count = 0
fail_count = 0

for i, filename in enumerate(files):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        # 1. 加载音频 (pydub 会自动检测原始采样率)
        audio = AudioSegment.from_wav(input_path)
        
        # 检查原始采样率
        original_sr = audio.frame_rate
        if original_sr == TARGET_SR:
            print(f"⚠️ 跳过 {filename}: 已经是 {original_sr} Hz")
            # 即使跳过，也复制一份到输出目录保持一致性
            audio.export(output_path, format="wav")
            success_count += 1
            continue
        
        print(f"[{i+1}/{len(files)}] 处理 {filename}: {original_sr} Hz -> {TARGET_SR} Hz")
        
        # 2. 降采样 (set_frame_rate 会重采样)
        audio_resampled = audio.set_frame_rate(TARGET_SR)
        
        # 3. 确保单声道 (ASR 通常只需要单声道，双声道转单声道更安全)
        if audio_resampled.channels == 2:
            audio_resampled = audio_resampled.set_channels(1)
            
        # 4. 导出
        # 参数说明：parameters=["-acodec", "pcm_s16le"] 确保保存为标准的 16bit PCM WAV
        audio_resampled.export(
            output_path, 
            format="wav",
            parameters=["-acodec", "pcm_s16le"]
        )
        
        success_count += 1
        
    except Exception as e:
        print(f"❌ 处理 {filename} 失败：{e}")
        fail_count += 1

print("\n" + "="*30)
print(f"✅ 处理完成！")
print(f"成功：{success_count} 个")
print(f"失败：{fail_count} 个")
print(f"新数据位于：{os.path.abspath(OUTPUT_DIR)}")
