import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 配置参数
input_file = "combined.wav"  # 你的长音频文件名
output_folder = "./split_wavs"      # 输出文件夹
min_silence_len = 1000              # 最小静音长度 (毫秒)，用于判定切割点 (对应 1.5 秒停顿，设 1000-1200 比较安全)
silence_thresh = -40                # 静音阈值 (dBFS)，低于这个值算静音，根据音量调整

# 创建输出目录
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"正在加载音频：{input_file} ...")
sound = AudioSegment.from_wav(input_file)

print("正在检测静音并切割...")
# split_on_silence 会返回一个音频片段列表
chunks = split_on_silence(
    sound,
    min_silence_len=min_silence_len,
    silence_thresh=silence_thresh,
    keep_silence=500  # 每个片段前后保留 500ms 的声音，防止切掉尾音
)

print(f"检测到 {len(chunks)} 个片段。开始保存...")

for i, chunk in enumerate(chunks):
    # 文件名格式：01.wav, 02.wav ...
    filename = f"{str(i+1).zfill(2)}.wav"
    filepath = os.path.join(output_folder, filename)
    
    # 导出
    chunk.export(filepath, format="wav")
    print(f"已保存：{filename} (时长：{len(chunk)/1000:.2f}秒)")

print("全部完成！")
