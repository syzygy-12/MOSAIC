import subprocess

def m4a_to_wav_ffmpeg(input_m4a, output_wav, sample_rate=16000, channels=1):
    """
    使用 ffmpeg 将 M4A 转换为 WAV
    """
    command = [
        "ffmpeg",
        "-i", input_m4a,
        "-ac", str(channels),      # 声道数
        "-ar", str(sample_rate),   # 采样率
        "-vn",                     # 忽略视频（纯音频）
        output_wav
    ]
    subprocess.run(command, check=True)
    print(f"转换完成：{output_wav}")

# 示例用法
if __name__ == "__main__":
    input_m4a = "../data/raw_audio/deep.m4a"
    output_wav = "../data/raw_audio/deep.wav"
    m4a_to_wav_ffmpeg(input_m4a, output_wav)