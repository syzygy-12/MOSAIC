import os
import librosa
import soundfile as sf
import subprocess

def extract_audio(video_path: str, output_wav_path: str, target_sr: int = 16000):
    print(f"🔍 Loading video from: {video_path}")
    
    # Step 1: 用 ffmpeg 直接提取音频
    temp_wav_path = "temp_audio.wav"
    
    try:
        # 使用ffmpeg命令行工具提取音频
        cmd = [
            "ffmpeg", "-i", video_path, 
            "-ac", "1",  # 转为单声道
            "-ar", "44100",  # 采样率
            "-y",  # 覆盖输出文件
            temp_wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg错误: {result.stderr}")
            return
            
    except FileNotFoundError:
        print("❌ 未找到ffmpeg，请安装ffmpeg")
        print("运行: sudo apt-get install ffmpeg  或  conda install ffmpeg")
        return

    # Step 2: 用 librosa 加载并转换采样率
    y, sr = librosa.load(temp_wav_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Step 3: 保存为目标 wav 文件
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sf.write(output_wav_path, y, sr)
    print(f"✅ Saved extracted audio to: {output_wav_path}")

    # 清理中间文件
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

# Example usage
if __name__ == "__main__":
    video_path = "../data/raw_video/output.mp4"
    output_audio_path = "../data/raw_audio/output.wav"
    extract_audio(video_path, output_audio_path)