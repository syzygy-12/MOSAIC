import os
import librosa
import soundfile as sf
import moviepy
print(moviepy.__file__)
from moviepy.editor import VideoFileClip

def extract_audio(video_path: str, output_wav_path: str, target_sr: int = 16000):
    print(f"🔍 Loading video from: {video_path}")
    
    # Step 1: 用 moviepy 读取音频
    video = VideoFileClip(video_path)
    audio = video.audio
    temp_wav_path = "temp_audio.wav"
    audio.write_audiofile(temp_wav_path, fps=44100, codec='pcm_s16le', verbose=False, logger=None)

    # Step 2: 用 librosa 加载并转换采样率 + 单声道
    y, sr = librosa.load(temp_wav_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Step 3: 保存为目标 wav 文件
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sf.write(output_wav_path, y, sr)
    print(f"✅ Saved extracted audio to: {output_wav_path}")

    # 清理中间文件
    os.remove(temp_wav_path)

# Example usage
if __name__ == "__main__":
    video_path = "../data/raw_video/qimeidi.mp4"
    output_audio_path = "../data/raw_audio/qimeidi.wav"
    extract_audio(video_path, output_audio_path)
