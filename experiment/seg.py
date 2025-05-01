import whisper
import librosa
import numpy as np
from typing import Optional, Dict, List

def load_and_preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    max_duration: Optional[float] = None
) -> np.ndarray:
    """加载并预处理音频文件"""
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # 自动裁剪静音部分（可选）
        # audio = librosa.effects.trim(audio, top_db=20)[0]
        
        if max_duration and len(audio) > target_sr * max_duration:
            audio = audio[:target_sr * max_duration]
            print(f"警告：音频超过{max_duration}秒，已截取前{max_duration}秒")
            
        return audio
    except Exception as e:
        raise RuntimeError(f"音频加载失败: {str(e)}")

def transcribe_with_boundaries(
    model: whisper.Whisper,
    audio: np.ndarray,
    language: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """带边界检测的语音转录"""
    try:
        result = model.transcribe(
            audio,
            word_timestamps=True,
            language=language,
            verbose=verbose
        )
        
        # 后处理：合并连续的空格单词
        for segment in result["segments"]:
            if "words" in segment:
                merged_words = []
                for word in segment["words"]:
                    if merged_words and word["word"].strip() == "":
                        merged_words[-1]["end"] = word["end"]
                    else:
                        merged_words.append(word)
                segment["words"] = merged_words
                
        return result
    except Exception as e:
        raise RuntimeError(f"转录失败: {str(e)}")

def print_results(result: Dict):
    """结构化打印结果"""
    for seg in result["segments"]:
        print(f"\n[Segment {seg['id']+1}] {seg['start']:.2f}s -> {seg['end']:.2f}s")
        print(f"Text: {seg['text'].strip()}")
        
        if "words" in seg:
            print("Word-level boundaries:")
            for word in seg["words"]:
                print(f"  {word['start']:.2f}s - {word['end']:.2f}s | {word['word'].strip()}")

if __name__ == "__main__":
    # 配置参数
    AUDIO_PATH = "../data/processed_clean/trump_clean.wav"
    MODEL_SIZE = "medium"  # small/medium/large
    MAX_DURATION = 180  # 最长处理时长（秒）
    
    try:
        # 1. 加载模型
        print(f"加载Whisper {MODEL_SIZE}模型...")
        model = whisper.load_model(MODEL_SIZE)
        
        # 2. 加载并预处理音频
        print("处理音频文件...")
        audio = load_and_preprocess_audio(AUDIO_PATH, max_duration=MAX_DURATION)
        
        # 3. 执行带边界检测的转录
        print("开始语音识别与边界检测...")
        result = transcribe_with_boundaries(model, audio)
        
        # 4. 输出结果
        print("\n" + "="*50)
        print_results(result)
        print("="*50)
        
        print(f"\n处理完成！总时长: {len(audio)/16000:.2f}秒")
        
    except Exception as e:
        print(f"错误发生: {str(e)}")