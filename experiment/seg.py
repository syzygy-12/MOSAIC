import stable_whisper
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
        
        if max_duration and len(audio) > target_sr * max_duration:
            audio = audio[:target_sr * max_duration]
            print(f"警告：音频超过{max_duration}秒，已截取前{max_duration}秒")
            
        return audio
    except Exception as e:
        raise RuntimeError(f"音频加载失败: {str(e)}")

def transcribe_with_boundaries(
    model,  # 移除了类型注解，或改用 stable_whisper.WhisperModel
    audio: np.ndarray,
    language: Optional[str] = None,
    verbose: bool = False
    ) -> Dict:
    """带字级边界检测的语音转录"""
    result = model.transcribe(
        audio,
        language=language,
        verbose=verbose,
        word_timestamps=True  # 启用字级时间戳
    )
    return result.to_dict()  # 转换为字典格式

def print_results(result: Dict):
    """结构化打印结果"""
    for seg in result["segments"]:
        print(f"\n[Segment {seg['id']+1}] {seg['start']:.2f}s -> {seg['end']:.2f}s")
        print(f"Text: {seg['text'].strip()}")
        
        if "words" in seg:
            print("Character-level boundaries:")
            for word in seg["words"]:
                print(f"  {word['start']:.2f}s - {word['end']:.2f}s | {word['word'].strip()}")

if __name__ == "__main__":
    AUDIO_PATH = "../data/processed_audio/deep.wav"  # 替换为实际音频路径
    MODEL_SIZE = "medium"  # tiny/base/small/medium/large
    MAX_DURATION = 180

    print(f"加载 stable-whisper {MODEL_SIZE} 模型...")
    model = stable_whisper.load_model(MODEL_SIZE)  # 正确加载方式
    
    print("处理音频文件...")
    audio = load_and_preprocess_audio(AUDIO_PATH, max_duration=MAX_DURATION)
    
    print("开始语音识别与字级边界检测...")
    result = transcribe_with_boundaries(model, audio, language="zh")
    
    print("\n" + "="*50)
    print(result)
    #print_results(result)
    print("="*50)
    
    print(f"\n处理完成！总时长: {len(audio)/16000:.2f}秒")