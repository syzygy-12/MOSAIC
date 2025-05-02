import subprocess
import os
import io
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment, silence
import noisereduce as nr
import webrtcvad
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
import whisper
import json
from pathlib import Path

RAW_AUDIO_DIR = "../data/raw_audio"
PROCESSED_AUDIO_DIR = "../data/processed_audio"

class ProcessedAudio:
    def __init__(self, name: str, audio: np.ndarray = None, frame_index: list[int] = None):
        """
        name: 音频文件名或路径（用于标识）
        audio: 预处理后的音频样本（1D numpy array）
        frame_index: 每20ms对应原始帧的下标（用于回溯）
        """
        self.name = name
        self.audio = audio
        self.frame_index = frame_index
        self.vec = None
        self.tokens = None

    def from_file(self, input_path: str, name: str) -> 'ProcessedAudio':
        """
        从原始媒体文件中创建 ProcessedAudio（自动格式转化+重采样，无需中间文件）
        """
        ffmpeg_cmd = [
            "ffmpeg", "-i", input_path,
            "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            "-vn", "-hide_banner", "-loglevel", "error", "-"
        ]

        try:
            proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            audio_bytes = io.BytesIO(proc.stdout)
            y, sr = sf.read(audio_bytes)
            if sr != 16000:
                raise ValueError(f"[ERROR] 采样率错误: 期望 16000 Hz，实际 {sr} Hz")
            if len(y) == 0:
                raise ValueError("[ERROR] 音频长度为0，跳过")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[FFMPEG ERROR] 无法处理文件 {input_path}:\n{e.stderr.decode(errors='ignore')}") from e
        except Exception as e:
            raise RuntimeError(f"[ERROR] 无法读取音频数据：{input_path}\n{str(e)}") from e

        frame_index = list(range(len(y) // int(16000 * 0.02)))  # 每20ms一帧
        print(f"[INFO] 读取音频 {name} 成功，采样率: {sr} Hz, 长度: {len(y) / sr:.2f}秒")
        self.audio = y
        self.name = name
        self.frame_index = frame_index
        return self
    
    def load_audio(self, name: str = "") -> 'ProcessedAudio':
        if self.name and self.name != "":
            name = self.name
        os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
        for fname in os.listdir(RAW_AUDIO_DIR):
            if not fname.lower().endswith(('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg')) or name != os.path.splitext(fname)[0]:
                continue
            input_path = os.path.join(RAW_AUDIO_DIR, fname)
            print(f"[INFO] Loading {name} from {input_path}...")
            return self.from_file(input_path, name)
        raise FileNotFoundError(f"[ERROR] 找不到音频文件 {name} 在 {RAW_AUDIO_DIR}")

    def denoise(self) -> 'ProcessedAudio':
        """ 应用噪声抑制 """
        self.audio = nr.reduce_noise(y=self.audio, sr=16000)
        return self

    def remove_silence_vad(self, aggressiveness=2, frame_duration_ms=20) -> 'ProcessedAudio':
        vad = webrtcvad.Vad(aggressiveness)
        samples = (self.audio * 32768).astype(np.int16)
        bytes_audio = samples.tobytes()

        frame_length = int(16000 * frame_duration_ms / 1000)  # 每帧多少采样点
        new_audio = []
        new_frame_index = []

        for i in range(0, len(samples), frame_length):
            start = i
            end = i + frame_length
            frame = bytes_audio[start * 2: end * 2]
            if len(frame) < frame_length * 2:
                break
            if vad.is_speech(frame, sample_rate=16000):
                # 添加音频
                new_audio.extend(samples[start:end])
                # 添加帧索引：一个20ms帧可能包含多个帧起点
                num_subframes = frame_length // int(16000 * 0.02)  # 理论上应该是1
                for j in range(num_subframes):
                    idx = start + j * int(16000 * 0.02)
                    new_frame_index.append(self.frame_index[idx // int(16000 * 0.02)])
        self.audio = np.array(new_audio).astype(np.float32) / 32768.0
        # 归一化响度
        self.audio = librosa.util.normalize(self.audio)
        self.frame_index = new_frame_index
        return self
    
    def pre_process(self) -> 'ProcessedAudio':
        """
        预处理音频：去噪、去静音、重采样
        """
        self.denoise()
        self.remove_silence_vad()
        # 保存
        os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
        output_path = os.path.join(PROCESSED_AUDIO_DIR, f"{self.name}.wav")
        sf.write(output_path, self.audio, 16000, format="WAV", subtype="PCM_16")
        print(f"[INFO] 预处理完成，保存到 {output_path}")
        return self

    def extract_feature(self, model_name: str = "facebook/wav2vec2-large-xlsr-53") -> 'ProcessedAudio':
        """
        提取 wav2vec2 特征（按15秒分段后拼接），若已有缓存则直接加载
        """
        feature_dir = "../data/feature"
        os.makedirs(feature_dir, exist_ok=True)
        feature_path = os.path.join(feature_dir, f"{self.name}.npy")

        if os.path.exists(feature_path):
            print(f"[INFO] Feature already exists for {self.name}, loading from {feature_path}")
            self.feature = np.load(feature_path)
            return self

        print(f"[INFO] Extracting features with {model_name} for {self.name}...")

        # 初始化模型和处理器（静态缓存防止重复加载）
        if not hasattr(self.__class__, "_wav2vec2_extractor"):
            self.__class__._wav2vec2_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.__class__._wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")

        extractor = self.__class__._wav2vec2_extractor
        model = self.__class__._wav2vec2_model

        # 分段：15s -> 240000 samples
        sr = 16000
        segment_len = sr * 15
        features = []

        for i in range(0, len(self.audio), segment_len):
            segment = self.audio[i:i + segment_len + 320]  # 额外加一帧（20ms = 320采样点）
            if len(segment) < 1000:
                continue

            inputs = extractor(segment, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(model.device)

            with torch.no_grad():
                outputs = model(input_values, output_hidden_states=True)
                hidden_states = outputs.hidden_states[4].squeeze(0).cpu().numpy()
                features.append(hidden_states)

        if not features:
            raise ValueError(f"[ERROR] No valid audio segments found for {self.name}")

        self.feature = np.concatenate(features, axis=0)
        np.save(feature_path, self.feature)
        print(f"[OK] Feature saved to {feature_path} with shape {self.feature.shape}")

        return self
    
    def sec_to_frame(t):  # 20ms 一帧
        return int(round(t * 50))

    def tokenize_with_whisper(self, model_size: str = "medium") -> 'ProcessedAudio':
        """
        使用 Whisper 对音频进行语音识别和分词，按帧编号（20ms/frame）保存。
        如果 token 已存在于 ../data/token 中则直接读取。
        需要修改，现在会把中文按照词粒度分词，会有两个字的短语，后期可能用经典方法修改
        """

        token_path = Path(f"../data/token/{self.name}.json")
        os.makedirs(token_path.parent, exist_ok=True)

        # ========== 1. 如果已有分词结果就直接读取 ==========
        if token_path.exists():
            with open(token_path, "r", encoding="utf-8") as f:
                self.tokens = json.load(f)
            print(f"[INFO] Loaded cached tokens from {token_path}")
            return self

        print(f"[INFO] Tokenizing {self.name} using Whisper ({model_size})...")

        # ========== 2. 加载 Whisper 模型 ==========
        if not hasattr(self.__class__, "_whisper_model"):
            self.__class__._whisper_model = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
        model = self.__class__._whisper_model

        result = model.transcribe(self.audio, word_timestamps=True, verbose=False)

        if "segments" not in result or not result["segments"]:
            raise ValueError(f"[ERROR] Whisper 没有在 {self.name} 中检测到有效语音")

        # ========== 3. 将时间（秒）映射为帧编号 ==========
        tokens = []
        for segment in result["segments"]:
            for word in segment.get("words", []):
                #print(word)
                tokens.append({
                    "start": int(max(0, round(word["start"] * 50))),
                    "end": int(round(word["end"] * 50)),
                    # 这里分词会有一个很明显的尾音，把下一个字的音头剪进来了
                    "text": word["word"].strip()
                })

        # ========== 4. 保存并赋值 ==========
        with open(token_path, "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)

        self.tokens = tokens
        print(f"[OK] Tokenized {self.name} into {len(tokens)} words. Saved to {token_path}")
        return self
    
    def tokenize(self, granularity: str = "word") -> 'ProcessedAudio':
        """
        不同粒度的分词，待实现，目前只支持 word 粒度
        granularity: "word" or "syllable"
        """
        if granularity == "word":
            return self.tokenize_with_whisper()
        else:
            raise NotImplementedError(f"[ERROR] Tokenization for '{granularity}' is not implemented yet.")
        return self


        
