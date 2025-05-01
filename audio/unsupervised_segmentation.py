import os
import librosa
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
from scipy.signal import find_peaks
from tqdm import tqdm
print(np.__version__)
print(np.arange(10))
print("🔊 Unsupervised Segmentation")
# Load pretrained multilingual Wav2Vec2

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_embeddings(audio: np.ndarray, sr=16000) -> np.ndarray:
    # Explicitly cast audio to float32 np.ndarray to avoid PyTorch/Numpy bugs
    audio = np.asarray(audio, dtype=np.float32)

    # Wrap in list → batch of 1
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state.squeeze(0)  # (T, D)

    return hidden_states.cpu().numpy()

def compute_change_signal(embeddings: np.ndarray) -> np.ndarray:
    """Compute L2 or cosine distance between consecutive frames"""
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    diffs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs) + 1e-6)  # normalize
    return diffs

def detect_boundaries(change_signal: np.ndarray, threshold=0.6, distance=15) -> np.ndarray:
    """Use peak detection to find segmentation boundaries"""
    peaks, _ = find_peaks(change_signal, height=threshold, distance=distance)
    return peaks

def segment_audio_by_boundaries(audio: np.ndarray, sr: int, boundaries: np.ndarray, frame_duration=0.02, output_dir="segments"):
    os.makedirs(output_dir, exist_ok=True)
    frame_samples = int(sr * frame_duration)

    time_boundaries = (boundaries * frame_samples).astype(int)
    time_boundaries = np.concatenate(([0], time_boundaries, [len(audio)]))

    print(f"🔪 Cutting into {len(time_boundaries)-1} segments")

    for i in range(len(time_boundaries) - 1):
        start = time_boundaries[i]
        end = time_boundaries[i + 1]
        segment = audio[start:end]
        if len(segment) > sr * 0.2:  # at least 200ms
            sf.write(os.path.join(output_dir, f"seg_{i:03d}.wav"), segment, sr)

def plot_segmentation(audio: np.ndarray, change_signal: np.ndarray, boundaries: np.ndarray, sr=16000, frame_duration=0.02):
    plt.figure(figsize=(14, 6))
    times = np.linspace(0, len(audio) / sr, num=len(change_signal))

    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, label="Waveform", alpha=0.7)
    plt.vlines(boundaries * frame_duration, ymin=-1, ymax=1, color='r', linestyle='--', label="Detected Boundaries")
    plt.title("Waveform with Segmentation Boundaries")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(times, change_signal, label="Embedding Jump Signal", color='g')
    plt.vlines(boundaries * frame_duration, ymin=0, ymax=1, color='r', linestyle='--')
    plt.title("Change Signal from Embedding (L2 distance)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def run_unsupervised_segmentation(audio_path: str, output_dir: str):
    print(f"🚀 Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)

    print("🔍 Extracting embeddings...")
    embeddings = extract_embeddings(y, sr)
    print(f"📐 Embedding shape: {embeddings.shape}")

    print("🧠 Computing change signal...")
    change_signal = compute_change_signal(embeddings)

    print("📌 Detecting boundaries...")
    boundaries = detect_boundaries(change_signal, threshold=0.6, distance=15)
    print(f"🧩 Found {len(boundaries)} boundaries")

    print("💾 Saving segments...")
    segment_audio_by_boundaries(y, sr, boundaries, output_dir=output_dir)

    print("📊 Plotting results...")
    plot_segmentation(y, change_signal, boundaries, sr=sr)

# Example usage
if __name__ == "__main__":
    print("🔊 Unsupervised Segmentation")
    input_audio = "../data/processed_clean/deep_clean.wav"  # 替换成你自己的文件路径
    output_dir = "../data/processed_cliped/deep_clips"
    run_unsupervised_segmentation(input_audio, output_dir)
