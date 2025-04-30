# preprocess_audio.py

import librosa
import noisereduce as nr
import webrtcvad
import numpy as np
import os
import soundfile as sf
from pydub import AudioSegment

def denoise_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply noise reduction to raw waveform"""
    reduced = nr.reduce_noise(y=y, sr=sr)
    return reduced

def vad_split(y: np.ndarray, sr: int, frame_ms=30, aggressiveness=2) -> np.ndarray:
    """Use WebRTC VAD to remove silent parts from audio"""
    vad = webrtcvad.Vad(aggressiveness)
    samples_per_frame = int(sr * frame_ms / 1000)
    num_frames = len(y) // samples_per_frame

    voiced_frames = []

    for i in range(num_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        frame = y[start:end]

        if len(frame) < samples_per_frame:
            break

        # Convert to 16-bit PCM bytes (required by webrtcvad)
        pcm_data = (frame * 32768).astype(np.int16).tobytes()
        is_voiced = vad.is_speech(pcm_data, sample_rate=sr)

        if is_voiced:
            voiced_frames.append(frame)

    if not voiced_frames:
        print("Warning: No speech detected.")
        return np.array([], dtype=np.float32)

    return np.concatenate(voiced_frames)

def preprocess_audio_file(input_path: str, output_path: str, target_sr=16000):
    print(f"Processing {input_path}")
    # Step 1: Load audio
    y, sr = librosa.load(input_path, sr=None)

    # Step 2: Resample to target sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Step 3: Denoise
    y_denoised = y #denoise_audio(y, sr)

    # Step 4: VAD (remove silence)
    y_clean = vad_split(y_denoised, sr)

    # Step 5: Normalize
    if len(y_clean) > 0:
        y_clean = y_clean / np.max(np.abs(y_clean))

    # Step 6: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_clean, sr)
    print(f"Saved preprocessed audio to {output_path}")

# Example usage
if __name__ == "__main__":
    input_file = "../data/raw_audio/Trump_WEF_2018.mp3"
    output_file = "../data/processed_clean/Trump_WEF_2018_clean.wav"
    preprocess_audio_file(input_file, output_file)
