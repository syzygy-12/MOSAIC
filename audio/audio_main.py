import os
from processed_audio import ProcessedAudio
from scipy.io import wavfile

RAW_AUDIO_DIR = "../data/raw_audio"
PROCESSED_AUDIO_DIR = "../data/processed_audio"
NAME = "qimeidi"

def process_all():
    os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
    for fname in os.listdir(RAW_AUDIO_DIR):
        if not fname.lower().endswith(('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg')):
            continue

        input_path = os.path.join(RAW_AUDIO_DIR, fname)
        name = os.path.splitext(fname)[0]
        if name != NAME:
            continue

        print(f"[INFO] Processing {name}...")
        try:
            audio = ProcessedAudio.from_file(input_path, name)
            audio.denoise()
            audio.remove_silence_vad()

            out_path = os.path.join(PROCESSED_AUDIO_DIR, f"{name}.wav")
            wav_int16 = (audio.audio * 32768).astype("int16")
            wavfile.write(out_path, 16000, wav_int16)
            audio.extract_feature()
            audio.tokenize()
            #print(audio.frame_index)

            print(f"[OK] Saved WAV to {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {e}")

if __name__ == "__main__":
    audio = ProcessedAudio(NAME)
    audio.load_audio(NAME)
