import numpy as np
from processed_audio import ProcessedAudio
from typing import Dict
from scipy.interpolate import interp1d
import os
import soundfile as sf
import librosa

DATANAME = "base"
TARGETNAME = "deep"

def resample_vector_flow(flow: np.ndarray, target_len: int) -> np.ndarray:

    if len(flow) == target_len:
        return flow

    x_old = np.linspace(0, 1, len(flow))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, flow, axis=0)
    return f(x_new)

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    #return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    # shape: (T, D)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    sim = (a_norm * b_norm).sum(axis=1).mean()  # mean cosine similarity
    return sim

def match_token_to_data(
    token_flow: np.ndarray,
    data_audio: ProcessedAudio,
    scale_range=(0.8, 1.2),
    step=1.1
) -> dict:
    best_score = -1
    best_info = {}

    token_len = len(token_flow)
    token_avg = token_flow.mean(axis=0)  # 提前计算 token 的平均向量
    scales = np.arange(scale_range[0], scale_range[1] + 1e-4, step - 1)

    for scale in scales:
        window_len = int(token_len * scale)
        if window_len > data_audio.len:
            print(f"Window length {window_len} exceeds data length {data_audio.len}. Skipping scale {scale}.")
            continue
        for i in range(data_audio.len - window_len):
            window = data_audio.extract_feature_from_segment(start=i, end=i + window_len)
            #window_avg = window.mean(axis=0)  # 当前窗口平均向量
            window_resampled = resample_vector_flow(window, token_len)  # 重采样到 token_len
            #score = similarity(token_avg, window_avg)  # 直接比较两个平均向量
            score = similarity(token_flow, window_resampled)  # 比较 token 平均向量和当前窗口平均向量
            if score > best_score:
                best_score = score
                best_info = {
                    "score": float(score),
                    "start": int(i),
                    "end": int(i + window_len),
                    "scale": float(scale)
                }
        
    print(f"Best match: {best_info}")
    return best_info

def match_all_tokens(target_name: str, data_name: str, output_dir: str = "../data/matched_wavs"):
    os.makedirs(output_dir, exist_ok=True)  # 确保输出文件夹存在

    target_audio = ProcessedAudio(target_name)
    target_audio.load_audio(target_name)
    target_audio.pre_process()
    # 只取前10s
    target_audio.audio = target_audio.audio[:10 * 16000]
    target_audio.audio = librosa.util.normalize(target_audio.audio)

    #target_audio.extract_feature()
    target_audio.tokenize()

    data_audio = ProcessedAudio(data_name)
    data_audio.load_audio(data_name)
    data_audio.pre_process()
    #data_audio.extract_feature()

    #target_flow = target_audio.feature
    #data_flow = data_audio.feature
    tokens = target_audio.tokens

    frame_duration = 0.02  # 每帧20ms
    sampling_rate = 16000  # 与预处理保持一致
    frame_size = int(sampling_rate * frame_duration)  # 每帧对应的采样点数：320

    final_audio = []

    matched = []
    for idx, token in enumerate(tokens):
        start, end = token["start"], token["end"]
        token_flow = target_audio.extract_feature_from_segment(start=start, end=end)

        match = match_token_to_data(token_flow, data_audio)
        match.update({
            "token": token["text"],
            "token_start": start,
            "token_end": end
        })
        matched.append(match)

        # ========== 保存匹配出来的 data 音频片段 ==========
        data_start_sample = match["start"] * frame_size
        data_end_sample = match["end"] * frame_size
        audio_snippet = data_audio.audio[data_start_sample : data_end_sample]
        # 按照scale缩放音频片段
        scale = match["scale"]
        audio_snippet = librosa.resample(audio_snippet, orig_sr=sampling_rate, target_sr=int(sampling_rate / scale))
        # 把audio_snippet插入到 final_audio 中
        final_audio.append(audio_snippet)

        # 构建保存路径（如：hello_05_013_1.1.wav）
        token_label = token["text"]
        scale = match["scale"]
        save_path = os.path.join(
            output_dir,
            f"{token_label}_{idx:03d}_{match['start']:04d}_{match['end']:04d}_x{scale:.2f}.wav"
        )

        sf.write(save_path, audio_snippet, samplerate=sampling_rate)
        print(f"[SAVED] Token '{token_label}' matched to {save_path}")
        # target音频片段保存
        target_start_sample = start * frame_size
        target_end_sample = end * frame_size
        target_snippet = target_audio.audio[target_start_sample : target_end_sample]
        target_save_path = os.path.join(
            output_dir,
            f"{token_label}_target_{idx:03d}_{start:04d}_{end:04d}.wav"
        )
        sf.write(target_save_path, target_snippet, samplerate=sampling_rate)
        print(f"[SAVED] Token '{token_label}' target to {target_save_path}")

    # ========== 合并最终的音频片段并保存 ==========
    final_audio = np.concatenate(final_audio, axis=0)  # 合并所有音频片段
    final_save_path = os.path.join(output_dir, f"_final_{target_name}_{data_name}.wav")
    sf.write(final_save_path, final_audio, samplerate=sampling_rate)
    print(f"[SAVED] Final audio saved to {final_save_path}")

    return matched

if __name__ == "__main__":
    result = match_all_tokens(TARGETNAME, DATANAME)
    for i, match in enumerate(result):
        print(f"{i}: Token '{match['token']}' matched with score {match['score']:.3f}, "
              f"data [{match['start']} → {match['end']}] (scale: {match['scale']:.2f})")
