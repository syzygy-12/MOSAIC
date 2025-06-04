import numpy as np
import os
import librosa
import soundfile as sf
from tqdm import tqdm
from typing import Dict
from scipy.interpolate import interp1d
from processed_audio import ProcessedAudio
import torch
from processed_audio import SCALE
from save_final_video import save_final_video

DATANAME = "trump2"
TARGETNAME = "deep"


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个特征序列的平均余弦相似度，使用GPU加速。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a_torch = torch.from_numpy(a).float().to(device)
    b_torch = torch.from_numpy(b).float().to(device)
    a_norm = a_torch / (a_torch.norm(dim=1, keepdim=True) + 1e-6)
    b_norm = b_torch / (b_torch.norm(dim=1, keepdim=True) + 1e-6)
    sim = (a_norm * b_norm).sum(dim=1).mean().item()
    return sim


def match_token_to_data(
    token_flow: np.ndarray,
    data_feature: np.ndarray,
    batch_size: int = 2048  # 可根据显存调整
) -> dict:
    """
    在 data_feature 中滑窗匹配 token_flow，返回最佳匹配信息。使用GPU分批加速，防止OOM。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    token_len = len(token_flow)
    data_len = len(data_feature)
    if data_len < token_len:
        return {}
    windows = np.lib.stride_tricks.sliding_window_view(data_feature, (token_len, data_feature.shape[1]))
    windows = windows.reshape(-1, token_len, data_feature.shape[1])
    token = torch.from_numpy(np.array(token_flow, copy=True)).float().to(device)  # (T, D)
    token_norm = token / (token.norm(dim=1, keepdim=True) + 1e-6)  # (T, D)

    best_score = -float('inf')
    best_idx = -1
    #print(f"🔍 Matching token of length {token_len} against data feature of length {data_len}...")
    #print(len(windows), "windows to match")

    for i in range(0, len(windows), batch_size):
        batch = windows[i:i+batch_size]
        batch_torch = torch.from_numpy(np.array(batch, copy=True)).float().to(device)  # (B, T, D)
        batch_norm = batch_torch / (batch_torch.norm(dim=2, keepdim=True) + 1e-6)  # (B, T, D)
        sim = (token_norm * batch_norm).sum(dim=2).mean(dim=1)  # (B,)
        max_sim, max_idx = torch.max(sim, dim=0)
        if max_sim.item() > best_score:
            best_score = max_sim.item()
            best_idx = i + max_idx.item()
        del batch_torch, batch_norm, sim  # 释放显存
        torch.cuda.empty_cache()

    if best_idx == -1:
        return {}
    return {
        "score": float(best_score),
        "start": int(best_idx),
        "end": int(best_idx + token_len)
    }


def match_all_tokens(target_name: str, data_name: str, output_dir: str = "../data/matched_wavs"):
    """
    对 target_name 的每个分词，在 data_name 的不同vecSet倍率中寻找最佳匹配片段，并保存结果。
    对于每个token，遍历所有scale，取相似度最大的匹配。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 载入目标音频及特征
    target_audio = ProcessedAudio(target_name)
    target_audio.load_audio(target_name)
    target_audio.pre_process()
    #target_audio.audio = target_audio.audio[:5 * 16000]
    target_audio.audio = librosa.util.normalize(target_audio.audio)
    target_audio.tokenize()
    target_audio.extract_feature()

    # 载入数据音频及特征
    data_audio = ProcessedAudio(data_name)
    data_audio.load_audio(data_name)
    data_audio.pre_process()
    data_audio.extract_feature()

    tokens = target_audio.tokens
    frame_duration = 0.02
    sampling_rate = 16000
    frame_size = int(sampling_rate * frame_duration)
    final_audio = []
    matched = []
    final_target = []

    for idx, token in enumerate(tqdm(tokens, desc="Matching tokens")):
        start, end = token["start"], token["end"]
        token_flow = target_audio.vec[start:end]
        if token_flow.size == 0:
            continue
        best_match = None
        best_scale_idx = None
        best_score = -1
        # 遍历所有scale，取最大相似度
        for scale_idx, data_feature in enumerate(data_audio.vecSet):
            match = match_token_to_data(token_flow, data_feature)
            if match and match["score"] > best_score:
                best_score = match["score"]
                best_match = match
                best_scale_idx = scale_idx
        if not best_match:
            continue
        scale = SCALE[best_scale_idx]
        best_match.update({
            "token": token["text"],
            "token_start": start,
            "token_end": end,
            "scale": float(scale)
        })
        matched.append(best_match)

        # 保存匹配出来的 data 音频片段
        data_start_sample = best_match["start"] * frame_size
        data_end_sample = best_match["end"] * frame_size
        audio_snippet = data_audio.audioSet[best_scale_idx][data_start_sample: data_end_sample]
        final_audio.append(audio_snippet)

        # 保存路径
        token_label = token["text"]
        save_path = os.path.join(
            output_dir,
            f"{token_label}_{idx:03d}_{best_match['start']:04d}_{best_match['end']:04d}_x{scale:.2f}.wav"
        )
        sf.write(save_path, audio_snippet, samplerate=sampling_rate)
        print(f"[SAVED] Token '{token_label}' matched to {save_path}")

        # 保存 target 音频片段
        target_start_sample = start * frame_size
        target_end_sample = end * frame_size
        target_snippet = target_audio.audio[target_start_sample: target_end_sample]
        final_target.append(target_snippet)
        target_save_path = os.path.join(
            output_dir,
            f"{token_label}_target_{idx:03d}_{start:04d}_{end:04d}.wav"
        )
        sf.write(target_save_path, target_snippet, samplerate=sampling_rate)
        print(f"[SAVED] Token '{token_label}' target to {target_save_path}")

    # 合并最终的音频片段并保存
    if final_audio:
        final_audio = crossfade_concat(final_audio, crossfade_ms=20, sr=sampling_rate)
        final_save_path = os.path.join(output_dir, f"_final_{target_name}_{data_name}.wav")
        sf.write(final_save_path, final_audio, samplerate=sampling_rate)
        print(f"[SAVED] Final audio saved to {final_save_path}")
    if final_target:
        final_target = crossfade_concat(final_target, crossfade_ms=20, sr=sampling_rate)
        final_target_save_path = os.path.join(output_dir, f"_final_target_{target_name}.wav")
        sf.write(final_target_save_path, final_target, samplerate=sampling_rate)
        print(f"[SAVED] Final target audio saved to {final_target_save_path}")

    # 保存最终视频
    save_final_video(
        target_audio=target_audio,
        data_audio=data_audio,
        matched=matched,
        output_path="../data/results/final_result.mp4"
    )

    return matched


def crossfade_concat(snippets, crossfade_ms=20, sr=16000):
    """
    Concatenate audio snippets with crossfade to smooth transitions.
    crossfade_ms: crossfade duration in milliseconds.
    """
    if not snippets:
        return np.array([])
    crossfade_samples = int(sr * crossfade_ms / 1000)
    output = snippets[0]
    for snippet in snippets[1:]:
        if crossfade_samples > 0 and len(output) >= crossfade_samples and len(snippet) >= crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            output[-crossfade_samples:] = (
                output[-crossfade_samples:] * fade_out +
                snippet[:crossfade_samples] * fade_in
            )
            output = np.concatenate([output, snippet[crossfade_samples:]])
        else:
            output = np.concatenate([output, snippet])
    return output


if __name__ == "__main__":
    result = match_all_tokens(TARGETNAME, DATANAME)
    for i, match in enumerate(result):
        print(f"{i}: Token '{match['token']}' matched with score {match['score']:.3f}, "
              f"data [{match['start']} → {match['end']}] (scale: {match['scale']:.2f})")