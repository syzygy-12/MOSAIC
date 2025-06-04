import os
import numpy as np
import librosa
import moviepy.editor as mp
import soundfile as sf
from typing import List, Dict
from processed_audio import SCALE, ProcessedAudio

SR = 16000  # 音频采样率

def save_final_video(
    target_audio: ProcessedAudio,
    data_audio: ProcessedAudio,
    matched: List[Dict],
    output_path: str = "../data/results/final_result.mp4",
    raw_audio_dir: str = "../data/raw_audio",
    raw_video_dir: str = "../data/raw_video"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载原始音频和视频
    data_audio_path = os.path.join(raw_audio_dir, f"{data_audio.name}.wav")
    data_video_path = os.path.join(raw_video_dir, f"{data_audio.name}.mp4")

    raw_data_audio, sr = librosa.load(data_audio_path)
    print("sampling rate:", sr)
    data_video_clip = mp.VideoFileClip(data_video_path)
    
    original_height, original_width = data_video_clip.size

    frame_size = int(sr * 0.02)

    final_audio = np.zeros(0, dtype=np.float32)
    final_video_segments = []

    prev_video_end_time = 0
    total_video_duration = 0.0

    for i, match in enumerate(matched):
        # 时间对齐
        token_start_frame = match["token_start"]
        token_end_frame = match["token_end"]
        raw_duration = token_end_frame - token_start_frame
        raw_start = target_audio.frame_index[token_start_frame]
        raw_end = min(target_audio.frame_index[token_end_frame], raw_start + raw_duration - 1)
        if raw_duration <= 2:
            print(f"Warning: match {i} has duration <= 2 frames, skipping")
            continue

        # 目标时间轴上的插入位置
        start_time = raw_start * 0.02
        end_time = raw_end * 0.02
        start_sample = int(start_time * sr)

        # 音频：匹配区域，不放缩，直接贴
        scale = match["scale"]
        scale_factor = float(scale)
        data_start_frame = match["start"]
        data_end_frame = match["end"]
        origin_data_start_frame = int(data_start_frame * scale_factor)
        origin_data_end_frame = int(data_end_frame * scale_factor)
        
        origin_data_duration = origin_data_end_frame - origin_data_start_frame
        
        if origin_data_end_frame >= len(data_audio.frame_index):
            print("Warning: origin_data_end_frame out of range")
            continue
        
        data_start = data_audio.frame_index[origin_data_start_frame]
        data_end = min(data_audio.frame_index[origin_data_end_frame],
                        data_start + origin_data_duration - 1)
        #print(data_start, data_end, len(data_audio.frame_index))
        
        data_start = int(data_start * frame_size)
        data_end = int(data_end * frame_size)
        snippet_audio = raw_data_audio[data_start:data_end]
        # 保存每个 snippet_audio 到本地
        snippet_dir = os.path.join(os.path.dirname(output_path), "snippets")
        os.makedirs(snippet_dir, exist_ok=True)

        snippet_filename = f"snippet_{i:03d}_{match['token']}.wav"
        snippet_path = os.path.join(snippet_dir, snippet_filename)

        sf.write(snippet_path, snippet_audio, sr)



        # 在音频中补零以对齐位置
        gap = start_sample - final_audio.shape[0]
        if gap > 0:
            final_audio = np.concatenate([final_audio, np.zeros(gap, dtype=np.float32)])
        final_audio = np.concatenate([final_audio, snippet_audio])

        # 视频：从 data_video 中提取，并按 scale 放缩
        clip_start = data_start / sr
        clip_end = data_end / sr
        # 这里clip_start, clip_end 是未放缩前的时间点
        original_duration = clip_end - clip_start
        scaled_duration = original_duration / scale_factor

        print(f"Processing match {i}: token={match['token']}, "
              f"start_time={start_time:.2f}, end_time={end_time:.2f}, "
              f"clip_start={clip_start:.2f}, clip_end={clip_end:.2f}, "
              f"scale_factor={scale_factor:.2f}") 

        # 若 total_video_duration < start_time，则插入填充片段（来自原视频后续）
        if total_video_duration < start_time:
            pad_duration = start_time - total_video_duration
            # 上一个 token 的 scale（如果是第一个 token，就设为1.0）
            prev_scale = float(matched[i - 1]["scale"]) if i > 0 else 1.0

            pad_source_duration = pad_duration * prev_scale
            pad_start = prev_video_end_time
            pad_end = pad_start + pad_source_duration

            if pad_end <= data_video_clip.duration:
                pad_clip = data_video_clip.subclip(pad_start, pad_end)
            else:
                pad_clip = data_video_clip.subclip(pad_start, data_video_clip.duration).fx(
                    mp.vfx.loop, duration=pad_source_duration
                )

            # 放缩 pad_clip，以让它播放 pad_duration 秒
            pad_clip = pad_clip.fx(mp.vfx.speedx, factor=prev_scale)
            #print("Padding clip duration:", pad_duration, "seconds")
            #print(pad_clip.duration, "seconds after speedx")

            final_video_segments.append(pad_clip)
            total_video_duration += pad_duration
            prev_video_end_time = pad_end

        # if total_video_duration < start_time:
        #     pad_duration = start_time - total_video_duration

        #     # 创建黑屏填充段
        #     black_clip = mp.ColorClip(size=(original_height, original_width), color=(0, 0, 0), duration=pad_duration)
        #     black_clip = black_clip.set_fps(data_video_clip.fps)

        #     final_video_segments.append(black_clip)
        #     total_video_duration += pad_duration

        # 拉伸视频 clip
        video_clip = data_video_clip.subclip(clip_start, clip_end).fx(mp.vfx.speedx, factor=scale_factor)
        # clip保存到本地
        video_clip_filename = f"video_clip_{i:03d}_{match['token']}.mp4"
        video_clip_path = os.path.join(snippet_dir, video_clip_filename)
        video_clip.write_videofile(video_clip_path, codec='libx264', audio_codec='aac')

        final_video_segments.append(video_clip)
        total_video_duration += scaled_duration
        prev_video_end_time = clip_end  # 更新为当前 clip 的结束时间（data 视频中的原始时间）


    # 保存音频
    audio_output_path = output_path.replace(".mp4", ".wav")
    sf.write(audio_output_path, final_audio, sr)

    # 合并视频
    final_video = mp.concatenate_videoclips(final_video_segments, method="compose")
    # 设置视频分辨率为原始视频的分辨率，拉伸成原始视频横纵比
    final_video = final_video.resize(newsize=(original_height, original_width))
    print("height:", original_height, "width:", original_width)
    final_audio_clip = mp.AudioFileClip(audio_output_path)
    final_video = final_video.set_audio(final_audio_clip)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    print(f"Saved final video to {output_path}")
