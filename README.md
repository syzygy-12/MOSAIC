收集数据：

收集视频，处理噪声，空白（保证数据多样性），大概1h左右。收集target音频。搞明白用什么格式比较舒服，这些格式有什么特点，处理这些音视频要用什么工具。

音频特征采集：

调研wav2vec等模型如何运作，如何使用，做初步的实验，观察是否有不错的音高音色无关性质。再研究filter，speaker-invariant，和multi-speaker数据集有什么用，是否有其他方法（比如对抗学习）。

音频裁剪匹配：

调研有没有比dp更好的方法，设计损失函数。这一步先不着急

视频动作匹配：

调研openpose如何使用，做实验

音视频剪辑：

研究库怎么用





文件结构：

```
MOSAIC/
│
├── audio/
│   ├── extract_features.py        # 需要加上15s分段再合并
│   ├── train_filter_model.py      # 训练音频特征过滤器
|   --- resample.py
|   --- convert_to_wav.py
|   --- segmenter.py
|   --- denoise.py
|   --- silence_remove.py
|   --- matching.py                # 返回(Segments, (begin_f, end_f, 速度倍率))
|   --- audio_main.py  
│
├── video/
│   ├── extract_frames.py          # 使用 OpenCV 分帧
│   ├── extract_pose.py            # 骨架提取（OpenPose or MediaPipe）
│   ├── match_poses.py             # 骨架姿态匹配（邻近搜索）
│   └── assemble_video.py          # 视频段拼接与输出
│
├── data/
│   ├── raw_audio/                 # 原始音频和视频
│   ├── query_audio/              # 用户输入的查询语音
│   └── processed/                # 提取的特征、剪辑片段等缓存
│
├── models/
│   └── filter_model.pth           # 保存训练好的过滤模型
│
├── main.py                        # 主入口，整合pipeline
└── README.md

```

一些问题：

音素、音节粒度的分词器。语言无关的语音片段边界检测问题
要把输入音频切成15s的小段再wav2vec
wav2vec的vec满足正交性？相似性曲线如何
说话人无关
改成vec flow，有无更好的速度匹配方法
中，英，日，粤

需要写个class，ProcessedAudio：
音频数据audio，每一帧（20ms）对应原始数据的哪一帧frame_index，文件名name，特征向量vec，分词帧segment_frame

这里特征向量需要支持本地读取or在线计算
