import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_syllable_boundaries(audio_path, sr=16000, plot=True):
    """基于频谱峰值检测音节边界
    
    参数:
        audio_path: 音频文件路径
        sr: 采样率 (默认16kHz)
        plot: 是否可视化结果
    """
    # 1. 加载音频
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 2. 提取Mel频谱 (比MFCC更适合音节检测)
    S = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        fmax=8000
    )
    log_S = librosa.power_to_db(S, ref=np.max)  # 转换为dB单位

    # 3. 计算频谱通量(Spectral Flux) - 检测频谱变化
    spectral_flux = np.sum(np.diff(log_S, axis=1)**2, axis=0)
    spectral_flux = np.insert(spectral_flux, 0, 0)  # 保持长度一致

    # 4. 平滑并归一化
    flux_smooth = librosa.util.normalize(
        np.convolve(spectral_flux, np.hanning(5), mode='valid')
    )

    # 5. 峰值检测 (关键参数调整点)
    peaks, _ = find_peaks(
        flux_smooth,
        height=np.mean(flux_smooth) + 0.5 * np.std(flux_smooth),  # 高度阈值
        distance=int(0.1 * sr / 256)  # 最小间隔0.1秒
    )
    
    # 转换为时间戳
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=256)
    
    # 6. 可视化
    if plot:
        plt.figure(figsize=(14, 8))
        
        # 波形图
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.vlines(times, -1, 1, color='r', linestyle='--', alpha=0.8)
        plt.title('Audio Waveform with Syllable Boundaries')
        
        # 频谱图
        plt.subplot(3, 1, 2)
        librosa.display.specshow(log_S, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        # 频谱通量
        plt.subplot(3, 1, 3)
        plt.plot(librosa.frames_to_time(range(len(flux_smooth)), sr=sr, hop_length=256), 
                flux_smooth, label='Spectral Flux')
        plt.vlines(times, 0, 1, color='r', linestyle='--', alpha=0.5)
        plt.title('Spectral Flux with Detected Peaks')
        plt.tight_layout()
        plt.show()

    return times

# 使用示例
boundaries = detect_syllable_boundaries("../data/processed_clean/deep_clean.wav")
print("Detected syllable boundaries (seconds):", boundaries)