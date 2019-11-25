from __future__ import print_function, division
import numpy as np
import librosa
from scipy.io import wavfile


def extract_spectrogram(audio_fpath, window_size=0.03, window_stride=0.015, n_fft=661):
    sampling_rate, audio = wavfile.read(audio_fpath)
    audio = np.mean(audio, axis=1)

    win_length = int(sampling_rate * window_size)
    hop_length = int(sampling_rate * window_stride)

    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window='hamming', center=False)
    S = np.power(np.abs(X[:, 2:]), 2)   
    return np.log(S)


def to_frames(s, width=31):
    height, audio_length = s.shape
    start_idx = 0
    frames = list()
    for end_idx in range(width, audio_length, width):
        frame = s[:, start_idx:end_idx].copy()
        frames.append(frame)
        start_idx = end_idx
    end_idx = audio_length

    last_idx = end_idx - start_idx
    frame = np.zeros((height, width))
    frame[:, :last_idx] = s[:, start_idx:end_idx]
    frames.append(frame)
    return frames, last_idx


def to_windows(sampling_rate, audio, window_size=0.03, window_stride=0.015):
    
    win_length = int(sampling_rate * window_size)
    hop_length = int(sampling_rate * window_stride)
    
    win_shift = win_length - hop_length
    num_windows = 1 + np.floor((audio.size - win_length) / win_shift).astype(np.int16)

    win_segments = np.zeros((win_length, num_windows))
    for i in range(num_windows):
        win_segments[:, i] = audio[i*win_shift:win_length+i*win_shift]
        
    return win_segments


def normalize(s: np.ndarray):
    return s - np.mean(s)


def calculate_energy(win_segments):
    return np.mean(np.power(win_segments,2), axis=0)


def extract_log_energy(audio_fpath, window_size=0.03, window_stride=0.015):
    sampling_rate, audio = wavfile.read(audio_fpath)
    audio = normalize(np.mean(audio, axis=1))

    win_segments = to_windows(sampling_rate, audio, window_size, window_stride)
    energy = calculate_energy(win_segments)
    return np.log(energy)
