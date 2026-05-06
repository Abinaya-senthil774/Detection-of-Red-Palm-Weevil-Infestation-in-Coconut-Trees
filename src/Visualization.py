"""
DESCRIPTION:
This script visualizes audio data by generating time-domain and frequency-domain 
plots for samples from two distinct categories ('clean' and 'infested'). For a 
limited number of .wav files (default is 2) from each specified directory, it 
creates three visual representations using the librosa and PyWavelets libraries: 
a standard Waveform, a logarithmic STFT Spectrogram, and a Continuous Wavelet 
Transform (CWT) Scalogram using the Morlet wavelet.

INPUTS:
- Audio files (.wav format) located in the directories specified by 'clean_folder' 
  and 'infested_folder'.
- Audio signals are loaded and resampled at a common sample rate of 22050 Hz.

OUTPUTS:
- Console print messages indicating which audio category is currently being visualized.
- Sequential Matplotlib figure pop-ups for each processed audio file, displaying 
  its Waveform, Spectrogram, and CWT Scalogram.
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import cwt, morlet2
'''
# === CONFIGURATION ===
clean_folder = r
infested_folder = r
sample_rate = 22050  # Common resample rate for audio

def load_wav_files(folder, max_files=2):
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    return [os.path.join(folder, f) for f in files[:max_files]]

def plot_waveform(y, sr, title):
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform: {title}')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr, title):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {title}')
    plt.tight_layout()
    plt.show()

def plot_cwt(y, sr, title):
    widths = np.arange(1, 128)
    wavelet = 'morl'
    cwtmatr, freqs = pywt.cwt(y, widths, wavelet, sampling_period=1/sr)
    plt.figure(figsize=(10, 3))
    plt.imshow(np.abs(cwtmatr), extent=[0, len(y)/sr, 1, 128], cmap='inferno', aspect='auto',
               vmax=np.percentile(np.abs(cwtmatr), 99))
    plt.title(f'CWT Scalogram: {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Scale')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# === PROCESS ===
def visualize_folder(folder, label):
    print(f"Visualizing {label} audio samples...")
    files = load_wav_files(folder)
    for file in files:
        y, sr = librosa.load(file, sr=sample_rate)
        name = os.path.basename(file)
        plot_waveform(y, sr, f"{label} - {name}")
        plot_spectrogram(y, sr, f"{label} - {name}")
        plot_cwt(y, sr, f"{label} - {name}")

# === EXECUTION ===
visualize_folder(clean_folder, 'CLEAN')
visualize_folder(infested_folder, 'INFESTED')
'''

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pywt



# === CONFIGURATION ===
clean_folder = r"C:\Users\sabin\Downloads\archive (8)\field\field\train\clean"
infested_folder = r"C:\Users\sabin\Downloads\archive (8)\field\field\train\infested"
sample_rate = 22050  # Common resample rate for audio

def load_wav_files(folder, max_files=2):
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    return [os.path.join(folder, f) for f in files[:max_files]]

def plot_waveform(y, sr, title):
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform: {title}')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr, title):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {title}')
    plt.tight_layout()
    plt.show()

def plot_cwt(y, sr, title):
    widths = np.arange(1, 128)
    wavelet = 'morl'
    cwtmatr, freqs = pywt.cwt(y, widths, wavelet, sampling_period=1/sr)
    plt.figure(figsize=(10, 3))
    plt.imshow(np.abs(cwtmatr), extent=[0, len(y)/sr, 1, 128], cmap='inferno', aspect='auto',
               vmax=np.percentile(np.abs(cwtmatr), 99))
    plt.title(f'CWT Scalogram: {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Scale')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# === PROCESS ===
def visualize_folder(folder, label):
    print(f"Visualizing {label} audio samples...")
    files = load_wav_files(folder)
    for file in files:
        y, sr = librosa.load(file, sr=sample_rate)
        name = os.path.basename(file)
        plot_waveform(y, sr, f"{label} - {name}")
        plot_spectrogram(y, sr, f"{label} - {name}")
        plot_cwt(y, sr, f"{label} - {name}")

# === EXECUTION ===
visualize_folder(clean_folder, 'clean')
visualize_folder(infested_folder, 'infested')
