"""
DESCRIPTION:
This script performs peak normalization on an audio file and visually compares 
the audio before and after the process. It grabs the first '.wav' file from a 
specified directory, scales its amplitude so the maximum absolute value is 1.0 
(preventing clipping while maximizing volume), and generates plots to compare 
the time-domain waveforms and amplitude histograms.

INPUTS:
- A directory specified by `folder_path` containing at least one '.wav' audio file.
- Audio signals are loaded using `librosa` at a sample rate of 22050 Hz.

OUTPUTS:
- A Matplotlib figure displaying four subplots:
  1. Original Waveform
  2. Normalized Waveform
  3. Original Amplitude Histogram
  4. Normalized Amplitude Histogram
- Console output indicating if no '.wav' files are found.
"""

# Re-import after kernel reset
import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# === CONFIG ===
folder_path = r"D:\CS\AI\PROJECT\coconut_project\cleaned_dataset\infested\folder_3"  # Example: a folder with .wav files
sample_rate = 22050

# === Normalization Function ===
def normalize_audio(y):
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

# === Visualization: waveform and amplitude histogram ===
def visualize_before_after(original, normalized, sr, filename):
    time = np.linspace(0, len(original) / sr, len(original))

    plt.figure(figsize=(14, 6))

    # Waveform comparison
    plt.subplot(2, 2, 1)
    plt.plot(time, original)
    plt.title(f"Original Waveform - {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 2, 2)
    plt.plot(time, normalized)
    plt.title(f"Normalized Waveform - {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Histogram comparison
    plt.subplot(2, 2, 3)
    plt.hist(original, bins=50, color='gray')
    plt.title("Original Amplitude Histogram")

    plt.subplot(2, 2, 4)
    plt.hist(normalized, bins=50, color='green')
    plt.title("Normalized Amplitude Histogram")

    plt.tight_layout()
    plt.show()

# === Process One File for Visualization ===
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
if wav_files:
    file_path = os.path.join(folder_path, wav_files[0])
    y, sr = librosa.load(file_path, sr=sample_rate)
    y_normalized = normalize_audio(y)
    visualize_before_after(y, y_normalized, sr, wav_files[0])
else:
    print("No .wav files found in the folder.")
