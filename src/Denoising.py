"""
DESCRIPTION:
This script performs batch audio denoising using Discrete Wavelet Transform (DWT). 
It recursively scans an input directory for all '.wav' files, applies wavelet-based 
noise reduction (using the 'db8' wavelet), and saves the processed files into a 
designated output directory while preserving the original subdirectory structure.

INPUTS:
- Raw audio files (.wav format) located in the directory specified by `input_root`.
- Audio signals are loaded at a sample rate of 22050 Hz.

OUTPUTS:
- Denoised audio files (.wav format) saved to the directory specified by `output_root`.
- Console print statements indicating the save path of each successfully processed file.
"""

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa.display
'''
# === CONFIG ===
input_path =   # <-- Replace with your test file path
output_path = r'D:\CS\AI\PROJECT\coconut_project\denoised_output2.wav'
sample_rate = 22050


# === WAVELET DENOISING ===
def wavelet_denoise(signal, wavelet='db8', level=5, threshold=0.2):
    # Perform discrete wavelet transform (DWT)
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Thresholding - set the coefficients below the threshold to zero
    coeffs_thresholded = [pywt.threshold(c, threshold * max(c)) for c in coeffs]

    # Reconstruct the signal from the thresholded coefficients
    return pywt.waverec(coeffs_thresholded, wavelet)


# === VISUALIZATION ===
def plot_waveform(y, sr, title):
    plt.figure(figsize=(10, 2))
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def plot_cwt(y, sr, title):
    widths = np.arange(1, 128)
    cwtmatr, _ = pywt.cwt(y, widths, 'morl', sampling_period=1 / sr)
    plt.figure(figsize=(10, 3))
    plt.imshow(np.abs(cwtmatr), extent=[0, len(y) / sr, 1, 128], cmap='inferno', aspect='auto')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Scale')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# === Frequency Spectrum Plot ===
def plot_frequency_spectrum(y, sr, title):
    # Calculate the spectrum
    D = np.abs(librosa.stft(y))
    freq = librosa.fft_frequencies(sr=sr)
    plt.figure(figsize=(10, 6))
    plt.semilogy(freq, np.mean(D, axis=1))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


# === Spectrogram Plot ===
def plot_spectrogram(y, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, cmap='inferno')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


# === PROCESS SINGLE FILE ===
y, sr = librosa.load(input_path, sr=sample_rate)

# Denoise using wavelet transform
y_denoised = wavelet_denoise(y)

# Save the result
sf.write(output_path, y_denoised, sr)
print(f"Saved denoised audio to: {output_path}")

# === VISUALIZE ===
plot_waveform(y, sr, "Original Waveform")
plot_waveform(y_denoised, sr, "Denoised Waveform")

# Plot Frequency Spectrum (Before & After)
plot_frequency_spectrum(y, sr, "Original Frequency Spectrum")
plot_frequency_spectrum(y_denoised, sr, "Denoised Frequency Spectrum")

plot_cwt(y, sr, "Original CWT")
plot_cwt(y_denoised, sr, "Denoised CWT")

# Compare the original and denoised waveform in the time domain
plt.figure(figsize=(10, 2))
plt.plot(y, label='Original')
plt.plot(y_denoised, label='Denoised', alpha=0.7)
plt.title('Original vs Denoised Time-domain Waveform')
plt.legend()
plt.tight_layout()
plt.show()

# Plot Spectrogram (Before & After)
plot_spectrogram(y, sr, "Original Spectrogram")
plot_spectrogram(y_denoised, sr, "Denoised Spectrogram")
'''
import os
import librosa
import soundfile as sf
import pywt
import numpy as np

# === CONFIG ===
input_root = r"C:\Users\sabin\Downloads\archive (8)\field\field\train\infested"
output_root = r"D:\CS\AI\PROJECT\coconut_project\cleaned_dataset\infested"
sample_rate = 22050

# === WAVELET DENOISING FUNCTION ===
def wavelet_denoise(signal, wavelet='db8', level=5, threshold=0.2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs_thresh = [pywt.threshold(c, threshold * max(c)) for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)

# === BATCH PROCESSING ===
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith(".wav"):
            input_path = os.path.join(root, file)

            # Read audio
            y, sr = librosa.load(input_path, sr=sample_rate)

            # Denoise
            y_denoised = wavelet_denoise(y)

            # Construct output path (maintain folder structure)
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save denoised audio
            sf.write(output_path, y_denoised[:len(y)], sr)
            print(f"Saved: {output_path}")
