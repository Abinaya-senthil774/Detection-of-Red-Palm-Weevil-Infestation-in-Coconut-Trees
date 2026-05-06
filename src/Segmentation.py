"""
DESCRIPTION:
This script performs event-based audio segmentation on a dataset of '.wav' files. 
It analyzes the audio to find active segments where the Root Mean Square (RMS) energy 
exceeds a predefined threshold (`RMS_THRESHOLD`). If an active segment lasts longer 
than the `MIN_SEGMENT_DURATION`, it is extracted and saved as a new individual '.wav' file.
The script processes files recursively within 'clean' and 'infested' subdirectories, 
preserving the original class labels by saving the extracted segments into corresponding 
output folders.

INPUTS:
- A directory specified by `input_root` containing subfolders 'clean' and 'infested' 
  with the original '.wav' files.
- Audio parameters defined at the top of the script (e.g., threshold, frame length).

OUTPUTS:
- Individual segmented '.wav' files saved into the directory specified by `output_root`, 
  maintaining 'clean' and 'infested' subfolders.
- Extracted segments are named systematically combining the original relative folder path, 
  original filename, and a segment index.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Parameters
RMS_THRESHOLD = 0.01     # You can tune this
MIN_SEGMENT_DURATION = 0.1  # seconds
FRAME_LENGTH = 2048
HOP_LENGTH = 512
SR = 22050

def event_based_segmentation(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH)

    segments = []
    is_active = False
    start = 0

    for i, e in enumerate(rms):
        if e > RMS_THRESHOLD and not is_active:
            start = i
            is_active = True
        elif e <= RMS_THRESHOLD and is_active:
            end = i
            if (end - start) * HOP_LENGTH / sr >= MIN_SEGMENT_DURATION:
                segments.append((start, end))
            is_active = False

    # Handle if file ends while still active
    if is_active:
        end = len(rms) - 1
        if (end - start) * HOP_LENGTH / sr >= MIN_SEGMENT_DURATION:
            segments.append((start, end))

    return segments

# Create output folders
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Traverse dataset
input_root = r"D:\CS\AI\PROJECT\coconut_project\cleaned_dataset"  # Your root folder
output_root = r"D:\CS\AI\PROJECT\coconut_project\segmented_audio"

for label in ['clean', 'infested']:
    input_label_path = os.path.join(input_root, label)
    output_label_path = os.path.join(output_root, label)
    ensure_dir(output_label_path)

    for root, _, files in os.walk(input_label_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=SR)

                segments = event_based_segmentation(y, sr)

                for idx, (start_frame, end_frame) in enumerate(segments):
                    start_sample = start_frame * HOP_LENGTH
                    end_sample = end_frame * HOP_LENGTH
                    segment_audio = y[start_sample:end_sample]

                    if len(segment_audio) < int(MIN_SEGMENT_DURATION * sr):
                        continue

                    # Unique naming: tree_foldername_filename_segmentID
                    rel_path = os.path.relpath(file_path, input_label_path)
                    base_name = os.path.splitext(rel_path.replace(os.sep, "_"))[0]
                    output_file = f"{base_name}_segment{idx}.wav"
                    output_path = os.path.join(output_label_path, output_file)

                    sf.write(output_path, segment_audio, sr)
