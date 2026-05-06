"""
DESCRIPTION:
This script traverses a structured dataset directory (e.g., containing 'clean' and 'infested' 
subfolders) and extracts acoustic features from each audio file. It calculates the mean 
of 13 Mel-frequency cepstral coefficients (MFCCs) and 7 Spectral Contrast bands across 
time. Finally, it compiles these features along with their corresponding class labels 
and filenames into a CSV file ('features_mfcc_contrast.csv') for machine learning training.

INPUTS:
- Audio files (.wav format) organized in class-labeled subdirectories (e.g., 'segmented_audio/clean').

OUTPUTS:
- 'features_mfcc_contrast.csv': A structured dataset ready for training models, containing 
  the feature columns, a 'label' column, and a 'file' column.
"""

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIGURATION ===
# Replace with the path to your cleaned or segmented audio folders
dataset_dir = r"D:\CS\AI\PROJECT\coconut_project\segmented_audio" 
output_csv_path = r"D:\CS\AI\PROJECT\coconut_project\features_mfcc_contrast.csv"
classes = ['clean', 'infested']  # Folder names representing the labels


# === FEATURE EXTRACTION FUNCTION ===
def extract_features(wav_file):
    """Extracts 13 MFCCs and 7 Spectral Contrast bands from an audio file."""
    try:
        # Load the .wav file (sr=None preserves the original sample rate)
        y, sr = librosa.load(wav_file, sr=None) 
        
        # Skip empty files or files that are too short to extract features properly
        if len(y) == 0:
            return None

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) 

        # Extract Spectral Contrast
        # Adjust parameters to avoid Nyquist error in spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bands=6
        )

        # Average the features along the time axis (to get a single 1D feature vector per file)
        mfcc_mean = np.mean(mfcc, axis=1)
        contrast_mean = np.mean(spectral_contrast, axis=1)

        # Combine MFCC and Spectral Contrast features into a single vector
        features = np.concatenate((mfcc_mean, contrast_mean), axis=0)

        return features
        
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return None


# === MAIN PROCESSING LOOP ===
feature_list = []

print(f"Scanning directory: {dataset_dir}")

for label in classes:
    class_dir = os.path.join(dataset_dir, label)
    
    if not os.path.exists(class_dir):
        print(f"Warning: Directory not found -> {class_dir}")
        continue
        
    # Get all .wav files in the class directory
    wav_files = []
    for root, dirs, files in os.walk(class_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
                
    print(f"Extracting features for class '{label}' ({len(wav_files)} files)...")
    
    # Process each file with a progress bar
    for file_path in tqdm(wav_files, desc=label):
        features = extract_features(file_path)
        
        if features is not None:
            # Create a dictionary for this row of data
            row_data = {}
            
            # Map MFCC features to columns (mfcc_1 to mfcc_13)
            for i in range(13):
                row_data[f"mfcc_{i + 1}"] = features[i]
                
            # Map Spectral Contrast features to columns (contrast_1 to contrast_7)
            for i in range(7):
                row_data[f"contrast_{i + 1}"] = features[13 + i]
                
            # Add metadata columns required by the training script
            row_data['label'] = label
            row_data['file'] = os.path.basename(file_path)
            
            feature_list.append(row_data)


# === SAVE TO CSV ===
if feature_list:
    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(feature_list)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully extracted features for {len(df)} files.")
    print(f"Dataset saved to: {output_csv_path}")
else:
    print("\nNo features were extracted. Please check your dataset directory.")
