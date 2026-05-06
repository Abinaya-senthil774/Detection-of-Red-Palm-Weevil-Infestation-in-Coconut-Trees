"""
DESCRIPTION:
This script evaluates a test dataset of audio files using a pre-trained PyTorch 
Feedforward Neural Network (FFNN). It scans a designated directory for '.wav' files, 
extracts 13 Mel-frequency cepstral coefficients (MFCCs) and 7 Spectral Contrast 
features per file using librosa, and scales the data. It then loads the saved 
model weights to perform binary classification inference, saving both the extracted 
features and the final predictions into CSV files.

INPUTS:
- Test audio files (.wav format) located within the 'test_dir' directory.
- A pre-trained PyTorch model weights file ('model_weights.pth').

OUTPUTS:
- 'extracted_features.csv': A CSV file containing the computed MFCC and Spectral Contrast features for the test data.
- 'predictions.csv': A CSV file mapping each test audio filename to its corresponding binary prediction (0.0 or 1.0).
"""

import os
import torch
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Function to extract MFCC and Spectral Contrast features from a wav file
import librosa

# Function to extract MFCC and Spectral Contrast features from a wav file
def extract_features(wav_file):
    # Load the .wav file
    y, sr = librosa.load(wav_file, sr=None)  # sr=None to preserve the original sample rate

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCCs

    # Adjust parameters to avoid Nyquist error in spectral contrast
    # Default parameters of spectral_contrast may cause an issue when number of bands exceeds Nyquist
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bands=6)

    # Average the features along the time axis (to get a single feature vector)
    mfcc_mean = np.mean(mfcc, axis=1)
    contrast_mean = np.mean(spectral_contrast, axis=1)

    # Combine MFCC and Spectral Contrast features into a single vector
    features = np.concatenate((mfcc_mean, contrast_mean), axis=0)

    return features



# Function to get all .wav files from the test folder (including subfolders)
def get_wav_files(test_dir):
    wav_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files


# List of test .wav files (replace this with your actual test file paths)
test_dir = r"C:\Users\sabin\Downloads\archive (8)\field\field\test" # Directory containing test folders
test_wav_files = get_wav_files(test_dir)

# Extract features for each test file and store them in a list
feature_list = []
file_names = []

for wav_file in test_wav_files:
    features = extract_features(wav_file)
    feature_list.append(features)
    file_names.append(wav_file.split("\\")[-1])  # Save only the filename, or use full path

# Create a DataFrame to store the features (same structure as training CSV)
df_test = pd.DataFrame(feature_list)
df_test.columns = [f"mfcc_{i + 1}" for i in range(13)] + [f"contrast_{i + 1}" for i in
                                                          range(7)]  # Adjust according to your features

df_test.to_csv(r"D:\CS\AI\PROJECT\coconut_project\extracted_features.csv", index=False)

print("Features saved to extracted_features.csv")
# Scale the features (same as during training)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(df_test)

# Convert the scaled features to PyTorch tensor
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


# Define the FFNN model class (same as during training)
class FFNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(FFNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 1)  # Output layer
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# Initialize the model with the same input dimension used during training
model = FFNN(input_dim=X_test_scaled.shape[1])

# Load the trained model weights
model.load_state_dict(torch.load(r"D:\CS\AI\PROJECT\coconut_project\model_weights.pth"))
model.eval()

# Initialize lists for predictions and file names
predictions = []

# Make predictions on the test set
with torch.no_grad():
    for i in range(len(X_test_tensor)):
        # Get the features for each test sample
        feature = X_test_tensor[i].unsqueeze(0)  # Add batch dimension
        output = model(feature).squeeze(1)
        predicted_label = (output > 0.5).float()  # Apply threshold for binary classification (0 or 1)

        # Append the predicted label
        predictions.append(predicted_label.item())

# Save predictions to a DataFrame with file names
df_predictions = pd.DataFrame({
    'file': file_names,
    'prediction': predictions
})

# Save DataFrame to CSV
df_predictions.to_csv("D:\CS\AI\PROJECT\coconut_project\predictions.csv", index=False)

print("Predictions saved to predictions.csv")
