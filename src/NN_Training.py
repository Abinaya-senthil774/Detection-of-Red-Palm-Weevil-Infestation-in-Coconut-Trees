"""
DESCRIPTION:
This script trains a Feedforward Neural Network (FFNN) using PyTorch for binary classification 
based on extracted audio features. It performs data preprocessing including label encoding, 
train-test splitting, addressing class imbalance using SMOTE, and standard scaling. 
The model is trained over multiple epochs, evaluated using accuracy, a confusion matrix, 
and a classification report (visualized via Seaborn heatmaps). Finally, the trained 
model and its weights are saved to disk.

INPUTS:
- A CSV file ('features_mfcc_contrast.csv') containing audio features and corresponding labels.

OUTPUTS:
- Console logs displaying training loss and accuracy per epoch.
- Printed evaluation metrics including Accuracy, Confusion Matrix, and Classification Report.
- Seaborn heatmap visualizations of the Confusion Matrix and Classification Report.
- Saved PyTorch model files ('model_full.pth' and 'model_weights.pth').
"""

#Neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load your dataset
df = pd.read_csv(r"D:\CS\AI\PROJECT\coconut_project\features_mfcc_contrast.csv")

# Drop the 'file' column (not needed for training)
df = df.drop(columns=['file'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
X = df.drop(columns=['label'])

# Train/test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing data
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define the Feedforward Neural Network (FFNN)
class FFNN(nn.Module):
    def __init__(self, input_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# Initialize the model
model = FFNN(input_dim=X_train_scaled.shape[1])

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).squeeze(1)  # squeeze to match the shape of labels
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()  # Binary classification threshold at 0.5
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {correct / total * 100:.2f}%")

import matplotlib.pyplot as plt
import seaborn as sns
# Evaluate the model
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze(1)
        predicted = (outputs > 0.5).float()
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Convert predictions and true labels to numpy for evaluation
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
class_report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
plt.figure(figsize=(10, 6))
sns.heatmap(class_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report')
plt.show()

# Print evaluation metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nAccuracy: ", accuracy_score(y_true, y_pred))
# Save the entire model (including the architecture)
torch.save(model, r"D:\CS\AI\PROJECT\coconut_project\model_full.pth")
# Save model weights (state_dict)
torch.save(model.state_dict(), r"D:\CS\AI\PROJECT\coconut_project\model_weights.pth")
print("Model saved successfully")
