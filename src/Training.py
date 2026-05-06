"""
DESCRIPTION:
This script performs binary classification using a Feedforward Neural Network (FFNN) 
implemented in PyTorch. It includes various commented-out machine learning models 
(XGBoost, LightGBM, CatBoost, GradientBoosting, AdaBoost, HistGradientBoosting, MLP, 
and Stacking) for reference. The active code loads an audio feature dataset, 
preprocesses the data (Label Encoding, Standard Scaling, and SMOTE for class balancing), 
trains the PyTorch neural network, visualizes the evaluation metrics (confusion matrix 
and classification report heatmaps), and saves the final trained model.

INPUTS:
- A CSV file ('features_mfcc_contrast.csv') containing extracted acoustic features.
  The dataset is expected to have a 'label' column, a 'file' column (which is dropped), 
  and numerical feature columns.

OUTPUTS:
- Console outputs displaying training progress (loss and accuracy per epoch).
- Final evaluation metrics printed to the console (Confusion Matrix, Classification Report, Accuracy).
- Visual plots (Seaborn heatmaps) for the Confusion Matrix and Classification Report.
- Saved PyTorch model files ('model_full.pth' for the entire model and 'model_weights.pth' for state dict).
"""

'''
#XGBOOST
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv(r"D:\CS\AI\PROJECT\coconut_project\features_mfcc_contrast.csv")
df = df.drop(columns=['file'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
X = df.drop(columns=['label'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss')

# Hyperparameter grid for tuning
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],          # Step size shrinkage
    'n_estimators': [100, 200, 500],             # Number of boosting rounds
    'max_depth': [3, 5, 7],                       # Maximum depth of trees
    'subsample': [0.8, 0.9, 1.0],                 # Fraction of samples used per boosting round
    'colsample_bytree': [0.8, 0.9, 1.0]           # Fraction of features used per tree
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters for XGBoost: ", grid_search.best_params_)

# Get the best model from the grid search
best_xgb_model = grid_search.best_estimator_

# Fit the best model on the training data
best_xgb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = best_xgb_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

'''
'''
#LIGHTGBM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
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

# Define LightGBM model with hyperparameters
lgb_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,  # Small learning rate
    max_depth=10,        # Limit the depth of the trees
    num_leaves=31,       # Set number of leaves per tree
    min_child_samples=20,  # Minimum samples per leaf
    objective='binary',
    random_state=42
)

# Train the model
lgb_model.fit(X_train_scaled, y_train_resampled)

# Predict on test set
y_pred = lgb_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot
importances = lgb_model.feature_importances_
features = X.columns

# Plot the importance
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('LightGBM Feature Importance')
plt.show()

'''
'''
#CATBOOST
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
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

# Define CatBoost model with hyperparameters
catboost_model = CatBoostClassifier(
    iterations=1000,  # Number of boosting iterations
    depth=10,         # Maximum depth of trees
    learning_rate=0.05,  # Learning rate
    l2_leaf_reg=3,    # L2 regularization
    loss_function='Logloss',  # Binary classification loss
    random_seed=42,   # Set random seed for reproducibility
    cat_features=[],  # No categorical features in the current dataset
    verbose=200       # Print progress every 200 iterations
)

# Train the model
catboost_model.fit(X_train_scaled, y_train_resampled)

# Predict on test set
y_pred = catboost_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot
importances = catboost_model.get_feature_importance()
features = X.columns

# Plot the importance
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('CatBoost Feature Importance')
plt.show()
'''
'''
#GradientBoosting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
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

# Define Gradient Boosting model with hyperparameters
gb_model = GradientBoostingClassifier(
    n_estimators=100,    # Number of boosting stages to be used
    learning_rate=0.1,   # Learning rate to shrink the contribution of each tree
    max_depth=3,         # Max depth of the individual trees
    random_state=42      # Random seed for reproducibility
)

# Train the model
gb_model.fit(X_train_scaled, y_train_resampled)

# Predict on test set
y_pred = gb_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot
importances = gb_model.feature_importances_
features = X.columns

# Plot the importance
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Gradient Boosting Feature Importance')
plt.show()
'''
'''
#Adaboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
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

# Define the base estimator (e.g., DecisionTreeClassifier)
base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)

# Initialize AdaBoostClassifier using the default base estimator (DecisionTree) but set parameters for it
ada_model = AdaBoostClassifier(
    n_estimators=100,           # Number of boosting rounds
    learning_rate=0.1,          # Weight applied to each estimator
    random_state=42             # Reproducibility
)

# Now, instead of specifying 'base_estimator', you can define the model as below:
# Create a pipeline of AdaBoost with a custom base estimator

# Train the model
ada_model.fit(X_train_scaled, y_train_resampled)

# Predict on the test set
y_pred = ada_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot (only available if base_estimator has it, like DecisionTree)
if hasattr(ada_model, 'feature_importances_'):
    importances = ada_model.feature_importances_
    features = X.columns

    # Plot the importance
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('AdaBoost Feature Importance')
    plt.show()
'''
'''
#HistogramBoosting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_hist_gradient_boosting  # Ensure the experimental version is enabled
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load dataset
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

# Initialize HistGradientBoostingClassifier
hist_model = HistGradientBoostingClassifier(
    max_iter=100,                # Number of boosting iterations
    learning_rate=0.1,           # Learning rate (controls the step size in gradient descent)
    max_depth=3,                 # Maximum depth of each tree
    random_state=42              # Random state for reproducibility
)

# Train the model
hist_model.fit(X_train_scaled, y_train_resampled)

# Predict on the test set
y_pred = hist_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot (only available if model supports it)
if hasattr(hist_model, 'feature_importances_'):
    importances = hist_model.feature_importances_
    features = X.columns

    # Plot the importance
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('HistGradientBoosting Feature Importance')
    plt.show()
    '''
'''
MLP : 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

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

# Initialize the MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, solver='adam', activation='relu')

# Train the model
mlp_model.fit(X_train_scaled, y_train_resampled)

# Predict on the test set
y_pred = mlp_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
'''
'''
#MLP+ XGBOOST+ CATBOOST
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
df = pd.read_csv(r"D:\CS\AI\PROJECT\coconut_project\features_mfcc_contrast.csv")

# Drop the 'file' column (not needed for training)
df = df.drop(columns=['file'])

# Encode labels
from sklearn.preprocessing import LabelEncoder
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

# Define the base models
xgb_model = XGBClassifier(eval_metric='mlogloss')
catboost_model = CatBoostClassifier(silent=True)
mlp_model = MLPClassifier(max_iter=500)

# Define the meta-model (Logistic Regression in this case)
meta_model = LogisticRegression()

# Define the stacking classifier with XGBoost, CatBoost, MLP as base models
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('catboost', catboost_model),
        ('mlp', mlp_model)
    ],
    final_estimator=meta_model
)

# Train the stacking classifier
stacking_model.fit(X_train_scaled, y_train_resampled)

# Make predictions
y_pred = stacking_model.predict(X_test_scaled)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
'''

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
