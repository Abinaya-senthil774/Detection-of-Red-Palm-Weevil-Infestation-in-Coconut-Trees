# Detection of Red-Palm Weevil Infestation in Coconut Trees Using Neural Network Systems

The red palm weevil (*Rhynchophorus ferrugineus*) is an invasive pest that causes severe, often hidden damage to coconut palms worldwide. Traditional visual inspections are ineffective for early-stage detection. This project proposes a neural network-based approach to detect infestations early by classifying the acoustic signals (bioacoustics) emitted by larvae feeding inside the tree trunk. 

By analyzing Mel-Frequency Cepstral Coefficients (MFCC) and spectral contrast features from audio recordings, our Feedforward Neural Network (FFNN) achieves an 85% testing accuracy, offering a scalable, low-cost solution for precision agriculture and early pest intervention.

##Key Fearures
*   **Non-Invasive Detection:** Uses audio signals to detect pests inside the tree trunk without causing harm to the tree.
*   **Advanced Preprocessing:** Utilizes Continuous Wavelet Transform (CWT), Singular Value Decomposition (SVD), and bandpass filtering to denoise and segment audio effectively.
*   **Custom Neural Network:** A lightweight PyTorch Feedforward Neural Network optimized for binary classification (Infested vs. Healthy).
*   **Class Balancing:** Implements SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance in the acoustic dataset.

## 📊 Dataset
The project utilizes the **Tree Vibes** dataset[cite: 1].
* **Total Records:** 5,153 audio samples[cite: 1].
* **Classes:** 731 infested sound files and 1,734 non-infested (clean) sound files[cite: 1].
* **Testing Data:** 692 test case files were provided in the dataset[cite: 1].
* **Source:** [Tree Vibes Dataset on Kaggle](https://www.kaggle.com/datasets/potamitis/treevibes/data)[cite: 1].

## 🧠 Model Architecture
The core classification model is a Feedforward Neural Network (FFNN) built in PyTorch, structured as follows:
1. **Input Layer:** Takes the 20-dimensional feature vector (13 MFCCs + 7 Spectral Contrast bands)[cite: 1].
2. **Hidden Layer 1:** 512 neurons with ReLU activation[cite: 1].
3. **Hidden Layer 2:** 256 neurons with ReLU activation[cite: 1].
4. **Hidden Layer 3:** 128 neurons with ReLU activation[cite: 1].
5. **Output Layer:** 1 neuron with Sigmoid activation for binary classification (Probability > 0.5 indicates infestation)[cite: 1].

## 🏆 Performance & Results
The proposed neural network model achieved strong overall classification performance, outperforming other tested ensemble methods like XGBoost and CatBoost[cite: 1].
* **Testing Accuracy:** 85.06%[cite: 1]
* **Training Accuracy:** 94.04%[cite: 1]
* **Precision (Infested):** 0.69[cite: 1]
* **Recall (Infested):** 0.80[cite: 1]
* **F1-Score (Infested):** 0.74[cite: 1]

The high recall value (0.80) is particularly significant for pest detection applications, ensuring that actual infestations are rarely missed (minimizing false negatives)[cite: 1].

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8+
* Virtual Environment (Recommended)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/coconut-weevil-acoustic-detection.git
cd coconut-weevil-acoustic-detection
