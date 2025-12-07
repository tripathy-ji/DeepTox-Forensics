# 🧪 AI Forensic Toxicology Predictor

**An Deep Learning tool for In Silico Toxicology.**
This project uses a Neural Network to predict the toxicity of chemical compounds (specifically **NR-AhR**, associated with liver toxicity and environmental toxins) based solely on their molecular structure.

## 📊 Performance
- **Target:** NR-AhR (Aryl Hydrocarbon Receptor)
- **Model Accuracy (ROC-AUC):** **0.93** (Forensic Grade)
- **Dataset:** NIH Tox21 Data Challenge (~10,000 compounds)
- **Tech Stack:** Python, PyTorch, RDKit, Rich

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/forensic-tox-ai.git
   cd forensic-tox-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

### 1. Train the Model
To retrain the neural network from scratch using the Tox21 dataset:
```bash
python train_tox.py
```
This will train on the GPU (if available) and save the best model to `best_tox_model.pth`.

### 2. Predict Toxicity (Forensic Scanner)
To scan new chemicals using the trained brain:
```bash
python predict_tox.py
```

**Example Input (Dioxin):** `Clc1cc2OC3=CC(=C(Cl)C=C3O2)Cl)Cl`

**AI Verdict:** 🔴 TOXIC (98.5%)

## 📂 Project Structure
- `train_tox.py`: Main training pipeline with Early Stopping & Dropout.
- `predict_tox.py`: Interactive scanner for forensic analysis.
- `best_tox_model.pth`: The trained PyTorch model weights.

---
Created by a Forensic Science Student at NFSU.
