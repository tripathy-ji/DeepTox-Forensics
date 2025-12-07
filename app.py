import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE
# ==========================================
# This MUST match the class used in train_tox.py exactly
class ToxNet(nn.Module):
    def __init__(self):
        super(ToxNet, self).__init__()
        # Layer 1
        self.layer1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.4)
        # Layer 2
        self.layer2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        # Layer 3
        self.layer3 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu(self.bn2(self.layer2(x))))
        x = self.sigmoid(self.layer3(x))
        return x

# ==========================================
# 2. LOAD MODEL AND CONFIGURATION
# ==========================================
DEVICE = torch.device("cpu") # Cloud servers use CPU
MODEL_PATH = "best_tox_model.pth"

# Initialize model
model = ToxNet().to(DEVICE)

# Setup Fingerprint Generator (New Method - No Warnings)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Load weights
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Successfully loaded weights from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("Did you change the ToxNet class? It must match train_tox.py exactly.")
else:
    print(f"⚠️ Warning: {MODEL_PATH} not found. Predictions will be random.")

model.eval()

# ==========================================
# 3. FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    smiles = data.get('smiles', '').strip()

    if not smiles:
        return jsonify({'error': 'Please enter a SMILES string.'}), 400

    # Process Molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': 'Invalid Chemical Structure.'}), 400

    # Generate Fingerprint (The Modern Way)
    fp = morgan_gen.GetFingerprint(mol)
    np_fp = np.array(fp, dtype=np.float32)
    
    # Add batch dimension (1, 2048)
    input_tensor = torch.tensor(np_fp).unsqueeze(0).to(DEVICE)

    # Inference
    try:
        with torch.no_grad():
            prediction = model(input_tensor)
            probability = prediction.item()
            
            # Forensic Threshold logic
            is_toxic = probability > 0.50
            label = "TOXIC" if is_toxic else "SAFE"
            
            return jsonify({
                'smiles': smiles,
                'prediction': label,
                'probability': f"{probability:.4f}",
                'is_toxic': is_toxic
            })
            
    except Exception as e:
        return jsonify({'error': f"Inference error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)