import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="DeepTox Forensics", page_icon="🧪")

st.title("🧪 DeepTox Forensic Scanner")
st.write("Predict chemical toxicity (NR-AhR) using Deep Learning.")

# --- 2. MODEL ARCHITECTURE (Must match training) ---
class ToxNet(nn.Module):
    def __init__(self):
        super(ToxNet, self).__init__()
        self.layer1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.4)
        self.layer2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu(self.bn2(self.layer2(x))))
        x = self.sigmoid(self.layer3(x))
        return x

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    device = torch.device("cpu") # Cloud servers use CPU usually
    model = ToxNet().to(device)
    
    # Load weights
    if os.path.exists("best_tox_model.pth"):
        model.load_state_dict(torch.load("best_tox_model.pth", map_location=device))
    else:
        st.error("Model file not found! Please upload best_tox_model.pth.")
        return None
    
    model.eval()
    return model

model = load_model()
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# --- 4. UI & PREDICTION ---
input_smiles = st.text_input("Enter SMILES String:", placeholder="e.g. CCO")

if st.button("Analyze Toxicity"):
    if not input_smiles:
        st.warning("Please enter a chemical string.")
    else:
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            st.error("Invalid Chemical Structure.")
        else:
            # Visualize
            st.image(Chem.Draw.MolToImage(mol), caption="Chemical Structure", width=300)
            
            # Predict
            fp = morgan_gen.GetFingerprint(mol)
            np_fp = np.array(fp, dtype=np.float32)
            tensor_fp = torch.FloatTensor(np_fp).unsqueeze(0) # CPU tensor
            
            with torch.no_grad():
                output = model(tensor_fp)
                prob = output.item()
            
            percent = prob * 100
            if percent > 50:
                st.error(f"🔴 TOXIC DETECTED ({percent:.2f}%)")
            else:
                st.success(f"🟢 SAFE ({percent:.2f}%)")