import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rich.console import Console
from rich.panel import Panel

console = Console()

# --- 1. DEFINE THE MODEL ARCHITECTURE (Must match training) ---
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

# --- 2. SETUP ---
MODEL_PATH = "best_tox_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained brain
model = ToxNet().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to "Evaluation Mode" (Turns off Dropout)
    console.print(f"[green]✅ Loaded forensic model from {MODEL_PATH}[/green]")
except FileNotFoundError:
    console.print(f"[bold red]❌ Error: {MODEL_PATH} not found![/bold red]")
    exit()

# Setup Fingerprint Generator (The same one used in training)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def predict_chemical(smiles_string):
    # 1. Convert Text -> Molecule
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None, "Invalid Chemical Structure"

    # 2. Molecule -> Numbers (Fingerprint)
    fp = morgan_gen.GetFingerprint(mol)
    np_fp = np.array(fp, dtype=np.float32)
    
    # 3. Numbers -> Tensor -> GPU
    tensor_fp = torch.FloatTensor(np_fp).unsqueeze(0).to(device) # Add batch dimension

    # 4. Predict
    with torch.no_grad():
        output = model(tensor_fp)
        probability = output.item()
    
    return probability, mol

# --- 3. INTERACTIVE LOOP ---
console.print(Panel.fit("[bold cyan]Forensic Toxicology Scanner[/bold cyan]\nType 'exit' to quit.", border_style="cyan"))

while True:
    user_input = input("\n🧪 Enter SMILES string: ").strip()
    
    if user_input.lower() in ['exit', 'quit']:
        break
    
    prob, result = predict_chemical(user_input)
    
    if prob is None:
        console.print("[bold red]❌ Invalid SMILES string. Try again.[/bold red]")
    else:
        # Interpret the result
        percentage = prob * 100
        color = "green" if percentage < 50 else "red"
        verdict = "SAFE" if percentage < 50 else "TOXIC"
        
        console.print(f"   Probability of Toxicity: [{color}]{percentage:.2f}%[/{color}]")
        console.print(f"   AI Verdict: [bold {color}]{verdict}[/bold {color}]")