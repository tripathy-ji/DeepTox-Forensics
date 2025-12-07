import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools, rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import copy

# --- RICH IMPORTS ---
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.live import Live

# Initialize Rich Console
console = Console()

# --- 1. CONFIGURATION & HARDWARE ---
SDF_PATH = r"E:\projects\tox\data\tox21_10k_data_all.sdf\tox21_10k_data_all.sdf"
TARGET_LABEL = 'NR-AhR'
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5

console.print(Panel.fit("[bold cyan]Forensic AI Training Tool[/bold cyan]\nTarget: [yellow]Liver Toxicity (NR-AhR)[/yellow]", title="System Startup"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
console.print(f"[bold green]✅ Hardware Detected:[/bold green] {gpu_name}")

# --- 2. DATA LOADING (With Spinner) ---
if not os.path.exists(SDF_PATH):
    console.print(f"[bold red]❌ CRITICAL ERROR:[/bold red] File not found at {SDF_PATH}")
    exit()

df = None
with console.status("[bold blue]⚗️  Reading Chemical Database...[/bold blue]", spinner="dots"):
    try:
        df = PandasTools.LoadSDF(SDF_PATH, embedProps=True, molColName='ROMol')
    except Exception as e:
        console.print(f"[bold red]❌ Error reading SDF:[/bold red] {e}")
        exit()

with console.status("[bold blue]🧹  Cleaning & Preprocessing Data...[/bold blue]", spinner="dots"):
    # Clean Data
    df['target'] = pd.to_numeric(df[TARGET_LABEL], errors='coerce')
    df = df.dropna(subset=['target'])
    df = df.reset_index(drop=True)

    # Generate Features (Morgan Fingerprints)
    # Optimization: Instantiate generator once outside the loop
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def mol_to_fp(mol):
        if mol is None: return np.zeros(2048)
        # Use the modern generator
        fp = morgan_gen.GetFingerprint(mol)
        return np.array(fp)

    X_data = np.array([mol_to_fp(m) for m in df['ROMol']])
    y_data = df['target'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

console.print(f"[green]✅ Data Ready![/green] Training on [bold]{len(X_train)}[/bold] compounds.")

# --- 3. PYTORCH SETUP ---
class ToxDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(ToxDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ToxDataset(X_test, y_test), batch_size=BATCH_SIZE)

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

model = ToxNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss()

# --- 4. TRAINING LOOP (With Rich Progress) ---
best_auc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience = 5
no_improve_count = 0

# Create a layout table for live updates
results_table = Table(title="Training Metrics", box=box.ROUNDED)
results_table.add_column("Epoch", justify="center", style="cyan")
results_table.add_column("Train Loss", justify="right", style="magenta")
results_table.add_column("Val AUC", justify="right", style="green")
results_table.add_column("Status", justify="left")

# Progress Bar Configuration
progress = Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn()
)

console.print("\n[bold]🚀 Starting Training Session...[/bold]")

# Using Live display to show the table growing
with Live(results_table, console=console, refresh_per_second=4):
    with progress:
        # Main Epoch Task
        epoch_task = progress.add_task("[green]Total Progress...", total=EPOCHS)
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            # Inner Batch Task (Shows detailed steps)
            batch_task = progress.add_task(f"  Epoch {epoch+1} Processing...", total=len(train_loader))
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Advance batch bar
                progress.update(batch_task, advance=1)
            
            # Remove batch bar after epoch finishes to keep it clean
            progress.remove_task(batch_task)
            
            # Validation
            avg_train_loss = running_loss / len(train_loader)
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            val_auc = roc_auc_score(all_labels, all_preds)
            
            # Logic for Best Model
            status_msg = ""
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_count = 0
                status_msg = "⭐ New Best!"
                torch.save(model.state_dict(), "best_tox_model.pth")
            else:
                no_improve_count += 1
                status_msg = f"No improve ({no_improve_count}/{patience})"
            
            # Add row to table
            results_table.add_row(f"{epoch+1}", f"{avg_train_loss:.4f}", f"{val_auc:.4f}", status_msg)
            
            # Advance Epoch Bar
            progress.update(epoch_task, advance=1)
            
            if no_improve_count >= patience:
                results_table.add_row("---", "---", "---", "[bold red]EARLY STOPPING[/bold red]")
                break

# --- 5. FINAL SUMMARY ---
console.print(Panel.fit(
    f"[bold]Training Complete![/bold]\n\n"
    f"🏆 Best ROC-AUC: [green]{best_auc:.4f}[/green]\n"
    f"💾 Model Saved: [yellow]best_tox_model.pth[/yellow]",
    title="Summary",
    border_style="green"
))