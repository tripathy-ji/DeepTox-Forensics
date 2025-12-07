import sys

print("1. Starting Diagnostic...")

try:
    print("   -> Importing Numpy...")
    import numpy
    print("   ✅ Numpy is OK.")
except Exception as e:
    print(f"   ❌ Numpy Failed: {e}")

try:
    print("   -> Importing Pandas...")
    import pandas
    print("   ✅ Pandas is OK.")
except Exception as e:
    print(f"   ❌ Pandas Failed: {e}")

try:
    print("   -> Importing Torch (This checks your GPU)...")
    import torch
    print(f"   ✅ Torch is OK. Version: {torch.__version__}")
    print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ GPU Detected: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ❌ Torch Failed: {e}")

try:
    print("   -> Importing RDKit (The usual suspect)...")
    from rdkit import Chem
    print("   ✅ RDKit is OK.")
except Exception as e:
    print(f"   ❌ RDKit Failed: {e}")

print("2. Diagnostic Complete.")