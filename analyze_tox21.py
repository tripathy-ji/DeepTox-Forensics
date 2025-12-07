import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
import os

# Define the file path
sdf_path = r"E:\projects\tox\data\tox21_10k_data_all.sdf\tox21_10k_data_all.sdf"

def analyze_tox21():
    print(f"Loading data from {sdf_path}...")
    
    # Load the SDF file into a Pandas DataFrame
    # embedProps=True is default, but good to be explicit if we want properties
    # molColName='ROMol' is default
    df = PandasTools.LoadSDF(sdf_path, embedProps=True, molColName='ROMol')
    
    print("Data loaded successfully.")
    
    # Inspect Labels: Print the first 5 rows specifically showing the columns Formula, NR-AhR, and SR-p53.
    # We also keep ROMol to ensure we have the molecule data, but for printing we select specific columns.
    columns_to_inspect = ['Formula', 'NR-AhR', 'SR-p53']
    
    # Check if columns exist
    missing_cols = [c for c in columns_to_inspect if c not in df.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} not found in dataset. Available columns: {df.columns.tolist()}")
    else:
        print("\n--- First 5 rows (Formula, NR-AhR, SR-p53) ---")
        print(df[columns_to_inspect].head(5))

    # Clean the Data: Check for any "missing" values (NaNs) in the NR-AhR column 
    # and report how many usable compounds we have for that specific toxicity target.
    if 'NR-AhR' in df.columns:
        # Convert to numeric if possible, errors='coerce' turns non-numeric to NaN
        # This helps if the SDF loaded them as strings
        df['NR-AhR_numeric'] = pd.to_numeric(df['NR-AhR'], errors='coerce')
        
        usable_count = df['NR-AhR_numeric'].notna().sum()
        print(f"\n--- NR-AhR Data Quality ---")
        print(f"Total rows: {len(df)}")
        print(f"Usable compounds for NR-AhR (non-NaN): {usable_count}")
        
        # Visualize: Use rdkit to draw the chemical structure of the very first toxic compound found in the dataset.
        # Assuming '1' indicates toxicity.
        toxic_compounds = df[df['NR-AhR_numeric'] == 1]
        
        if not toxic_compounds.empty:
            first_toxic_mol = toxic_compounds.iloc[0]['ROMol']
            first_toxic_id = toxic_compounds.iloc[0].get('DSSTox_CID', 'Unknown_ID') # Try to get an ID if available
            
            print(f"\n--- Visualization ---")
            print(f"Found {len(toxic_compounds)} toxic compounds for NR-AhR.")
            print(f"Visualizing the first toxic compound (Index: {toxic_compounds.index[0]})...")
            
            # Draw the molecule
            # We will save it to a file since we are in a script environment
            img = Draw.MolToImage(first_toxic_mol, legend=f"First Toxic Compound (NR-AhR)\nID: {first_toxic_id}")
            img_path = "first_toxic_compound.png"
            img.save(img_path)
            print(f"Image saved to {os.path.abspath(img_path)}")
        else:
            print("No toxic compounds (NR-AhR == 1) found.")
    else:
        print("Column 'NR-AhR' not found, cannot perform cleaning or visualization based on it.")

if __name__ == "__main__":
    analyze_tox21()
