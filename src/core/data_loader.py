"""
Data loading functions for molecules from various sources.
"""

import pandas as pd
from pathlib import Path
from .molecule import DrugMolecule


def load_drug_dataset(df):
    """
    Load drug dataset from DataFrame and create DrugMolecule objects.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Drug Name', 'CAS Registry Number', 
                          'SMILES', 'XYZ']

    Returns:
        list: List of DrugMolecule objects
    """
    molecules = []
    failed = []

    for idx, row in df.iterrows():
        try:
            mol = DrugMolecule(
                name=row['Drug Name'],
                cas=row.get('CAS Registry Number', 'N/A'),
                smiles=row['SMILES'],
                xyz_string=row.get('XYZ', '')
            )
            molecules.append(mol)
        except Exception as e:
            failed.append((row['Drug Name'], str(e)))
            print(f"⚠ Failed to parse {row['Drug Name']}: {e}")

    print(f"\n✓ Successfully loaded {len(molecules)} molecules")
    if failed:
        print(f"✗ Failed to load {len(failed)} molecules")

    return molecules


def load_molecules_from_fda_csv(csv_path='FDA_Approved_structures.csv', max_molecules=50, 
                                min_atoms=5, max_atoms=100):
    """
    Load molecules from FDA Approved Structures CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the FDA_Approved_structures.csv file
    max_molecules : int
        Maximum number of molecules to load (for faster processing)
    min_atoms : int
        Minimum number of heavy atoms (filter out very small molecules)
    max_atoms : int
        Maximum number of heavy atoms (filter out very large molecules)
    
    Returns:
    --------
    list : List of DrugMolecule objects
    """
    print(f"📂 Loading FDA Approved Drugs from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Found {len(df)} total drugs in database")
        
        molecules = []
        skipped = 0
        
        for idx, row in df.iterrows():
            if len(molecules) >= max_molecules:
                break
            
            name = row['Name']
            smiles = row['SMILES']
            
            # Basic validation
            if pd.isna(smiles) or not smiles or len(smiles) < 5:
                skipped += 1
                continue
            
            # Check molecule size
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    skipped += 1
                    continue
                
                n_atoms = mol.GetNumHeavyAtoms()
                if n_atoms < min_atoms or n_atoms > max_atoms:
                    skipped += 1
                    continue
                
                # Create DrugMolecule object
                drug_mol = DrugMolecule(
                    name=name,
                    cas='N/A',  # FDA database doesn't include CAS
                    smiles=smiles,
                    xyz_string=''  # Will be generated from SMILES
                )
                molecules.append(drug_mol)
                
            except Exception as e:
                skipped += 1
                continue
        
        print(f"✓ Successfully loaded {len(molecules)} molecules")
        print(f"⏭️ Skipped {skipped} molecules (invalid SMILES or wrong size)")
        
        return molecules
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_path}")
        print("   Please provide the FDA_Approved_structures.csv file")
        return []
    except Exception as e:
        print(f"❌ Error loading FDA database: {e}")
        return []


def create_sample_dataset():
    """Create a sample dataset for testing."""
    sample_data = [
        {
            'Drug Name': 'Aspirin',
            'CAS Registry Number': '50-78-2',
            'SMILES': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'XYZ': ''
        },
        {
            'Drug Name': 'Caffeine', 
            'CAS Registry Number': '58-08-2',
            'SMILES': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'XYZ': ''
        },
        {
            'Drug Name': 'Ibuprofen',
            'CAS Registry Number': '15687-27-1', 
            'SMILES': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'XYZ': ''
        }
    ]
    
    df = pd.DataFrame(sample_data)
    return load_drug_dataset(df)