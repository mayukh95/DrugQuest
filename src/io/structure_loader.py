"""
Structure loading tools for DFT optimized coordinates.

This module handles loading of XYZ coordinates from DFT optimization results
and mapping them to RDKit molecules.
"""

from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def load_xyz_coordinates(mol_name, base_dir='optimized_molecules'):
    """
    Load XYZ coordinates from DFT optimization output file.
    
    Parameters:
    -----------
    mol_name : str
        Name of the molecule (folder name)
    base_dir : str, optional
        Base directory containing optimization results (default: 'optimized_molecules')
        
    Returns:
    --------
    dict or None
        Dictionary with keys 'atoms' (list of symbols) and 'coords' (numpy array of xyz)
        Returns None if file not found.
    """
    # Handle clean names vs directory names (which often have _optimized)
    # The convention in the notebook seems to be that folders might be named
    # either 'Molecule' or 'Molecule_optimized', and contain 'Molecule_optimized.xyz'
    
    clean_name = mol_name.replace('_optimized', '')
    
    # Try multiple path variations to be robust
    paths_to_try = [
        Path(base_dir) / mol_name / f"{clean_name}_optimized.xyz",
        Path(base_dir) / f"{clean_name}_optimized" / f"{clean_name}_optimized.xyz",
        Path(base_dir) / clean_name / f"{clean_name}_optimized.xyz"
    ]
    
    xyz_path = None
    for p in paths_to_try:
        if p.exists():
            xyz_path = p
            break
            
    if xyz_path is None:
        return None
        
    try:
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
            
        n_atoms = int(lines[0].strip())
        atoms, coords = [], []
        
        # Parse XYZ content (skip first 2 lines: count and comment)
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
                
        return {'atoms': atoms, 'coords': np.array(coords)}
    except Exception as e:
        print(f"Error reading XYZ file for {mol_name}: {e}")
        return None


def create_mol_with_dft_coords(smiles, mol_name, base_dir='optimized_molecules'):
    """
    Create RDKit molecule with DFT-optimized coordinates if available.
    
    Parameters:
    -----------
    smiles : str
        SMILES string
    mol_name : str
        Name of the molecule
    base_dir : str, optional
        Base directory for optimized structures
        
    Returns:
    --------
    rdkit.Chem.Mol
        Molecule with 3D conformer (DFT coordinates if found, else MMFF)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    mol = Chem.AddHs(mol)
    
    # Try to load DFT coordinates
    xyz_data = load_xyz_coordinates(mol_name, base_dir)
    
    if xyz_data is None or mol.GetNumAtoms() != len(xyz_data['atoms']):
        # Fallback to MMFF optimization if DFT not found or atom count mismatch
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass  # Keep 2D if embedding fails
        return mol
        
    # Apply DFT coordinates to conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, coord in enumerate(xyz_data['coords']):
        conf.SetAtomPosition(i, coord)
        
    mol.AddConformer(conf, assignId=True)
    return mol
