"""
3D Shape Similarity Analysis using Open3DAlign (O3A) and Shape Tanimoto.

This module provides functions for computing 3D shape similarity between molecules
and visualizing aligned molecular structures.
"""

import numpy as np
import pandas as pd
import py3Dmol
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdShapeHelpers


def compute_shape_similarity(mol1, mol2):
    """
    Compute 3D shape similarity between two molecules using O3A alignment.
    
    Parameters:
    -----------
    mol1 : rdkit.Chem.Mol
        First molecule (must have 3D conformer)
    mol2 : rdkit.Chem.Mol
        Second molecule (must have 3D conformer)
    
    Returns:
    --------
    float
        Shape Tanimoto similarity (0-1, where 1 is identical)
    """
    if not mol1 or not mol2 or mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
        return 0.0
    try:
        # O3A Alignment
        o3a = rdMolAlign.GetO3A(mol2, mol1)
        o3a.Align()
        return 1 - rdShapeHelpers.ShapeTanimotoDist(mol1, mol2)
    except:
        return 0.0


def compute_shape_matrix(molecules, max_molecules=50):
    """
    Compute pairwise 3D shape similarity matrix for a list of molecules.
    
    Parameters:
    -----------
    molecules : list
        List of DrugMolecule objects
    max_molecules : int, optional
        Maximum number of molecules to process (default: 50)
    
    Returns:
    --------
    pd.DataFrame
        Symmetric matrix of shape similarities
    """
    print("⏳ Calculating 3D Shape Similarity Matrix (O3A Alignment)...")
    
    # Limit to top N molecules for speed
    mols_to_shape = molecules[:max_molecules]
    names = [m.name for m in mols_to_shape]
    n = len(mols_to_shape)
    
    shape_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            s = compute_shape_similarity(mols_to_shape[i].mol, mols_to_shape[j].mol)
            shape_mat[i, j] = shape_mat[j, i] = s
    
    sim_shape = pd.DataFrame(shape_mat, index=names, columns=names)
    print(f"✅ Computed Shape Matrix: {sim_shape.shape}")
    
    return sim_shape


def visualize_shape_alignment(molecules, mol1_name, mol2_name):
    """
    Visualize 3D shape alignment between two molecules.
    
    Parameters:
    -----------
    molecules : list
        List of DrugMolecule objects to search from
    mol1_name : str
        Name of the first molecule (fixed reference)
    mol2_name : str
        Name of the second molecule (mobile, will be aligned)
    """
    # Find molecule objects
    obj1 = next((m for m in molecules if m.name == mol1_name), None)
    obj2 = next((m for m in molecules if m.name == mol2_name), None)
    
    if not obj1 or not obj2 or not obj1.mol or not obj2.mol:
        print("❌ Could not load molecules")
        return
    
    # Create copies for alignment visualization
    m1 = Chem.Mol(obj1.mol)
    m2 = Chem.Mol(obj2.mol)
    
    try:
        o3a = rdMolAlign.GetO3A(m2, m1)
        rmsd = o3a.Align()
        shape_sim = 1 - rdShapeHelpers.ShapeTanimotoDist(m1, m2)
    except Exception as e:
        print(f"Alignment failed: {e}")
        return
    
    # Create 3D viewer
    view = py3Dmol.view(width=800, height=500)
    
    # Add Mol 1 (Fixed) - Cyan
    view.addModel(Chem.MolToMolBlock(m1), 'mol')
    view.setStyle({'model': 0}, {'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.15}})
    
    # Add Mol 2 (Aligned) - Magenta
    view.addModel(Chem.MolToMolBlock(m2), 'mol')
    view.setStyle({'model': 1}, {'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.15}})
    
    view.zoomTo()
    
    display(HTML(f"""
    <div style='text-align:center; font-family:sans-serif'>
        <h4>🧊 Shape Overlay</h4>
        <b>{mol1_name}</b> (Cyan) vs <b>{mol2_name}</b> (Magenta)<br>
        Shape Sim: <b>{shape_sim:.3f}</b> | RMSD: <b>{rmsd:.2f} Å</b>
    </div>
    """))
    view.show()


def create_shape_alignment_widget(molecules):
    """
    Create an interactive widget for shape alignment visualization.
    
    Parameters:
    -----------
    molecules : list
        List of DrugMolecule objects
    
    Returns:
    --------
    ipywidgets.VBox
        Interactive widget for molecule selection and alignment
    """
    mol_names = sorted([m.name for m in molecules])
    
    # Set defaults
    ref_def = 'Ibuprofen' if 'Ibuprofen' in mol_names else mol_names[0]
    cand_def = 'Naproxen' if 'Naproxen' in mol_names else (mol_names[1] if len(mol_names) > 1 else mol_names[0])
    
    d1 = widgets.Dropdown(options=mol_names, value=ref_def, description='Fixed:')
    d2 = widgets.Dropdown(options=mol_names, value=cand_def, description='Mobile:')
    b_align = widgets.Button(description='Align 3D', button_style='primary')
    out_align = widgets.Output()
    
    def on_click_align(_):
        with out_align:
            clear_output(wait=True)
            visualize_shape_alignment(molecules, d1.value, d2.value)
    
    b_align.on_click(on_click_align)
    
    return widgets.VBox([
        widgets.HBox([d1, d2, b_align]),
        out_align
    ])
