"""
Interactive 3D Pharmacophore Viewer Widget.

This module provides an interactive widget for exploring 3D pharmacophore features
of molecules in a dataset.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import py3Dmol
from rdkit import Chem


# Constants for pharmacophore colors
PHARM_COLORS = {
    'Donor': '0x3498db', 'Acceptor': '0xe74c3c', 'Aromatic': '0x9b59b6',
    'Hydrophobe': '0x2ecc71', 'LumpedHydrophobe': '0x27ae60',
    'PosIonizable': '0xf39c12', 'NegIonizable': '0xe67e22'
}


def create_pharmacophore_explorer(df, visualize_func, default_molecule=None):
    """
    Create an interactive widget for exploring 3D pharmacophores of molecules.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing molecule data with 'Molecule_Name' and 'SMILES' columns
    visualize_func : callable
        Function to visualize pharmacophore, signature: visualize_func(mol_name, smiles)
    default_molecule : str, optional
        Default molecule to display (default: first in list)
    
    Returns:
    --------
    ipywidgets.VBox
        Interactive widget with dropdown and visualization
    """
    molecule_names = sorted(df['Molecule_Name'].tolist())
    
    # Set default
    if default_molecule and default_molecule in molecule_names:
        default_value = default_molecule
    else:
        default_value = molecule_names[0] if molecule_names else None
    
    mol_dropdown = widgets.Dropdown(
        options=molecule_names,
        value=default_value,
        description='Molecule:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='300px')
    )
    
    view_btn = widgets.Button(
        description='🔬 View 3D Pharmacophore',
        button_style='primary'
    )
    
    output = widgets.Output()
    
    def on_view_click(b):
        with output:
            clear_output(wait=True)
            mol_name = mol_dropdown.value
            mol_row = df[df['Molecule_Name'] == mol_name].iloc[0]
            visualize_func(mol_name, mol_row['SMILES'])
    
    view_btn.on_click(on_view_click)
    
    return widgets.VBox([
        widgets.HTML("<h4>🔍 Interactive Molecule Explorer</h4>"),
        widgets.HBox([mol_dropdown, view_btn]),
        output
    ])


def compare_pharmacophores_grid(reference_mol, hit_mols, df, mol_creator_func, feature_extractor_func):
    """
    Compare reference molecule with multiple hits in a grid visualization.
    
    Parameters:
    -----------
    reference_mol : str
        Name of the reference molecule
    hit_mols : list
        List of molecule names to compare against reference
    df : pd.DataFrame
        DataFrame containing molecule data
    mol_creator_func : callable
        Function to create RDKit mol with 3D coords: func(smiles, mol_name)
    feature_extractor_func : callable
        Function to extract pharmacophore features: func(mol)
    """
    n_hits = len(hit_mols)
    if n_hits == 0:
        print("No hit molecules provided.")
        return

    # Create grid view settings
    n_cols = min(3, n_hits + 1)
    n_rows = (n_hits + 1 + n_cols - 1) // n_cols
    
    view = py3Dmol.view(width=1000, height=350 * n_rows, viewergrid=(n_rows, n_cols))
    
    all_mols = [reference_mol] + hit_mols
    
    for idx, mol_name in enumerate(all_mols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        # Try to find the molecule with robust name matching
        mol_row = None
        
        # 1. Try exact match
        rows = df[df['Molecule_Name'] == mol_name]
        if not rows.empty:
            mol_row = rows.iloc[0]
        
        # 2. Try appending _optimized
        if mol_row is None:
            rows = df[df['Molecule_Name'] == f"{mol_name}_optimized"]
            if not rows.empty:
                mol_row = rows.iloc[0]
                
        # 3. Try removing _optimized
        if mol_row is None:
            clean_name = mol_name.replace('_optimized', '')
            rows = df[df['Molecule_Name'] == clean_name]
            if not rows.empty:
                mol_row = rows.iloc[0]
        
        if mol_row is None:
            print(f"Molecule '{mol_name}' not found in DataFrame (tried exact, +_optimized, -_optimized)")
            continue

        try:
            # Use the actual name found in the DataFrame to load coordinates
            actual_name = mol_row['Molecule_Name']
            mol = mol_creator_func(mol_row['SMILES'], actual_name)
            
            if mol is None:
                continue
                
            mol_display = Chem.RemoveHs(mol)
            mol_block = Chem.MolToMolBlock(mol_display)
            features = feature_extractor_func(mol)
            
            view.addModel(mol_block, 'mol', viewer=(row_idx, col_idx))
            view.setStyle({}, {'stick': {'radius': 0.1, 'color': '0xaaaaaa'}}, viewer=(row_idx, col_idx))
            view.setBackgroundColor('white', viewer=(row_idx, col_idx))
            
            for feat in features:
                color = PHARM_COLORS.get(feat['type'], '0x999999')
                if 'coords' in feat:
                    x, y, z = feat['coords']
                    view.addSphere({'center': {'x': float(x), 'y': float(y), 'z': float(z)},
                                    'radius': 0.8, 'color': color, 'opacity': 0.7}, viewer=(row_idx, col_idx))
            
            view.zoomTo(viewer=(row_idx, col_idx))
            
        except IndexError:
            print(f"Molecule {mol_name} not found in DataFrame")
            continue
        except Exception as e:
            print(f"Error processing {mol_name}: {str(e)}")
            continue
    
    # Create labels
    labels = [f"🎯 {reference_mol} (Reference)"] + [f"{m}" for m in hit_mols]
    
    display(widgets.HTML(f"<h3 style='text-align:center'>Pharmacophore Comparison</h3>"))
    
    label_html = "<div style='display:flex; flex-wrap:wrap; justify-content:center'>"
    for i, label in enumerate(labels):
        style = "font-weight:bold; color:green" if i == 0 else ""
        label_html += f"<span style='margin:5px 20px; {style}'>{label}</span>"
    label_html += "</div>"
    
    display(widgets.HTML(label_html))
    view.show()


def visualize_3d_pharmacophore(mol_name, smiles, mol_creator_func, feature_extractor_func):
    """
    Professional 3D pharmacophore visualization (Side-by-side view).
    
    Parameters:
    -----------
    mol_name : str
        Name of the molecule
    smiles : str
        SMILES string
    mol_creator_func : callable
        Function to create RDKit mol with 3D coords: func(smiles, mol_name)
    feature_extractor_func : callable
        Function to extract pharmacophore features: func(mol)
    """
    mol = mol_creator_func(smiles, mol_name)
    if mol is None:
        print(f"❌ Could not load {mol_name}")
        return
    
    mol_display = Chem.RemoveHs(mol)
    mol_block = Chem.MolToMolBlock(mol_display)
    features = feature_extractor_func(mol)
    
    # Create side-by-side view
    view = py3Dmol.view(width=1000, height=500, viewergrid=(1, 2))
    
    # LEFT: Molecule only
    view.addModel(mol_block, 'mol', viewer=(0, 0))
    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}}, viewer=(0, 0))
    view.setBackgroundColor('white', viewer=(0, 0))
    
    # RIGHT: Molecule + Pharmacophore
    view.addModel(mol_block, 'mol', viewer=(0, 1))
    view.setStyle({'stick': {'radius': 0.1, 'color': '0xaaaaaa'}}, viewer=(0, 1))
    view.setBackgroundColor('white', viewer=(0, 1))
    
    for feat in features:
        color = PHARM_COLORS.get(feat['type'], '0x999999')
        if 'coords' in feat:
            x, y, z = feat['coords']
            view.addSphere({'center': {'x': float(x), 'y': float(y), 'z': float(z)},
                            'radius': 1.0, 'color': color, 'opacity': 0.7}, viewer=(0, 1))
            view.addLabel(feat['type'][:3], {
                'position': {'x': float(x), 'y': float(y), 'z': float(z) + 1.2},
                'fontSize': 12, 'fontColor': 'black', 'backgroundColor': 'white', 
                'backgroundOpacity': 0.8
            }, viewer=(0, 1))
    
    view.zoomTo(viewer=(0, 0))
    view.zoomTo(viewer=(0, 1))
    
    clean_name = mol_name.replace('_optimized', '')
    display(widgets.HTML(f"<h3 style='text-align:center'>{clean_name}</h3>"))
    display(widgets.HTML("<div style='text-align:center'><b>3D Structure</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>3D Pharmacophore</b></div>"))
    view.show()
    
    # Print feature summary
    feat_counts = {}
    for f in features:
        feat_counts[f['type']] = feat_counts.get(f['type'], 0) + 1
    print(f"\n📊 Pharmacophore Features: {len(features)} total")
    for ft, c in sorted(feat_counts.items()):
        print(f"   • {ft}: {c}")

