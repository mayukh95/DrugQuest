"""
Consensus similarity scoring tools.

This module provides functions for aggregating multiple similarity metrics
into a consensus score for ranking molecules.
"""

import pandas as pd

def compute_consensus_scores(reference_mol, sim_morgan, sim_pharm2d, sim_pharm3d,
                             weights=None):
    """
    Compute weighted consensus similarity scores for a reference molecule.
    
    Parameters:
    -----------
    reference_mol : str
        Name of the reference molecule
    sim_morgan : pd.DataFrame
        Morgan/ECFP similarity matrix
    sim_pharm2d : pd.DataFrame
        2D Pharmacophore similarity matrix
    sim_pharm3d : pd.DataFrame
        3D Pharmacophore similarity matrix
    weights : dict, optional
        Weights for each metric. Default: {'morgan': 0.33, 'pharm2d': 0.33, 'pharm3d': 0.34}
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with individual and consensus scores, sorted by consensus (descending)
    """
    if weights is None:
        weights = {'morgan': 0.33, 'pharm2d': 0.33, 'pharm3d': 0.34}
        
    results = []
    
    # Check if reference is in matrices
    if reference_mol not in sim_morgan.index:
        # Try to find it with suffix handling if needed (though we cleaned names earlier)
        found = False
        for name in [f"{reference_mol}_optimized", reference_mol.replace('_optimized', '')]:
            if name in sim_morgan.index:
                reference_mol = name
                found = True
                break
        if not found:
            print(f"❌ Reference molecule '{reference_mol}' not found in similarity matrices")
            return pd.DataFrame()
            
    for mol in sim_morgan.index:
        if mol == reference_mol:
            continue
        
        morgan_score = sim_morgan.loc[reference_mol, mol]
        
        # Handle cases where indices might differ slightly
        pharm2d_score = 0
        if mol in sim_pharm2d.columns:
            if reference_mol in sim_pharm2d.index:
                pharm2d_score = sim_pharm2d.loc[reference_mol, mol]
            elif f"{reference_mol}_optimized" in sim_pharm2d.index:
                pharm2d_score = sim_pharm2d.loc[f"{reference_mol}_optimized", mol]
                
        pharm3d_score = 0
        if mol in sim_pharm3d.columns:
            if reference_mol in sim_pharm3d.index:
                pharm3d_score = sim_pharm3d.loc[reference_mol, mol]
            elif f"{reference_mol}_optimized" in sim_pharm3d.index:
                pharm3d_score = sim_pharm3d.loc[f"{reference_mol}_optimized", mol]
        
        consensus = (weights['morgan'] * morgan_score + 
                     weights['pharm2d'] * pharm2d_score + 
                     weights['pharm3d'] * pharm3d_score)
        
        results.append({
            'Molecule': mol,
            'Morgan': morgan_score,
            'Pharm2D': pharm2d_score,
            'Pharm3D': pharm3d_score,
            'Consensus': consensus
        })
    
    return pd.DataFrame(results).sort_values('Consensus', ascending=False)
