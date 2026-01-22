"""
Pharmacophore feature extraction tools.
"""

import numpy as np
from pathlib import Path
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures

# Initialize Feature Factory (singleton-like pattern)
_FEATURE_FACTORY = None

def get_feature_factory():
    """Get or initialize the RDKit ChemicalFeatureFactory."""
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
        try:
            _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))
        except IOError:
            print(f"Warning: Could not find BaseFeatures.fdef at {fdef_name}")
            return None
    return _FEATURE_FACTORY


def extract_pharm_features(mol, factory=None):
    """
    Extract pharmacophore features from a molecule with 3D coordinates.
    
    Parameters:
    -----------
    mol : rdkit.Chem.Mol
        Molecule with 3D conformer
    factory : ChemicalFeatures.FeatureFactory, optional
        Pharmacophore feature factory (default: uses RDKit BaseFeatures)
        
    Returns:
    --------
    list of dict
        List of features, each containing 'type', 'coords', and 'atom_ids'
    """
    if mol is None or mol.GetNumConformers() == 0:
        return []
    
    if factory is None:
        factory = get_feature_factory()
        if factory is None:
            return []
            
    features = []
    # RDKit's GetFeaturesForMol returns distinct features
    for feat in factory.GetFeaturesForMol(mol):
        atom_ids = feat.GetAtomIds()
        conf = mol.GetConformer()
        
        # Calculate geometric center of the feature
        coords = [[conf.GetAtomPosition(a).x, conf.GetAtomPosition(a).y, 
                   conf.GetAtomPosition(a).z] for a in atom_ids]
                   
        features.append({
            'type': feat.GetFamily(),
            'coords': np.array(coords).mean(axis=0),
            'atom_ids': list(atom_ids)
        })
        
    return features
