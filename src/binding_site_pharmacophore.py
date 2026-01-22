"""
Binding Site Pharmacophore Analysis Module

This module extracts pharmacophore-like features from protein binding sites
and scores drug molecules based on complementarity.
"""

import numpy as np
import json
from collections import defaultdict
from pathlib import Path

# Amino acid classification for pharmacophore features
RESIDUE_PROPERTIES = {
    # H-bond Donors (backbone NH + sidechain)
    'HBD': ['ARG', 'LYS', 'HIS', 'ASN', 'GLN', 'SER', 'THR', 'TYR', 'TRP'],
    # H-bond Acceptors (backbone C=O + sidechain)
    'HBA': ['ASP', 'GLU', 'ASN', 'GLN', 'SER', 'THR', 'TYR', 'HIS'],
    # Hydrophobic residues
    'HYDROPHOBIC': ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'],
    # Aromatic residues (π-stacking)
    'AROMATIC': ['PHE', 'TYR', 'TRP', 'HIS'],
    # Positively charged (at pH 7.4)
    'POS_CHARGE': ['ARG', 'LYS'],
    # Negatively charged (at pH 7.4)
    'NEG_CHARGE': ['ASP', 'GLU']
}

# Key atoms for each feature type per residue
FEATURE_ATOMS = {
    'ARG': {'donor': ['NE', 'NH1', 'NH2'], 'pos': ['CZ']},
    'LYS': {'donor': ['NZ'], 'pos': ['NZ']},
    'HIS': {'donor': ['ND1', 'NE2'], 'acceptor': ['ND1', 'NE2'], 'aromatic': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']},
    'ASN': {'donor': ['ND2'], 'acceptor': ['OD1']},
    'GLN': {'donor': ['NE2'], 'acceptor': ['OE1']},
    'SER': {'donor': ['OG'], 'acceptor': ['OG']},
    'THR': {'donor': ['OG1'], 'acceptor': ['OG1']},
    'TYR': {'donor': ['OH'], 'acceptor': ['OH'], 'aromatic': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']},
    'TRP': {'donor': ['NE1'], 'aromatic': ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']},
    'ASP': {'acceptor': ['OD1', 'OD2'], 'neg': ['CG']},
    'GLU': {'acceptor': ['OE1', 'OE2'], 'neg': ['CD']},
    'PHE': {'aromatic': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']},
    'ALA': {'hydrophobic': ['CB']},
    'VAL': {'hydrophobic': ['CG1', 'CG2']},
    'LEU': {'hydrophobic': ['CD1', 'CD2']},
    'ILE': {'hydrophobic': ['CD1', 'CG2']},
    'MET': {'hydrophobic': ['CE', 'SD']},
    'PRO': {'hydrophobic': ['CG', 'CD']}
}


class BindingSitePharmacophore:
    """Extract and analyze pharmacophore features from a protein binding site."""
    
    def __init__(self, binding_site_residues, structure, chain_id):
        """
        Initialize with binding site information.
        
        Args:
            binding_site_residues: List of (resname, resnum) tuples
            structure: BioPython Structure object
            chain_id: Chain identifier
        """
        self.residues = binding_site_residues
        self.structure = structure
        self.chain_id = chain_id
        self.features = []
        self.feature_summary = {}
        
    def extract_features(self):
        """Extract all pharmacophore features from binding site residues."""
        self.features = []
        
        # Get the chain
        chain = None
        for model in self.structure:
            for c in model:
                if c.get_id() == self.chain_id:
                    chain = c
                    break
        
        if chain is None:
            print(f"Chain {self.chain_id} not found")
            return
        
        # Process each binding site residue
        for resname, resnum in self.residues:
            residue = None
            for res in chain:
                if res.get_id()[1] == resnum:
                    residue = res
                    break
            
            if residue is None:
                continue
            
            # Extract features based on residue type
            self._extract_residue_features(residue, resname, resnum)
        
        # Calculate summary
        self._calculate_summary()
        
        return self.features
    
    def _extract_residue_features(self, residue, resname, resnum):
        """Extract pharmacophore features from a single residue."""
        
        atoms = {atom.get_name(): atom.get_coord() for atom in residue.get_atoms()}
        
        # Get feature atoms for this residue type
        if resname not in FEATURE_ATOMS:
            return
        
        feature_def = FEATURE_ATOMS[resname]
        
        # H-bond Donors -> Ligand needs Acceptor
        if 'donor' in feature_def:
            coords = [atoms[a] for a in feature_def['donor'] if a in atoms]
            if coords:
                centroid = np.mean(coords, axis=0)
                self.features.append({
                    'type': 'Acceptor',  # Inverse: site donor -> ligand acceptor
                    'site_type': 'Donor',
                    'coords': centroid.tolist(),
                    'residue': f"{resname}{resnum}",
                    'tolerance': 1.5
                })
        
        # H-bond Acceptors -> Ligand needs Donor
        if 'acceptor' in feature_def:
            coords = [atoms[a] for a in feature_def['acceptor'] if a in atoms]
            if coords:
                centroid = np.mean(coords, axis=0)
                self.features.append({
                    'type': 'Donor',  # Inverse: site acceptor -> ligand donor
                    'site_type': 'Acceptor',
                    'coords': centroid.tolist(),
                    'residue': f"{resname}{resnum}",
                    'tolerance': 1.5
                })
        
        # Aromatic -> Ligand needs Aromatic
        if 'aromatic' in feature_def:
            coords = [atoms[a] for a in feature_def['aromatic'] if a in atoms]
            if coords:
                centroid = np.mean(coords, axis=0)
                self.features.append({
                    'type': 'Aromatic',
                    'site_type': 'Aromatic',
                    'coords': centroid.tolist(),
                    'residue': f"{resname}{resnum}",
                    'tolerance': 1.5
                })
        
        # Hydrophobic -> Ligand needs Hydrophobic
        if 'hydrophobic' in feature_def:
            coords = [atoms[a] for a in feature_def['hydrophobic'] if a in atoms]
            if coords:
                centroid = np.mean(coords, axis=0)
                self.features.append({
                    'type': 'Hydrophobe',
                    'site_type': 'Hydrophobic',
                    'coords': centroid.tolist(),
                    'residue': f"{resname}{resnum}",
                    'tolerance': 2.0
                })
        
        # Positive charge -> Ligand needs Negative (NegIonizable)
        if 'pos' in feature_def:
            coords = [atoms[a] for a in feature_def['pos'] if a in atoms]
            if coords:
                centroid = np.mean(coords, axis=0)
                self.features.append({
                    'type': 'NegIonizable',  # Inverse
                    'site_type': 'PosCharge',
                    'coords': centroid.tolist(),
                    'residue': f"{resname}{resnum}",
                    'tolerance': 2.0
                })
        
        # Negative charge -> Ligand needs Positive (PosIonizable)
        if 'neg' in feature_def:
            coords = [atoms[a] for a in feature_def['neg'] if a in atoms]
            if coords:
                centroid = np.mean(coords, axis=0)
                self.features.append({
                    'type': 'PosIonizable',  # Inverse
                    'site_type': 'NegCharge',
                    'coords': centroid.tolist(),
                    'residue': f"{resname}{resnum}",
                    'tolerance': 2.0
                })
    
    def _calculate_summary(self):
        """Calculate feature type summary."""
        self.feature_summary = defaultdict(int)
        for feat in self.features:
            self.feature_summary[feat['type']] += 1
        self.feature_summary = dict(self.feature_summary)
    
    def get_required_ligand_features(self):
        """Get the pharmacophore features required in a ligand."""
        return self.feature_summary
    
    def save_pharmacophore(self, filename):
        """Save pharmacophore model to JSON."""
        data = {
            'chain': self.chain_id,
            'n_residues': len(self.residues),
            'n_features': len(self.features),
            'feature_summary': self.feature_summary,
            'features': self.features
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return filename


def score_molecule_against_site(ligand_features, site_pharmacophore, method='count'):
    """
    Score a molecule's pharmacophore against binding site requirements.
    
    Args:
        ligand_features: List of ligand pharmacophore features from RDKit
        site_pharmacophore: BindingSitePharmacophore object
        method: 'count' or 'weighted'
    
    Returns:
        score: Complementarity score (0-1)
        matched: List of matched features
    """
    required = site_pharmacophore.get_required_ligand_features()
    
    # Count ligand features by type
    ligand_counts = defaultdict(int)
    for feat in ligand_features:
        ligand_counts[feat['type']] += 1
    
    # Calculate match score
    total_required = sum(required.values())
    if total_required == 0:
        return 0.0, []
    
    matched_count = 0
    matched_features = []
    
    for feat_type, count in required.items():
        available = ligand_counts.get(feat_type, 0)
        matches = min(available, count)
        matched_count += matches
        if matches > 0:
            matched_features.append({
                'type': feat_type,
                'required': count,
                'available': available,
                'matched': matches
            })
    
    score = matched_count / total_required
    
    return score, matched_features
