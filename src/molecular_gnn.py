"""
Molecular GNN for Binding Site-Guided Drug Discovery

This module implements a Graph Neural Network that:
1. Encodes molecules into a latent space based on their graph structure
2. Translates binding site pharmacophore into an "ideal ligand profile"
3. Scores molecules by how well they match the ideal profile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from collections import defaultdict
import json


# =============================================================================
# MOLECULAR GRAPH FEATURIZATION
# =============================================================================

# Atom features
ATOM_FEATURES = {
    'atom_type': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Other'],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
    'num_hs': [0, 1, 2, 3, 4]
}

# Bond features
BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True]
}


def one_hot_encode(value, allowed_values):
    """One-hot encode a value."""
    encoding = [0] * len(allowed_values)
    if value in allowed_values:
        encoding[allowed_values.index(value)] = 1
    else:
        encoding[-1] = 1  # 'Other' category
    return encoding


def get_atom_features(atom):
    """Extract features from an RDKit atom."""
    features = []
    
    # Atom type
    symbol = atom.GetSymbol()
    features.extend(one_hot_encode(symbol, ATOM_FEATURES['atom_type']))
    
    # Degree
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_FEATURES['degree']))
    
    # Formal charge
    features.extend(one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    
    # Hybridization
    hyb = str(atom.GetHybridization())
    features.extend(one_hot_encode(hyb, ATOM_FEATURES['hybridization']))
    
    # Is aromatic
    features.extend(one_hot_encode(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    
    # Is in ring
    features.extend(one_hot_encode(atom.IsInRing(), ATOM_FEATURES['is_in_ring']))
    
    # Number of Hs
    features.extend(one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
    
    return features


def get_bond_features(bond):
    """Extract features from an RDKit bond."""
    features = []
    
    # Bond type
    bt = str(bond.GetBondType())
    features.extend(one_hot_encode(bt, BOND_FEATURES['bond_type']))
    
    # Is conjugated
    features.extend(one_hot_encode(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated']))
    
    # Is in ring
    features.extend(one_hot_encode(bond.IsInRing(), BOND_FEATURES['is_in_ring']))
    
    return features


def mol_to_graph(mol):
    """Convert RDKit molecule to PyTorch Geometric Data object."""
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Get bond indices and features
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_feat = get_bond_features(bond)
        
        # Add both directions (undirected graph)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)
    
    if len(edge_index) == 0:
        # Handle molecules with no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_FEATURES['bond_type']) + 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =============================================================================
# BINDING SITE TO IDEAL LIGAND PROFILE
# =============================================================================

class BindingSiteProfiler:
    """
    Translates binding site pharmacophore into an ideal ligand property profile.
    
    Binding Site Feature → Required Ligand Property
    ─────────────────────────────────────────────────
    Arg (+)              → Need HBA, acidic groups
    Asp/Glu (-)          → Need HBD, basic groups  
    Phe/Tyr/Trp          → Need aromatic rings
    Leu/Val/Ile          → Need hydrophobic groups
    Ser/Thr/Asn          → Need H-bond capable
    """
    
    def __init__(self, binding_site_model):
        self.model = binding_site_model
        self.required_features = binding_site_model.get('required_ligand_features', {})
        self.site_classification = binding_site_model.get('site_classification', {})
        
    def compute_ideal_profile(self):
        """Compute the ideal ligand property profile from binding site."""
        
        # Count binding site features
        n_donors = len(self.site_classification.get('donors', []))
        n_acceptors = len(self.site_classification.get('acceptors', []))
        n_hydrophobic = len(self.site_classification.get('hydrophobic', []))
        n_aromatic = len(self.site_classification.get('aromatic', []))
        n_positive = len(self.site_classification.get('positive', []))
        n_negative = len(self.site_classification.get('negative', []))
        
        # Translate to ideal ligand properties
        # These are heuristic mappings based on medicinal chemistry principles
        
        ideal_profile = {
            # From binding site donors -> ligand needs acceptors
            'HBA': min(n_donors, 10),  # Cap at 10
            
            # From binding site acceptors -> ligand needs donors  
            'HBD': min(n_acceptors, 5),  # Cap at 5
            
            # From hydrophobic residues -> need hydrophobic character
            # LogP roughly scales with hydrophobic contact area
            'LogP': 2.0 + 0.2 * n_hydrophobic,  # Base + contribution
            
            # From aromatic residues -> need aromatic rings
            'AromaticRings': min(n_aromatic, 4),  # Cap at 4
            
            # From charged residues -> may need charged groups
            'NegCharge': 1 if n_positive > 0 else 0,  # Need - if site has +
            'PosCharge': 1 if n_negative > 0 else 0,  # Need + if site has -
            
            # General MW based on binding site size
            'MW': 200 + 20 * (n_hydrophobic + n_aromatic),  # Rough estimate
            
            # TPSA related to H-bonding capacity
            'TPSA': 20 + 15 * (n_donors + n_acceptors),
            
            # Rotatable bonds for flexibility
            'RotatableBonds': 3 + max(0, n_hydrophobic - 5),
        }
        
        # Clamp to drug-like ranges
        ideal_profile['LogP'] = min(max(ideal_profile['LogP'], 0), 5)
        ideal_profile['MW'] = min(max(ideal_profile['MW'], 150), 500)
        ideal_profile['TPSA'] = min(max(ideal_profile['TPSA'], 20), 140)
        
        return ideal_profile
    
    def to_tensor(self):
        """Convert ideal profile to a tensor for the model."""
        profile = self.compute_ideal_profile()
        
        # Normalize features to [0, 1] range
        normalized = [
            profile['HBA'] / 10,
            profile['HBD'] / 5,
            profile['LogP'] / 5,
            profile['AromaticRings'] / 4,
            profile['NegCharge'],
            profile['PosCharge'],
            profile['MW'] / 500,
            profile['TPSA'] / 140,
            profile['RotatableBonds'] / 10,
        ]
        
        return torch.tensor(normalized, dtype=torch.float)


# =============================================================================
# MOLECULAR GNN MODEL
# =============================================================================

class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular encoding.
    
    Architecture:
    ┌───────────────┐
    │ Atom Features │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  GCN Layer 1  │ ── Message passing
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  GCN Layer 2  │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  GCN Layer 3  │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ Global Pooling│ ── Aggregate node features
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │   MLP Head    │ ── Project to latent space
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ Latent Vector │ (64-dim)
    └───────────────┘
    """
    
    def __init__(self, node_features, hidden_dim=128, latent_dim=64, num_layers=3):
        super(MolecularGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # MLP head for latent projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN message passing
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Project to latent space
        latent = self.mlp(x)
        
        return latent


class BindingSiteEncoder(nn.Module):
    """
    Encode binding site pharmacophore requirements into latent space.
    
    Maps the ideal ligand profile to the same latent space as molecules.
    """
    
    def __init__(self, profile_dim=9, latent_dim=64):
        super(BindingSiteEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(profile_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, profile):
        return self.encoder(profile)


class BindingAffinityPredictor(nn.Module):
    """
    Complete model that predicts binding affinity between molecules and binding site.
    
    ┌─────────────────┐     ┌─────────────────┐
    │    Molecule     │     │  Binding Site   │
    │   (Graph)       │     │ (Pharmacophore) │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  Molecular GNN  │     │  Site Encoder   │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────────────────────────────┐
    │        Latent Space (64-dim)            │
    │   mol_latent          site_latent       │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │   Similarity Score (cosine / MLP)       │
    └─────────────────────────────────────────┘
    """
    
    def __init__(self, node_features, hidden_dim=128, latent_dim=64):
        super(BindingAffinityPredictor, self).__init__()
        
        self.mol_encoder = MolecularGNN(node_features, hidden_dim, latent_dim)
        self.site_encoder = BindingSiteEncoder(profile_dim=9, latent_dim=latent_dim)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def encode_molecule(self, mol_data):
        """Get latent representation of molecule."""
        return self.mol_encoder(mol_data)
    
    def encode_site(self, site_profile):
        """Get latent representation of binding site requirements."""
        return self.site_encoder(site_profile)
    
    def forward(self, mol_data, site_profile):
        """Predict binding affinity score."""
        mol_latent = self.encode_molecule(mol_data)
        site_latent = self.encode_site(site_profile)
        
        # Concatenate and predict
        combined = torch.cat([mol_latent, site_latent.expand(mol_latent.size(0), -1)], dim=1)
        score = self.predictor(combined)
        
        return score, mol_latent, site_latent
    
    def compute_similarity(self, mol_latent, site_latent):
        """Compute cosine similarity between molecule and site latents."""
        mol_norm = F.normalize(mol_latent, p=2, dim=1)
        site_norm = F.normalize(site_latent, p=2, dim=0)
        return torch.mm(mol_norm, site_norm.unsqueeze(1)).squeeze()


# =============================================================================
# PROPERTY-BASED SCORING (NO TRAINING REQUIRED)
# =============================================================================

class PropertyBasedScorer:
    """
    Score molecules based on property matching with binding site requirements.
    
    This is a simpler approach that doesn't require training:
    1. Compute ideal ligand profile from binding site
    2. Calculate molecular properties for each molecule
    3. Score based on property similarity
    """
    
    def __init__(self, binding_site_model):
        self.profiler = BindingSiteProfiler(binding_site_model)
        self.ideal_profile = self.profiler.compute_ideal_profile()
        
    def compute_mol_properties(self, mol):
        """Compute properties for a molecule."""
        if mol is None:
            return None
        
        try:
            props = {
                'HBA': Descriptors.NumHAcceptors(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'LogP': Descriptors.MolLogP(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'MW': Descriptors.MolWt(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            }
            
            # Check for ionizable groups
            smiles = Chem.MolToSmiles(mol)
            props['NegCharge'] = 1 if 'C(=O)O' in smiles or 'S(=O)(=O)O' in smiles else 0
            props['PosCharge'] = 1 if '[NH3+]' in smiles or '[NH2+]' in smiles or mol.GetFormalCharge() > 0 else 0
            
            return props
        except:
            return None
    
    def score_molecule(self, mol):
        """Score a molecule against the ideal profile."""
        props = self.compute_mol_properties(mol)
        if props is None:
            return 0.0, {}
        
        # Compute property match scores
        scores = {}
        
        # Discrete features (count-based)
        for feat in ['HBA', 'HBD', 'AromaticRings']:
            ideal = self.ideal_profile[feat]
            actual = props[feat]
            # Score based on how close we are (1.0 = perfect match)
            if ideal == 0:
                scores[feat] = 1.0 if actual == 0 else 0.5
            else:
                scores[feat] = max(0, 1 - abs(ideal - actual) / ideal)
        
        # Continuous features (Gaussian similarity)
        for feat, sigma in [('LogP', 1.0), ('MW', 100), ('TPSA', 30)]:
            ideal = self.ideal_profile[feat]
            actual = props[feat]
            scores[feat] = np.exp(-((ideal - actual) ** 2) / (2 * sigma ** 2))
        
        # Binary features
        for feat in ['NegCharge', 'PosCharge']:
            ideal = self.ideal_profile[feat]
            actual = props[feat]
            scores[feat] = 1.0 if ideal == actual else 0.5
        
        # Weighted average
        weights = {
            'HBA': 2.0, 'HBD': 2.0, 'LogP': 1.5, 'AromaticRings': 1.5,
            'MW': 1.0, 'TPSA': 1.0, 'NegCharge': 1.0, 'PosCharge': 1.0
        }
        
        total_weight = sum(weights.values())
        overall_score = sum(scores[f] * weights[f] for f in scores) / total_weight
        
        return overall_score, {
            'property_scores': scores,
            'actual_properties': props,
            'ideal_profile': self.ideal_profile
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_atom_features_dim():
    """Calculate the dimension of atom feature vector."""
    dim = 0
    for key, values in ATOM_FEATURES.items():
        dim += len(values)
    return dim


def prepare_dataset(df, smiles_col='SMILES'):
    """Convert DataFrame of molecules to PyTorch Geometric dataset."""
    dataset = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            data = mol_to_graph(mol)
            if data is not None:
                data.idx = idx
                data.smiles = smiles
                dataset.append(data)
    
    return dataset


def load_binding_site_model(filepath):
    """Load binding site pharmacophore model from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)
