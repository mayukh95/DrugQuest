"""
Data loading utilities for molecular datasets and configuration files.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ..core.molecule import DrugMolecule
from ..core.config import Config


class MoleculeLoader:
    """Load molecules from various file formats."""
    
    @staticmethod
    def from_smiles_file(file_path: Union[str, Path], 
                        smiles_column='SMILES', 
                        name_column='Name'):
        """Load molecules from CSV file containing SMILES."""
        df = pd.read_csv(file_path)
        
        molecules = []
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            name = row.get(name_column, f"mol_{idx}")
            
            try:
                mol_obj = DrugMolecule(smiles=smiles, name=name)
                molecules.append(mol_obj)
            except Exception as e:
                print(f"Warning: Could not load molecule {name}: {e}")
        
        print(f"✓ Loaded {len(molecules)} molecules from {file_path}")
        return molecules
    
    @staticmethod
    def from_sdf(file_path: Union[str, Path]):
        """Load molecules from SDF file."""
        molecules = []
        
        supplier = Chem.SDMolSupplier(str(file_path))
        for i, mol in enumerate(supplier):
            if mol is not None:
                # Get molecule name
                name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{i}"
                
                # Create DrugMolecule object
                smiles = Chem.MolToSmiles(mol)
                mol_obj = DrugMolecule(smiles=smiles, name=name)
                
                # Copy properties
                for prop_name in mol.GetPropNames():
                    value = mol.GetProp(prop_name)
                    mol_obj.set_property(prop_name, value)
                
                molecules.append(mol_obj)
        
        print(f"✓ Loaded {len(molecules)} molecules from {file_path}")
        return molecules
    
    @staticmethod
    def from_mol2(file_path: Union[str, Path]):
        """Load molecules from MOL2 file."""
        molecules = []
        
        # Read MOL2 file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split by molecules (MOL2 format uses @<TRIPOS>MOLECULE)
        mol_blocks = content.split('@<TRIPOS>MOLECULE')[1:]
        
        for i, mol_block in enumerate(mol_blocks):
            mol_block = '@<TRIPOS>MOLECULE' + mol_block
            
            try:
                mol = Chem.MolFromMol2Block(mol_block)
                if mol is not None:
                    name = f"mol_{i}"
                    smiles = Chem.MolToSmiles(mol)
                    mol_obj = DrugMolecule(smiles=smiles, name=name)
                    molecules.append(mol_obj)
            except Exception as e:
                print(f"Warning: Could not load molecule {i}: {e}")
        
        print(f"✓ Loaded {len(molecules)} molecules from {file_path}")
        return molecules
    
    @staticmethod
    def from_json(file_path: Union[str, Path]):
        """Load molecules from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        molecules = []
        
        if isinstance(data, list):
            # List of molecule objects
            for item in data:
                if 'smiles' in item:
                    name = item.get('name', f"mol_{len(molecules)}")
                    mol_obj = DrugMolecule(smiles=item['smiles'], name=name)
                    
                    # Add properties
                    for key, value in item.items():
                        if key not in ['smiles', 'name']:
                            mol_obj.set_property(key, value)
                    
                    molecules.append(mol_obj)
        
        elif isinstance(data, dict):
            # Dictionary format
            for name, mol_data in data.items():
                if 'smiles' in mol_data:
                    mol_obj = DrugMolecule(smiles=mol_data['smiles'], name=name)
                    
                    # Add properties
                    for key, value in mol_data.items():
                        if key != 'smiles':
                            mol_obj.set_property(key, value)
                    
                    molecules.append(mol_obj)
        
        print(f"✓ Loaded {len(molecules)} molecules from {file_path}")
        return molecules


class DatasetLoader:
    """Load and prepare molecular datasets for analysis."""
    
    @staticmethod
    def load_drugbank_subset(file_path: Union[str, Path], max_molecules=1000):
        """Load subset of DrugBank database."""
        if Path(file_path).suffix.lower() == '.sdf':
            molecules = MoleculeLoader.from_sdf(file_path)
        else:
            molecules = MoleculeLoader.from_smiles_file(file_path)
        
        # Filter and limit
        filtered_molecules = []
        for mol_obj in molecules[:max_molecules]:
            try:
                # Basic filtering
                mol = mol_obj.mol
                if mol is not None:
                    mw = rdMolDescriptors.CalcExactMolWt(mol)
                    if 150 <= mw <= 800:  # Reasonable drug-like size
                        filtered_molecules.append(mol_obj)
            except Exception:
                continue
        
        print(f"✓ Loaded and filtered {len(filtered_molecules)} drug-like molecules")
        return filtered_molecules
    
    @staticmethod
    def load_chembl_subset(file_path: Union[str, Path], 
                          activity_threshold=None, max_molecules=1000):
        """Load subset of ChEMBL database with activity data."""
        # Load molecules
        df = pd.read_csv(file_path)
        
        molecules = []
        for idx, row in df.iterrows():
            if len(molecules) >= max_molecules:
                break
            
            try:
                smiles = row['SMILES']
                name = row.get('Name', f"chembl_{idx}")
                
                mol_obj = DrugMolecule(smiles=smiles, name=name)
                
                # Add ChEMBL-specific properties
                if 'Activity' in row:
                    mol_obj.set_property('activity', row['Activity'])
                if 'Target' in row:
                    mol_obj.set_property('target', row['Target'])
                if 'IC50' in row:
                    mol_obj.set_property('ic50', row['IC50'])
                
                # Filter by activity if specified
                if activity_threshold is not None:
                    if 'Activity' in row and row['Activity'] < activity_threshold:
                        continue
                
                molecules.append(mol_obj)
                
            except Exception as e:
                print(f"Warning: Could not load molecule {idx}: {e}")
        
        print(f"✓ Loaded {len(molecules)} bioactive molecules from ChEMBL")
        return molecules
    
    @staticmethod
    def load_zinc_subset(file_path: Union[str, Path], 
                        lead_like=True, max_molecules=1000):
        """Load subset of ZINC database."""
        molecules = MoleculeLoader.from_smiles_file(file_path)
        
        # Apply lead-like filters if requested
        if lead_like:
            filtered_molecules = []
            
            for mol_obj in molecules[:max_molecules]:
                try:
                    mol = mol_obj.mol
                    mw = rdMolDescriptors.CalcExactMolWt(mol)
                    logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
                    
                    # Lead-like criteria (relaxed Lipinski)
                    if (250 <= mw <= 350 and 
                        -1 <= logp <= 3 and 
                        rdMolDescriptors.CalcNumHBD(mol) <= 3 and
                        rdMolDescriptors.CalcNumHBA(mol) <= 6):
                        
                        filtered_molecules.append(mol_obj)
                        
                except Exception:
                    continue
            
            molecules = filtered_molecules
        
        print(f"✓ Loaded {len(molecules)} lead-like molecules from ZINC")
        return molecules[:max_molecules]


class ConfigLoader:
    """Load and validate configuration files."""
    
    @staticmethod
    def load_config(file_path: Union[str, Path]):
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Create Config object with loaded data
        config = Config()
        
        # Update configuration
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
        
        print(f"✓ Loaded configuration from {file_path}")
        return config
    
    @staticmethod
    def load_dft_parameters(file_path: Union[str, Path]):
        """Load DFT calculation parameters."""
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        # Validate required parameters
        required_keys = ['method', 'basis', 'convergence_threshold']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required DFT parameter: {key}")
        
        print(f"✓ Loaded DFT parameters from {file_path}")
        return params
    
    @staticmethod
    def load_optimization_settings(file_path: Union[str, Path]):
        """Load optimization settings."""
        with open(file_path, 'r') as f:
            settings = json.load(f)
        
        # Set defaults
        defaults = {
            'max_iterations': 100,
            'convergence_threshold': 1e-6,
            'step_size': 0.01,
            'line_search': True
        }
        
        for key, default_value in defaults.items():
            if key not in settings:
                settings[key] = default_value
                print(f"Using default value for {key}: {default_value}")
        
        print(f"✓ Loaded optimization settings from {file_path}")
        return settings


class BatchLoader:
    """Load multiple datasets and combine them."""
    
    def __init__(self):
        """Initialize batch loader."""
        self.datasets = {}
    
    def add_dataset(self, name: str, file_path: Union[str, Path], 
                   loader_type='auto', **kwargs):
        """Add a dataset to the batch."""
        file_path = Path(file_path)
        
        # Determine loader type automatically
        if loader_type == 'auto':
            suffix = file_path.suffix.lower()
            if suffix == '.sdf':
                molecules = MoleculeLoader.from_sdf(file_path)
            elif suffix == '.json':
                molecules = MoleculeLoader.from_json(file_path)
            elif suffix in ['.csv', '.tsv']:
                molecules = MoleculeLoader.from_smiles_file(file_path, **kwargs)
            elif suffix == '.mol2':
                molecules = MoleculeLoader.from_mol2(file_path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        
        else:
            # Use specific loader
            loader_map = {
                'sdf': MoleculeLoader.from_sdf,
                'smiles': MoleculeLoader.from_smiles_file,
                'json': MoleculeLoader.from_json,
                'mol2': MoleculeLoader.from_mol2
            }
            
            if loader_type not in loader_map:
                raise ValueError(f"Unknown loader type: {loader_type}")
            
            loader_func = loader_map[loader_type]
            molecules = loader_func(file_path, **kwargs)
        
        self.datasets[name] = molecules
        print(f"✓ Added dataset '{name}' with {len(molecules)} molecules")
    
    def combine_datasets(self, dataset_names: List[str] = None):
        """Combine multiple datasets into one."""
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        combined_molecules = []
        
        for name in dataset_names:
            if name in self.datasets:
                molecules = self.datasets[name]
                
                # Add dataset label to each molecule
                for mol_obj in molecules:
                    mol_obj.set_property('dataset', name)
                
                combined_molecules.extend(molecules)
        
        print(f"✓ Combined {len(combined_molecules)} molecules from {len(dataset_names)} datasets")
        return combined_molecules
    
    def get_dataset_summary(self):
        """Get summary of all loaded datasets."""
        summary = {}
        
        for name, molecules in self.datasets.items():
            summary[name] = {
                'count': len(molecules),
                'avg_mw': 0,
                'avg_logp': 0
            }
            
            if molecules:
                try:
                    mws = [rdMolDescriptors.CalcExactMolWt(mol.mol) for mol in molecules]
                    logps = [rdMolDescriptors.CalcCrippenDescriptors(mol.mol)[0] for mol in molecules]
                    
                    summary[name]['avg_mw'] = sum(mws) / len(mws)
                    summary[name]['avg_logp'] = sum(logps) / len(logps)
                    
                except Exception:
                    pass
        
        return summary