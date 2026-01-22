"""
Helper functions and utilities for molecular analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski

from ..core.molecule import DrugMolecule


def validate_molecule(smiles: str, name: str = None) -> Optional[DrugMolecule]:
    """Validate a SMILES string and create a DrugMolecule object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None
        
        # Basic validation
        if mol.GetNumAtoms() == 0:
            print(f"Empty molecule: {smiles}")
            return None
        
        # Create DrugMolecule object
        mol_name = name if name else f"mol_{hash(smiles) % 10000}"
        drug_mol = DrugMolecule(smiles=smiles, name=mol_name)
        
        return drug_mol
        
    except Exception as e:
        print(f"Error validating molecule {smiles}: {e}")
        return None


def calculate_molecular_descriptors(molecules: List[DrugMolecule]) -> pd.DataFrame:
    """Calculate comprehensive molecular descriptors for a list of molecules."""
    descriptors_data = []
    
    for mol_obj in molecules:
        try:
            mol = mol_obj.mol
            
            descriptors = {
                'name': mol_obj.name,
                'smiles': mol_obj.smiles,
                
                # Basic properties
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'heavy_atom_count': mol.GetNumHeavyAtoms(),
                'atom_count': mol.GetNumAtoms(),
                
                # Lipinski descriptors
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                
                # Structural features
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                'saturated_rings': Descriptors.NumSaturatedRings(mol),
                'heterocycles': Descriptors.NumHeterocycles(mol),
                
                # Complexity measures
                'molecular_complexity': Descriptors.BertzCT(mol),
                'molecular_framework': rdMolDescriptors.CalcNumSpiroAtoms(mol),
                'bridgehead_atoms': rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                
                # Electronic properties
                'formal_charge': Chem.GetFormalCharge(mol),
                'radical_electrons': Descriptors.NumRadicalElectrons(mol),
                
                # Pharmacophore features
                'lipinski_violations': calculate_lipinski_violations(mol),
                'drug_likeness_score': calculate_drug_likeness_score(mol),
                'lead_likeness_score': calculate_lead_likeness_score(mol),
                
                # Additional descriptors
                'molar_refractivity': Descriptors.MolMR(mol),
                'balaban_j': Descriptors.BalabanJ(mol),
                'kappa1': Descriptors.Kappa1(mol),
                'kappa2': Descriptors.Kappa2(mol),
                'kappa3': Descriptors.Kappa3(mol),
                'chi0v': Descriptors.Chi0v(mol),
                'chi1v': Descriptors.Chi1v(mol),
                'chi2v': Descriptors.Chi2v(mol),
                'chi3v': Descriptors.Chi3v(mol),
                'chi4v': Descriptors.Chi4v(mol),
                'hall_kier_alpha': Descriptors.HallKierAlpha(mol),
                
                # Surface area and volume estimates
                'labutte_asa': Descriptors.LabuteASA(mol),
                'peoe_vsa1': Descriptors.PEOE_VSA1(mol),
                'smr_vsa1': Descriptors.SMR_VSA1(mol),
                'estate_vsa1': Descriptors.EState_VSA1(mol),
                
                # Fragment-based descriptors
                'fr_alkyl_halide': Descriptors.fr_alkyl_halide(mol),
                'fr_aryl_methyl': Descriptors.fr_aryl_methyl(mol),
                'fr_benzene': Descriptors.fr_benzene(mol),
                'fr_ester': Descriptors.fr_ester(mol),
                'fr_ether': Descriptors.fr_ether(mol),
                'fr_halogen': Descriptors.fr_halogen(mol),
                'fr_ketone': Descriptors.fr_ketone(mol),
                'fr_amide': Descriptors.fr_amide(mol),
                'fr_aniline': Descriptors.fr_aniline(mol),
                'fr_phenol': Descriptors.fr_phenol(mol)
            }
            
            descriptors_data.append(descriptors)
            
        except Exception as e:
            print(f"Error calculating descriptors for {mol_obj.name}: {e}")
            continue
    
    return pd.DataFrame(descriptors_data)


def calculate_lipinski_violations(mol) -> int:
    """Calculate number of Lipinski rule violations."""
    violations = 0
    
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    
    return violations


def calculate_drug_likeness_score(mol) -> float:
    """Calculate a simple drug-likeness score (0-1 scale)."""
    score = 1.0
    
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    
    # Molecular weight penalty
    if mw < 150 or mw > 800:
        score *= 0.5
    elif mw < 200 or mw > 600:
        score *= 0.8
    
    # LogP penalty
    if logp < -1 or logp > 6:
        score *= 0.3
    elif logp < 0 or logp > 5:
        score *= 0.7
    
    # TPSA penalty
    if tpsa < 20 or tpsa > 200:
        score *= 0.5
    elif tpsa < 40 or tpsa > 140:
        score *= 0.8
    
    # Rotatable bonds penalty
    if rotatable_bonds > 15:
        score *= 0.3
    elif rotatable_bonds > 10:
        score *= 0.7
    
    # Lipinski violations
    lipinski_violations = calculate_lipinski_violations(mol)
    if lipinski_violations > 1:
        score *= 0.5
    elif lipinski_violations == 1:
        score *= 0.8
    
    return score


def calculate_lead_likeness_score(mol) -> float:
    """Calculate lead-likeness score based on relaxed criteria."""
    score = 1.0
    
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    # Lead-like criteria (more stringent than drug-like)
    if not (250 <= mw <= 350):
        score *= 0.5
    if not (-1 <= logp <= 3):
        score *= 0.5
    if hbd > 3:
        score *= 0.7
    if hba > 6:
        score *= 0.7
    
    return score


def filter_molecules_by_properties(molecules: List[DrugMolecule], 
                                 filters: Dict[str, Tuple[float, float]]) -> List[DrugMolecule]:
    """Filter molecules based on property ranges."""
    filtered = []
    
    for mol_obj in molecules:
        try:
            mol = mol_obj.mol
            include = True
            
            for prop, (min_val, max_val) in filters.items():
                if prop == 'molecular_weight':
                    value = Descriptors.ExactMolWt(mol)
                elif prop == 'logp':
                    value = Descriptors.MolLogP(mol)
                elif prop == 'tpsa':
                    value = Descriptors.TPSA(mol)
                elif prop == 'hbd':
                    value = Descriptors.NumHDonors(mol)
                elif prop == 'hba':
                    value = Descriptors.NumHAcceptors(mol)
                elif prop == 'rotatable_bonds':
                    value = Descriptors.NumRotatableBonds(mol)
                else:
                    continue  # Skip unknown properties
                
                if not (min_val <= value <= max_val):
                    include = False
                    break
            
            if include:
                filtered.append(mol_obj)
                
        except Exception:
            continue  # Skip problematic molecules
    
    return filtered


def identify_functional_groups(mol) -> Dict[str, int]:
    """Identify and count functional groups in a molecule."""
    functional_groups = {
        'alcohol': Descriptors.fr_Al_OH(mol),
        'aldehyde': Descriptors.fr_aldehyde(mol),
        'amide': Descriptors.fr_amide(mol),
        'amine': Descriptors.fr_Ar_N(mol) + Descriptors.fr_NH0(mol) + Descriptors.fr_NH1(mol) + Descriptors.fr_NH2(mol),
        'aromatic_ring': Descriptors.fr_benzene(mol),
        'carboxylic_acid': Descriptors.fr_COO(mol),
        'ester': Descriptors.fr_ester(mol),
        'ether': Descriptors.fr_ether(mol),
        'halogen': Descriptors.fr_halogen(mol),
        'ketone': Descriptors.fr_ketone(mol),
        'nitro': Descriptors.fr_nitro(mol),
        'phenol': Descriptors.fr_phenol(mol),
        'sulfide': Descriptors.fr_sulfide(mol),
        'sulfonamide': Descriptors.fr_sulfonamd(mol),
        'thiocarbonyl': Descriptors.fr_thiocyan(mol)
    }
    
    return functional_groups


def calculate_similarity_matrix(molecules: List[DrugMolecule], 
                              method='tanimoto') -> np.ndarray:
    """Calculate similarity matrix between molecules."""
    from rdkit.Chem import DataStructs
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    
    n_mols = len(molecules)
    similarity_matrix = np.zeros((n_mols, n_mols))
    
    # Generate fingerprints
    fingerprints = []
    for mol_obj in molecules:
        fp = GetMorganFingerprintAsBitVect(mol_obj.mol, radius=2, nBits=2048)
        fingerprints.append(fp)
    
    # Calculate similarities
    for i in range(n_mols):
        for j in range(i, n_mols):
            if method == 'tanimoto':
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            elif method == 'dice':
                sim = DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j])
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    return similarity_matrix


def find_most_similar_molecules(target_molecule: DrugMolecule, 
                               molecule_library: List[DrugMolecule],
                               top_k=10) -> List[Tuple[DrugMolecule, float]]:
    """Find most similar molecules in a library."""
    from rdkit.Chem import DataStructs
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    
    # Generate target fingerprint
    target_fp = GetMorganFingerprintAsBitVect(target_molecule.mol, radius=2, nBits=2048)
    
    similarities = []
    for mol_obj in molecule_library:
        try:
            mol_fp = GetMorganFingerprintAsBitVect(mol_obj.mol, radius=2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(target_fp, mol_fp)
            similarities.append((mol_obj, similarity))
        except Exception:
            continue
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def format_results(results: Dict[str, Any], precision=3) -> Dict[str, Any]:
    """Format numerical results for display."""
    formatted = {}
    
    for key, value in results.items():
        if isinstance(value, float):
            formatted[key] = round(value, precision)
        elif isinstance(value, np.ndarray):
            formatted[key] = np.round(value, precision).tolist()
        elif isinstance(value, list):
            if value and isinstance(value[0], (int, float)):
                formatted[key] = [round(v, precision) if isinstance(v, float) else v for v in value]
            else:
                formatted[key] = value
        elif isinstance(value, dict):
            formatted[key] = format_results(value, precision)
        else:
            formatted[key] = value
    
    return formatted


def generate_molecular_report(molecules: List[DrugMolecule], 
                            title="Molecular Analysis Report") -> str:
    """Generate a text-based molecular analysis report."""
    if not molecules:
        return "No molecules to analyze."
    
    report = f"{title}\n{'=' * len(title)}\n\n"
    
    # Summary statistics
    descriptors_df = calculate_molecular_descriptors(molecules)
    
    report += f"Dataset Summary:\n"
    report += f"  Total molecules: {len(molecules)}\n"
    report += f"  Average molecular weight: {descriptors_df['molecular_weight'].mean():.1f} g/mol\n"
    report += f"  Average LogP: {descriptors_df['logp'].mean():.2f}\n"
    report += f"  Average TPSA: {descriptors_df['tpsa'].mean():.1f} Ų\n\n"
    
    # Drug-likeness analysis
    drug_like_count = sum(1 for _, row in descriptors_df.iterrows() 
                         if row['lipinski_violations'] <= 1)
    lead_like_count = sum(1 for _, row in descriptors_df.iterrows() 
                         if row['lead_likeness_score'] >= 0.7)
    
    report += f"Drug-likeness Analysis:\n"
    report += f"  Drug-like molecules (≤1 Lipinski violation): {drug_like_count} ({drug_like_count/len(molecules)*100:.1f}%)\n"
    report += f"  Lead-like molecules: {lead_like_count} ({lead_like_count/len(molecules)*100:.1f}%)\n\n"
    
    # Property distributions
    report += f"Property Distributions:\n"
    for prop in ['molecular_weight', 'logp', 'tpsa', 'rotatable_bonds']:
        values = descriptors_df[prop]
        report += f"  {prop.replace('_', ' ').title()}:\n"
        report += f"    Min: {values.min():.2f}, Max: {values.max():.2f}\n"
        report += f"    Mean: {values.mean():.2f}, Std: {values.std():.2f}\n"
    
    report += "\n"
    
    # Top molecules by drug-likeness
    top_molecules = descriptors_df.nlargest(5, 'drug_likeness_score')
    report += "Top 5 Drug-like Molecules:\n"
    for idx, row in top_molecules.iterrows():
        report += f"  {row['name']}: Score = {row['drug_likeness_score']:.3f}\n"
    
    return report


def cleanup_molecules(molecules: List[DrugMolecule], 
                     remove_salts=True, 
                     standardize=True) -> List[DrugMolecule]:
    """Clean up and standardize molecule structures."""
    from rdkit.Chem import SaltRemover
    from rdkit.Chem.MolStandardize import rdMolStandardize
    
    cleaned_molecules = []
    
    if remove_salts:
        salt_remover = SaltRemover.SaltRemover()
    
    if standardize:
        normalizer = rdMolStandardize.Normalizer()
        uncharger = rdMolStandardize.Uncharger()
    
    for mol_obj in molecules:
        try:
            mol = mol_obj.mol
            
            # Remove salts
            if remove_salts:
                mol = salt_remover.StripMol(mol)
            
            # Standardize
            if standardize:
                mol = normalizer.normalize(mol)
                mol = uncharger.uncharge(mol)
            
            # Create new DrugMolecule object with cleaned structure
            new_smiles = Chem.MolToSmiles(mol)
            cleaned_mol = DrugMolecule(smiles=new_smiles, name=mol_obj.name)
            
            # Copy properties
            for key, value in mol_obj.get_properties().items():
                cleaned_mol.set_property(key, value)
            
            cleaned_molecules.append(cleaned_mol)
            
        except Exception as e:
            print(f"Warning: Could not clean molecule {mol_obj.name}: {e}")
            # Keep original if cleaning fails
            cleaned_molecules.append(mol_obj)
    
    return cleaned_molecules