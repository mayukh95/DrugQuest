# binding_score.py
# COX-2 Binding Score Calculation Module

from rdkit import Chem
from rdkit.Chem import Descriptors

# COX-2 binding site pharmacophore requirements (from structural analysis)
# These are based on the binding site features, NOT any specific ligand
COX2_BINDING_SITE = {
    'HBA_range': (1, 6),        # H-bond acceptors for Arg120, Tyr355, Ser530
    'HBD_range': (0, 4),        # H-bond donors 
    'LogP_range': (1.0, 5.5),   # Hydrophobic pocket compatibility
    'AromaticRings_range': (1, 4),  # π-stacking with Tyr385, Trp387
    'TPSA_range': (20, 140),    # Polar surface area for membrane permeability
    'MW_range': (150, 600),     # Size constraint for binding pocket
    'RotatableBonds_max': 10,   # Flexibility constraint
}


def compute_unbiased_binding_score(mol):
    """
    Score molecules based on COX-2 binding site compatibility.
    Uses pharmacophore requirements, NOT similarity to any reference ligand.
    """
    if mol is None:
        return 0.0
    
    scores = []
    weights = []
    smiles = Chem.MolToSmiles(mol)
    
    # 1. Carboxylic Acid/Acidic Group (important for Arg120 interaction)
    # Check for various acidic groups, not just carboxylic acid
    has_carboxylic = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[OH]')) or 'C(=O)[O-]' in smiles
    has_sulfonamide = mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N'))
    has_sulfonic = mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)[OH]'))
    has_phosphate = mol.HasSubstructMatch(Chem.MolFromSmarts('P(=O)([OH])'))
    
    acidic_score = 1.0 if (has_carboxylic or has_sulfonamide or has_sulfonic or has_phosphate) else 0.4
    scores.append(acidic_score)
    weights.append(3.0)  # Important but not the only criterion
    
    # 2. H-Bond Acceptors (range-based, not target-based)
    hba = Descriptors.NumHAcceptors(mol)
    hba_min, hba_max = COX2_BINDING_SITE['HBA_range']
    if hba_min <= hba <= hba_max:
        hba_score = 1.0
    elif hba < hba_min:
        hba_score = max(0.3, 1.0 - (hba_min - hba) * 0.2)
    else:
        hba_score = max(0.3, 1.0 - (hba - hba_max) * 0.1)
    scores.append(hba_score)
    weights.append(2.0)
    
    # 3. H-Bond Donors (range-based)
    hbd = Descriptors.NumHDonors(mol)
    hbd_min, hbd_max = COX2_BINDING_SITE['HBD_range']
    if hbd_min <= hbd <= hbd_max:
        hbd_score = 1.0
    else:
        hbd_score = max(0.3, 1.0 - abs(hbd - hbd_max) * 0.15)
    scores.append(hbd_score)
    weights.append(2.0)
    
    # 4. LogP (hydrophobic pocket compatibility)
    logp = Descriptors.MolLogP(mol)
    logp_min, logp_max = COX2_BINDING_SITE['LogP_range']
    if logp_min <= logp <= logp_max:
        logp_score = 1.0
    elif logp < logp_min:
        logp_score = max(0.2, 1.0 - (logp_min - logp) * 0.2)
    else:
        logp_score = max(0.2, 1.0 - (logp - logp_max) * 0.15)
    scores.append(logp_score)
    weights.append(2.5)
    
    # 5. Aromatic Rings (π-stacking potential)
    arom = Descriptors.NumAromaticRings(mol)
    arom_min, arom_max = COX2_BINDING_SITE['AromaticRings_range']
    if arom_min <= arom <= arom_max:
        arom_score = 1.0
    elif arom == 0:
        arom_score = 0.4  # No aromatic rings is a disadvantage
    else:
        arom_score = max(0.5, 1.0 - (arom - arom_max) * 0.1)
    scores.append(arom_score)
    weights.append(2.0)
    
    # 6. Molecular Weight (binding pocket size constraint)
    mw = Descriptors.MolWt(mol)
    mw_min, mw_max = COX2_BINDING_SITE['MW_range']
    if mw_min <= mw <= mw_max:
        mw_score = 1.0
    elif mw < mw_min:
        mw_score = 0.5
    else:
        mw_score = max(0.3, 1.0 - (mw - mw_max) / 200)
    scores.append(mw_score)
    weights.append(1.5)
    
    # 7. TPSA (membrane permeability and binding)
    tpsa = Descriptors.TPSA(mol)
    tpsa_min, tpsa_max = COX2_BINDING_SITE['TPSA_range']
    if tpsa_min <= tpsa <= tpsa_max:
        tpsa_score = 1.0
    elif tpsa < tpsa_min:
        tpsa_score = 0.6
    else:
        tpsa_score = max(0.3, 1.0 - (tpsa - tpsa_max) / 100)
    scores.append(tpsa_score)
    weights.append(1.5)
    
    # 8. Rotatable Bonds (flexibility for binding)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    if rot_bonds <= COX2_BINDING_SITE['RotatableBonds_max']:
        rot_score = 1.0
    else:
        rot_score = max(0.4, 1.0 - (rot_bonds - 10) * 0.1)
    scores.append(rot_score)
    weights.append(1.0)
    
    # Weighted average
    total_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return total_score
