"""
Advanced ADMET prediction with multiple drug-likeness filters.
"""

import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, Lipinski


class ADMETPredictor:
    """Advanced ADMET prediction with multiple drug-likeness filters."""
    
    def __init__(self):
        """Initialize ADMET predictor."""
        pass
    
    def predict_properties(self, mol):
        """Predict ADMET properties for a given molecule."""
        if mol is None:
            raise ValueError("Invalid molecule object")
        
        self.mol = mol
        self.mol_no_h = Chem.RemoveHs(mol)
        self._descriptors = None
        self._calculate_descriptors()
        
        # Return essential properties for drug discovery
        desc = self._descriptors
        lipinski = self.check_lipinski_ro5()
        veber = self.check_veber()
        
        return {
            'Molecular_Weight': desc['MW'],
            'LogP': desc['LogP'],
            'TPSA': desc['TPSA'],
            'HBD': desc['HBD'],
            'HBA': desc['HBA'],
            'Rotatable_Bonds': desc['RotBonds'],
            'QED': desc['QED'],
            'Lipinski_Violations': lipinski['violations'],
            'Lipinski_Pass': lipinski['passed'],
            'Veber_Pass': veber['passed'],
            'Drug_Like_QED': desc['QED'] > 0.5,
            'Oral_Bioavailability': veber['passed'] and lipinski['passed']
        }
    
    def _calculate_descriptors(self):
        """Calculate all molecular descriptors."""
        mol = self.mol_no_h
        
        self._descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'QED': QED.qed(mol),
            'HeavyAtoms': mol.GetNumHeavyAtoms(),
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            'RingCount': Descriptors.RingCount(mol),
            'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
            'PEOE_VSA14': Descriptors.PEOE_VSA14(mol)
        }
        
        # Alias for backward compatibility
        self._descriptors['RotatableBonds'] = self._descriptors['RotBonds']
        
        try:
            from rdkit.Contrib.SA_Score import sascorer
            self._descriptors['SyntheticAccessibility'] = sascorer.calculateScore(mol)
        except ImportError:
            self._descriptors['SyntheticAccessibility'] = None
    
    def check_lipinski_ro5(self):
        """Lipinski Rule of Five."""
        desc = self._descriptors
        violations = []
        
        if desc['MW'] > 500:
            violations.append('MW > 500')
        if desc['LogP'] > 5:
            violations.append('LogP > 5')
        if desc['HBD'] > 5:
            violations.append('HBD > 5')
        if desc['HBA'] > 10:
            violations.append('HBA > 10')
        
        return {
            'passed': len(violations) <= 1,  # Allow 1 violation
            'violations': len(violations),
            'details': violations,
            'rule': f"Lipinski Ro5 ({len(violations)}/4 violations)"
        }
    
    def check_veber(self):
        """Veber oral bioavailability rules."""
        desc = self._descriptors
        violations = []
        
        if desc['TPSA'] > 140:
            violations.append('TPSA > 140')
        if desc['RotBonds'] > 10:
            violations.append('RotBonds > 10')
        
        return {
            'passed': len(violations) == 0,
            'violations': len(violations),
            'details': violations,
            'rule': f"Veber ({len(violations)}/2 violations)"
        }
    
    def check_ghose(self):
        """Ghose filter."""
        desc = self._descriptors
        violations = []
        
        if not (160 <= desc['MW'] <= 480):
            violations.append('MW not in 160-480')
        if not (-0.4 <= desc['LogP'] <= 5.6):
            violations.append('LogP not in -0.4 to 5.6')
        if not (20 <= desc['HeavyAtoms'] <= 70):
            violations.append('Heavy atoms not in 20-70')
        
        return {
            'passed': len(violations) == 0,
            'violations': len(violations),
            'details': violations,
            'rule': f"Ghose ({len(violations)}/3 violations)"
        }
    
    def check_egan(self):
        """Egan filter."""
        desc = self._descriptors
        violations = []
        
        if not (desc['TPSA'] <= 131.6):
            violations.append('TPSA > 131.6')
        if not (-1 <= desc['LogP'] <= 6):
            violations.append('LogP not in -1 to 6')
        
        return {
            'passed': len(violations) == 0,
            'violations': len(violations),
            'details': violations,
            'rule': f"Egan ({len(violations)}/2 violations)"
        }
    
    def check_muegge(self):
        """Muegge filter."""
        desc = self._descriptors
        violations = []
        
        if not (200 <= desc['MW'] <= 600):
            violations.append('MW not in 200-600')
        if not (-2 <= desc['LogP'] <= 5):
            violations.append('LogP not in -2 to 5')
        if not (desc['TPSA'] <= 150):
            violations.append('TPSA > 150')
        if not (desc['RingCount'] <= 7):
            violations.append('Rings > 7')
        
        return {
            'passed': len(violations) == 0,
            'violations': len(violations),
            'details': violations,
            'rule': f"Muegge ({len(violations)}/4 violations)"
        }
    
    def check_leadlike(self):
        """Lead-like filter."""
        desc = self._descriptors
        violations = []
        
        if not (250 <= desc['MW'] <= 350):
            violations.append('MW not in 250-350')
        if not (desc['LogP'] <= 4):
            violations.append('LogP > 4')
        if not (desc['RotBonds'] <= 7):
            violations.append('RotBonds > 7')
        
        return {
            'passed': len(violations) == 0,
            'violations': len(violations),
            'details': violations,
            'rule': f"Lead-like ({len(violations)}/3 violations)"
        }
    
    def check_pains(self):
        """
        PAINS (Pan-Assay Interference) filter.
        
        Detects common promiscuous substructures that frequently give 
        false positives in biochemical assays.
        
        Reference: Baell & Holloway, J. Med. Chem. 2010
        """
        # Common PAINS patterns (subset of most important ones)
        pains_patterns = {
            'quinone': '[#6]1([#8])=[#6][#6]([#8])=[#6][#6]=[#6]1',  # Quinones
            'catechol': 'c1cc(O)c(O)cc1',  # Catechols
            'rhodanine': 'S=C1SC(=O)N(C1)C',  # Rhodanines
            'michael_acceptor': '[CH2]=[CH][C,c](=O)',  # Michael acceptors
            'thiourea': 'NC(=S)N',  # Thioureas
            'hydrazone': 'C=NN',  # Hydrazones
            'azo': 'N=N',  # Azo compounds
            'nitro_aromatic': 'c[N+](=O)[O-]',  # Nitroaromatics
            'aldehyde': '[CH]=O',  # Aldehydes (reactive)
            'epoxide': 'C1OC1',  # Epoxides
            'hydroxylamine': 'NO',  # Hydroxylamines
            'enone': 'C=CC(=O)C',  # α,β-unsaturated ketones
        }
        
        alerts = []
        for name, smarts in pains_patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and self.mol.HasSubstructMatch(pattern):
                alerts.append(f"{name}")
        
        return {
            'passed': len(alerts) == 0,
            'alerts': alerts,
            'rule': f"PAINS ({len(alerts)} alerts)"
        }
    
    def predict_aggregator_risk(self):
        """
        Predict aggregator risk based on Shoichet lab research.
        
        Key features for aggregation:
        - High LogP (> 3) - hydrophobic
        - Low TPSA (< 75) - poor aqueous solvation
        - Large fused aromatic systems
        - Few hydrogen bond donors/acceptors
        
        Reference: Irwin et al. J. Med. Chem. 2015
        """
        desc = self._descriptors
        
        risk_score = 0
        factors = []
        
        # LogP > 3 is major aggregation risk
        if desc['LogP'] > 4:
            risk_score += 2
            factors.append('Very high LogP (>4)')
        elif desc['LogP'] > 3:
            risk_score += 1
            factors.append('High LogP (>3)')
        
        # Large aromatic systems
        if desc['AromaticRings'] > 3:
            risk_score += 2
            factors.append('Many aromatic rings (>3)')
        elif desc['AromaticRings'] > 2:
            risk_score += 1
            factors.append('Multiple aromatic rings (>2)')
        
        # Low TPSA indicates poor aqueous solvation
        if desc['TPSA'] < 50:
            risk_score += 1
            factors.append('Low TPSA (<50)')
        
        # Heavy atom count (larger molecules aggregate more)
        if desc['HeavyAtoms'] > 35:
            risk_score += 1
            factors.append('Large molecule (>35 heavy atoms)')
        
        # Low fraction sp3 (flat molecules aggregate)
        if desc['FractionCsp3'] < 0.2:
            risk_score += 1
            factors.append('Low Fsp3 (<0.2, flat molecule)')
        
        if risk_score >= 4:
            risk = 'High'
        elif risk_score >= 2:
            risk = 'Medium'
        else:
            risk = 'Low'
        
        return {
            'risk': risk,
            'score': risk_score,
            'factors': factors
        }
    
    def get_full_report(self):
        """Get comprehensive ADMET report."""
        filters = {
            'lipinski': self.check_lipinski_ro5(),
            'veber': self.check_veber(),
            'ghose': self.check_ghose(),
            'egan': self.check_egan(),
            'muegge': self.check_muegge(),
            'leadlike': self.check_leadlike()
        }
        
        structural_alerts = {
            'pains': self.check_pains()
        }
        
        predictions = {
            'aggregator': self.predict_aggregator_risk()
        }
        
        return {
            'descriptors': self._descriptors.copy(),
            'filters': filters,
            'structural_alerts': structural_alerts,
            'predictions': predictions
        }
    
    def plot_radar_chart(self, title="Drug-likeness Profile"):
        """Create radar chart comparing to ideal drug-like profile."""
        desc = self._descriptors
        
        # Properties for radar chart (normalized to 0-1)
        props = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'QED']
        max_vals = {'MW': 500, 'LogP': 5, 'TPSA': 140, 'HBD': 5, 'HBA': 10, 'RotBonds': 10, 'QED': 1}
        
        # Normalize values
        values = [min(desc.get(p, 0) / max_vals[p], 1.2) for p in props]
        values.append(values[0])  # Close the radar
        props_closed = props + [props[0]]
        
        # Ideal drug-like reference profile
        ideal_values = [0.6, 0.5, 0.5, 0.4, 0.6, 0.4, 0.7, 0.6]
        
        fig = go.Figure()
        
        # Molecule trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=props_closed,
            fill='toself',
            name='This Molecule',
            line_color='#667eea',
            fillcolor='rgba(102, 126, 234, 0.4)'
        ))
        
        # Ideal drug-like trace
        fig.add_trace(go.Scatterpolar(
            r=ideal_values,
            theta=props_closed,
            fill='toself',
            name='Ideal Drug-like',
            line_color='#38a169',
            fillcolor='rgba(56, 161, 105, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.2]),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            title=dict(text=title, font=dict(size=14)),
            width=500,
            height=450,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig