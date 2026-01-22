"""
Drug molecule data structure with RDKit integration.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from .config import Config


class DrugMolecule:
    """
    Container class for drug molecule data with methods for parsing and validation.

    Attributes:
        name (str): Drug name
        cas (str): CAS registry number
        smiles (str): SMILES string
        xyz (str): XYZ coordinate block
        mol (Chem.Mol): RDKit molecule object
        optimized_coords (np.ndarray): DFT-optimized coordinates
        docking_score (float): Binding affinity from docking
    """

    def __init__(self, name, cas, smiles, xyz_string=''):
        self.name = name
        self.cas = cas
        self.smiles = smiles
        self.xyz_raw = xyz_string
        self.mol = None
        self.optimized_coords = None
        self.docking_score = None
        self.properties = {}
        self.initial_coords = None
        self.elements = []
        self._admet_predictor = None

        # Parse SMILES to RDKit mol
        self._parse_smiles()

        # Parse XYZ coordinates
        self._parse_xyz()

    def _parse_smiles(self):
        """Convert SMILES to RDKit molecule object with hydrogens."""
        self.mol = Chem.MolFromSmiles(self.smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES for {self.name}: {self.smiles}")
        self.mol = Chem.AddHs(self.mol)  # Add explicit hydrogens for 3D

    def _parse_xyz(self):
        """
        Parse XYZ coordinate string into numpy array.
        XYZ format: First line = atom count, second line = comment, then "Element X Y Z"
        """
        if not self.xyz_raw or self.xyz_raw.strip() == '':
            # No coordinates provided, generate with RDKit
            # Use robust embedding with fallbacks
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.maxIterations = 500
            
            embed_result = AllChem.EmbedMolecule(self.mol, params)
            
            if embed_result == -1:
                # Fallback: try with random coordinates
                params.useRandomCoords = True
                embed_result = AllChem.EmbedMolecule(self.mol, params)
            
            if embed_result == -1:
                # Last resort: legacy embedding
                embed_result = AllChem.EmbedMolecule(self.mol, randomSeed=42)
            
            if embed_result == -1:
                raise ValueError(f"Failed to generate 3D coordinates for {self.name}")
            
            # Optimize with force field
            try:
                mmff_result = AllChem.MMFFOptimizeMolecule(self.mol, maxIters=500)
                if mmff_result == -1:
                    AllChem.UFFOptimizeMolecule(self.mol, maxIters=500)
            except:
                try:
                    AllChem.UFFOptimizeMolecule(self.mol, maxIters=500)
                except:
                    pass  # Use unoptimized geometry
            
            conf = self.mol.GetConformer()
            self.initial_coords = conf.GetPositions()

            # Extract elements
            self.elements = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
            return

        # Parse existing XYZ string
        lines = self.xyz_raw.strip().split('\n')
        n_atoms = int(lines[0])
        coords = []
        elements = []

        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        self.initial_coords = np.array(coords)
        self.elements = elements

    def calculate_descriptors(self):
        """Calculate molecular descriptors for drug-likeness filtering."""
        mol_no_h = Chem.RemoveHs(self.mol)  # Most descriptors ignore H

        self.properties = {
            'MW': Descriptors.MolWt(mol_no_h),
            'LogP': Descriptors.MolLogP(mol_no_h),
            'TPSA': Descriptors.TPSA(mol_no_h),
            'HBD': Descriptors.NumHDonors(mol_no_h),
            'HBA': Descriptors.NumHAcceptors(mol_no_h),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol_no_h),
            'AromaticRings': Descriptors.NumAromaticRings(mol_no_h),
            'Rings': Descriptors.RingCount(mol_no_h),
            'FractionCSP3': Descriptors.FractionCSP3(mol_no_h),
            'QED': QED.qed(mol_no_h),  # Drug-likeness score (0-1)
        }
        return self.properties

    def get_mol_with_hydrogens(self):
        """Get molecule with explicit hydrogens."""
        if self.mol is None:
            return None
        mol_h = Chem.AddHs(self.mol)
        return mol_h
    
    def get_mol_without_hydrogens(self):
        """Get molecule without hydrogens (implicit H)."""
        if self.mol is None:
            return None
        return Chem.RemoveHs(self.mol)
    
    def get_xyz_with_hydrogens(self):
        """Get XYZ string including all hydrogens."""
        if self.mol is None or self.initial_coords is None:
            return None
        
        # Make sure we have a molecule with H
        mol_h = Chem.AddHs(self.mol)
        
        # Check if coords match hydrogen-included molecule
        if len(self.elements) == mol_h.GetNumAtoms():
            coords = self.initial_coords
            elements = self.elements
        else:
            # Generate new coordinates with H
            AllChem.EmbedMolecule(mol_h, randomSeed=42)
            try:
                AllChem.MMFFOptimizeMolecule(mol_h)
            except:
                pass
            conf = mol_h.GetConformer()
            coords = conf.GetPositions()
            elements = [a.GetSymbol() for a in mol_h.GetAtoms()]
        
        # Build XYZ string
        xyz = f"{len(elements)}\n{self.name} - with hydrogens\n"
        for el, coord in zip(elements, coords):
            xyz += f"{el} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
        
        return xyz
    
    def count_atoms(self, include_hydrogens=True):
        """Count atoms in the molecule."""
        if self.mol is None:
            return 0
        if include_hydrogens:
            mol_h = Chem.AddHs(self.mol)
            return mol_h.GetNumAtoms()
        else:
            return self.mol.GetNumHeavyAtoms()

    def passes_filters(self):
        """Check if molecule passes Lipinski-like ADMET filters."""
        if not self.properties:
            self.calculate_descriptors()

        checks = [
            Config.MW_MIN <= self.properties['MW'] <= Config.MW_MAX,
            Config.LOGP_MIN <= self.properties['LogP'] <= Config.LOGP_MAX,
            self.properties['TPSA'] <= Config.TPSA_MAX,
            self.properties['HBD'] <= Config.HBD_MAX,
            self.properties['HBA'] <= Config.HBA_MAX,
            self.properties['RotatableBonds'] <= Config.ROTATABLE_BONDS_MAX,
        ]
        return all(checks)

    def get_admet_report(self):
        """Get ADMET report using the enhanced predictor."""
        if not hasattr(self, '_admet_predictor') or self._admet_predictor is None:
            from .admet_predictor import ADMETPredictor
            self._admet_predictor = ADMETPredictor(self.mol)
        return self._admet_predictor.get_full_report()

    def print_admet_report(self):
        """Print formatted ADMET report."""
        if not hasattr(self, '_admet_predictor') or self._admet_predictor is None:
            from .admet_predictor import ADMETPredictor
            self._admet_predictor = ADMETPredictor(self.mol)
        
        report = self._admet_predictor.get_full_report()
        desc = report['descriptors']
        
        print("=" * 55)
        print("🧬 ADMET PREDICTION REPORT")
        print("=" * 55)
        print(f"  MW: {desc['MW']:.1f} | LogP: {desc['LogP']:.2f} | TPSA: {desc['TPSA']:.1f}")
        print(f"  HBD: {desc['HBD']} | HBA: {desc['HBA']} | RotBonds: {desc['RotatableBonds']}")
        print(f"  QED: {desc['QED']:.3f} | SA: {desc.get('SyntheticAccessibility', 'N/A')}")
        print("-" * 55)
        for name, result in report['filters'].items():
            status = "✅" if result['passed'] else "❌"
            print(f"  {status} {result['rule']}")
        print("-" * 55)
        if 'structural_alerts' in report:
            pains = report['structural_alerts']['pains']
            if not pains['passed']:
                print(f"  ⚠️ PAINS: {', '.join(pains['alerts'])}")
            else:
                print("  ✅ No PAINS alerts")
        if 'predictions' in report:
            print(f"  Aggregator Risk: {report['predictions']['aggregator']['risk']}")
        print("=" * 55)

    def show_admet_radar(self):
        """Display radar chart comparing to ideal drug-like profile."""
        if not hasattr(self, '_admet_predictor') or self._admet_predictor is None:
            from .admet_predictor import ADMETPredictor
            self._admet_predictor = ADMETPredictor(self.mol)
        fig = self._admet_predictor.plot_radar_chart(f"{self.name} - Drug-likeness")
        return fig