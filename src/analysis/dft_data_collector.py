
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Try to import RDKit for ADMET calculations
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class DFTDataCollector:
    """
    Collects and aggregates DFT calculation results from wavefunction files (.npz)
    into a comprehensive dataset for analysis, including ADMET properties.
    
    Now includes comprehensive atom-based property extraction:
    - Mulliken, Löwdin, Hirshfeld charges
    - Local reactivity indices (Fukui functions)
    - ESP at nuclei
    - Bond order statistics
    """
    
    def __init__(self, output_dir, save_atom_data=True):
        """
        Initialize DFTDataCollector.
        
        Args:
            output_dir: Directory containing optimized molecule folders
            save_atom_data: If True, saves per-atom data as JSON files alongside CSV
        """
        self.output_dir = Path(output_dir)
        self.save_atom_data = save_atom_data
        self.atom_data_cache = {}  # Store per-atom data for optional export
        
    def collect_data(self, include_atom_summary=True):
        """
        Scans the output directory for optimized wavefunction files and compiles stats.
        Returns a Pandas DataFrame with DFT and ADMET properties.
        """
        data_records = []
        
        # Find all wavefunction files recursively
        # Pattern: */wavefunctions/*_optimized_wavefunction.npz
        wfn_files = list(self.output_dir.rglob("*_optimized_wavefunction.npz"))
        
        print(f"Found {len(wfn_files)} wavefunction files in {self.output_dir}")
        
        for wfn_file in wfn_files:
            try:
                record = self._process_single_file(wfn_file)
                if record:
                    data_records.append(record)
            except Exception as e:
                print(f"Error processing {wfn_file.name}: {e}")
                
        # Create DataFrame
        df = pd.DataFrame(data_records)
        
        # Reorder columns for logical flow if data exists
        if not df.empty:
            preferred_order = [
                # Identification
                'Molecule_Name', 'SMILES', 'Formula', 'Num_Atoms',
                # DFT Electronic
                'Total_Energy_Ha', 'HOMO_eV', 'LUMO_eV', 'Gap_eV', 
                'Dipole_Moment_Debye', 'Polarizability_au3', 'Volume_A3', 'PSA_A2',
                'Hardness_eV', 'Electrophilicity_Index', 'Nucleophilicity_Index',
                # Fukui Reactivity Indices
                'Max_Fukui_f_plus', 'Max_Fukui_f_minus', 'Max_Fukui_f_radical',
                'Fukui_Plus_Atom', 'Fukui_Minus_Atom', 'Fukui_Radical_Atom',
                'Max_Spin_Density',
                # ESP Statistics
                'ESP_Min_au', 'ESP_Max_au', 'ESP_Variance',
                'ESP_Avg_Pos_au', 'ESP_Avg_Neg_au',
                # ========== ATOM-BASED CHARGE STATISTICS ==========
                # Mulliken Charges
                'Mulliken_Min', 'Mulliken_Max', 'Mulliken_Range', 'Mulliken_Std',
                'Mulliken_Most_Positive_Atom', 'Mulliken_Most_Negative_Atom',
                # Löwdin Charges
                'Lowdin_Min', 'Lowdin_Max', 'Lowdin_Range', 'Lowdin_Std',
                'Lowdin_Most_Positive_Atom', 'Lowdin_Most_Negative_Atom',
                # Hirshfeld Charges
                'Hirshfeld_Min', 'Hirshfeld_Max', 'Hirshfeld_Range', 'Hirshfeld_Std',
                'Hirshfeld_Most_Positive_Atom', 'Hirshfeld_Most_Negative_Atom',
                # Local Reactivity
                'Max_Local_Electrophilicity', 'Max_Local_Nucleophilicity',
                'Electrophilic_Site_Atom', 'Nucleophilic_Site_Atom',
                # Bond Order Statistics
                'Avg_Bond_Order', 'Max_Bond_Order', 'Num_Strong_Bonds',
                # ========== SEQUENTIAL PER-ATOM DATA ==========
                'Atom_Index_Sequence', 'Element_Sequence',
                'Mulliken_Charges_Seq', 'Lowdin_Charges_Seq', 'Hirshfeld_Charges_Seq',
                'Fukui_Plus_Seq', 'Fukui_Minus_Seq', 'Fukui_Radical_Seq',
                'Local_Electrophilicity_Seq', 'Local_Nucleophilicity_Seq',
                'Spin_Density_Seq', 'ESP_At_Nuclei_Seq',
                # ========== END ATOM-BASED ==========
                # ADMET Properties
                'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 
                'AromaticRings', 'FractionCsp3', 'QED',
                # Drug-likeness Filters
                'Lipinski_Pass', 'Lipinski_Violations', 
                'Veber_Pass', 'Ghose_Pass', 'Egan_Pass', 'Muegge_Pass', 'LeadLike_Pass',
                # ADMET Predictions
                'BBB_Penetration', 'hERG_Risk', 'Solubility', 'CYP_Inhibition',
                'Aggregator_Risk', 'PAINS_Alerts', 'Oral_Bioavailability'
            ]
            
            # Get columns that actually exist in the dataframe
            existing_cols = [c for c in preferred_order if c in df.columns]
            # Add remaining columns
            remaining_cols = [c for c in df.columns if c not in existing_cols]
            
            df = df[existing_cols + remaining_cols]
            
        return df
    
    def _calculate_admet_properties(self, smiles):
        """Calculate ADMET properties from SMILES string."""
        if not RDKIT_AVAILABLE or not smiles:
            return {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            mol_no_h = Chem.RemoveHs(mol)
            mol_with_h = Chem.AddHs(mol)
            
            # Basic descriptors
            props = {
                'MW': Descriptors.MolWt(mol_no_h),
                'LogP': Descriptors.MolLogP(mol_no_h),
                'TPSA': Descriptors.TPSA(mol_no_h),
                'HBD': Descriptors.NumHDonors(mol_no_h),
                'HBA': Descriptors.NumHAcceptors(mol_no_h),
                'RotatableBonds': Descriptors.NumRotatableBonds(mol_no_h),
                'AromaticRings': Descriptors.NumAromaticRings(mol_no_h),
                'FractionCsp3': Descriptors.FractionCSP3(mol_no_h),
                'HeavyAtoms': mol_no_h.GetNumHeavyAtoms(),
                'RingCount': Descriptors.RingCount(mol_no_h),
                'QED': QED.qed(mol_no_h)
            }
            
            # Lipinski Rule of 5
            lipinski_violations = 0
            if props['MW'] > 500: lipinski_violations += 1
            if props['LogP'] > 5: lipinski_violations += 1
            if props['HBD'] > 5: lipinski_violations += 1
            if props['HBA'] > 10: lipinski_violations += 1
            props['Lipinski_Violations'] = lipinski_violations
            props['Lipinski_Pass'] = lipinski_violations <= 1
            
            # Veber rules
            veber_pass = props['TPSA'] <= 140 and props['RotatableBonds'] <= 10
            props['Veber_Pass'] = veber_pass
            
            # Ghose filter
            ghose_pass = (160 <= props['MW'] <= 480 and 
                         -0.4 <= props['LogP'] <= 5.6 and 
                         20 <= props['HeavyAtoms'] <= 70)
            props['Ghose_Pass'] = ghose_pass
            
            # Egan filter
            egan_pass = props['TPSA'] <= 131.6 and -1 <= props['LogP'] <= 6
            props['Egan_Pass'] = egan_pass
            
            # Muegge filter
            muegge_pass = (200 <= props['MW'] <= 600 and 
                          -2 <= props['LogP'] <= 5 and 
                          props['TPSA'] <= 150 and 
                          props['RingCount'] <= 7)
            props['Muegge_Pass'] = muegge_pass
            
            # Lead-like
            leadlike_pass = (250 <= props['MW'] <= 350 and 
                            props['LogP'] <= 4 and 
                            props['RotatableBonds'] <= 7)
            props['LeadLike_Pass'] = leadlike_pass
            
            # BBB Penetration (Egan/Clark rules)
            bbb_score = 0
            if props['TPSA'] < 90: bbb_score += 1
            if 1 <= props['LogP'] <= 3: bbb_score += 1
            if props['MW'] < 450: bbb_score += 1
            if props['HBD'] <= 2: bbb_score += 1
            props['BBB_Penetration'] = 'High' if bbb_score >= 4 else ('Moderate' if bbb_score >= 2 else 'Low')
            
            # hERG Risk (Aronov model)
            n_basic = len(mol_with_h.GetSubstructMatches(Chem.MolFromSmarts('[#7;+,H1,H2]')))
            herg_risk = 0
            if props['LogP'] > 3: herg_risk += 1
            if props['LogP'] > 4: herg_risk += 1
            if n_basic > 0: herg_risk += 1
            if n_basic > 1 and props['LogP'] > 3: herg_risk += 1
            if props['MW'] > 400 and props['TPSA'] < 75: herg_risk += 1
            props['hERG_Risk'] = 'High' if herg_risk >= 3 else ('Moderate' if herg_risk >= 1 else 'Low')
            
            # Solubility (GSK 4/400 rule)
            sol_score = 0
            if props['LogP'] < 3: sol_score += 2
            elif props['LogP'] < 4: sol_score += 1
            if props['MW'] < 400: sol_score += 1
            if props['AromaticRings'] <= 3: sol_score += 1
            if props['TPSA'] > 50: sol_score += 1
            props['Solubility'] = 'Good' if sol_score >= 4 else ('Moderate' if sol_score >= 2 else 'Poor')
            
            # CYP Inhibition risk
            cyp_risk = 0
            if props['LogP'] > 3: cyp_risk += 1
            if props['AromaticRings'] > 2: cyp_risk += 1
            if props['MW'] > 350: cyp_risk += 1
            props['CYP_Inhibition'] = 'High' if cyp_risk >= 3 else ('Moderate' if cyp_risk >= 1 else 'Low')
            
            # Aggregator Risk (Shoichet model)
            agg_risk = 0
            if props['LogP'] > 4: agg_risk += 2
            elif props['LogP'] > 3: agg_risk += 1
            if props['AromaticRings'] > 3: agg_risk += 2
            elif props['AromaticRings'] > 2: agg_risk += 1
            if props['TPSA'] < 50: agg_risk += 1
            if props['HeavyAtoms'] > 35: agg_risk += 1
            if props['FractionCsp3'] < 0.2: agg_risk += 1
            props['Aggregator_Risk'] = 'High' if agg_risk >= 4 else ('Medium' if agg_risk >= 2 else 'Low')
            
            # PAINS check (simplified)
            pains_patterns = {
                'quinone': '[#6]1([#8])=[#6][#6]([#8])=[#6][#6]=[#6]1',
                'catechol': 'c1cc(O)c(O)cc1',
                'michael_acceptor': '[CH2]=[CH][C,c](=O)',
                'nitro_aromatic': 'c[N+](=O)[O-]',
                'aldehyde': '[CH]=O',
            }
            pains_alerts = []
            for name, smarts in pains_patterns.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol_with_h.HasSubstructMatch(pattern):
                    pains_alerts.append(name)
            props['PAINS_Alerts'] = ','.join(pains_alerts) if pains_alerts else 'None'
            
            # Oral Bioavailability (combined Lipinski + Veber)
            props['Oral_Bioavailability'] = 'Yes' if (props['Lipinski_Pass'] and props['Veber_Pass']) else 'No'
            
            # Remove intermediate keys not needed in CSV
            props.pop('HeavyAtoms', None)
            props.pop('RingCount', None)
            
            return props
            
        except Exception as e:
            print(f"ADMET calculation error: {e}")
            return {}
    
    def _get_atom_label(self, elements, idx):
        """Create an atom label like 'C5' or 'O12' from element and index."""
        if elements is not None and idx < len(elements):
            elem = str(elements[idx])
            return f"{elem}{idx+1}"  # 1-indexed for readability
        return f"Atom{idx+1}"
    
    def _extract_charge_statistics(self, charges, elements, charge_type):
        """
        Extract summary statistics from atomic charges.
        
        Returns dict with keys like:
        - {charge_type}_Min, {charge_type}_Max, {charge_type}_Range, {charge_type}_Std
        - {charge_type}_Most_Positive_Atom, {charge_type}_Most_Negative_Atom
        """
        if charges is None or len(charges) == 0:
            return {}
        
        charges = np.array(charges)
        
        min_val = float(np.min(charges))
        max_val = float(np.max(charges))
        
        stats = {
            f'{charge_type}_Min': min_val,
            f'{charge_type}_Max': max_val,
            f'{charge_type}_Range': max_val - min_val,
            f'{charge_type}_Std': float(np.std(charges)),
        }
        
        # Identify most positive and negative atoms
        max_idx = int(np.argmax(charges))
        min_idx = int(np.argmin(charges))
        
        stats[f'{charge_type}_Most_Positive_Atom'] = self._get_atom_label(elements, max_idx)
        stats[f'{charge_type}_Most_Negative_Atom'] = self._get_atom_label(elements, min_idx)
        
        return stats

    def _process_single_file(self, file_path):
        """Extracts properties from a single .npz file, including atom-based data."""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # infer molecule name from directory or filename
            # Structure usually: .../MoleculeName/wavefunctions/MoleculeName_optimized_wfn.npz
            mol_name = file_path.parent.parent.name
            
            record = {
                'Molecule_Name': mol_name,
                'File_Path': str(file_path)
            }
            
            # Get element symbols for atom labeling
            elements = data['elements'] if 'elements' in data else None
            if elements is not None:
                record['Num_Atoms'] = len(elements)
            
            # --- 1. Basic Electronic Structure ---
            if 'energy' in data:
                record['Total_Energy_Ha'] = float(data['energy'])
            if 'dipole_magnitude' in data:
                record['Dipole_Moment_Debye'] = float(data['dipole_magnitude'])
            
            # HOMO/LUMO/Gap
            if 'mo_energy' in data and 'mo_occ' in data:
                mo_energy = data['mo_energy']
                mo_occ = data['mo_occ']
                
                # Identify HOMO/LUMO
                occ_idx = np.where(mo_occ > 0)[0]
                if len(occ_idx) > 0:
                    homo_idx = occ_idx[-1]
                    lumo_idx = homo_idx + 1 if homo_idx + 1 < len(mo_energy) else homo_idx
                    
                    homo_ev = mo_energy[homo_idx] * 27.2114
                    lumo_ev = mo_energy[lumo_idx] * 27.2114
                    gap_ev = lumo_ev - homo_ev
                    
                    record['HOMO_eV'] = homo_ev
                    record['LUMO_eV'] = lumo_ev
                    record['Gap_eV'] = gap_ev
            
            # --- 2. Conceptual DFT Descriptors ---
            if 'chemical_hardness' in data:
                record['Hardness_eV'] = float(data['chemical_hardness'])
            elif 'HOMO_eV' in record:
                ip = -record['HOMO_eV']
                ea = -record['LUMO_eV']
                record['Hardness_eV'] = (ip - ea) / 2
                
            if 'electrophilicity_index' in data:
                record['Electrophilicity_Index'] = float(data['electrophilicity_index'])
            elif 'HOMO_eV' in record:
                ip = -record['HOMO_eV']
                ea = -record['LUMO_eV']
                neg = (ip + ea) / 2
                hardness = (ip - ea) / 2
                record['Electrophilicity_Index'] = (neg**2)/(2*hardness) if hardness > 0 else 0
                
            if 'nucleophilicity_index' in data:
                record['Nucleophilicity_Index'] = float(data['nucleophilicity_index'])
                
            # --- 3. Molecular Properties ---
            if 'polarizability' in data:
                record['Polarizability_au3'] = float(data['polarizability'])
            if 'molecular_volume' in data:
                record['Volume_A3'] = float(data['molecular_volume'])
            if 'polar_surface_area' in data:
                record['PSA_A2'] = float(data['polar_surface_area'])
                
            # --- 4. Reactivity & Toxicity Indicators (Max Values + Atom Labels) ---
            # Fukui f+ (electrophilic attack susceptibility)
            if 'fukui_plus' in data:
                fp = data['fukui_plus']
                if fp.size > 0:
                    record['Max_Fukui_f_plus'] = float(np.max(fp))
                    max_idx = int(np.argmax(fp))
                    record['Fukui_Plus_Atom'] = self._get_atom_label(elements, max_idx)
            
            # Fukui f- (nucleophilic attack susceptibility)
            if 'fukui_minus' in data:
                fm = data['fukui_minus']
                if fm.size > 0:
                    record['Max_Fukui_f_minus'] = float(np.max(np.abs(fm)))
                    max_idx = int(np.argmax(np.abs(fm)))
                    record['Fukui_Minus_Atom'] = self._get_atom_label(elements, max_idx)
            
            # Fukui f0 (radical attack susceptibility)
            if 'fukui_radical' in data:
                fr = data['fukui_radical']
                if fr.size > 0:
                    record['Max_Fukui_f_radical'] = float(np.max(fr))
                    max_idx = int(np.argmax(fr))
                    record['Fukui_Radical_Atom'] = self._get_atom_label(elements, max_idx)
                    
            if 'spin_densities' in data:
                sd = data['spin_densities']
                record['Max_Spin_Density'] = float(np.max(np.abs(sd))) if sd.size > 0 else 0
                
            # --- 5. Electrostatic Potential Stats ---
            if 'esp_min' in data:
                record['ESP_Min_au'] = float(data['esp_min'])
            if 'esp_max' in data:
                record['ESP_Max_au'] = float(data['esp_max'])
            if 'esp_variance' in data:
                record['ESP_Variance'] = float(data['esp_variance'])
            if 'esp_positive_avg' in data:
                record['ESP_Avg_Pos_au'] = float(data['esp_positive_avg'])
            if 'esp_negative_avg' in data:
                record['ESP_Avg_Neg_au'] = float(data['esp_negative_avg'])
            
            # ============================================================
            # --- 6. ATOM-BASED CHARGE STATISTICS (NEW) ---
            # ============================================================
            
            # 6a. Mulliken Charges
            if 'mulliken_charges' in data:
                mulliken = data['mulliken_charges']
                record.update(self._extract_charge_statistics(mulliken, elements, 'Mulliken'))
            
            # 6b. Löwdin Charges
            if 'lowdin_charges' in data:
                lowdin = data['lowdin_charges']
                record.update(self._extract_charge_statistics(lowdin, elements, 'Lowdin'))
            
            # 6c. Hirshfeld Charges
            if 'hirshfeld_charges' in data:
                hirshfeld = data['hirshfeld_charges']
                record.update(self._extract_charge_statistics(hirshfeld, elements, 'Hirshfeld'))
            
            # 6d. Local Electrophilicity/Nucleophilicity
            if 'local_electrophilicity' in data:
                le = data['local_electrophilicity']
                if le.size > 0:
                    record['Max_Local_Electrophilicity'] = float(np.max(le))
                    max_idx = int(np.argmax(le))
                    record['Electrophilic_Site_Atom'] = self._get_atom_label(elements, max_idx)
            
            if 'local_nucleophilicity' in data:
                ln = data['local_nucleophilicity']
                if ln.size > 0:
                    record['Max_Local_Nucleophilicity'] = float(np.max(ln))
                    max_idx = int(np.argmax(ln))
                    record['Nucleophilic_Site_Atom'] = self._get_atom_label(elements, max_idx)
            
            # 6e. Bond Order Statistics (from Mayer bond orders)
            if 'mayer_bond_orders' in data:
                bo = data['mayer_bond_orders']
                if bo.size > 0:
                    # Get upper triangle (exclude diagonal and lower triangle to avoid double-counting)
                    upper_tri = np.triu(bo, k=1)
                    nonzero_bonds = upper_tri[upper_tri > 0.5]  # Only consider meaningful bonds
                    
                    if len(nonzero_bonds) > 0:
                        record['Avg_Bond_Order'] = float(np.mean(nonzero_bonds))
                        record['Max_Bond_Order'] = float(np.max(nonzero_bonds))
                        record['Num_Strong_Bonds'] = int(np.sum(nonzero_bonds > 1.5))  # Double/triple bonds
            
            # ============================================================
            # --- 6f. SEQUENTIAL PER-ATOM DATA (Full Arrays as Strings) ---
            # ============================================================
            # Store complete atom-by-atom data as semicolon-separated values
            # Format: "value1;value2;value3;..." ordered by atom index
            
            if elements is not None:
                n_atoms = len(elements)
                
                # Atom index sequence (1-indexed for readability)
                record['Atom_Index_Sequence'] = ';'.join([str(i+1) for i in range(n_atoms)])
                
                # Element sequence
                record['Element_Sequence'] = ';'.join([str(e) for e in elements])
                
                # Helper function to format array as semicolon-separated string
                def arr_to_seq(arr, decimals=6):
                    if arr is None or len(arr) == 0:
                        return ''
                    return ';'.join([f'{float(v):.{decimals}f}' for v in arr])
                
                # Charge sequences
                if 'mulliken_charges' in data:
                    record['Mulliken_Charges_Seq'] = arr_to_seq(data['mulliken_charges'])
                
                if 'lowdin_charges' in data:
                    record['Lowdin_Charges_Seq'] = arr_to_seq(data['lowdin_charges'])
                
                if 'hirshfeld_charges' in data:
                    record['Hirshfeld_Charges_Seq'] = arr_to_seq(data['hirshfeld_charges'])
                
                # Fukui function sequences
                if 'fukui_plus' in data:
                    record['Fukui_Plus_Seq'] = arr_to_seq(data['fukui_plus'])
                
                if 'fukui_minus' in data:
                    record['Fukui_Minus_Seq'] = arr_to_seq(data['fukui_minus'])
                
                if 'fukui_radical' in data:
                    record['Fukui_Radical_Seq'] = arr_to_seq(data['fukui_radical'])
                
                # Local reactivity sequences
                if 'local_electrophilicity' in data:
                    record['Local_Electrophilicity_Seq'] = arr_to_seq(data['local_electrophilicity'])
                
                if 'local_nucleophilicity' in data:
                    record['Local_Nucleophilicity_Seq'] = arr_to_seq(data['local_nucleophilicity'])
                
                # Spin density sequence
                if 'spin_densities' in data:
                    record['Spin_Density_Seq'] = arr_to_seq(data['spin_densities'])
                
                # ESP at nuclei sequence
                if 'esp_at_nuclei' in data:
                    record['ESP_At_Nuclei_Seq'] = arr_to_seq(data['esp_at_nuclei'], decimals=4)
            
            # ============================================================
            # --- 7. Store Full Atom Data for Optional JSON Export ---
            # ============================================================
            if self.save_atom_data and elements is not None:
                atom_data = {
                    'molecule_name': mol_name,
                    'elements': [str(e) for e in elements],
                    'num_atoms': len(elements),
                }
                
                # Add per-atom arrays
                for key in ['mulliken_charges', 'lowdin_charges', 'hirshfeld_charges',
                           'fukui_plus', 'fukui_minus', 'fukui_radical',
                           'local_electrophilicity', 'local_nucleophilicity',
                           'spin_densities', 'esp_at_nuclei', 'electron_density_at_nuclei']:
                    if key in data:
                        arr = data[key]
                        atom_data[key] = [float(x) for x in arr]
                
                # Add coordinates
                if 'coords' in data:
                    atom_data['coordinates'] = data['coords'].tolist()
                
                # Add bond order matrices (for edge features)
                if 'mayer_bond_orders' in data:
                    atom_data['mayer_bond_orders'] = data['mayer_bond_orders'].tolist()
                
                if 'wiberg_indices' in data:
                    atom_data['wiberg_indices'] = data['wiberg_indices'].tolist()
                
                # Store in cache
                self.atom_data_cache[mol_name] = atom_data
                
            # --- 8. Get SMILES and calculate ADMET ---
            smiles = None
            log_file = file_path.parent.parent / "optimization.log"
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if "SMILES:" in line:
                                smiles = line.split("SMILES:")[1].strip()
                                record['SMILES'] = smiles
                            if "Formula:" in line:
                                record['Formula'] = line.split("Formula:")[1].strip()
                            if smiles:
                                break
                except:
                    pass
            
            # --- 9. Calculate ADMET Properties ---
            if smiles:
                admet_props = self._calculate_admet_properties(smiles)
                record.update(admet_props)
            
            return record
            
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None
    
    def save_atom_data_json(self, output_path=None):
        """
        Save all per-atom data to a JSON file for detailed analysis.
        
        Args:
            output_path: Path for output JSON. If None, saves to output_dir/atom_data.json
        
        Returns:
            Path to saved JSON file
        """
        if not self.atom_data_cache:
            print("No atom data cached. Run collect_data() first.")
            return None
        
        if output_path is None:
            output_path = self.output_dir / "atom_level_data.json"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.atom_data_cache, f, indent=2)
        
        print(f"💾 Saved atom-level data for {len(self.atom_data_cache)} molecules to {output_path}")
        return output_path
    
    def get_atom_data_dataframe(self, molecule_name=None):
        """
        Get atom-level data as a DataFrame for a specific molecule or all molecules.
        
        Args:
            molecule_name: Name of molecule to get data for. If None, returns all.
        
        Returns:
            DataFrame with per-atom properties
        """
        if molecule_name:
            if molecule_name not in self.atom_data_cache:
                print(f"No atom data for {molecule_name}")
                return pd.DataFrame()
            
            atom_data = self.atom_data_cache[molecule_name]
            df = pd.DataFrame({
                'atom_index': range(1, atom_data['num_atoms'] + 1),
                'element': atom_data['elements'],
            })
            
            # Add all available per-atom properties
            for key in ['mulliken_charges', 'lowdin_charges', 'hirshfeld_charges',
                       'fukui_plus', 'fukui_minus', 'fukui_radical',
                       'local_electrophilicity', 'local_nucleophilicity',
                       'spin_densities', 'esp_at_nuclei', 'electron_density_at_nuclei']:
                if key in atom_data:
                    df[key] = atom_data[key]
            
            df.insert(0, 'molecule', molecule_name)
            return df
        
        # Return data for all molecules
        all_dfs = []
        for mol_name in self.atom_data_cache:
            mol_df = self.get_atom_data_dataframe(mol_name)
            if not mol_df.empty:
                all_dfs.append(mol_df)
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()
    
    def save_full_analysis_json(self, df_results=None, output_path='dft_full_analysis.json'):
        """
        Save a comprehensive JSON file containing ALL data:
        - Molecular-level properties (from CSV columns)
        - Per-atom properties (charges, Fukui, ESP, etc.)
        - Bond order matrices (Mayer, Wiberg)
        - 3D coordinates
        
        This creates ONE file with everything needed for GNN training.
        
        Args:
            df_results: DataFrame from collect_data(). If None, runs collect_data() first.
            output_path: Path for output JSON file
        
        Returns:
            Path to saved JSON file
        """
        if df_results is None:
            df_results = self.collect_data()
        
        if not self.atom_data_cache:
            print("⚠️ No atom data cached. Make sure save_atom_data=True in constructor.")
            return None
        
        full_data = {}
        
        for idx, row in df_results.iterrows():
            mol_name = row['Molecule_Name']
            
            # 1. Molecular-level properties (from CSV)
            mol_data = {
                'molecular_properties': {}
            }
            
            # Add all molecular-level columns
            for col in df_results.columns:
                if col not in ['Molecule_Name', 'File_Path'] and not col.endswith('_Seq'):
                    val = row[col]
                    # Convert numpy types to Python types
                    if pd.notna(val):
                        if hasattr(val, 'item'):
                            val = val.item()
                        mol_data['molecular_properties'][col] = val
            
            # 2. Atom-level properties (from cache)
            if mol_name in self.atom_data_cache:
                atom_cache = self.atom_data_cache[mol_name]
                
                # Basic info
                mol_data['num_atoms'] = atom_cache.get('num_atoms', 0)
                mol_data['elements'] = atom_cache.get('elements', [])
                
                # Per-atom arrays
                mol_data['atom_properties'] = {}
                for key in ['mulliken_charges', 'lowdin_charges', 'hirshfeld_charges',
                           'fukui_plus', 'fukui_minus', 'fukui_radical',
                           'local_electrophilicity', 'local_nucleophilicity',
                           'spin_densities', 'esp_at_nuclei', 'electron_density_at_nuclei']:
                    if key in atom_cache:
                        mol_data['atom_properties'][key] = atom_cache[key]
                
                # 3D Coordinates
                if 'coordinates' in atom_cache:
                    mol_data['coordinates'] = atom_cache['coordinates']
                
                # 3. Bond order matrices
                mol_data['bond_orders'] = {}
                if 'mayer_bond_orders' in atom_cache:
                    mol_data['bond_orders']['mayer'] = atom_cache['mayer_bond_orders']
                if 'wiberg_indices' in atom_cache:
                    mol_data['bond_orders']['wiberg'] = atom_cache['wiberg_indices']
            
            full_data[mol_name] = mol_data
        
        # Save to JSON
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(full_data, f, indent=2)
        
        print(f"💾 Saved comprehensive analysis to '{output_path}'")
        print(f"   └─ {len(full_data)} molecules")
        print(f"   └─ Molecular properties: {len(mol_data['molecular_properties'])} columns")
        print(f"   └─ Atom properties: {len(mol_data.get('atom_properties', {}))} arrays")
        print(f"   └─ Bond orders: {list(mol_data.get('bond_orders', {}).keys())}")
        
        return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect DFT results into CSV")
    parser.add_argument("--output_dir", default="dft_data", help="Root directory of DFT outputs")
    parser.add_argument("--csv_name", default="dft_analysis_results.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    
    collector = DFTDataCollector(args.output_dir)
    df = collector.collect_data()
    
    if not df.empty:
        out_path = Path(args.csv_name).resolve()
        df.to_csv(out_path, index=False)
        print(f"✅ Successfully saved consolidated data to: {out_path}")
        print(f"   Total Layout: {df.shape[0]} molecules x {df.shape[1]} properties")
    else:
        print("⚠️ No data collected. Check your output directory path.")

if __name__ == "__main__":
    main()

