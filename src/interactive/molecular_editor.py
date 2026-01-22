"""
Interactive Molecular Editor with SMILES input and 3D preview.

Features:
- SMILES input with real-time validation
- 3D structure preview with py3Dmol
- 2D structure preview with RDKit
- Molecular property calculator with ADMET indicators
- Substructure highlighting (toxicophores)
- Similarity search against loaded molecules
- Conformer explorer
- History/Undo functionality
- Copy to clipboard
- Download SDF/PNG
- Common functional group additions
"""

import ipywidgets as widgets
from IPython.display import display, HTML, SVG, clear_output
import io
import base64

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from rdkit.Chem.QED import qed
    QED_AVAILABLE = True
except ImportError:
    QED_AVAILABLE = False

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False


class MolecularEditor:
    """
    Interactive molecular editor with SMILES input and 3D preview.
    
    Features:
    - SMILES input with real-time validation
    - 2D and 3D structure preview
    - Molecular property calculator with ADMET indicators
    - Similarity search
    - Conformer explorer
    - Substructure/toxicophore highlighting
    - History with undo
    - Copy/Download options
    """
    
    # Common functional groups for quick addition
    FUNCTIONAL_GROUPS = {
        'Methyl (-CH3)': 'C',
        'Hydroxyl (-OH)': 'O',
        'Amino (-NH2)': 'N',
        'Carboxyl (-COOH)': 'C(=O)O',
        'Fluoro (-F)': 'F',
        'Chloro (-Cl)': 'Cl',
        'Phenyl (-Ph)': 'c1ccccc1',
        'Acetyl (-COCH3)': 'C(=O)C',
    }
    
    # Example molecules (fallback)
    EXAMPLE_MOLECULES = {
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Acetaminophen': 'CC(=O)NC1=CC=C(C=C1)O',
        'Benzene': 'c1ccccc1',
    }
    
    # Toxicophore SMARTS patterns (common structural alerts)
    TOXICOPHORE_PATTERNS = {
        'Nitro aromatic': '[$(N(=O)~O)]~c',
        'Aldehyde': '[CH1]=O',
        'Michael acceptor': '[CH]=[CH][C,S,N]=O',
        'Epoxide': 'C1OC1',
        'Halogenated alkene': '[Cl,Br,I]C=C',
        'Acyl halide': 'C(=O)[Cl,Br,I]',
        'Thiocarbonyl': 'C=S',
        'Hydrazine': 'NN',
    }
    
    def __init__(self, molecules_list=None, drug_molecule_class=None):
        """
        Initialize the molecular editor.
        
        Parameters:
        -----------
        molecules_list : list, optional
            List to add new molecules to
        drug_molecule_class : class, optional
            DrugMolecule class for creating new molecules
        """
        self.molecules_list = molecules_list if molecules_list is not None else []
        self.drug_molecule_class = drug_molecule_class
        self.current_mol = None
        self.current_smiles = ''
        self.loaded_molecules_dict = {}
        
        # History for undo
        self.history = []
        self.history_index = -1
        self.max_history = 20
        
        # Conformers
        self.conformers = []
        self.current_conformer_idx = 0
        
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Create editor interface widgets."""
        
        # Show hydrogens checkbox
        self.show_h_checkbox = widgets.Checkbox(
            value=False,
            description='Show H',
            indent=False,
            layout=widgets.Layout(width='80px')
        )
        self.show_h_checkbox.observe(
            lambda c: self._show_3d_structure() if c['name'] == 'value' else None, 
            names='value'
        )
        
        # SMILES input
        self.smiles_input = widgets.Textarea(
            value='',
            placeholder='Enter SMILES string (e.g., CCO for ethanol)',
            description='',
            layout=widgets.Layout(width='100%', height='50px')
        )
        self.smiles_input.observe(self._on_smiles_change, names='value')
        
        # Name input
        self.name_input = widgets.Text(
            value='New Molecule',
            description='Name:',
            style={'description_width': '45px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Molecule dropdown
        self.example_select = widgets.Dropdown(
            options=['-- Select Molecule --'] + list(self.EXAMPLE_MOLECULES.keys()),
            value='-- Select Molecule --',
            description='',
            layout=widgets.Layout(width='200px')
        )
        self.example_select.observe(self._load_example, names='value')
        
        # Highlight dropdown
        self.highlight_select = widgets.Dropdown(
            options=['None'] + list(self.TOXICOPHORE_PATTERNS.keys()),
            value='None',
            description='Highlight:',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='200px')
        )
        self.highlight_select.observe(lambda c: self._show_2d_structure(), names='value')
        
        # Buttons row 1
        self.validate_btn = widgets.Button(
            description='✓ Validate',
            button_style='info',
            layout=widgets.Layout(width='90px')
        )
        self.validate_btn.on_click(self._validate_smiles)
        
        self.undo_btn = widgets.Button(
            description='↩️',
            button_style='',
            layout=widgets.Layout(width='40px'),
            tooltip='Undo'
        )
        self.undo_btn.on_click(self._undo)
        
        self.redo_btn = widgets.Button(
            description='↪️',
            button_style='',
            layout=widgets.Layout(width='40px'),
            tooltip='Redo'
        )
        self.redo_btn.on_click(self._redo)
        
        self.copy_btn = widgets.Button(
            description='📋 Copy',
            button_style='',
            layout=widgets.Layout(width='80px'),
            tooltip='Copy SMILES to clipboard'
        )
        self.copy_btn.on_click(self._copy_smiles)
        
        # Action buttons
        self.add_btn = widgets.Button(
            description='➕ Add to List',
            button_style='success',
            layout=widgets.Layout(width='120px')
        )
        self.add_btn.on_click(self._add_to_molecules)
        
        self.download_sdf_btn = widgets.Button(
            description='📥 SDF',
            button_style='',
            layout=widgets.Layout(width='70px'),
            tooltip='Download as SDF'
        )
        self.download_sdf_btn.on_click(self._download_sdf)
        
        self.download_png_btn = widgets.Button(
            description='📥 PNG',
            button_style='',
            layout=widgets.Layout(width='70px'),
            tooltip='Download 2D image'
        )
        self.download_png_btn.on_click(self._download_png)
        
        self.clear_btn = widgets.Button(
            description='🗑️',
            button_style='warning',
            layout=widgets.Layout(width='40px'),
            tooltip='Clear editor'
        )
        self.clear_btn.on_click(self._clear_editor)
        
        self.similarity_btn = widgets.Button(
            description='🔍 Find Similar',
            button_style='primary',
            layout=widgets.Layout(width='120px')
        )
        self.similarity_btn.on_click(self._find_similar)
        
        # Conformer controls
        self.conformer_label = widgets.HTML(value="<b>Conformer:</b> -")
        self.prev_conf_btn = widgets.Button(description='◀', layout=widgets.Layout(width='35px'))
        self.next_conf_btn = widgets.Button(description='▶', layout=widgets.Layout(width='35px'))
        self.gen_conf_btn = widgets.Button(
            description='Generate Conformers',
            button_style='',
            layout=widgets.Layout(width='140px')
        )
        self.prev_conf_btn.on_click(self._prev_conformer)
        self.next_conf_btn.on_click(self._next_conformer)
        self.gen_conf_btn.on_click(self._generate_conformers)
        
        # Functional group buttons
        self.fg_buttons = []
        for fg_name in list(self.FUNCTIONAL_GROUPS.keys()):
            btn = widgets.Button(
                description=fg_name.split('(')[0].strip(),
                layout=widgets.Layout(width='auto', height='26px'),
                button_style=''
            )
            btn.on_click(lambda b, fg=fg_name: self._add_functional_group(fg))
            self.fg_buttons.append(btn)
        
        # Output areas
        self.structure_output = widgets.Output(layout=widgets.Layout(min_height='280px'))
        self.props_output = widgets.Output()
        self.status_output = widgets.Output()
        self.structure_3d_output = widgets.Output(layout=widgets.Layout(min_height='280px'))
        self.similarity_output = widgets.Output()
        self.download_output = widgets.Output()
    
    def _on_smiles_change(self, change):
        """React to SMILES input changes."""
        smiles = change['new'].strip()
        if smiles and smiles != self.current_smiles:
            self._validate_and_preview(smiles)
            self._add_to_history(smiles)
    
    def _add_to_history(self, smiles):
        """Add SMILES to history for undo/redo."""
        if self.history and self.history[-1] == smiles:
            return  # Don't add duplicates
        
        # Trim future history if we're not at the end
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(smiles)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.history_index = len(self.history) - 1
    
    def _undo(self, b=None):
        """Go back in history."""
        if self.history_index > 0:
            self.history_index -= 1
            smiles = self.history[self.history_index]
            self.smiles_input.unobserve(self._on_smiles_change, names='value')
            self.smiles_input.value = smiles
            self.smiles_input.observe(self._on_smiles_change, names='value')
            self._validate_and_preview(smiles)
    
    def _redo(self, b=None):
        """Go forward in history."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            smiles = self.history[self.history_index]
            self.smiles_input.unobserve(self._on_smiles_change, names='value')
            self.smiles_input.value = smiles
            self.smiles_input.observe(self._on_smiles_change, names='value')
            self._validate_and_preview(smiles)
    
    def _copy_smiles(self, b=None):
        """Copy SMILES to clipboard using JavaScript."""
        if self.current_smiles:
            js = f"""
            <script>
            navigator.clipboard.writeText("{self.current_smiles}").then(function() {{
                console.log('Copied!');
            }});
            </script>
            """
            with self.status_output:
                clear_output(wait=True)
                display(HTML(js))
                print(f"📋 Copied to clipboard: {self.current_smiles}")
    
    def _validate_and_preview(self, smiles):
        """Validate SMILES and show preview."""
        if not RDKIT_AVAILABLE:
            with self.status_output:
                clear_output(wait=True)
                print("❌ RDKit not available")
            return
        
        with self.status_output:
            clear_output(wait=True)
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print("❌ Invalid SMILES")
                    self.current_mol = None
                    return
                
                self.current_mol = mol
                self.current_smiles = smiles
                self.conformers = []  # Reset conformers
                self.current_conformer_idx = 0
                
                # Canonicalize
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                print(f"✅ Valid | Atoms: {mol.GetNumHeavyAtoms()} | Bonds: {mol.GetNumBonds()}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                self.current_mol = None
        
        # Update previews
        self._show_2d_structure()
        self._show_3d_structure()
        self._show_properties()
    
    def _validate_smiles(self, b=None):
        """Manual validation button."""
        smiles = self.smiles_input.value.strip()
        if smiles:
            self._validate_and_preview(smiles)
            self._add_to_history(smiles)
    
    def _show_2d_structure(self):
        """Display 2D structure with optional highlighting."""
        with self.structure_output:
            clear_output(wait=True)
            
            if not RDKIT_AVAILABLE:
                print("RDKit not available")
                return
            
            if self.current_mol is None:
                display(HTML("<div style='color:#888; padding:40px; text-align:center;'>Enter a SMILES to see structure</div>"))
                return
            
            try:
                mol = self.current_mol
                highlight_atoms = []
                highlight_bonds = []
                
                # Check for toxicophore highlighting
                highlight_type = self.highlight_select.value
                if highlight_type != 'None' and highlight_type in self.TOXICOPHORE_PATTERNS:
                    pattern = Chem.MolFromSmarts(self.TOXICOPHORE_PATTERNS[highlight_type])
                    if pattern:
                        matches = mol.GetSubstructMatches(pattern)
                        for match in matches:
                            highlight_atoms.extend(match)
                
                # Draw molecule
                drawer = rdMolDraw2D.MolDraw2DSVG(340, 260)
                drawer.drawOptions().addAtomIndices = False
                drawer.drawOptions().addStereoAnnotation = True
                
                if highlight_atoms:
                    # Highlight matched atoms in red
                    colors = {i: (1, 0.4, 0.4) for i in highlight_atoms}
                    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, 
                                       highlightAtomColors=colors)
                else:
                    drawer.DrawMolecule(mol)
                
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
                
                display(SVG(svg))
                
                if highlight_atoms:
                    display(HTML(f"<div style='color:#ef4444; font-size:12px;'>⚠️ {highlight_type} detected!</div>"))
                    
            except Exception as e:
                print(f"⚠️ 2D rendering failed: {e}")
    
    def _show_3d_structure(self):
        """Display 3D structure."""
        with self.structure_3d_output:
            clear_output(wait=True)
            
            if not RDKIT_AVAILABLE or not PY3DMOL_AVAILABLE:
                print("RDKit or py3Dmol not available")
                return
            
            if self.current_mol is None:
                display(HTML("<div style='color:#888; padding:40px; text-align:center;'>No molecule</div>"))
                return
            
            try:
                show_h = self.show_h_checkbox.value
                
                # Use conformer if available, else generate
                if self.conformers:
                    mol_3d = self.conformers[self.current_conformer_idx]
                    self.conformer_label.value = f"<b>Conformer:</b> {self.current_conformer_idx + 1}/{len(self.conformers)}"
                else:
                    mol_3d = Chem.AddHs(self.current_mol)
                    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_3d)
                    except:
                        AllChem.UFFOptimizeMolecule(mol_3d)
                    self.conformer_label.value = "<b>Conformer:</b> 1/1"
                
                if not show_h:
                    mol_3d = Chem.RemoveHs(mol_3d)
                
                mol_block = Chem.MolToMolBlock(mol_3d)
                
                viewer = py3Dmol.view(width=340, height=260)
                viewer.addModel(mol_block, 'mol')
                viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                viewer.setBackgroundColor('#f8f9fa')
                viewer.zoomTo()
                viewer.show()
                
            except Exception as e:
                print(f"⚠️ 3D generation failed: {e}")
    
    def _generate_conformers(self, b=None):
        """Generate multiple conformers."""
        if not RDKIT_AVAILABLE or self.current_mol is None:
            return
        
        with self.status_output:
            clear_output(wait=True)
            print("⏳ Generating conformers...")
        
        try:
            mol = Chem.AddHs(self.current_mol)
            
            # Generate conformers
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 0
            
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
            
            # Optimize each
            for conf_id in conf_ids:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                except:
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
            
            # Store conformers
            self.conformers = []
            for conf_id in conf_ids:
                conf_mol = Chem.Mol(mol)
                conf_mol.RemoveAllConformers()
                conf_mol.AddConformer(mol.GetConformer(conf_id), assignId=True)
                self.conformers.append(conf_mol)
            
            self.current_conformer_idx = 0
            
            with self.status_output:
                clear_output(wait=True)
                print(f"✅ Generated {len(self.conformers)} conformers")
            
            self._show_3d_structure()
            
        except Exception as e:
            with self.status_output:
                clear_output(wait=True)
                print(f"❌ Conformer generation failed: {e}")
    
    def _prev_conformer(self, b=None):
        """Show previous conformer."""
        if self.conformers and self.current_conformer_idx > 0:
            self.current_conformer_idx -= 1
            self._show_3d_structure()
    
    def _next_conformer(self, b=None):
        """Show next conformer."""
        if self.conformers and self.current_conformer_idx < len(self.conformers) - 1:
            self.current_conformer_idx += 1
            self._show_3d_structure()
    
    def _show_properties(self):
        """Display molecular properties with ADMET indicators."""
        with self.props_output:
            clear_output(wait=True)
            
            if not RDKIT_AVAILABLE or self.current_mol is None:
                return
            
            mol = self.current_mol
            
            props = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
            }
            
            qed_val = 0.5
            if QED_AVAILABLE:
                try:
                    qed_val = qed(mol)
                    props['QED'] = qed_val
                except:
                    props['QED'] = 'N/A'
            
            # Build HTML with traffic lights
            html = "<div style='font-size:12px; line-height:1.6;'>"
            html += "<b>📊 Properties</b><hr style='margin:4px 0;'>"
            html += f"MW: <b>{props['MW']:.1f}</b> Da<br>"
            html += f"LogP: <b>{props['LogP']:.2f}</b><br>"
            html += f"TPSA: <b>{props['TPSA']:.1f}</b> Ų<br>"
            html += f"HBD/HBA: <b>{props['HBD']}/{props['HBA']}</b><br>"
            html += f"RotBonds: <b>{props['RotBonds']}</b><br>"
            if isinstance(props.get('QED'), float):
                html += f"QED: <b>{props['QED']:.3f}</b><br>"
            
            html += "<hr style='margin:6px 0;'>"
            
            # Lipinski
            violations = 0
            if props['MW'] > 500: violations += 1
            if props['LogP'] > 5: violations += 1
            if props['HBD'] > 5: violations += 1
            if props['HBA'] > 10: violations += 1
            
            if violations == 0:
                html += "Lipinski: <span style='color:#10b981;'>✅ Pass</span><br>"
            else:
                html += f"Lipinski: <span style='color:#ef4444;'>⚠️ {violations}v</span><br>"
            
            # ADMET Traffic Lights
            html += "<hr style='margin:6px 0;'><b>ADMET Indicators</b><br>"
            
            # BBB penetration (based on Egan et al. + Clark rules)
            # Good BBB: TPSA < 90, LogP 1-3, MW < 450, HBD < 3
            bbb_score = 0
            if props['TPSA'] < 90: bbb_score += 1
            if 1 <= props['LogP'] <= 3: bbb_score += 1  
            if props['MW'] < 450: bbb_score += 1
            if props['HBD'] <= 2: bbb_score += 1
            
            if bbb_score >= 4:
                bbb = "🟢 High"
            elif bbb_score >= 2:
                bbb = "🟡 Moderate"
            else:
                bbb = "🔴 Low"
            html += f"BBB: {bbb}<br>"
            
            # hERG risk (Aronov model - lipophilic bases are high risk)
            # Risk factors: LogP > 3, basic N, MW > 400, low TPSA
            n_basic = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7;+,H1,H2]')))
            herg_risk = 0
            if props['LogP'] > 3: herg_risk += 1
            if props['LogP'] > 4: herg_risk += 1
            if n_basic > 0: herg_risk += 1
            if n_basic > 1 and props['LogP'] > 3: herg_risk += 1
            if props['MW'] > 400 and props['TPSA'] < 75: herg_risk += 1
            
            if herg_risk >= 3:
                herg = "🔴 High Risk"
            elif herg_risk >= 1:
                herg = "🟡 Moderate"
            else:
                herg = "🟢 Low"
            html += f"hERG: {herg}<br>"
            
            # Solubility (GSK 4/400 rule + aromatic rings)
            # Good: LogP < 4, MW < 400, aromatic rings < 4
            arom_rings = Descriptors.NumAromaticRings(mol)
            sol_score = 0
            if props['LogP'] < 3: sol_score += 2
            elif props['LogP'] < 4: sol_score += 1
            if props['MW'] < 400: sol_score += 1
            if arom_rings <= 3: sol_score += 1
            if props['TPSA'] > 50: sol_score += 1  # Polar groups help
            
            if sol_score >= 4:
                sol = "🟢 Good"
            elif sol_score >= 2:
                sol = "🟡 Moderate"
            else:
                sol = "🔴 Poor"
            html += f"Solubility: {sol}<br>"
            
            # CYP Inhibition risk (simple heuristic)
            # High risk: lipophilic, large aromatic systems
            cyp_risk = 0
            if props['LogP'] > 3: cyp_risk += 1
            if arom_rings > 2: cyp_risk += 1
            if props['MW'] > 350: cyp_risk += 1
            
            if cyp_risk >= 3:
                cyp = "🔴 High"
            elif cyp_risk >= 1:
                cyp = "🟡 Moderate"
            else:
                cyp = "🟢 Low"
            html += f"CYP Inhib: {cyp}"
            
            html += "</div>"
            display(HTML(html))
    
    def _find_similar(self, b=None):
        """Find similar molecules in loaded library."""
        with self.similarity_output:
            clear_output(wait=True)
            
            if not RDKIT_AVAILABLE or self.current_mol is None:
                print("No molecule to compare")
                return
            
            if not self.molecules_list:
                print("No molecules loaded for comparison")
                return
            
            try:
                # Generate fingerprint for current molecule
                fp1 = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2, nBits=2048)
                
                similarities = []
                for mol in self.molecules_list:
                    if hasattr(mol, 'mol') and mol.mol is not None:
                        rdkit_mol = mol.mol
                    elif hasattr(mol, 'smiles'):
                        rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                    else:
                        continue
                    
                    if rdkit_mol is None:
                        continue
                    
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=2048)
                    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                    name = mol.name if hasattr(mol, 'name') else str(mol)
                    similarities.append((name, sim))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Display top 5
                html = "<div style='font-size:12px;'><b>🔍 Similar Molecules</b><br>"
                for name, sim in similarities[:5]:
                    color = '#10b981' if sim > 0.7 else '#f59e0b' if sim > 0.4 else '#888'
                    html += f"<span style='color:{color};'>●</span> {name}: <b>{sim*100:.0f}%</b><br>"
                html += "</div>"
                display(HTML(html))
                
            except Exception as e:
                print(f"Error: {e}")
    
    def _download_sdf(self, b=None):
        """Create download link for SDF file."""
        if not RDKIT_AVAILABLE or self.current_mol is None:
            return
        
        with self.download_output:
            clear_output(wait=True)
            
            try:
                mol_3d = Chem.AddHs(self.current_mol)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
                
                sdf_str = Chem.MolToMolBlock(mol_3d)
                sdf_b64 = base64.b64encode(sdf_str.encode()).decode()
                
                name = self.name_input.value.replace(' ', '_')
                html = f"""
                <a download="{name}.sdf" 
                   href="data:chemical/x-mdl-molfile;base64,{sdf_b64}" 
                   style="display:inline-block; padding:6px 12px; background:#667eea; 
                          color:white; border-radius:4px; text-decoration:none; font-size:12px;">
                    📥 Download {name}.sdf
                </a>
                """
                display(HTML(html))
            except Exception as e:
                print(f"Error: {e}")
    
    def _download_png(self, b=None):
        """Create download link for PNG image."""
        if not RDKIT_AVAILABLE or self.current_mol is None:
            return
        
        with self.download_output:
            clear_output(wait=True)
            
            try:
                img = Draw.MolToImage(self.current_mol, size=(400, 300))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                name = self.name_input.value.replace(' ', '_')
                html = f"""
                <a download="{name}.png" 
                   href="data:image/png;base64,{img_b64}" 
                   style="display:inline-block; padding:6px 12px; background:#10b981; 
                          color:white; border-radius:4px; text-decoration:none; font-size:12px;">
                    📥 Download {name}.png
                </a>
                """
                display(HTML(html))
            except Exception as e:
                print(f"Error: {e}")
    
    def _load_example(self, change):
        """Load a molecule from dropdown."""
        name = change['new']
        
        # Check loaded molecules first
        if name in self.loaded_molecules_dict:
            self.smiles_input.value = self.loaded_molecules_dict[name]
            self.name_input.value = name
        # Fallback to examples
        elif name in self.EXAMPLE_MOLECULES:
            self.smiles_input.value = self.EXAMPLE_MOLECULES[name]
            self.name_input.value = name
    
    def _add_functional_group(self, fg_name):
        """Add a functional group hint."""
        fg_smiles = self.FUNCTIONAL_GROUPS.get(fg_name, '')
        
        with self.status_output:
            clear_output(wait=True)
            print(f"💡 Add '{fg_smiles}' to your SMILES")
            print(f"   E.g.: CCO + Cl → CCCl")
    
    def _add_to_molecules(self, b=None):
        """Add current molecule to the working list."""
        with self.status_output:
            clear_output(wait=True)
            
            if self.current_mol is None:
                print("❌ No valid molecule to add")
                return
            
            name = self.name_input.value.strip() or "Edited Molecule"
            smiles = self.current_smiles
            
            if self.drug_molecule_class is not None:
                try:
                    new_mol = self.drug_molecule_class(
                        name=name,
                        cas='User-Edited',
                        smiles=smiles
                    )
                    
                    if new_mol.mol is not None:
                        self.molecules_list.append(new_mol)
                        self.loaded_molecules_dict[name] = smiles
                        self.example_select.options = ['-- Select Molecule --'] + list(self.loaded_molecules_dict.keys())
                        print(f"✅ Added '{name}' ({len(self.molecules_list)} total)")
                    else:
                        print(f"❌ Failed to create DrugMolecule")
                        
                except Exception as e:
                    print(f"⚠️ Error: {e}")
            else:
                self.molecules_list.append({
                    'name': name,
                    'smiles': smiles,
                    'mol': self.current_mol
                })
                self.loaded_molecules_dict[name] = smiles
                self.example_select.options = ['-- Select Molecule --'] + list(self.loaded_molecules_dict.keys())
                print(f"✅ Added '{name}' ({len(self.molecules_list)} total)")
    
    def _clear_editor(self, b=None):
        """Clear the editor."""
        self.smiles_input.value = ''
        self.name_input.value = 'New Molecule'
        self.current_mol = None
        self.current_smiles = ''
        self.conformers = []
        
        with self.structure_output:
            clear_output()
        with self.structure_3d_output:
            clear_output()
        with self.props_output:
            clear_output()
        with self.status_output:
            clear_output()
            print("🗑️ Cleared")
        with self.similarity_output:
            clear_output()
        with self.download_output:
            clear_output()
    
    def load_molecules(self, molecules, drug_molecule_class=None):
        """
        Set the molecules list and update dropdown.
        """
        self.molecules_list = molecules
        if drug_molecule_class is not None:
            self.drug_molecule_class = drug_molecule_class
        
        if molecules and len(molecules) > 0:
            self.loaded_molecules_dict = {}
            for mol in molecules:
                if hasattr(mol, 'name') and hasattr(mol, 'smiles'):
                    self.loaded_molecules_dict[mol.name] = mol.smiles
            
            if self.loaded_molecules_dict:
                self.example_select.options = ['-- Select Molecule --'] + list(self.loaded_molecules_dict.keys())
    
    def display(self):
        """Display the molecular editor interface."""
        
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); 
                    padding: 14px 18px; border-radius: 12px; margin-bottom: 14px;'>
            <h3 style='color: white; margin: 0; font-size: 18px;'>✏️ Molecular Editor</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 12px;'>
                Draw, edit, and analyze molecules with ADMET predictions
            </p>
        </div>
        """)
        
        # Top row: dropdown + name + history + copy
        top_row = widgets.HBox([
            self.example_select,
            self.name_input,
            self.undo_btn,
            self.redo_btn,
            self.copy_btn,
            self.show_h_checkbox
        ], layout=widgets.Layout(gap='6px', margin='0 0 10px 0'))
        
        # SMILES input
        smiles_row = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<span style='font-size:12px;'><b>SMILES:</b></span>"),
                self.smiles_input
            ], layout=widgets.Layout(flex='1')),
            widgets.VBox([
                self.validate_btn,
                self.clear_btn
            ], layout=widgets.Layout(gap='4px'))
        ], layout=widgets.Layout(gap='8px', margin='0 0 8px 0'))
        
        self.status_output.layout = widgets.Layout(max_height='40px')
        
        # Functional groups
        fg_row = widgets.HBox(self.fg_buttons, layout=widgets.Layout(gap='4px', margin='0 0 10px 0', flex_wrap='wrap'))
        
        # Highlight dropdown
        highlight_row = widgets.HBox([
            self.highlight_select,
            self.similarity_btn
        ], layout=widgets.Layout(gap='8px', margin='0 0 10px 0'))
        
        # Structure panels
        panel_2d = widgets.VBox([
            widgets.HTML("<b style='font-size:12px;'>2D Structure</b>"),
            self.structure_output
        ], layout=widgets.Layout(
            width='360px', padding='8px',
            border='1px solid #e0e0e0', border_radius='8px'
        ))
        
        # Conformer controls
        conf_controls = widgets.HBox([
            self.prev_conf_btn,
            self.conformer_label,
            self.next_conf_btn,
            self.gen_conf_btn
        ], layout=widgets.Layout(gap='4px', align_items='center'))
        
        panel_3d = widgets.VBox([
            widgets.HTML("<b style='font-size:12px;'>3D Structure</b>"),
            self.structure_3d_output,
            conf_controls
        ], layout=widgets.Layout(
            width='360px', padding='8px',
            border='1px solid #e0e0e0', border_radius='8px'
        ))
        
        panel_props = widgets.VBox([
            self.props_output,
            widgets.HTML("<hr style='margin:8px 0;'>"),
            self.similarity_output
        ], layout=widgets.Layout(
            width='180px', padding='8px',
            border='1px solid #e0e0e0', border_radius='8px'
        ))
        
        views_row = widgets.HBox([panel_2d, panel_3d, panel_props], layout=widgets.Layout(gap='10px'))
        
        # Actions row
        actions_row = widgets.HBox([
            self.add_btn,
            self.download_sdf_btn,
            self.download_png_btn,
            self.download_output
        ], layout=widgets.Layout(gap='8px', margin='12px 0 0 0', align_items='center'))
        
        # Assemble
        interface = widgets.VBox([
            header,
            top_row,
            smiles_row,
            self.status_output,
            fg_row,
            highlight_row,
            views_row,
            actions_row
        ], layout=widgets.Layout(max_width='950px'))
        
        return interface
    
    def get_current_molecule(self):
        """Get the current molecule as RDKit mol object."""
        return self.current_mol
    
    def get_current_smiles(self):
        """Get the current SMILES string."""
        return self.current_smiles
