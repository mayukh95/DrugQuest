"""
Interactive Molecule Comparator - Side-by-side comparison with 3D views and radar charts.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class MoleculeComparator:
    """Compare two molecules side-by-side with 3D views and property comparison."""
    
    def __init__(self, molecules_list=None):
        """
        Initialize the molecule comparator.
        
        Parameters:
        -----------
        molecules_list : list, optional
            List of DrugMolecule objects
        """
        self.molecules = {}
        self.mol_names = []
        
        # Setup widgets FIRST
        self._setup_widgets()
        
        # Then load molecules if provided
        if molecules_list:
            self.load_molecules(molecules_list)
    
    def _setup_widgets(self):
        """Create comparison interface widgets."""
        # Molecule selection dropdowns
        self.mol1_select = widgets.Dropdown(
            options=self.mol_names,
            value=self.mol_names[0] if self.mol_names else None,
            description='Molecule 1:',
            style={'description_width': '90px'},
            layout=widgets.Layout(width='250px')
        )
        
        self.mol2_select = widgets.Dropdown(
            options=self.mol_names,
            value=self.mol_names[1] if len(self.mol_names) > 1 else (self.mol_names[0] if self.mol_names else None),
            description='Molecule 2:',
            style={'description_width': '90px'},
            layout=widgets.Layout(width='250px')
        )
        
        # Compare button
        self.compare_btn = widgets.Button(
            description='🔍 Compare',
            button_style='primary',
            layout=widgets.Layout(width='120px', height='36px')
        )
        self.compare_btn.on_click(self._compare_molecules)
        
        # Output areas
        self.mol1_output = widgets.Output()
        self.mol2_output = widgets.Output()
        self.comparison_output = widgets.Output()
        self.radar_output = widgets.Output()
    
    def load_molecules(self, molecules_list):
        """
        Load molecules into comparator.
        
        Parameters:
        -----------
        molecules_list : list
            List of DrugMolecule objects
        """
        self.molecules = {m.name: m for m in molecules_list}
        self.mol_names = list(self.molecules.keys())
        
        # Update dropdowns if they exist
        if hasattr(self, 'mol1_select'):
            self.mol1_select.options = self.mol_names
            self.mol2_select.options = self.mol_names
            
            if self.mol_names:
                self.mol1_select.value = self.mol_names[0]
                self.mol2_select.value = self.mol_names[1] if len(self.mol_names) > 1 else self.mol_names[0]
    
    def _get_mol_xyz(self, mol):
        """Generate XYZ string from molecule."""
        xyz_str = f"{len(mol.elements)}\n{mol.name}\n"
        coords = mol.optimized_coords if hasattr(mol, 'optimized_coords') and mol.optimized_coords is not None else mol.initial_coords
        for el, coord in zip(mol.elements, coords):
            xyz_str += f"{el} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
        return xyz_str
    
    def _show_molecule(self, mol, output_widget):
        """Display molecule in 3D viewer."""
        with output_widget:
            clear_output(wait=True)
            
            if not mol.properties:
                mol.calculate_descriptors()
            
            # Info header
            print(f"🔬 {mol.name}")
            smiles_display = mol.smiles[:35] + '...' if len(mol.smiles) > 35 else mol.smiles
            print(f"   Formula: {smiles_display}")
            print(f"   Atoms: {len(mol.elements)}")
            
            # 3D viewer
            if PY3DMOL_AVAILABLE and hasattr(mol, 'elements') and hasattr(mol, 'initial_coords'):
                view = py3Dmol.view(width=350, height=280)
                view.addModel(self._get_mol_xyz(mol), 'xyz')
                view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                view.setBackgroundColor('#f8f9fa')
                view.zoomTo()
                view.show()
    
    def _compare_molecules(self, b=None):
        """Compare two selected molecules."""
        mol1 = self.molecules.get(self.mol1_select.value)
        mol2 = self.molecules.get(self.mol2_select.value)
        
        if not mol1 or not mol2:
            return
        
        # Ensure properties are calculated
        if not mol1.properties:
            mol1.calculate_descriptors()
        if not mol2.properties:
            mol2.calculate_descriptors()
        
        # Show 3D structures
        self._show_molecule(mol1, self.mol1_output)
        self._show_molecule(mol2, self.mol2_output)
        
        # Property comparison table
        with self.comparison_output:
            clear_output(wait=True)
            
            props = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'QED']
            
            print(f"\n{'Property':<15} {'Mol 1':<12} {'Mol 2':<12} {'Δ':>10}")
            print("─" * 52)
            
            for prop in props:
                v1 = mol1.properties.get(prop, 0)
                v2 = mol2.properties.get(prop, 0)
                diff = v2 - v1 if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else 'N/A'
                
                if isinstance(v1, float):
                    print(f"{prop:<15} {v1:<12.2f} {v2:<12.2f} {diff:>+10.2f}")
                else:
                    print(f"{prop:<15} {v1:<12} {v2:<12} {str(diff):>10}")
        
        # Radar chart
        self._plot_radar_chart(mol1, mol2)
    
    def _plot_radar_chart(self, mol1, mol2):
        """Create radar chart comparing molecular properties."""
        with self.radar_output:
            clear_output(wait=True)
            
            if not PLOTLY_AVAILABLE:
                print("Plotly not available for radar chart")
                return
            
            # Normalize properties for radar chart
            props = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'QED']
            
            # Reference values for normalization (drug-like ranges)
            ref_max = {'MW': 500, 'LogP': 5, 'TPSA': 140, 'HBD': 5, 'HBA': 10, 'RotBonds': 10, 'QED': 1}
            
            # Ideal drug values (optimal for drug-likeness)
            # MW: ~350 (sweet spot), LogP: 2-3, TPSA: 60-80, HBD: 1-2, HBA: 4-5, RotBonds: 3-5, QED: 0.7+
            ideal_values = {'MW': 350, 'LogP': 2.5, 'TPSA': 70, 'HBD': 2, 'HBA': 5, 'RotBonds': 4, 'QED': 0.8}
            
            values1 = []
            values2 = []
            values_ideal = []
            
            for prop in props:
                v1 = mol1.properties.get(prop, 0)
                v2 = mol2.properties.get(prop, 0)
                v_ideal = ideal_values.get(prop, 0)
                max_val = ref_max.get(prop, 1)
                
                # Normalize to 0-1 scale
                values1.append(min(v1 / max_val, 1.0) if max_val > 0 else 0)
                values2.append(min(v2 / max_val, 1.0) if max_val > 0 else 0)
                values_ideal.append(min(v_ideal / max_val, 1.0) if max_val > 0 else 0)
            
            # Close the radar
            values1.append(values1[0])
            values2.append(values2[0])
            values_ideal.append(values_ideal[0])
            props_closed = props + [props[0]]
            
            fig = go.Figure()
            
            # Add Ideal Drug reference first (so it appears behind)
            fig.add_trace(go.Scatterpolar(
                r=values_ideal,
                theta=props_closed,
                fill='toself',
                name='💊 Ideal Drug',
                line_color='#10b981',
                line_width=2,
                line_dash='dash',
                fillcolor='rgba(16, 185, 129, 0.15)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=values1,
                theta=props_closed,
                fill='toself',
                name=mol1.name,
                line_color='#667eea',
                fillcolor='rgba(102, 126, 234, 0.3)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=values2,
                theta=props_closed,
                fill='toself',
                name=mol2.name,
                line_color='#f56565',
                fillcolor='rgba(245, 101, 101, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                title=dict(
                    text='Property Comparison (Normalized)',
                    font=dict(size=14)
                ),
                width=450,
                height=400,
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            fig.show()
    
    def display(self):
        """Display the comparison interface."""
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 16px; border-radius: 12px; margin-bottom: 16px;'>
            <h3 style='color: white; margin: 0;'>🔬 Molecule Comparison Tool</h3>
            <p style='color: rgba(255,255,255,0.85); margin: 8px 0 0 0; font-size: 13px;'>
                Compare molecular properties and structures side-by-side
            </p>
        </div>
        """)
        
        selection_row = widgets.HBox([
            self.mol1_select, 
            self.mol2_select, 
            self.compare_btn
        ], layout=widgets.Layout(gap='12px', margin='0 0 16px 0'))
        
        structures_row = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<b style='color: #667eea;'>Molecule 1</b>"),
                self.mol1_output
            ], layout=widgets.Layout(
                width='380px', padding='12px', 
                border='2px solid #667eea', border_radius='10px'
            )),
            widgets.VBox([
                widgets.HTML("<b style='color: #f56565;'>Molecule 2</b>"),
                self.mol2_output
            ], layout=widgets.Layout(
                width='380px', padding='12px',
                border='2px solid #f56565', border_radius='10px'
            ))
        ], layout=widgets.Layout(gap='20px', margin='0 0 16px 0'))
        
        analysis_row = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<b>📊 Property Comparison</b>"),
                self.comparison_output
            ], layout=widgets.Layout(
                width='380px', padding='12px',
                border='1px solid #e0e0e0', border_radius='10px'
            )),
            widgets.VBox([
                widgets.HTML("<b>📈 Radar Chart</b>"),
                self.radar_output
            ], layout=widgets.Layout(
                width='480px', padding='12px',
                border='1px solid #e0e0e0', border_radius='10px'
            ))
        ], layout=widgets.Layout(gap='20px'))
        
        interface = widgets.VBox([
            header,
            selection_row,
            structures_row,
            analysis_row
        ], layout=widgets.Layout(max_width='900px'))
        
        # Show initial comparison if molecules are loaded
        if self.mol_names and len(self.mol_names) >= 1:
            self._compare_molecules()
        
        return interface