"""
Interactive molecule explorer for browsing and analyzing molecular datasets.
"""

import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from rdkit.Chem import rdMolDescriptors

from ..visualization.molecular_viewer import MolecularViewer3D


class MoleculeExplorer:
    """Interactive widget for exploring and analyzing molecules."""
    
    def __init__(self):
        """Initialize molecule explorer."""
        self.molecules = []
        self.current_molecule = None
        self.viewer_3d = MolecularViewer3D()
        
        # Create output widgets
        self.output = widgets.Output()
        self.viewer_output = widgets.Output()
        
        # Create controls
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all widgets for the interface."""
        # Molecule selector
        self.molecule_dropdown = widgets.Dropdown(
            description='Molecule:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        self.molecule_dropdown.observe(self._on_molecule_change, names='value')
        
        # Display options
        self.show_3d = widgets.Checkbox(
            value=True,
            description='Show 3D structure',
            layout=widgets.Layout(width='180px')
        )
        
        self.show_properties = widgets.Checkbox(
            value=True,
            description='Show properties',
            layout=widgets.Layout(width='150px')
        )
        
        self.show_admet = widgets.Checkbox(
            value=True,
            description='Show ADMET',
            layout=widgets.Layout(width='130px')
        )
        
        self.show_radar = widgets.Checkbox(
            value=False,
            description='Show radar chart',
            layout=widgets.Layout(width='150px')
        )
        
        # Export button
        self.export_button = widgets.Button(
            description='📁 Export Data',
            button_style='info',
            layout=widgets.Layout(width='140px')
        )
        self.export_button.on_click(self._export_data)
        
        # Bind events
        for checkbox in [self.show_3d, self.show_properties, self.show_admet, self.show_radar]:
            checkbox.observe(self._update_display, names='value')
    
    def load_molecules(self, molecules):
        """Load molecules into explorer."""
        self.molecules = molecules
        
        # Update dropdown options
        molecule_names = [mol.name for mol in molecules]
        self.molecule_dropdown.options = molecule_names
        
        if molecule_names:
            self.molecule_dropdown.value = molecule_names[0]
            self.current_molecule = molecules[0]
            self._update_display()
    
    def _on_molecule_change(self, change):
        """Handle molecule selection change."""
        if change['type'] == 'change' and change['name'] == 'value':
            molecule_name = change['new']
            self.current_molecule = next(
                (mol for mol in self.molecules if mol.name == molecule_name), None
            )
            if self.current_molecule:
                self._update_display()
    
    def _update_display(self, change=None):
        """Update the display based on current selections."""
        if not self.current_molecule:
            return
        
        with self.output:
            clear_output(wait=True)
            
            mol = self.current_molecule
            
            # Display basic info
            print(f"🧬 {mol.name}")
            print("=" * 50)
            print(f"SMILES: {mol.smiles}")
            print(f"Atoms: {mol.mol.GetNumAtoms()}")
            print()
            
            # Show properties if requested
            if self.show_properties.value:
                properties = mol.get_properties()
                if properties:
                    print("📊 Molecular Properties:")
                    print("-" * 25)
                    for key, value in properties.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.3f}")
                        else:
                            print(f"  {key}: {value}")
                    print()
            
            # Show ADMET if requested
            if self.show_admet.value:
                self._show_admet_analysis(mol)
            
        # Show 3D structure if requested
        if self.show_3d.value:
            with self.viewer_output:
                clear_output(wait=True)
                try:
                    fig = self.viewer_3d.plot_molecule(mol)
                    fig.show()
                except Exception as e:
                    print(f"3D visualization error: {e}")
        else:
            with self.viewer_output:
                clear_output(wait=True)
        
        # Show radar chart if requested
        if self.show_radar.value:
            self._show_radar_chart(mol)
    
    def _show_admet_analysis(self, mol):
        """Show ADMET analysis for molecule."""
        print("💊 ADMET Analysis:")
        print("-" * 18)
        
        # Calculate basic ADMET properties
        mw = rdMolDescriptors.CalcExactMolWt(mol.mol)
        logp = rdMolDescriptors.CalcCrippenDescriptors(mol.mol)[0]
        tpsa = rdMolDescriptors.CalcTPSA(mol.mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol.mol)
        hba = rdMolDescriptors.CalcNumHBA(mol.mol)
        
        # Lipinski rule of five
        lipinski_violations = 0
        if mw > 500:
            lipinski_violations += 1
        if logp > 5:
            lipinski_violations += 1
        if hbd > 5:
            lipinski_violations += 1
        if hba > 10:
            lipinski_violations += 1
        
        print(f"  Molecular Weight: {mw:.2f} Da {'✅' if mw <= 500 else '❌'}")
        print(f"  LogP: {logp:.2f} {'✅' if logp <= 5 else '❌'}")
        print(f"  TPSA: {tpsa:.2f} Ų")
        print(f"  H-bond Donors: {hbd} {'✅' if hbd <= 5 else '❌'}")
        print(f"  H-bond Acceptors: {hba} {'✅' if hba <= 10 else '❌'}")
        print(f"  Lipinski Violations: {lipinski_violations} {'✅' if lipinski_violations <= 1 else '❌'}")
        
        # Drug-likeness assessment
        if lipinski_violations <= 1:
            print("  🎯 Drug-like: YES")
        else:
            print("  ⚠️ Drug-like: NO")
        
        print()
    
    def _show_radar_chart(self, mol):
        """Show radar chart of molecular properties."""
        with self.output:
            # Calculate normalized properties for radar chart
            mw = rdMolDescriptors.CalcExactMolWt(mol.mol)
            logp = rdMolDescriptors.CalcCrippenDescriptors(mol.mol)[0]
            tpsa = rdMolDescriptors.CalcTPSA(mol.mol)
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol.mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol.mol)
            
            # Normalize to 0-1 scale
            properties = {
                'MW (0-800)': min(mw / 800, 1.0),
                'LogP (-3-7)': (logp + 3) / 10,
                'TPSA (0-200)': min(tpsa / 200, 1.0),
                'RotBonds (0-15)': min(rotbonds / 15, 1.0),
                'ArRings (0-5)': min(aromatic_rings / 5, 1.0)
            }
            
            # Create radar chart
            categories = list(properties.keys())
            values = list(properties.values())
            
            # Add first value at end to close the plot
            categories += [categories[0]]
            values += [values[0]]
            
            # Angles for each category
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
            ax.fill(angles, values, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories[:-1])
            ax.set_ylim(0, 1)
            ax.set_title(f'Property Profile: {mol.name}', y=1.08)
            ax.grid(True)
            
            plt.tight_layout()
            plt.show()
    
    def _export_data(self, button):
        """Export current molecule data."""
        if not self.current_molecule:
            with self.output:
                print("❌ No molecule selected for export")
            return
        
        with self.output:
            print("📁 Exporting molecule data...")
            
            mol = self.current_molecule
            
            # Create export data
            export_data = {
                'name': mol.name,
                'smiles': mol.smiles,
                'properties': mol.get_properties()
            }
            
            # Save to file
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"molecule_export_{mol.name}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported to: {filename}")
    
    def display(self):
        """Display the molecule explorer interface."""
        # Selector area
        selector_area = widgets.HBox([
            self.molecule_dropdown
        ])
        
        # Options area
        options_area = widgets.VBox([
            widgets.HTML("<h3>📊 Display Options</h3>"),
            self.show_3d,
            self.show_properties,
            self.show_admet,
            self.show_radar
        ])
        
        # Control buttons
        controls_area = widgets.HBox([
            self.export_button
        ])
        
        # Layout
        left_panel = widgets.VBox([
            options_area,
            controls_area
        ], layout=widgets.Layout(width='300px'))
        
        right_panel = widgets.VBox([
            self.output,
            self.viewer_output
        ])
        
        main_area = widgets.HBox([left_panel, right_panel])
        
        interface = widgets.VBox([
            widgets.HTML("<h2>🔬 Molecule Explorer</h2>"),
            selector_area,
            main_area
        ])
        
        return interface