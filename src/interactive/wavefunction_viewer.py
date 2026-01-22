"""
Interactive Wavefunction Visualization - Load and visualize molecular orbitals.
"""

import numpy as np
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class WavefunctionVisualizer:
    """
    Interactive Wavefunction Visualization for molecular orbitals.
    
    Features:
    - Scan directories for saved wavefunction files
    - Load .npz wavefunction data
    - MO energy diagram visualization
    - MO coefficient distribution plots
    - Cube file visualization (HOMO/LUMO)
    - 3D isosurface plotting with Plotly
    """
    
    def __init__(self, base_dir=None, mo_visualizer=None):
        """
        Initialize the wavefunction visualizer.
        
        Parameters:
        -----------
        base_dir : str or Path, optional
            Base directory for scanning wavefunction files
        mo_visualizer : object, optional
            OrbitalVisualizer instance from src.analysis for actual visualization
        """
        self.base_dir = Path(base_dir) if base_dir else Path('./optimized_molecules')
        self.mo_visualizer = mo_visualizer
        self.wavefunction_data = None
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Create viewer interface widgets."""
        # Output widget
        self.wfn_output = widgets.Output()
        
        # Directory input
        self.wfn_base_dir_input = widgets.Text(
            value=str(self.base_dir),
            placeholder='Enter base directory path (e.g., ./optimized_molecules)',
            description='Base Directory:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        # Scan button
        self.scan_wfn_button = widgets.Button(
            description='🔍 Scan for Wavefunctions',
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )
        self.scan_wfn_button.on_click(self._scan_wavefunctions)
        
        # Molecule selector
        self.wfn_mol_select = widgets.Dropdown(
            options=[],
            description='Molecule:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Load button
        self.load_wfn_button = widgets.Button(
            description='📂 Load Wavefunction',
            button_style='info',
            layout=widgets.Layout(width='200px')
        )
        self.load_wfn_button.on_click(self._load_and_visualize)
        
        # Visualization options
        self.n_orbitals_slider = widgets.IntSlider(
            value=10, min=5, max=20, step=1,
            description='Orbitals to show:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.isovalue_slider = widgets.FloatSlider(
            value=0.02, min=0.01, max=0.1, step=0.005,
            description='Isovalue:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.3f'
        )
    
    def set_mo_visualizer(self, mo_visualizer):
        """
        Set the molecular orbital visualizer.
        
        Parameters:
        -----------
        mo_visualizer : OrbitalVisualizer
            Instance of OrbitalVisualizer from src.analysis
        """
        self.mo_visualizer = mo_visualizer
    
    def _scan_wavefunctions(self, b=None):
        """Scan the base directory for available wavefunction files."""
        with self.wfn_output:
            clear_output(wait=True)
            base_dir = Path(self.wfn_base_dir_input.value)
            self.base_dir = base_dir # Update base_dir
            
            if not base_dir.is_dir():
                print(f"❌ Base directory not found: {base_dir}")
                self.wfn_mol_select.options = []
                return
            
            print(f"Scanning for wavefunctions in: {base_dir}")
            available_molecules = []
            
            # Recursively find all directories ending in '_optimized'
            for mol_dir in sorted(base_dir.rglob('*_optimized')):
                if mol_dir.is_dir():
                    wfn_dir = mol_dir / 'wavefunctions'
                    if wfn_dir.exists():
                        # Look for .npz file
                        npz_files = list(wfn_dir.glob('*_optimized_wavefunction.npz'))
                        if npz_files:
                            molecule_name = mol_dir.name.replace('_optimized', '')
                            available_molecules.append(molecule_name)
            
            if available_molecules:
                self.wfn_mol_select.options = available_molecules
                print(f"✅ Found {len(available_molecules)} molecules with wavefunctions:")
                for mol in available_molecules:
                    print(f"   • {mol}")
            else:
                print(f"⚠️ No wavefunction files found in {base_dir}")
                print("   Expected structure: base_dir/molecule_optimized/wavefunctions/molecule_optimized_wavefunction.npz")
                self.wfn_mol_select.options = []
    
    def load_wavefunction(self, wfn_file):
        """
        Load wavefunction data from .npz file.
        
        Parameters:
        -----------
        wfn_file : str or Path
            Path to wavefunction .npz file
        
        Returns:
        --------
        dict : Wavefunction data or None if failed
        """
        wfn_file = Path(wfn_file)
        
        if not wfn_file.exists():
            print(f"❌ File not found: {wfn_file}")
            return None
        
        try:
            data = np.load(wfn_file, allow_pickle=True)
            self.wavefunction_data = dict(data)
            print(f"✅ Loaded wavefunction from: {wfn_file.name}")
            
            # Print available keys
            print(f"   Available data: {list(data.keys())}")
            
            return self.wavefunction_data
        except Exception as e:
            print(f"❌ Error loading wavefunction: {e}")
            return None
    
    def _load_and_visualize(self, b=None):
        """Load and visualize the selected wavefunction."""
        with self.wfn_output:
            clear_output(wait=True)
            
            if not self.wfn_mol_select.options:
                print("⚠️ No molecules available. Please scan for wavefunctions first.")
                return
            
            molecule_name = self.wfn_mol_select.value
            if not molecule_name:
                print("⚠️ Please select a molecule.")
                return
            
            base_dir = Path(self.wfn_base_dir_input.value)
            wfn_dir = base_dir / f"{molecule_name}_optimized" / "wavefunctions"
            wfn_file = wfn_dir / f"{molecule_name}_optimized_wavefunction.npz"
            
            if not wfn_file.exists():
                print(f"❌ Wavefunction file not found: {wfn_file}")
                return
            
            # Load wavefunction
            wfn_data = self.load_wavefunction(wfn_file)
            if wfn_data is None:
                return
            
            # Use mo_visualizer if available
            if self.mo_visualizer is not None:
                print(f"\nLoading wavefunction into visualizer...")
                success = self.mo_visualizer.load_wavefunction(wfn_file)
                
                if not success:
                    print("❌ Failed to load wavefunction into visualizer")
                    return
                
                # Plot MO energy diagram
                print("\n📊 Plotting MO Energy Diagram...")
                self.mo_visualizer.plot_mo_energy_diagram(
                    n_orbitals=self.n_orbitals_slider.value,
                    save_path=wfn_dir / f"{molecule_name}_mo_diagram.png"
                )
                
                # Plot MO coefficient distribution
                print("\n📊 Plotting MO Coefficients...")
                self.mo_visualizer.plot_mo_coefficients(
                    save_path=wfn_dir / f"{molecule_name}_mo_coefficients.png"
                )
                
                # Visualize all saved cube files
                print("\n📊 Visualizing Cube Files...")
                self.mo_visualizer.visualize_all_saved_mos(wfn_dir, molecule_name)
                
                # 3D visualization of HOMO
                homo_cube = wfn_dir / f"{molecule_name}_optimized_HOMO.cube"
                if homo_cube.exists():
                    print("\n🎨 Creating 3D HOMO visualization...")
                    cube_data = self.mo_visualizer.read_cube_file(homo_cube)
                    self.mo_visualizer.plot_cube_isosurface_3d(
                        cube_data, 
                        isovalue=self.isovalue_slider.value,
                        save_path=str(wfn_dir / f"{molecule_name}_HOMO_3d.html")
                    )
                
                # 3D visualization of LUMO
                lumo_cube = wfn_dir / f"{molecule_name}_optimized_LUMO.cube"
                if lumo_cube.exists():
                    print("\n🎨 Creating 3D LUMO visualization...")
                    cube_data = self.mo_visualizer.read_cube_file(lumo_cube)
                    self.mo_visualizer.plot_cube_isosurface_3d(
                        cube_data, 
                        isovalue=self.isovalue_slider.value,
                        save_path=str(wfn_dir / f"{molecule_name}_LUMO_3d.html")
                    )
                
                print(f"\n✅ Visualization complete! Files saved to: {wfn_dir}")
            else:
                # Basic visualization without mo_visualizer
                self._basic_visualization(wfn_data, molecule_name, wfn_dir)
    
    def _basic_visualization(self, wfn_data, molecule_name, wfn_dir):
        """Basic visualization when mo_visualizer is not available."""
        print("\n📊 Wavefunction Summary:")
        
        # Show MO energies if available
        if 'mo_energy' in wfn_data:
            mo_energies = wfn_data['mo_energy']
            print(f"   Total MOs: {len(mo_energies)}")
            
            # Find HOMO/LUMO
            if 'mo_occ' in wfn_data:
                occ = wfn_data['mo_occ']
                homo_idx = np.where(occ > 0)[0][-1]
                lumo_idx = homo_idx + 1 if homo_idx + 1 < len(mo_energies) else homo_idx
                
                print(f"   HOMO (MO {homo_idx}): {mo_energies[homo_idx]*27.2114:.4f} eV")
                print(f"   LUMO (MO {lumo_idx}): {mo_energies[lumo_idx]*27.2114:.4f} eV")
                print(f"   HOMO-LUMO Gap: {(mo_energies[lumo_idx]-mo_energies[homo_idx])*27.2114:.4f} eV")
            
            # Plot energy levels
            if MATPLOTLIB_AVAILABLE:
                self._plot_mo_energy_diagram(mo_energies, wfn_data.get('mo_occ'))
        
        # Check for cube files
        cube_files = list(wfn_dir.glob('*.cube'))
        if cube_files:
            print(f"\n📁 Available cube files:")
            for cf in cube_files:
                print(f"   • {cf.name}")
        
        print(f"\n⚠️ For full visualization, set mo_visualizer:")
        print(f"   from src.analysis import OrbitalVisualizer")
        print(f"   viewer.set_mo_visualizer(OrbitalVisualizer())")
    
    def _plot_mo_energy_diagram(self, mo_energies, mo_occ=None):
        """Plot simple MO energy diagram."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        n_show = min(self.n_orbitals_slider.value, len(mo_energies))
        
        # Convert to eV
        energies_ev = mo_energies * 27.2114
        
        # Find HOMO index
        homo_idx = 0
        if mo_occ is not None:
            occupied = np.where(mo_occ > 0)[0]
            if len(occupied) > 0:
                homo_idx = occupied[-1]
        
        # Get range around HOMO
        start = max(0, homo_idx - n_show // 2)
        end = min(len(energies_ev), start + n_show)
        
        indices = list(range(start, end))
        energies = energies_ev[start:end]
        
        # Plot lines
        for i, (idx, e) in enumerate(zip(indices, energies)):
            color = '#2196f3' if idx <= homo_idx else '#f44336'
            ax.hlines(y=e, xmin=i-0.3, xmax=i+0.3, colors=color, linewidth=3)
            
            # Mark HOMO/LUMO
            if idx == homo_idx:
                ax.annotate('HOMO', (i+0.4, e), fontsize=10, fontweight='bold')
            elif idx == homo_idx + 1:
                ax.annotate('LUMO', (i+0.4, e), fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Molecular Orbital Index', fontsize=12)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title('MO Energy Diagram', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(indices)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def display(self):
        """Display the wavefunction visualization interface."""
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
            <h3 style='color: white; margin: 0;'>🧬 Interactive Wavefunction Visualization</h3>
            <p style='color: rgba(255,255,255,0.85); margin: 5px 0 0 0; font-size: 14px;'>
                Load and visualize molecular orbitals from saved DFT wavefunctions
            </p>
        </div>
        """)
        
        controls_row1 = widgets.HBox([
            self.wfn_base_dir_input, 
            self.scan_wfn_button
        ])
        
        controls_row2 = widgets.HBox([
            self.wfn_mol_select, 
            self.load_wfn_button
        ])
        
        controls_row3 = widgets.HBox([
            self.n_orbitals_slider, 
            self.isovalue_slider
        ])
        
        interface = widgets.VBox([
            header,
            controls_row1,
            controls_row2,
            controls_row3,
            self.wfn_output
        ])
        
        return interface
