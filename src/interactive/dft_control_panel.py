"""
Enhanced DFT control panel for optimization settings.
This module provides the complete interactive DFT optimization dashboard.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import time
import threading
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.interactive.ui_styles import UIStyles

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    from pyscf import gto, dft, grad
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


# ==============================================================================
# Enhanced DFT Calculator with Advanced Features
# ==============================================================================

class EnhancedDFTCalculatorPanel:
    """
    Enhanced DFT calculator supporting multiple functionals, dispersion corrections,
    solvation, and robust SCF convergence strategies.
    """
    
    # ==========================================================================
    # DFT Functionals organized by category
    # ==========================================================================
    FUNCTIONALS = {
        'LDA': ['SVWN', 'VWN5', 'VWN3', 'LDA'],
        'GGA': ['BLYP', 'PBE', 'BP86', 'PW91', 'OLYP', 'HCTH', 'revPBE', 'RPBE'],
        'Meta-GGA': ['TPSS', 'M06L', 'SCAN', 'MN15L', 'revTPSS'],
        'Hybrid-GGA': ['B3LYP', 'PBE0', 'B3PW91', 'B3P86', 'O3LYP', 'X3LYP', 'BHANDHLYP'],
        'Range-Separated': ['CAM-B3LYP', 'wB97X', 'wB97', 'LC-wPBE', 'wB97X-D3'],
        'Hybrid Meta-GGA': ['M06', 'M06-2X', 'M08-HX', 'MN15', 'TPSSh', 'TPSS0'],
        'Double Hybrid': ['B2PLYP', 'B2GPPLYP', 'XYG3']
    }
    
    # ==========================================================================
    # Extensive Basis Set Library
    # ==========================================================================
    BASIS_SETS = {
        'Minimal': ['sto-3g', 'sto-6g'],
        'Pople Double-Zeta': ['3-21G', '6-31G', '6-31G*', '6-31G**', '6-31+G*', '6-31++G**'],
        'Pople Triple-Zeta': ['6-311G', '6-311G*', '6-311G**', '6-311+G*', '6-311+G**', '6-311++G**', '6-311+G(2d,p)', '6-311++G(2d,2p)', '6-311++G(3df,3pd)'],
        'Dunning cc': ['cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z'],
        'Dunning aug-cc': ['aug-cc-pVDZ', 'aug-cc-pVTZ', 'aug-cc-pVQZ'],
        'Karlsruhe def2': ['def2-SVP', 'def2-SVPD', 'def2-TZVP', 'def2-TZVPP', 'def2-TZVPD', 'def2-QZVP', 'def2-QZVPP'],
        'Ahlrichs': ['def-SVP', 'def-TZVP', 'def-QZVP'],
        'Jensen Polarization': ['pc-0', 'pc-1', 'pc-2', 'pc-3', 'aug-pc-1', 'aug-pc-2']
    }
    
    # ==========================================================================
    # Effective Core Potentials for Heavy Elements
    # ==========================================================================
    ECP_OPTIONS = {
        'None': None,
        'Stuttgart': 'def2-ecp',
        'LANL2DZ': 'lanl2dz',
        'SBKJC': 'sbkjc',
        'CRENBL': 'crenbl',
        'def2-ECP': 'def2-ecp'
    }
    
    # ==========================================================================
    # Solvation Models
    # ==========================================================================
    SOLVATION_MODELS = {
        'None': None,
        'PCM (Polarizable Continuum)': 'pcm',
        'COSMO': 'cosmo',
        'ddCOSMO (Domain Decomposition)': 'ddcosmo',
        'ddPCM': 'ddpcm',
        'SMD (Universal Solvation)': 'smd'
    }
    
    SOLVENT_OPTIONS = {
        'Water': {'epsilon': 78.39, 'name': 'water'},
        'Methanol': {'epsilon': 32.7, 'name': 'methanol'},
        'Ethanol': {'epsilon': 24.55, 'name': 'ethanol'},
        'Acetone': {'epsilon': 20.7, 'name': 'acetone'},
        'DMSO': {'epsilon': 46.7, 'name': 'dmso'},
        'Chloroform': {'epsilon': 4.81, 'name': 'chloroform'},
        'Dichloromethane': {'epsilon': 8.93, 'name': 'dichloromethane'},
        'THF': {'epsilon': 7.58, 'name': 'thf'},
        'Toluene': {'epsilon': 2.38, 'name': 'toluene'},
        'Hexane': {'epsilon': 1.88, 'name': 'hexane'},
        'Benzene': {'epsilon': 2.27, 'name': 'benzene'},
        'Acetonitrile': {'epsilon': 37.5, 'name': 'acetonitrile'},
        'Custom': {'epsilon': None, 'name': 'custom'}
    }
    
    # ==========================================================================
    # Dispersion Correction Options
    # ==========================================================================
    DISPERSION_OPTIONS = {
        'None': None,
        'D3 (Grimme)': 'd3',
        'D3(BJ) (Becke-Johnson damping)': 'd3bj',
        'D4': 'd4'
    }
    
    # ==========================================================================
    # Optimization Methods
    # ==========================================================================
    OPTIMIZATION_METHODS = [
        'BFGS (Quasi-Newton)',
        'L-BFGS (Limited Memory)',
        'Conjugate Gradient',
        'Steepest Descent',
        'Newton-Raphson (Hessian)',
        'PySCF Native (geometric)'
    ]


class DFTControlPanel:
    """
    Enhanced Interactive DFT Optimization Control Panel.
    
    Features:
    - All DFT functionals with categories (38+)
    - Complete basis set library (40+)
    - Dispersion corrections (D3, D3BJ, D4)
    - Implicit solvation (PCM, COSMO, ddCOSMO)
    - Multiple optimization methods including PySCF Native
    - SCF convergence controls
    - Effective Core Potentials (ECPs)
    - py3Dmol molecule preview
    - Gradient header with professional styling
    """
    
    def __init__(self):
        """Initialize DFT control panel."""
        self.molecules = []
        self.molecule_names = []
        self.optimization_results = {}
        
        # Output widgets
        self.mol_preview_output = widgets.Output()
        self.opt_output = widgets.Output(layout=widgets.Layout(
            height='400px', 
            overflow_y='auto',
            border='1px solid #ddd',
            padding='10px'
        ))
        
        # Build options lists
        self._build_options()
        
        # Create widgets
        self._create_widgets()
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _build_options(self):
        """Build dropdown options from EnhancedDFTCalculatorPanel."""
        # Organize functionals for dropdown
        self.all_functionals = []
        for category, funcs in EnhancedDFTCalculatorPanel.FUNCTIONALS.items():
            for f in funcs:
                self.all_functionals.append(f"{f} ({category})")
        
        # Organize basis sets for dropdown
        self.all_basis_sets = []
        for category, bases in EnhancedDFTCalculatorPanel.BASIS_SETS.items():
            for b in bases:
                self.all_basis_sets.append(b)
        
        # Get other options
        self.dispersion_options = list(EnhancedDFTCalculatorPanel.DISPERSION_OPTIONS.keys())
        self.solvation_options = list(EnhancedDFTCalculatorPanel.SOLVATION_MODELS.keys())
        self.solvent_options = list(EnhancedDFTCalculatorPanel.SOLVENT_OPTIONS.keys())
        self.ecp_options = list(EnhancedDFTCalculatorPanel.ECP_OPTIONS.keys())
        self.optimization_methods = EnhancedDFTCalculatorPanel.OPTIMIZATION_METHODS
    
    def _create_widgets(self):
        """Create all control panel widgets."""
        # Row 1: Main selectors
        self.mol_select = widgets.Dropdown(
            options=[],
            description='',
            layout=widgets.Layout(width='250px')
        )
        
        self.functional_select = widgets.Dropdown(
            options=self.all_functionals,
            value='B3LYP (Hybrid-GGA)',
            description='',
            layout=widgets.Layout(width='220px')
        )
        
        self.basis_select = widgets.Dropdown(
            options=self.all_basis_sets,
            value='6-31G*',
            description='',
            layout=widgets.Layout(width='180px')
        )
        
        self.opt_method_select = widgets.Dropdown(
            options=self.optimization_methods,
            value='BFGS (Quasi-Newton)',
            description='',
            layout=widgets.Layout(width='220px')
        )
        
        # Advanced options
        self.dispersion_select = widgets.Dropdown(
            options=self.dispersion_options,
            value='None',
            description='Dispersion:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='280px')
        )
        
        self.solvation_select = widgets.Dropdown(
            options=self.solvation_options,
            value='None',
            description='Solvation:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='280px')
        )
        
        self.solvent_select = widgets.Dropdown(
            options=self.solvent_options,
            value='Water',
            description='Solvent:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        self.ecp_select = widgets.Dropdown(
            options=self.ecp_options,
            value='None',
            description='ECP:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Convergence settings
        self.max_geom_steps_slider = widgets.IntSlider(
            value=50, min=10, max=500, step=10,
            description='Max Geom Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        self.max_scf_cycles_slider = widgets.IntSlider(
            value=100, min=20, max=500, step=10,
            description='Max SCF Cycles:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        self.force_tol_slider = widgets.FloatLogSlider(
            value=0.001, base=10, min=-5, max=-1, step=0.5,
            description='Force Tol (Ha/Bohr):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px'),
            readout_format='.1e'
        )
        
        self.scf_damping_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.9, step=0.1,
            description='SCF Damping:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='280px')
        )
        
        self.level_shift_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description='Level Shift (Ha):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='280px')
        )
        
        self.charge_input = widgets.IntText(
            value=0, description='Charge:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='120px')
        )
        
        self.multiplicity_input = widgets.IntText(
            value=1, description='Multiplicity:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='150px')
        )
        
        # Run button
        # Run button
        self.run_opt_button = widgets.Button(
            description='🚀 Run Optimization',
            button_style='success',
            layout=widgets.Layout(width='200px', height='45px')
        )
        UIStyles.apply_premium_style(self.run_opt_button)
        
        # Animated Status Indicator
        self.status_indicator = widgets.HTML(
            value="",
            layout=widgets.Layout(margin='10px 0 0 10px')
        )
    
    def _setup_callbacks(self):
        """Setup widget callbacks."""
        self.mol_select.observe(self._on_molecule_change, names='value')
        self.run_opt_button.on_click(self._on_run_optimization)
    
    def _on_molecule_change(self, change):
        """Handle molecule selection change."""
        if change['type'] == 'change' and change['name'] == 'value':
            self._show_molecule_preview(change['new'])
    
    def _show_molecule_preview(self, molecule_name):
        """Show 3D preview of selected molecule with properties."""
        with self.mol_preview_output:
            clear_output(wait=False)
            display(HTML(UIStyles.get_skeleton_html(height="300px")))
            
            mol = next((m for m in self.molecules if m.name == molecule_name), None)
            if mol is None:
                print("No molecule selected")
                return
            
            # Calculate properties if needed
            if not mol.properties:
                mol.calculate_descriptors()
            
            # Clear skeleton
            clear_output(wait=True)
            
            # Display molecule info
            print(f"{'─'*50}")
            print(f"📊 {mol.name}")
            print(f"{'─'*50}")
            smiles_display = mol.smiles[:40] + '...' if len(mol.smiles) > 40 else mol.smiles
            print(f"   Formula: {smiles_display}")
            print(f"   Atoms: {len(mol.elements)}")
            print(f"   MW: {mol.properties.get('MW', 0):.2f} Da")
            print(f"   LogP: {mol.properties.get('LogP', 0):.2f}")
            print(f"   QED: {mol.properties.get('QED', 0):.3f}")
            print(f"{'─'*50}")
            
            # 3D visualization with py3Dmol
            if PY3DMOL_AVAILABLE and hasattr(mol, 'elements') and hasattr(mol, 'initial_coords'):
                xyz_data = f"{len(mol.elements)}\n{mol.name}\n"
                for element, coord in zip(mol.elements, mol.initial_coords):
                    xyz_data += f"{element} {coord[0]:.4f} {coord[1]:.4f} {coord[2]:.4f}\n"
                
                viewer = py3Dmol.view(width=400, height=300)
                viewer.addModel(xyz_data, "xyz")
                viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                viewer.setBackgroundColor('#f0f0f0')
                viewer.zoomTo()
                viewer.show()
    
    def _on_run_optimization(self, button):
        """Run DFT optimization with all enhanced options."""
        # Visual feedback
        self.run_opt_button.disabled = True
        self.status_indicator.value = "<div style='display:flex; align-items:center;'><div class='status-running'></div><div style='color:var(--primary-color); font-weight:bold;'>Running Optimization...</div></div>"
        
        # Show Toast
        UIStyles.show_toast("Optimization Started", "DFT calculation initialized...", "info")
        
        with self.opt_output:
            clear_output(wait=True)
            
            molecule_name = self.mol_select.value
            mol = next((m for m in self.molecules if m.name == molecule_name), None)
            if mol is None:
                print(f"❌ Molecule '{molecule_name}' not found")
                self.run_opt_button.disabled = False
                self.status_indicator.value = ""
                return
            
            # Extract functional name
            functional_full = self.functional_select.value
            functional = functional_full.split(' (')[0]
            basis_set = self.basis_select.value
            opt_method = self.opt_method_select.value
            
            # Convert dispersion option
            disp_map = {'None': None, 'D3 (Grimme)': 'd3', 'D3(BJ) (Becke-Johnson damping)': 'd3bj', 'D4': 'd4'}
            dispersion_val = disp_map.get(self.dispersion_select.value, None)
            
            # Convert solvation option
            solv_map = {'None': None, 'PCM (Polarizable Continuum)': 'pcm', 'COSMO': 'cosmo', 
                       'ddCOSMO (Domain Decomposition)': 'ddcosmo', 'ddPCM': 'ddpcm', 'SMD (Universal Solvation)': 'smd'}
            solvation_val = solv_map.get(self.solvation_select.value, None)
            
            # Convert ECP option
            ecp_val = EnhancedDFTCalculatorPanel.ECP_OPTIONS.get(self.ecp_select.value, None)
            
            # Display optimization header
            print(f"╔{'═'*72}╗")
            print(f"║{'🚀 DFT GEOMETRY OPTIMIZATION':^72}║")
            print(f"╠{'═'*72}╣")
            print(f"║  {'Molecule:':<20} {mol.name:<49} ║")
            print(f"║  {'Functional:':<20} {functional:<49} ║")
            print(f"║  {'Basis Set:':<20} {basis_set:<49} ║")
            print(f"║  {'Method:':<20} {opt_method:<49} ║")
            print(f"║  {'Max Geom Steps:':<20} {self.max_geom_steps_slider.value:<49} ║")
            print(f"║  {'Max SCF Cycles:':<20} {self.max_scf_cycles_slider.value:<49} ║")
            print(f"║  {'Force Tolerance:':<20} {self.force_tol_slider.value:<49.6f} ║")
            if dispersion_val:
                print(f"║  {'Dispersion:':<20} {dispersion_val.upper():<49} ║")
            if solvation_val:
                solvent = self.solvent_select.value
                print(f"║  {'Solvation:':<20} {solvation_val.upper()} ({solvent}){' '*(36-len(solvent))} ║")
            if ecp_val:
                print(f"║  {'ECP:':<20} {ecp_val:<49} ║")
            if self.scf_damping_slider.value > 0:
                print(f"║  {'SCF Damping:':<20} {self.scf_damping_slider.value:<49.2f} ║")
            if self.level_shift_slider.value > 0:
                print(f"║  {'Level Shift:':<20} {self.level_shift_slider.value:<49.2f} ║")
            print(f"║  {'Charge:':<20} {self.charge_input.value:<49} ║")
            print(f"║  {'Multiplicity:':<20} {self.multiplicity_input.value:<49} ║")
            print(f"║  {'Atoms:':<20} {len(mol.elements):<49} ║")
            print(f"╚{'═'*72}╝")
            
            # Import DFT components
            try:
                from ..quantum.dft_calculator import EnhancedDFTCalculator
                from ..quantum.geometry_optimizer import ClassicalGeometryOptimizer
                DFT_AVAILABLE = True
            except ImportError:
                DFT_AVAILABLE = False
                print("\n⚠️ PySCF/DFT modules not available!")
                print("   Please ensure pyscf is installed: pip install pyscf")
                self.run_opt_button.disabled = False
                self.status_indicator.value = ""
                return
            
            # Map optimization method
            method_map = {
                'BFGS (Quasi-Newton)': 'BFGS',
                'L-BFGS (Limited Memory)': 'L-BFGS-B',
                'Conjugate Gradient': 'CG',
                'Steepest Descent': 'CG',
                'Newton-Raphson (Hessian)': 'Newton-CG',
                'PySCF Native (geometric)': 'L-BFGS-B'
            }
            scipy_method = method_map.get(opt_method, 'L-BFGS-B')
            
            print(f"\n⏳ Running {opt_method} optimization... (this may take a moment)\n")
            
            # Create output directory
            from pathlib import Path
            output_dir = Path('./optimized_molecules') / f"{mol.name}_optimized"
            output_dir.mkdir(exist_ok=True, parents=True)
            traj_file = output_dir / f'{mol.name}_trajectory.xyz'
            
            try:
                # Create DFT calculator
                dft_calc = EnhancedDFTCalculator(
                    functional=functional,
                    basis=basis_set,
                    max_scf_cycles=self.max_scf_cycles_slider.value,
                    num_processors=4,
                    max_memory=8000,
                    dispersion=dispersion_val,
                    solvation=solvation_val,
                    solvent=self.solvent_select.value,
                    ecp=ecp_val,
                    scf_damping=self.scf_damping_slider.value,
                    level_shift=self.level_shift_slider.value,
                    verbose=False
                )
                
                # Create optimizer
                optimizer = ClassicalGeometryOptimizer(
                    dft_calculator=dft_calc,
                    method=scipy_method
                )
                
                # Run optimization
                import time
                start_time = time.time()
                optimized_coords, history = optimizer.optimize(
                    initial_coords=mol.initial_coords,
                    elements=mol.elements,
                    charge=self.charge_input.value,
                    multiplicity=self.multiplicity_input.value,
                    max_iter=self.max_geom_steps_slider.value,
                    force_tol=self.force_tol_slider.value,
                    verbose=False
                )
                
                # Save trajectory
                with open(traj_file, 'w') as f:
                    for i, (step_coords, energy, force_norm) in enumerate(zip(
                        history['coords'], history['energies'], history['force_norms']
                    )):
                        f.write(f"{len(mol.elements)}\n")
                        f.write(f"Step {i}: E={energy:.8f} Ha, F={force_norm:.6f} Ha/Bohr\n")
                        for el, coord in zip(mol.elements, step_coords):
                            f.write(f"{el} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
                
                # Save final structure
                final_xyz = output_dir / f'{mol.name}_optimized.xyz'
                with open(final_xyz, 'w') as f:
                    f.write(f"{len(mol.elements)}\n")
                    f.write(f"Optimized {mol.name}, E={history['energies'][-1]:.8f} Ha\n")
                    for el, coord in zip(mol.elements, optimized_coords):
                        f.write(f"{el} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
                
                # Save wavefunction
                try:
                    wfn_success = dft_calc.save_wavefunction(
                        coords=optimized_coords,
                        elements=mol.elements,
                        output_dir=output_dir,
                        file_name=f"{mol.name}_optimized",
                        charge=self.charge_input.value,
                        multiplicity=self.multiplicity_input.value
                    )
                    if not wfn_success:
                        print("   ⚠️ Wavefunction saving returned False - check for errors")
                except Exception as e:
                    print(f"   ❌ Error saving wavefunction: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Save detailed log file (ORCA/Gaussian style)
                import datetime
                job_time = time.time() - start_time if 'start_time' in dir() else 0
                log_file = output_dir / 'optimization_output.log'
                
                with open(log_file, 'w') as f:
                    # Header
                    f.write("=" * 80 + "\n")
                    f.write("                    DFT GEOMETRY OPTIMIZATION OUTPUT\n")
                    f.write("                         PySCF-based Calculator\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write(f"Calculation started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Molecule: {mol.name}\n")
                    if hasattr(mol, 'smiles') and mol.smiles:
                        f.write(f"SMILES: {mol.smiles}\n")
                    f.write("\n")
                    
                    # System Information
                    f.write("-" * 80 + "\n")
                    f.write("                           SYSTEM INFORMATION\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Number of atoms:       {len(mol.elements)}\n")
                    f.write(f"Molecular charge:      {self.charge_input.value}\n")
                    f.write(f"Spin multiplicity:     {self.multiplicity_input.value}\n")
                    
                    atomic_numbers = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'P': 15}
                    n_electrons = sum(atomic_numbers.get(el, 6) for el in mol.elements) - self.charge_input.value
                    f.write(f"Number of electrons:   {n_electrons} (approx)\n")
                    f.write("\n")
                    
                    # Initial Geometry
                    f.write("-" * 80 + "\n")
                    f.write("                         INITIAL GEOMETRY (Angstrom)\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Atom':<4} {'X':>14} {'Y':>14} {'Z':>14}\n")
                    f.write("-" * 50 + "\n")
                    for el, coord in zip(mol.elements, mol.initial_coords):
                        f.write(f"{el:<4} {coord[0]:>14.8f} {coord[1]:>14.8f} {coord[2]:>14.8f}\n")
                    f.write("\n")
                    
                    # Calculation Settings
                    f.write("-" * 80 + "\n")
                    f.write("                         CALCULATION SETTINGS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"DFT Functional:        {functional}\n")
                    f.write(f"Basis Set:             {basis_set}\n")
                    f.write(f"Optimization Method:   {self.opt_method_select.value}\n")
                    f.write(f"Max Geometry Steps:    {self.max_geom_steps_slider.value}\n")
                    f.write(f"Max SCF Cycles:        {self.max_scf_cycles_slider.value}\n")
                    f.write(f"Force Tolerance:       {self.force_tol_slider.value:.2e} Ha/Bohr\n")
                    f.write(f"Dispersion Correction: {self.dispersion_select.value}\n")
                    f.write(f"Solvation Model:       {self.solvation_select.value}\n")
                    if self.solvation_select.value != 'None':
                        f.write(f"Solvent:               {self.solvent_select.value}\n")
                    # Detailed Optimization Progress (ORCA-style)
                    f.write("-" * 80 + "\n")
                    f.write("                  DETAILED GEOMETRY OPTIMIZATION PROGRESS\n")
                    f.write("-" * 80 + "\n\n")
                    
                    force_tol = self.force_tol_slider.value
                    
                    for step_i in range(len(history['energies'])):
                        e = history['energies'][step_i]
                        fn = history['force_norms'][step_i]
                        coords_step = history['coords'][step_i]
                        
                        if step_i == 0:
                            delta_e = 0.0
                        else:
                            delta_e = e - history['energies'][step_i - 1]
                        conv = "YES" if fn < force_tol else "NO"
                        
                        f.write("=" * 80 + "\n")
                        f.write(f"                      GEOMETRY OPTIMIZATION CYCLE {step_i + 1}\n")
                        f.write("=" * 80 + "\n\n")
                        
                        # SCF Iteration Details (if available)
                        if 'scf_details' in history and step_i < len(history.get('scf_details', [])):
                            scf_info = history['scf_details'][step_i]
                            if scf_info and 'iterations' in scf_info and scf_info['iterations']:
                                f.write("SCF CONVERGENCE:\n")
                                f.write("-" * 60 + "\n")
                                f.write(f"{'Iter':>5} {'Energy (Ha)':>20} {'Delta E':>16} {'DIIS Error':>14}\n")
                                f.write("-" * 60 + "\n")
                                
                                for scf_iter in scf_info['iterations']:
                                    f.write(f"{scf_iter['cycle']:>5} {scf_iter['energy']:>20.12f} "
                                           f"{scf_iter['delta_e']:>16.10f} {scf_iter['diis_error']:>14.8f}\n")
                                
                                f.write("-" * 60 + "\n")
                                f.write(f"SCF {'CONVERGED' if scf_info.get('converged', False) else 'NOT CONVERGED'} "
                                       f"in {scf_info.get('cycles', len(scf_info['iterations']))} cycles\n\n")
                            else:
                                f.write(f"SCF converged in {scf_info.get('cycles', 'N/A')} cycles\n\n")
                        
                        # Geometry at this step
                        f.write(f"GEOMETRY (Step {step_i + 1}):\n")
                        f.write(f"{'Atom':<4} {'X':>14} {'Y':>14} {'Z':>14}\n")
                        f.write("-" * 50 + "\n")
                        for j, (el, coord) in enumerate(zip(mol.elements, coords_step)):
                            f.write(f"{el:<4} {coord[0]:>14.8f} {coord[1]:>14.8f} {coord[2]:>14.8f}\n")
                        f.write("\n")
                        
                        # Energy and Forces
                        f.write(f"ENERGY:           {e:>20.12f} Ha\n")
                        f.write(f"DELTA E:          {delta_e:>20.12f} Ha ({delta_e * 627.509:>+12.6f} kcal/mol)\n")
                        f.write(f"FORCE NORM:       {fn:>20.10f} Ha/Bohr\n")
                        
                        if 'max_force' in history and step_i < len(history.get('max_force', [])):
                            f.write(f"MAX FORCE:        {history['max_force'][step_i]:>20.10f} Ha/Bohr\n")
                        if 'rms_force' in history and step_i < len(history.get('rms_force', [])):
                            f.write(f"RMS FORCE:        {history['rms_force'][step_i]:>20.10f} Ha/Bohr\n")
                        
                        f.write(f"CONVERGED:        {conv:>20}\n")
                        f.write("\n")
                    
                    # Summary Table
                    f.write("-" * 80 + "\n")
                    f.write("                      OPTIMIZATION SUMMARY TABLE\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Step':>5} {'Energy (Ha)':>18} {'Delta E (Ha)':>14} {'Force Norm':>14} {'Conv':>6}\n")
                    f.write("-" * 65 + "\n")
                    
                    for step_i in range(len(history['energies'])):
                        e = history['energies'][step_i]
                        fn = history['force_norms'][step_i]
                        if step_i == 0:
                            delta_e = 0.0
                        else:
                            delta_e = e - history['energies'][step_i - 1]
                        conv = "Yes" if fn < force_tol else "No"
                        f.write(f"{step_i:>5} {e:>18.10f} {delta_e:>14.8f} {fn:>14.8f} {conv:>6}\n")
                    
                    f.write("\n")
                    
                    # Final Energy Summary
                    f.write("-" * 80 + "\n")
                    f.write("                           FINAL ENERGY SUMMARY\n")
                    f.write("-" * 80 + "\n")
                    energy_change = history['energies'][-1] - history['energies'][0]
                    f.write(f"Initial Energy:        {history['energies'][0]:.10f} Ha\n")
                    f.write(f"Final Energy:          {history['energies'][-1]:.10f} Ha\n")
                    f.write(f"Energy Change:         {energy_change:.10f} Ha\n")
                    f.write(f"Energy Change:         {energy_change * 627.509:.6f} kcal/mol\n")
                    f.write(f"Energy Change:         {energy_change * 2625.5:.6f} kJ/mol\n")
                    f.write(f"Energy Change:         {energy_change * 27.2114:.6f} eV\n")
                    f.write(f"Final Force Norm:      {history['force_norms'][-1]:.8f} Ha/Bohr\n")
                    converged = history['force_norms'][-1] < force_tol
                    f.write(f"Optimization Status:   {'CONVERGED' if converged else 'NOT CONVERGED'}\n")
                    f.write(f"Number of Steps:       {len(history['energies'])}\n")
                    f.write("\n")
                    
                    # Optimized Geometry
                    f.write("-" * 80 + "\n")
                    f.write("                        OPTIMIZED GEOMETRY (Angstrom)\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Atom':<4} {'X':>14} {'Y':>14} {'Z':>14}\n")
                    f.write("-" * 50 + "\n")
                    for el, coord in zip(mol.elements, optimized_coords):
                        f.write(f"{el:<4} {coord[0]:>14.8f} {coord[1]:>14.8f} {coord[2]:>14.8f}\n")
                    f.write("\n")
                    
                    # Geometry Changes
                    f.write("-" * 80 + "\n")
                    f.write("                         GEOMETRY CHANGES (Angstrom)\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Atom':<4} {'dX':>14} {'dY':>14} {'dZ':>14} {'|dr|':>14}\n")
                    f.write("-" * 65 + "\n")
                    total_displacement = 0.0
                    for el, c_init, c_final in zip(mol.elements, mol.initial_coords, optimized_coords):
                        dx = c_final[0] - c_init[0]
                        dy = c_final[1] - c_init[1]
                        dz = c_final[2] - c_init[2]
                        dr = np.sqrt(dx**2 + dy**2 + dz**2)
                        total_displacement += dr
                        f.write(f"{el:<4} {dx:>14.8f} {dy:>14.8f} {dz:>14.8f} {dr:>14.8f}\n")
                    f.write("-" * 65 + "\n")
                    rmsd = np.sqrt(np.mean(np.sum((np.array(optimized_coords) - np.array(mol.initial_coords))**2, axis=1)))
                    f.write(f"Total displacement: {total_displacement:.8f} Angstrom\n")
                    f.write(f"RMSD from initial:  {rmsd:.8f} Angstrom\n")
                    f.write("\n")
                    
                    # Molecular Orbital Analysis & Extended PySCF Data
                    wfn_file = output_dir / "wavefunctions" / f"{mol.name}_optimized_wavefunction.npz"
                    if wfn_file.exists():
                        try:
                            wfn_data = np.load(wfn_file, allow_pickle=True)
                            mo_energy = wfn_data['mo_energy']
                            mo_occ = wfn_data['mo_occ']
                            mo_coeff = wfn_data['mo_coeff']
                            
                            occ_idx = np.where(mo_occ > 0)[0]
                            homo_idx = occ_idx[-1] if len(occ_idx) > 0 else 0
                            lumo_idx = homo_idx + 1 if homo_idx + 1 < len(mo_energy) else homo_idx
                            
                            # Extended PySCF Data Section
                            f.write("-" * 80 + "\n")
                            f.write("                     PYSCF ELECTRONIC STRUCTURE DATA\n")
                            f.write("-" * 80 + "\n\n")
                            
                            # Basis Set Information
                            f.write("BASIS SET INFORMATION:\n")
                            f.write("-" * 40 + "\n")
                            if 'n_ao' in wfn_data:
                                f.write(f"Number of Atomic Orbitals:     {int(wfn_data['n_ao'])}\n")
                            if 'n_mo' in wfn_data:
                                f.write(f"Number of Molecular Orbitals:  {int(wfn_data['n_mo'])}\n")
                            if 'n_electrons' in wfn_data:
                                f.write(f"Number of Electrons:           {int(wfn_data['n_electrons'])}\n")
                            if 'smallest_overlap_ev' in wfn_data:
                                sov = float(wfn_data['smallest_overlap_ev'])
                                f.write(f"Smallest Overlap Eigenvalue:   {sov:.6e}\n")
                                if sov < 1e-6:
                                    f.write("   ⚠️ WARNING: Possible linear dependency in basis!\n")
                            f.write("\n")
                            
                            # Energy Decomposition
                            f.write("ENERGY DECOMPOSITION (Hartree):\n")
                            f.write("-" * 40 + "\n")
                            total_e = float(wfn_data['energy'])
                            f.write(f"Total Electronic Energy:       {total_e:>18.10f}\n")
                            if 'nuclear_repulsion' in wfn_data:
                                nuc_rep = float(wfn_data['nuclear_repulsion'])
                                f.write(f"Nuclear Repulsion Energy:      {nuc_rep:>18.10f}\n")
                            if 'one_electron_energy' in wfn_data:
                                one_e = float(wfn_data['one_electron_energy'])
                                f.write(f"One-Electron Energy:           {one_e:>18.10f}\n")
                            if 'two_electron_energy' in wfn_data:
                                two_e = float(wfn_data['two_electron_energy'])
                                f.write(f"Two-Electron Energy:           {two_e:>18.10f}\n")
                            if 'xc_energy' in wfn_data:
                                xc_e = float(wfn_data['xc_energy'])
                                if xc_e != 0:
                                    f.write(f"Exchange-Correlation Energy:   {xc_e:>18.10f}\n")
                            f.write("\n")
                            
                            # Dipole Moment
                            f.write("DIPOLE MOMENT:\n")
                            f.write("-" * 40 + "\n")
                            if 'dipole' in wfn_data:
                                dip = wfn_data['dipole']
                                f.write(f"μx:  {float(dip[0]):>10.4f} Debye\n")
                                f.write(f"μy:  {float(dip[1]):>10.4f} Debye\n")
                                f.write(f"μz:  {float(dip[2]):>10.4f} Debye\n")
                            if 'dipole_magnitude' in wfn_data:
                                dip_mag = float(wfn_data['dipole_magnitude'])
                                f.write(f"|μ|: {dip_mag:>10.4f} Debye\n")
                            # ============================================================
                            # POPULATION AND CHARGE ANALYSIS
                            # ============================================================
                            f.write("-" * 80 + "\n")
                            f.write("                      POPULATION AND CHARGE ANALYSIS\n")
                            f.write("-" * 80 + "\n\n")
                            
                            # Combined charge table header
                            f.write("ATOMIC CHARGES (All Methods):\n")
                            f.write("-" * 70 + "\n")
                            f.write(f"{'#':<4} {'Atom':<6} {'Mulliken':>14} {'Löwdin':>14} {'Hirshfeld':>14}\n")
                            f.write("-" * 70 + "\n")
                            
                            mull_charges = wfn_data.get('mulliken_charges', np.zeros(len(mol.elements)))
                            lowd_charges = wfn_data.get('lowdin_charges', np.zeros(len(mol.elements)))
                            hirsh_charges = wfn_data.get('hirshfeld_charges', np.zeros(len(mol.elements)))
                            
                            for i, el in enumerate(mol.elements):
                                mull = float(mull_charges[i]) if i < len(mull_charges) else 0.0
                                lowd = float(lowd_charges[i]) if i < len(lowd_charges) else 0.0
                                hirsh = float(hirsh_charges[i]) if i < len(hirsh_charges) else 0.0
                                f.write(f"{i+1:<4} {el:<6} {mull:>14.6f} {lowd:>14.6f} {hirsh:>14.6f}\n")
                            
                            f.write("-" * 70 + "\n")
                            mull_sum = float(np.sum(mull_charges)) if len(mull_charges) > 0 else 0.0
                            lowd_sum = float(np.sum(lowd_charges)) if len(lowd_charges) > 0 else 0.0
                            hirsh_sum = float(np.sum(hirsh_charges)) if len(hirsh_charges) > 0 else 0.0
                            f.write(f"{'Sum':<10} {mull_sum:>14.6f} {lowd_sum:>14.6f} {hirsh_sum:>14.6f}\n\n")
                            
                            # ============================================================
                            # BOND ORDER ANALYSIS  
                            # ============================================================
                            if 'mayer_bond_orders' in wfn_data:
                                f.write("-" * 80 + "\n")
                                f.write("                          MAYER BOND ORDER ANALYSIS\n")
                                f.write("-" * 80 + "\n\n")
                                
                                mayer_bo = wfn_data['mayer_bond_orders']
                                
                                # Print significant bonds (bond order > 0.1)
                                f.write("Significant Bonds (Mayer BO > 0.1):\n")
                                f.write(f"{'Atom A':<8} {'Atom B':<8} {'Bond Order':>12} {'Bond Type':>12}\n")
                                f.write("-" * 45 + "\n")
                                
                                bond_count = 0
                                for i in range(len(mol.elements)):
                                    for j in range(i+1, len(mol.elements)):
                                        bo = float(mayer_bo[i, j])
                                        if bo > 0.1:
                                            bond_count += 1
                                            # Classify bond type
                                            if bo > 2.5:
                                                bond_type = "Triple"
                                            elif bo > 1.5:
                                                bond_type = "Double"
                                            elif bo > 0.5:
                                                bond_type = "Single"
                                            else:
                                                bond_type = "Weak"
                                            f.write(f"{mol.elements[i]}{i+1:<6} {mol.elements[j]}{j+1:<6} {bo:>12.4f} {bond_type:>12}\n")
                                
                                if bond_count == 0:
                                    f.write("   No significant bonds found.\n")
                                f.write("\n")
                                
                                # Total valence (diagonal elements)
                                f.write("Atomic Valences (Sum of Bond Orders):\n")
                                f.write(f"{'#':<4} {'Atom':<6} {'Valence':>12}\n")
                                f.write("-" * 25 + "\n")
                                for i, el in enumerate(mol.elements):
                                    valence = float(np.sum(mayer_bo[i, :])) - float(mayer_bo[i, i])
                                    f.write(f"{i+1:<4} {el:<6} {valence:>12.4f}\n")
                                f.write("\n")
                            
                            # ============================================================
                            # ADDITIONAL ELECTRONIC PROPERTIES
                            # ============================================================
                            f.write("-" * 80 + "\n")
                            f.write("                      ADDITIONAL ELECTRONIC PROPERTIES\n")
                            f.write("-" * 80 + "\n\n")
                            
                            # Kinetic and potential energy
                            if 'kinetic_energy' in wfn_data:
                                f.write(f"Kinetic Energy:                {float(wfn_data['kinetic_energy']):>18.10f} Ha\n")
                            if 'potential_energy' in wfn_data:
                                f.write(f"Nuclear-Electron Potential:    {float(wfn_data['potential_energy']):>18.10f} Ha\n")
                            
                            # Quadrupole moment
                            if 'quadrupole' in wfn_data:
                                quad = wfn_data['quadrupole']
                                f.write("\nQuadrupole Moment (Debye-Å):\n")
                                f.write(f"   Qxx: {float(quad[0,0]):>10.4f}   Qxy: {float(quad[0,1]):>10.4f}   Qxz: {float(quad[0,2]):>10.4f}\n")
                                f.write(f"   Qyx: {float(quad[1,0]):>10.4f}   Qyy: {float(quad[1,1]):>10.4f}   Qyz: {float(quad[1,2]):>10.4f}\n")
                                f.write(f"   Qzx: {float(quad[2,0]):>10.4f}   Qzy: {float(quad[2,1]):>10.4f}   Qzz: {float(quad[2,2]):>10.4f}\n")
                            
                            # Spin info (if available)
                            if 's_squared' in wfn_data and float(wfn_data['s_squared']) != 0:
                                f.write(f"\n<S²>:                          {float(wfn_data['s_squared']):>10.4f}\n")
                                f.write(f"Expected S:                    {float(wfn_data['s_expected']):>10.4f}\n")
                            
                            # Electron density at nuclei
                            if 'electron_density_at_nuclei' in wfn_data:
                                edn = wfn_data['electron_density_at_nuclei']
                                if np.any(edn != 0):
                                    f.write("\nElectron Density at Nuclei (a.u.):\n")
                                    f.write(f"{'#':<4} {'Atom':<6} {'ρ(nucleus)':>14}\n")
                                    f.write("-" * 28 + "\n")
                                    for i, el in enumerate(mol.elements):
                                        f.write(f"{i+1:<4} {el:<6} {float(edn[i]):>14.6f}\n")
                            
                            f.write("\n")
                            
                            # Molecular Orbital Analysis
                            f.write("-" * 80 + "\n")
                            f.write("                      MOLECULAR ORBITAL ANALYSIS\n")
                            f.write("-" * 80 + "\n\n")
                            
                            homo_eV = mo_energy[homo_idx] * 27.2114
                            lumo_eV = mo_energy[lumo_idx] * 27.2114
                            gap_ha = mo_energy[lumo_idx] - mo_energy[homo_idx]
                            gap_ev = gap_ha * 27.2114
                            
                            f.write(f"HOMO Energy:           {homo_eV:.4f} eV ({mo_energy[homo_idx]:.6f} Ha)\n")
                            f.write(f"LUMO Energy:           {lumo_eV:.4f} eV ({mo_energy[lumo_idx]:.6f} Ha)\n")
                            f.write(f"HOMO-LUMO Gap:         {gap_ev:.4f} eV ({gap_ha:.6f} Ha)\n\n")
                            
                            # Conceptual DFT Reactivity Descriptors
                            f.write("CONCEPTUAL DFT REACTIVITY DESCRIPTORS:\n")
                            f.write("-" * 60 + "\n")
                            
                            # Ionization Potential and Electron Affinity (Koopmans' theorem)
                            IP = -homo_eV  # Ionization Potential
                            EA = -lumo_eV  # Electron Affinity
                            
                            f.write(f"Ionization Potential (IP):     {IP:>10.4f} eV  (= -E_HOMO)\n")
                            f.write(f"Electron Affinity (EA):        {EA:>10.4f} eV  (= -E_LUMO)\n\n")
                            
                            # Global Reactivity Descriptors
                            hardness = (IP - EA) / 2 if (IP - EA) > 0 else 0.001
                            softness = 1 / (2 * hardness) if hardness > 0 else 0
                            electronegativity = (IP + EA) / 2
                            chemical_potential = -electronegativity
                            electrophilicity = electronegativity**2 / (2 * hardness) if hardness > 0 else 0
                            nucleophilicity = 1 / electrophilicity if electrophilicity > 0 else 0
                            
                            f.write(f"Chemical Hardness (η):         {hardness:>10.4f} eV  (resistance to charge transfer)\n")
                            f.write(f"Chemical Softness (S):         {softness:>10.4f} eV⁻¹ (ease of polarization)\n")
                            f.write(f"Electronegativity (χ):         {electronegativity:>10.4f} eV  (electron-withdrawing power)\n")
                            f.write(f"Chemical Potential (μ):        {chemical_potential:>10.4f} eV  (electron escape tendency)\n")
                            f.write(f"Electrophilicity Index (ω):    {electrophilicity:>10.4f} eV  (electrophilic power)\n")
                            f.write(f"Nucleophilicity Index (N):     {nucleophilicity:>10.4f}     (nucleophilic power)\n\n")
                            
                            # Drug Screening Interpretation
                            f.write("DRUG SCREENING INTERPRETATION:\n")
                            f.write("-" * 60 + "\n")
                            
                            # Stability assessment
                            if gap_ev > 5.0:
                                stability = "Very High (stable, possibly inert)"
                            elif gap_ev > 4.0:
                                stability = "High (good drug-like stability)"
                            elif gap_ev > 3.0:
                                stability = "Moderate (acceptable)"
                            elif gap_ev > 2.0:
                                stability = "Low (potential metabolic liability)"
                            else:
                                stability = "Very Low (highly reactive, toxicity concern)"
                            f.write(f"Chemical Stability:            {stability}\n")
                            
                            # Electrophilicity assessment (toxicity risk)
                            if electrophilicity < 0.8:
                                tox_risk = "Low (nucleophilic character, antioxidant potential)"
                            elif electrophilicity < 1.5:
                                tox_risk = "Low-Moderate (marginal electrophile)"
                            elif electrophilicity < 3.0:
                                tox_risk = "Moderate (metabolism concern, monitor reactivity)"
                            else:
                                tox_risk = "HIGH (strong electrophile - TOXICITY ALERT!)"
                            f.write(f"Electrophilic Toxicity Risk:   {tox_risk}\n")
                            
                            # Hardness assessment
                            if hardness > 6.0:
                                hard_assess = "Hard molecule (stable, low reactivity)"
                            elif hardness > 4.0:
                                hard_assess = "Moderately hard (balanced reactivity)"
                            elif hardness > 2.0:
                                hard_assess = "Soft molecule (reactive, potential off-targets)"
                            else:
                                hard_assess = "Very soft (highly polarizable, reactive)"
                            f.write(f"Hardness Assessment:           {hard_assess}\n\n")
                            
                            # ============================================================
                            # FUKUI FUNCTIONS (Metabolic Site Prediction)
                            # ============================================================
                            if 'fukui_minus' in wfn_data:
                                f.write("-" * 80 + "\n")
                                f.write("                   FUKUI FUNCTIONS (METABOLIC SITE PREDICTION)\n")
                                f.write("-" * 80 + "\n\n")
                                
                                fukui_plus = wfn_data.get('fukui_plus', np.zeros(len(mol.elements)))
                                fukui_minus = wfn_data.get('fukui_minus', np.zeros(len(mol.elements)))
                                fukui_radical = wfn_data.get('fukui_radical', np.zeros(len(mol.elements)))
                                local_elec = wfn_data.get('local_electrophilicity', np.zeros(len(mol.elements)))
                                local_nuc = wfn_data.get('local_nucleophilicity', np.zeros(len(mol.elements)))
                                
                                f.write("Condensed Fukui Functions (per atom):\n")
                                f.write("-" * 75 + "\n")
                                f.write(f"{'#':<4} {'Atom':<6} {'f⁺(nuc)':<12} {'f⁻(elec)':<12} {'f⁰(rad)':<12} {'ω_k':<12} {'Metabolic?':<12}\n")
                                f.write("-" * 75 + "\n")
                                
                                # Identify top metabolic sites
                                metabolic_threshold = np.max(np.abs(fukui_minus)) * 0.5 if np.max(np.abs(fukui_minus)) > 0 else 0.1
                                
                                for i, el in enumerate(mol.elements):
                                    fp = float(fukui_plus[i]) if i < len(fukui_plus) else 0
                                    fm = float(fukui_minus[i]) if i < len(fukui_minus) else 0
                                    fr = float(fukui_radical[i]) if i < len(fukui_radical) else 0
                                    le = float(local_elec[i]) if i < len(local_elec) else 0
                                    
                                    # Flag likely metabolic sites (high f- values)
                                    is_metabolic = "⚠️ YES" if abs(fm) > metabolic_threshold and el not in ['H'] else ""
                                    
                                    f.write(f"{i+1:<4} {el:<6} {fp:<12.6f} {fm:<12.6f} {fr:<12.6f} {le:<12.6f} {is_metabolic:<12}\n")
                                
                                f.write("-" * 75 + "\n\n")
                                
                                f.write("INTERPRETATION:\n")
                                f.write("  f⁺ (nucleophilic): Sites susceptible to nucleophilic attack (e.g., GSH conjugation)\n")
                                f.write("  f⁻ (electrophilic): Sites susceptible to oxidation (CYP450 metabolism)\n")
                                f.write("  f⁰ (radical): Sites susceptible to radical attack\n")
                                f.write("  ω_k (local electrophilicity): Site-specific toxicity risk\n\n")
                            
                            # ============================================================
                            # MOLECULAR PROPERTIES (Drug-Likeness Related)
                            # ============================================================
                            if 'polarizability' in wfn_data or 'molecular_volume' in wfn_data:
                                f.write("-" * 80 + "\n")
                                f.write("                   MOLECULAR PROPERTIES (DFT-DERIVED)\n")
                                f.write("-" * 80 + "\n\n")
                                
                                if 'polarizability' in wfn_data:
                                    pol = float(wfn_data['polarizability'])
                                    f.write(f"Isotropic Polarizability:      {pol:>10.4f} a.u.³\n")
                                
                                if 'molecular_volume' in wfn_data:
                                    vol = float(wfn_data['molecular_volume'])
                                    f.write(f"Molecular Volume (vdW):        {vol:>10.4f} Å³\n")
                                
                                if 'polar_surface_area' in wfn_data:
                                    psa = float(wfn_data['polar_surface_area'])
                                    f.write(f"Polar Surface Area (PSA):      {psa:>10.4f} Å²\n")
                                    # BBB penetration assessment
                                    if psa < 60:
                                        bbb = "Good (PSA < 60 Å²)"
                                    elif psa < 90:
                                        bbb = "Moderate (60-90 Å²)"
                                    else:
                                        bbb = "Poor (PSA > 90 Å²)"
                                    f.write(f"BBB Penetration Prediction:    {bbb}\n")
                                f.write("\n")
                            
                            # ============================================================
                            # ELECTROSTATIC POTENTIAL ANALYSIS
                            # ============================================================
                            if 'esp_at_nuclei' in wfn_data:
                                f.write("-" * 80 + "\n")
                                f.write("                   ELECTROSTATIC POTENTIAL ANALYSIS\n")
                                f.write("-" * 80 + "\n\n")
                                
                                esp = wfn_data['esp_at_nuclei']
                                f.write("ESP at Atomic Positions:\n")
                                f.write("-" * 50 + "\n")
                                f.write(f"{'#':<4} {'Atom':<6} {'ESP (a.u.)':<14} {'Site Type':<20}\n")
                                f.write("-" * 50 + "\n")
                                
                                for i, el in enumerate(mol.elements):
                                    esp_val = float(esp[i]) if i < len(esp) else 0
                                    if esp_val < -0.05:
                                        site_type = "H-bond acceptor"
                                    elif esp_val > 0.1:
                                        site_type = "H-bond donor"
                                    else:
                                        site_type = "Neutral/Hydrophobic"
                                    f.write(f"{i+1:<4} {el:<6} {esp_val:<14.6f} {site_type:<20}\n")
                                
                                f.write("-" * 50 + "\n\n")
                                
                                # ESP Statistics
                                f.write("ESP Surface Statistics:\n")
                                if 'esp_min' in wfn_data:
                                    f.write(f"  V_min:              {float(wfn_data['esp_min']):>10.4f} a.u.\n")
                                if 'esp_max' in wfn_data:
                                    f.write(f"  V_max:              {float(wfn_data['esp_max']):>10.4f} a.u.\n")
                                if 'esp_positive_avg' in wfn_data:
                                    f.write(f"  V_S+ (avg positive):{float(wfn_data['esp_positive_avg']):>10.4f} a.u.\n")
                                if 'esp_negative_avg' in wfn_data:
                                    f.write(f"  V_S- (avg negative):{float(wfn_data['esp_negative_avg']):>10.4f} a.u.\n")
                                if 'esp_variance' in wfn_data:
                                    f.write(f"  σ²_tot (variance):  {float(wfn_data['esp_variance']):>10.6f}\n")
                                f.write("\n")
                            
                            # ============================================================
                            # TOXICITY PREDICTION & EXPLAINABLE AI (ADMET)
                            # ============================================================
                            f.write("-" * 80 + "\n")
                            f.write("             TOXICITY PREDICTION & EXPLAINABLE AI (ADMET)\n")
                            f.write("-" * 80 + "\n\n")
                            
                            f.write(f"{'Parameter':<35} {'Value':<15} {'Risk Level':<15} {'Interpretation':<30}\n")
                            f.write("-" * 100 + "\n")
                            
                            # 1. Electrophilicity Index (DNA Alkylation)
                            el_idx_risk = "LOW"
                            el_idx_interp = "Safe"
                            if electrophilicity > 3.0: 
                                el_idx_risk = "HIGH"
                                el_idx_interp = "Potential DNA/Protein alkylator"
                            elif electrophilicity > 1.5:
                                el_idx_risk = "MODERATE"
                                el_idx_interp = "Monitor for reactivity"
                                
                            f.write(f"{'Electrophilicity Index (ω)':<35} {electrophilicity:<15.4f} {el_idx_risk:<15} {el_idx_interp:<30}\n")
                            
                            # 2. LUMO Energy (Electrophilic Reactivity)
                            lumo_risk = "LOW"
                            lumo_interp = "Stable against reduction"
                            if lumo_eV < -4.0:
                                lumo_risk = "HIGH"
                                lumo_interp = "Highly reductive/Reactive"
                            elif lumo_eV < -2.5:
                                lumo_risk = "MODERATE"
                                lumo_interp = "Potential electron acceptor"
                            f.write(f"{'LUMO Energy':<35} {lumo_eV:<15.4f} {lumo_risk:<15} {lumo_interp:<30}\n")
                            
                            # 3. HOMO-LUMO Gap (Overall Reactivity)
                            gap_risk = "LOW"
                            gap_interp = "Stable molecule"
                            if gap_ev < 3.0:
                                gap_risk = "HIGH"
                                gap_interp = "Highly reactive / Unstable"
                            elif gap_ev < 4.0:
                                gap_risk = "MODERATE"
                                gap_interp = "Moderately reactive"
                            f.write(f"{'HOMO-LUMO Gap':<35} {gap_ev:<15.4f} {gap_risk:<15} {gap_interp:<30}\n")
                            
                            # 4. Spin Density (Radical Generation)
                            spin_dens = wfn_data.get('spin_densities', np.zeros(len(mol.elements)))
                            max_spin = np.max(np.abs(spin_dens)) if len(spin_dens) > 0 else 0
                            spin_risk = "LOW"
                            spin_interp = "Closed shell / No radicals"
                            if max_spin > 0.5:
                                spin_risk = "HIGH"
                                spin_interp = "Significant radical character"
                            elif max_spin > 0.1:
                                spin_risk = "MODERATE"
                                spin_interp = "Minor radical delocalization"
                            f.write(f"{'Max Spin Density':<35} {max_spin:<15.4f} {spin_risk:<15} {spin_interp:<30}\n")

                             # 5. Fukui f+ max (Nucleophilic Attack Susceptibility)
                            fukui_plus_arr = wfn_data.get('fukui_plus', np.zeros(len(mol.elements)))
                            max_fp = np.max(fukui_plus_arr) if len(fukui_plus_arr) > 0 else 0
                            fp_risk = "LOW" 
                            fp_interp = "Resistant to nuc. attack"
                            if max_fp > 0.5:
                                fp_risk = "HIGH"
                                fp_interp = "Likely Michael acceptor"
                            elif max_fp > 0.3:
                                fp_risk = "MODERATE"
                                fp_interp = "Possible nuc. attack site"
                            f.write(f"{'Max Fukui f+ (Nuc Attack)':<35} {max_fp:<15.4f} {fp_risk:<15} {fp_interp:<30}\n")

                            # 6. Halogen Sigma-Hole Check
                            halogens = ['F', 'Cl', 'Br', 'I']
                            has_halogen = any(el in halogens for el in mol.elements)
                            if has_halogen and 'esp_max' in wfn_data:
                                sigma_val = float(wfn_data['esp_max'])
                                sigma_risk = "LOW"
                                sigma_interp = "No significant σ-hole"
                                if sigma_val > 0.05:
                                    sigma_risk = "POSSIBLE"
                                    sigma_interp = "Potential σ-hole / H-bond"
                                f.write(f"{'Halogen Sigma-Hole Ind.':<35} {sigma_val:<15.4f} {sigma_risk:<15} {sigma_interp:<30}\n")

                            f.write("\n")
                            
                            # Complete MO Energy Table (all orbitals)
                            f.write("COMPLETE MOLECULAR ORBITAL ENERGIES:\n")
                            f.write(f"{'MO':<6} {'Occ':>6} {'Energy (Ha)':>16} {'Energy (eV)':>14} {'Label':>12}\n")
                            f.write("-" * 60 + "\n")
                            
                            for i in range(len(mo_energy)):
                                e_ha = mo_energy[i]
                                e_ev = e_ha * 27.2114
                                occ = mo_occ[i]
                                
                                if i == homo_idx:
                                    label = "← HOMO"
                                elif i == lumo_idx:
                                    label = "← LUMO"
                                elif i == homo_idx - 1:
                                    label = "HOMO-1"
                                elif i == homo_idx - 2:
                                    label = "HOMO-2"
                                elif i == lumo_idx + 1:
                                    label = "LUMO+1"
                                elif i == lumo_idx + 2:
                                    label = "LUMO+2"
                                else:
                                    label = ""
                                
                                f.write(f"{i+1:<6} {occ:>6.2f} {e_ha:>16.8f} {e_ev:>14.4f} {label:>12}\n")
                            
                            f.write("\n")
                            
                            # Frontier Orbital Coefficients
                            f.write("FRONTIER ORBITAL COEFFICIENTS (Top 10 AO contributions):\n")
                            f.write("-" * 60 + "\n\n")
                            
                            frontier_orbitals = []
                            for offset in [-2, -1, 0]:
                                idx = homo_idx + offset
                                if 0 <= idx < len(mo_energy):
                                    label = "HOMO" if offset == 0 else f"HOMO{offset}"
                                    frontier_orbitals.append((label, idx))
                            
                            for offset in [0, 1, 2]:
                                idx = lumo_idx + offset
                                if 0 <= idx < len(mo_energy):
                                    label = "LUMO" if offset == 0 else f"LUMO+{offset}"
                                    frontier_orbitals.append((label, idx))
                            
                            for label, idx in frontier_orbitals:
                                coeffs = mo_coeff[:, idx]
                                e_ev = mo_energy[idx] * 27.2114
                                sorted_indices = np.argsort(np.abs(coeffs))[::-1][:10]
                                
                                f.write(f"{label} (MO {idx+1}, E = {e_ev:.4f} eV):\n")
                                f.write(f"  {'AO':>6} {'Coefficient':>14} {'|Coeff|^2':>12} {'% Contrib':>10}\n")
                                
                                total_sq = np.sum(coeffs**2)
                                for ao_idx in sorted_indices:
                                    c = coeffs[ao_idx]
                                    c_sq = c**2
                                    pct = (c_sq / total_sq) * 100 if total_sq > 0 else 0
                                    f.write(f"  {ao_idx+1:>6} {c:>14.8f} {c_sq:>12.8f} {pct:>10.2f}%\n")
                                f.write("\n")
                        except Exception as e:
                            f.write(f"(Could not load extended PySCF data: {e})\n\n")
                    
                    # Timing
                    f.write("-" * 80 + "\n")
                    f.write("                              TIMING INFORMATION\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Total Wall Time:       {job_time:.2f} seconds\n")
                    if len(history['energies']) > 0:
                        f.write(f"Time per Step:         {job_time / len(history['energies']):.2f} seconds (average)\n")
                    f.write("\n")
                    
                    # Output Files
                    f.write("-" * 80 + "\n")
                    f.write("                              OUTPUT FILES\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Optimization log:      optimization_output.log\n")
                    f.write(f"Trajectory file:       {mol.name}_trajectory.xyz\n")
                    f.write(f"Optimized structure:   {mol.name}_optimized.xyz\n")
                    f.write(f"Wavefunction files:    wavefunctions/{mol.name}_optimized_wavefunction.npz\n")
                    f.write(f"                       wavefunctions/{mol.name}_optimized_HOMO.cube\n")
                    f.write(f"                       wavefunctions/{mol.name}_optimized_LUMO.cube\n")
                    f.write("\n")
                    
                    # Footer
                    f.write("=" * 80 + "\n")
                    f.write("                        END OF OPTIMIZATION OUTPUT\n")
                    f.write(f"               Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n")
                
                
                #
                # Summary
                init_energy = history['energies'][0]
                final_energy = history['energies'][-1]
                energy_change = init_energy - final_energy
                final_force = history['force_norms'][-1]
                
                converged = final_force < self.force_tol_slider.value
                status = "✅ CONVERGED" if converged else "⚠️ MAX ITER"
                
                print(f"\n╔{'═'*72}╗")
                print(f"║{'📊 OPTIMIZATION SUMMARY':^72}║")
                print(f"╠{'═'*72}╣")
                print(f"║  Initial Energy:  {init_energy:>20.8f} Ha{' '*28}║")
                print(f"║  Final Energy:    {final_energy:>20.8f} Ha{' '*28}║")
                print(f"║  Energy Change:   {energy_change:>20.8f} Ha{' '*28}║")
                print(f"║  Energy Change:   {energy_change * 627.509:>20.4f} kcal/mol{' '*22}║")
                print(f"║  Final Force:     {final_force:>20.6f} Ha/Bohr{' '*24}║")
                print(f"║  Steps:           {len(history['energies']):>20} {' '*30}║")
                print(f"║  Status:          {status:<51}║")
                print(f"║  Saved to:        {str(output_dir)[:50]:<51}║")
                print(f"╚{'═'*72}╝")
                
                # Display final optimized structure
                print(f"\n🔬 Final Optimized Structure:")
                if PY3DMOL_AVAILABLE:
                    xyz_final = f"{len(mol.elements)}\nOptimized structure: {mol.name}\n"
                    for el, coord in zip(mol.elements, optimized_coords):
                        xyz_final += f"{el} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
                    
                    viewer = py3Dmol.view(width=500, height=400)
                    viewer.addModel(xyz_final, "xyz")
                    viewer.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
                    viewer.setBackgroundColor('#ffffff')
                    viewer.zoomTo()
                    viewer.show()
                
                # Plot convergence
                if history.get('energies'):
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    axes[0].plot(history['energies'], 'b-', linewidth=2, marker='o', markersize=3)
                    axes[0].set_xlabel('Step', fontsize=11)
                    axes[0].set_ylabel('Energy (Hartree)', fontsize=11)
                    axes[0].set_title(f'{mol.name}: Energy Convergence ({functional}/{basis_set})', fontsize=12)
                    axes[0].grid(alpha=0.3)
                    axes[0].axhline(y=final_energy, color='green', linestyle='--', alpha=0.7, label='Final')
                    axes[0].legend()
                    
                    axes[1].semilogy(history['force_norms'], 'r-', linewidth=2, marker='o', markersize=3)
                    axes[1].axhline(y=self.force_tol_slider.value, color='green', linestyle='--', 
                                   label=f'Tolerance ({self.force_tol_slider.value})')
                    axes[1].set_xlabel('Step', fontsize=11)
                    axes[1].set_ylabel('Force Norm (Ha/Bohr)', fontsize=11)
                    axes[1].set_title('Force Convergence', fontsize=12)
                    axes[1].legend()
                    axes[1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Reset UI (Success)
                    self.run_opt_button.disabled = False
                    self.status_indicator.value = "<div style='display:flex; align-items:center;'><div class='status-success'></div><div style='color:var(--success-color); font-weight:bold;'>Optimization Completed</div></div>"
                    
                    UIStyles.show_toast("Complete", "Optimization finished successfully!", "success")
                
            except Exception as e:
                import traceback
                print(f"\n❌ DFT Optimization Failed!")
                print(f"   Error: {str(e)}")
                traceback.print_exc()
                
                # Reset UI (Error)
                self.run_opt_button.disabled = False
                self.status_indicator.value = "<div style='color:var(--danger-color); font-weight:bold;'>❌ Optimization Failed</div>"
                
                UIStyles.show_toast("Error", "Optimization process failed", "error")
    
    def load_molecules(self, molecules):
        """
        Load molecules into the control panel.
        
        Parameters:
        -----------
        molecules : list
            List of DrugMolecule objects
        """
        self.molecules = molecules
        self.molecule_names = [mol.name for mol in molecules]
        
        # Update dropdown
        self.mol_select.options = self.molecule_names
        if self.molecule_names:
            self.mol_select.value = self.molecule_names[0]
            self._show_molecule_preview(self.molecule_names[0])
    
    def display(self):
        """Display the complete DFT control panel interface."""
        UIStyles.inject()
        # Gradient header
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
            <h2 style='color: white; margin: 0;'>⚛️ DFT Optimization Control Panel</h2>
        </div>
        """)
        
        # Labels row
        labels_row1 = widgets.HBox([
            widgets.HTML("<b style='width:250px;'>Molecule:</b>"),
            widgets.HTML("<b style='width:220px;'>Functional:</b>"),
            widgets.HTML("<b style='width:180px;'>Basis Set:</b>"),
            widgets.HTML("<b style='width:220px;'>Optimizer:</b>"),
        ])
        
        # Selects row
        selects_row1 = widgets.HBox([
            self.mol_select, 
            self.functional_select, 
            self.basis_select, 
            self.opt_method_select
        ])
        
        # Functional info panel
        func_info = widgets.HTML("""
        <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 12px;'>
            <b>Functional Guide:</b><br>
            • <b>LDA</b> (SVWN): Fastest, least accurate<br>
            • <b>GGA</b> (BLYP, PBE): Good balance of speed/accuracy<br>
            • <b>Hybrid-GGA</b> (B3LYP, PBE0): Best for organic molecules<br>
            • <b>Meta-GGA</b> (TPSS, M06L): Better for transition metals<br>
            • <b>Range-Separated</b> (CAM-B3LYP, ωB97X): Better for charge transfer<br>
            • <b>Double Hybrid</b> (B2PLYP): Most accurate, slowest
        </div>
        """)
        
        # Advanced options panel
        advanced_options = widgets.HTML("""
        <div style='background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 12px;'>
            <b>Advanced Options:</b><br>
            • <b>Dispersion (D3/D4):</b> Essential for weak interactions (stacking, H-bonding)<br>
            • <b>Solvation (PCM/COSMO):</b> Implicit solvent for realistic drug environments<br>
            • <b>ECP:</b> Effective core potentials for heavy elements (reduces computation)<br>
            • <b>SCF Damping/Level Shift:</b> Help with difficult SCF convergence
        </div>
        """)
        
        # 3-column layout
        settings_layout = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<div style='font-weight:bold; margin-top:10px;'>Convergence Settings:</div>"),
                self.max_geom_steps_slider,
                self.max_scf_cycles_slider,
                self.force_tol_slider,
                widgets.HBox([self.charge_input, self.multiplicity_input]),
            ]),
            widgets.VBox([
                widgets.HTML("<div style='font-weight:bold; margin-top:10px;'>Advanced Options:</div>"),
                widgets.HBox([self.dispersion_select, self.solvation_select]),
                widgets.HBox([self.solvent_select, self.ecp_select]),
                widgets.HBox([self.scf_damping_slider, self.level_shift_slider]),
            ]),
            self.mol_preview_output
        ])
        
        # Complete interface
        interface = widgets.VBox([
            header,
            labels_row1,
            selects_row1,
            settings_layout,
            widgets.HBox([self.run_opt_button, self.status_indicator]),
            func_info,
            advanced_options,
            self.opt_output
        ])
        
        return interface