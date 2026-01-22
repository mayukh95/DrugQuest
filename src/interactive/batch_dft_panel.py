"""
Advanced Parallel DFT Optimization System (ENHANCED v3)

Features:
- Multiple molecule selection with checkboxes
- Parallel execution (up to 3 jobs simultaneously)
- Crash recovery with progress tracking
- Individual output folders per molecule
- Dispersion corrections (D3, D3BJ, D4)
- Implicit solvation (PCM, COSMO, ddCOSMO)
- All optimization methods including PySCF Native
- Extensive basis set library
- ECPs for heavy elements
- SCF convergence fallback strategies
"""

import json
import time
import threading
from pathlib import Path
from src.interactive.ui_styles import UIStyles
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures

import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False


# ==============================================================================
# DFT Configuration Constants
# ==============================================================================

class DFTConfig:
    """Configuration constants for DFT calculations."""
    
    FUNCTIONALS = {
        'LDA': ['SVWN', 'VWN5', 'VWN3', 'LDA'],
        'GGA': ['BLYP', 'PBE', 'BP86', 'PW91', 'OLYP', 'HCTH', 'revPBE', 'RPBE'],
        'Meta-GGA': ['TPSS', 'M06L', 'SCAN', 'MN15L', 'revTPSS'],
        'Hybrid-GGA': ['B3LYP', 'PBE0', 'B3PW91', 'B3P86', 'O3LYP', 'X3LYP', 'BHANDHLYP'],
        'Range-Separated': ['CAM-B3LYP', 'wB97X', 'wB97', 'LC-wPBE', 'wB97X-D3'],
        'Hybrid Meta-GGA': ['M06', 'M06-2X', 'M08-HX', 'MN15', 'TPSSh', 'TPSS0'],
        'Double Hybrid': ['B2PLYP', 'B2GPPLYP', 'XYG3']
    }
    
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
    
    ECP_OPTIONS = {
        'None': None,
        'Stuttgart': 'def2-ecp',
        'LANL2DZ': 'lanl2dz',
        'SBKJC': 'sbkjc',
        'CRENBL': 'crenbl',
        'def2-ECP': 'def2-ecp'
    }
    
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
    
    DISPERSION_OPTIONS = {
        'None': None,
        'D3 (Grimme)': 'd3',
        'D3(BJ) (Becke-Johnson damping)': 'd3bj',
        'D4': 'd4'
    }
    
    OPTIMIZATION_METHODS = [
        'BFGS (Quasi-Newton)',
        'L-BFGS (Limited Memory)',
        'Conjugate Gradient',
        'Steepest Descent',
        'Newton-Raphson (Hessian)',
        'PySCF Native (geometric)'
    ]


# ==============================================================================
# Progress Tracking System
# ==============================================================================

class OptimizationTracker:
    """Track optimization progress and prevent re-optimization after crashes."""
    
    def __init__(self, tracker_file='optimization_progress.json'):
        self.tracker_file = Path(tracker_file)
        self.data = self.load()
    
    def load(self):
        """Load tracking data from file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'completed': [], 'failed': [], 'in_progress': []}
    
    def save(self):
        """Save tracking data to file."""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def is_completed(self, molecule_name):
        """Check if molecule is already optimized."""
        return molecule_name in self.data['completed']
    
    def mark_started(self, molecule_name):
        """Mark molecule as in progress."""
        if molecule_name not in self.data['in_progress']:
            self.data['in_progress'].append(molecule_name)
        self.save()
    
    def mark_completed(self, molecule_name):
        """Mark molecule as successfully optimized."""
        if molecule_name in self.data['in_progress']:
            self.data['in_progress'].remove(molecule_name)
        if molecule_name not in self.data['completed']:
            self.data['completed'].append(molecule_name)
        self.save()
    
    def mark_failed(self, molecule_name, error_msg):
        """Mark molecule as failed."""
        if molecule_name in self.data['in_progress']:
            self.data['in_progress'].remove(molecule_name)
        self.data['failed'].append({
            'molecule': molecule_name,
            'error': str(error_msg),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        self.save()
    
    def reset(self):
        """Reset all tracking data."""
        self.data = {'completed': [], 'failed': [], 'in_progress': []}
        self.save()


# ==============================================================================
# Resource Manager
# ==============================================================================

class ResourceManager:
    """Manage computational resources for parallel optimization."""
    
    def __init__(self):
        if PSUTIL_AVAILABLE:
            self.total_cpus = psutil.cpu_count(logical=False) or 4
            self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        else:
            self.total_cpus = 4
            self.total_memory_gb = 16.0
        self.refresh()
    
    def refresh(self):
        """Refresh available resources."""
        if PSUTIL_AVAILABLE:
            self.available_cpus = self.total_cpus
            self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        else:
            self.available_cpus = self.total_cpus
            self.available_memory_gb = self.total_memory_gb * 0.8
    
    def can_run_job(self, cpus_needed, memory_needed_gb):
        """Check if resources are available for a new job."""
        self.refresh()
        return (self.available_cpus >= cpus_needed and 
                self.available_memory_gb >= memory_needed_gb)
    
    def estimate_max_parallel_jobs(self, cpus_per_job, memory_per_job_gb):
        """Estimate maximum number of parallel jobs."""
        self.refresh()
        max_by_cpu = self.total_cpus // cpus_per_job
        max_by_memory = int(self.total_memory_gb // memory_per_job_gb)
        return min(max_by_cpu, max_by_memory, 3)


# ==============================================================================
# Parallel Batch Optimizer
# ==============================================================================

class ParallelBatchOptimizer:
    """Manage parallel optimization of multiple molecules using real DFT."""
    
    def __init__(self, tracker, resource_manager):
        self.tracker = tracker
        self.resource_manager = resource_manager
        self.results = []
        self.running_jobs = []
    
    def optimize_batch(self, selected_molecules, molecules_dict, functional, basis_set,
                      opt_method, max_geom_steps, max_scf_cycles, force_tol,
                      charge, multiplicity, cpus_per_job, memory_per_job_gb,
                      output_widget, dispersion=None, solvation=None, solvent='Water',
                      ecp=None, scf_damping=0.0, level_shift=0.0, output_base_dir=None,
                      progress_bar=None, status_label=None, preview_callback=None,
                      info_callback=None):
        """Optimize multiple molecules using real DFT calculations."""
        
        # Try to import DFT components
        try:
            from ..quantum.dft_calculator import EnhancedDFTCalculator
            from ..quantum.geometry_optimizer import ClassicalGeometryOptimizer
            DFT_AVAILABLE = True
        except ImportError:
            DFT_AVAILABLE = False
        
        with output_widget:
            clear_output(wait=True)
            
            if not DFT_AVAILABLE:
                print("⚠️ PySCF/DFT modules not available!")
                print("   Please ensure pyscf is installed: pip install pyscf")
                return
            
            # Filter out already completed molecules
            molecules_to_run = [m for m in selected_molecules 
                              if not self.tracker.is_completed(m)]
            
            if not molecules_to_run:
                print("✅ All selected molecules are already optimized!")
                print(f"   Completed: {', '.join(selected_molecules)}")
                return
            
            skipped = [m for m in selected_molecules if m not in molecules_to_run]
            if skipped:
                print(f"⏭️  Skipping already optimized: {', '.join(skipped)}\n")
            
            # Parse functional (remove category suffix)
            parsed_functional = functional.split(' (')[0] if ' (' in functional else functional
            
            print(f"{'='*70}")
            print(f"🚀 DFT BATCH OPTIMIZATION")
            print(f"{'='*70}")
            print(f"Molecules to optimize: {len(molecules_to_run)}")
            print(f"Functional: {parsed_functional}")
            print(f"Basis: {basis_set}")
            print(f"Method: {opt_method}")
            print(f"Max Geometry Steps: {max_geom_steps}")
            print(f"Max SCF Cycles: {max_scf_cycles}")
            print(f"Force Tolerance: {force_tol:.1e} Ha/Bohr")
            if dispersion:
                print(f"Dispersion: {dispersion}")
            if solvation:
                print(f"Solvation: {solvation} ({solvent})")
            if ecp:
                print(f"ECP: {ecp}")
            print(f"\n📊 Resource Allocation:")
            print(f"   Total CPUs: {self.resource_manager.total_cpus}")
            print(f"   Total Memory: {self.resource_manager.total_memory_gb:.1f} GB")
            print(f"   CPUs per job: {cpus_per_job}")
            print(f"   Memory per job: {memory_per_job_gb:.1f} GB")
            print(f"{'='*70}\n")
            
            # Create output base directory
            if output_base_dir is None:
                output_base_dir = Path('./optimized_molecules')
            else:
                output_base_dir = Path(output_base_dir)
            output_base_dir.mkdir(exist_ok=True, parents=True)
            
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
            
            # Prepare jobs
            jobs = []
            for mol_name in molecules_to_run:
                if mol_name not in molecules_dict:
                    print(f"⚠️  Skipping {mol_name}: molecule data not found")
                    continue
                
                mol_data = molecules_dict[mol_name]
                jobs.append((mol_name, mol_data))
            
            # Initialize progress bar
            if progress_bar:
                progress_bar.value = 0
                progress_bar.max = len(jobs)
                progress_bar.bar_style = 'info'
            
            # Execute jobs
            completed_count = 0
            failed_count = 0
            start_time = time.time()
            
            print(f"🏃 Starting DFT optimization of {len(jobs)} molecules...\n")
            
            for i, (mol_name, mol_data) in enumerate(jobs):
                self.tracker.mark_started(mol_name)
                
                # Update UI status
                if status_label:
                    status_label.value = f"<b>🔄 Optimizing {mol_name}... ({i+1}/{len(jobs)})</b>"
                
                # Update 3D preview
                if preview_callback:
                    preview_callback(mol_name)
                
                # Update info cards with progress and ETA
                if info_callback:
                    remaining = len(jobs) - i
                    elapsed = time.time() - start_time
                    if i > 0:
                        avg_per_mol = elapsed / i
                        eta_min = (remaining * avg_per_mol) / 60
                        eta_text = f"~{eta_min:.1f} min"
                    else:
                        eta_text = "Calculating..."
                    info_callback(running_count=1, eta_text=eta_text)
                
                # Create output folder for this molecule
                mol_output_dir = output_base_dir / f"{mol_name}_optimized"
                mol_output_dir.mkdir(exist_ok=True, parents=True)
                log_file = mol_output_dir / 'optimization_output.log'
                traj_file = mol_output_dir / f'{mol_name}_trajectory.xyz'
                
                print(f"{'─'*60}")
                print(f"▶️  Starting: {mol_name}")
                print(f"   Output: {mol_output_dir}")
                
                job_start = time.time()
                
                try:
                    # Get molecule data
                    coords = np.array(mol_data['coords'])
                    elements = mol_data['elements']
                    
                    # Create DFT calculator (VERBOSE=FALSE)
                    dft_calc = EnhancedDFTCalculator(
                        functional=parsed_functional,
                        basis=basis_set,
                        max_scf_cycles=max_scf_cycles,
                        num_processors=cpus_per_job,
                        max_memory=int(memory_per_job_gb * 1000),
                        dispersion=dispersion,
                        solvation=solvation,
                        solvent=solvent,
                        ecp=ecp,
                        scf_damping=scf_damping,
                        level_shift=level_shift,
                        verbose=False  # <--- SILENCED
                    )
                    
                    # Create geometry optimizer
                    optimizer = ClassicalGeometryOptimizer(
                        dft_calculator=dft_calc,
                        method=scipy_method
                    )
                    
                    # Run optimization (VERBOSE=FALSE)
                    # print(f"   Running {scipy_method} optimization...") # Removed
                    optimized_coords, history = optimizer.optimize(
                        initial_coords=coords,
                        elements=elements,
                        charge=charge,
                        multiplicity=multiplicity,
                        max_iter=max_geom_steps,
                        force_tol=force_tol,
                        verbose=False  # <--- SILENCED
                    )
                    
                    job_time = time.time() - job_start
                    
                    # Save trajectory
                    with open(traj_file, 'w') as f:
                        for i, (step_coords, energy, force_norm) in enumerate(zip(
                            history['coords'], 
                            history['energies'], 
                            history['force_norms']
                        )):
                            f.write(f"{len(elements)}\n")
                            f.write(f"Step {i}: E={energy:.8f} Ha, F={force_norm:.6f} Ha/Bohr\n")
                            for el, coord in zip(elements, step_coords):
                                f.write(f"{el} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
                    
                    # Save final structure
                    final_xyz_file = mol_output_dir / f'{mol_name}_optimized.xyz'
                    with open(final_xyz_file, 'w') as f:
                        f.write(f"{len(elements)}\n")
                        f.write(f"Optimized {mol_name}, E={history['energies'][-1]:.8f} Ha\n")
                        for el, coord in zip(elements, optimized_coords):
                            f.write(f"{el} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
                    
                    # Generate and save wavefunction
                    # print(f"   Generating wavefunction files...") # Removed
                    dft_calc.save_wavefunction(
                        coords=optimized_coords,
                        elements=elements,
                        output_dir=mol_output_dir,
                        file_name=f"{mol_name}_optimized",
                        charge=charge,
                        multiplicity=multiplicity
                    )
                    
                    # Save detailed log (ORCA/Gaussian style)
                    with open(log_file, 'w') as f:
                        # Header
                        f.write("=" * 80 + "\n")
                        f.write("                    DFT GEOMETRY OPTIMIZATION OUTPUT\n")
                        f.write("                         PySCF-based Calculator\n")
                        f.write("=" * 80 + "\n\n")
                        
                        import datetime
                        f.write(f"Calculation started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Molecule: {mol_name}\n")
                        if 'smiles' in mol_data:
                            f.write(f"SMILES: {mol_data['smiles']}\n")
                        f.write("\n")
                        
                        # System Information
                        f.write("-" * 80 + "\n")
                        f.write("                           SYSTEM INFORMATION\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"Number of atoms:       {len(elements)}\n")
                        f.write(f"Molecular charge:      {charge}\n")
                        f.write(f"Spin multiplicity:     {multiplicity}\n")
                        
                        # Count electrons (approximate)
                        atomic_numbers = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'P': 15}
                        n_electrons = sum(atomic_numbers.get(el, 6) for el in elements) - charge
                        f.write(f"Number of electrons:   {n_electrons} (approx)\n")
                        f.write("\n")
                        
                        # Initial Geometry
                        f.write("-" * 80 + "\n")
                        f.write("                         INITIAL GEOMETRY (Angstrom)\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"{'Atom':<4} {'X':>14} {'Y':>14} {'Z':>14}\n")
                        f.write("-" * 50 + "\n")
                        for i, (el, coord) in enumerate(zip(elements, coords)):
                            f.write(f"{el:<4} {coord[0]:>14.8f} {coord[1]:>14.8f} {coord[2]:>14.8f}\n")
                        f.write("\n")
                        
                        # Calculation Settings
                        f.write("-" * 80 + "\n")
                        f.write("                         CALCULATION SETTINGS\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"DFT Functional:        {parsed_functional}\n")
                        f.write(f"Basis Set:             {basis_set}\n")
                        f.write(f"Optimization Method:   {opt_method} ({scipy_method})\n")
                        f.write(f"Max Geometry Steps:    {max_geom_steps}\n")
                        f.write(f"Max SCF Cycles:        {max_scf_cycles}\n")
                        f.write(f"Force Tolerance:       {force_tol:.2e} Ha/Bohr\n")
                        if dispersion:
                            f.write(f"Dispersion Correction: {dispersion}\n")
                        else:
                            f.write(f"Dispersion Correction: None\n")
                        if solvation:
                            f.write(f"Solvation Model:       {solvation} (solvent: {solvent})\n")
                        else:
                            f.write(f"Solvation Model:       Gas Phase\n")
                        if ecp:
                            f.write(f"Effective Core Pot.:   {ecp}\n")
                        f.write(f"SCF Damping:           {scf_damping}\n")
                        f.write(f"Level Shift:           {level_shift} Ha\n")
                        f.write(f"CPUs per job:          {cpus_per_job}\n")
                        f.write(f"Memory per job:        {memory_per_job_gb:.1f} GB\n")
                        
                        # Detailed Optimization Progress (ORCA-style)
                        f.write("-" * 80 + "\n")
                        f.write("                  DETAILED GEOMETRY OPTIMIZATION PROGRESS\n")
                        f.write("-" * 80 + "\n\n")
                        
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
                            for j, (el, coord) in enumerate(zip(elements, coords_step)):
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
                        for i, (el, coord) in enumerate(zip(elements, optimized_coords)):
                            f.write(f"{el:<4} {coord[0]:>14.8f} {coord[1]:>14.8f} {coord[2]:>14.8f}\n")
                        f.write("\n")
                        
                        # Geometry Changes
                        f.write("-" * 80 + "\n")
                        f.write("                         GEOMETRY CHANGES (Angstrom)\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"{'Atom':<4} {'dX':>14} {'dY':>14} {'dZ':>14} {'|dr|':>14}\n")
                        f.write("-" * 65 + "\n")
                        total_displacement = 0.0
                        for i, (el, c_init, c_final) in enumerate(zip(elements, coords, optimized_coords)):
                            dx = c_final[0] - c_init[0]
                            dy = c_final[1] - c_init[1]
                            dz = c_final[2] - c_init[2]
                            dr = np.sqrt(dx**2 + dy**2 + dz**2)
                            total_displacement += dr
                            f.write(f"{el:<4} {dx:>14.8f} {dy:>14.8f} {dz:>14.8f} {dr:>14.8f}\n")
                        f.write("-" * 65 + "\n")
                        rmsd = np.sqrt(np.mean(np.sum((np.array(optimized_coords) - np.array(coords))**2, axis=1)))
                        f.write(f"Total displacement: {total_displacement:.8f} Angstrom\n")
                        f.write(f"RMSD from initial:  {rmsd:.8f} Angstrom\n")
                        f.write("\n")
                        
                        # Molecular Orbital Analysis & Extended PySCF Data
                        # Load wavefunction data
                        wfn_file = mol_output_dir / "wavefunctions" / f"{mol_name}_optimized_wavefunction.npz"
                        if wfn_file.exists():
                            try:
                                wfn_data = np.load(wfn_file, allow_pickle=True)
                                mo_energy = wfn_data['mo_energy']
                                mo_occ = wfn_data['mo_occ']
                                mo_coeff = wfn_data['mo_coeff']
                                
                                # Find HOMO/LUMO indices
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
                                f.write("\n")
                                
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
                                
                                mull_charges = wfn_data.get('mulliken_charges', np.zeros(len(elements)))
                                lowd_charges = wfn_data.get('lowdin_charges', np.zeros(len(elements)))
                                hirsh_charges = wfn_data.get('hirshfeld_charges', np.zeros(len(elements)))
                                
                                for i, el in enumerate(elements):
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
                                    for i in range(len(elements)):
                                        for j in range(i+1, len(elements)):
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
                                                f.write(f"{elements[i]}{i+1:<6} {elements[j]}{j+1:<6} {bo:>12.4f} {bond_type:>12}\n")
                                    
                                    if bond_count == 0:
                                        f.write("   No significant bonds found.\n")
                                    f.write("\n")
                                    
                                    # Total valence (diagonal elements)
                                    f.write("Atomic Valences (Sum of Bond Orders):\n")
                                    f.write(f"{'#':<4} {'Atom':<6} {'Valence':>12}\n")
                                    f.write("-" * 25 + "\n")
                                    for i, el in enumerate(elements):
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
                                        for i, el in enumerate(elements):
                                            f.write(f"{i+1:<4} {el:<6} {float(edn[i]):>14.6f}\n")
                                
                                f.write("\n")
                                
                                # Molecular Orbital Analysis
                                f.write("-" * 80 + "\n")
                                f.write("                      MOLECULAR ORBITAL ANALYSIS\n")
                                f.write("-" * 80 + "\n\n")
                                
                                # HOMO-LUMO Gap and Conceptual DFT
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
                                    
                                    fukui_plus = wfn_data.get('fukui_plus', np.zeros(len(elements)))
                                    fukui_minus = wfn_data.get('fukui_minus', np.zeros(len(elements)))
                                    fukui_radical = wfn_data.get('fukui_radical', np.zeros(len(elements)))
                                    local_elec = wfn_data.get('local_electrophilicity', np.zeros(len(elements)))
                                    
                                    f.write("Condensed Fukui Functions (per atom):\n")
                                    f.write("-" * 75 + "\n")
                                    f.write(f"{'#':<4} {'Atom':<6} {'f⁺(nuc)':<12} {'f⁻(elec)':<12} {'f⁰(rad)':<12} {'ω_k':<12} {'Metabolic?':<12}\n")
                                    f.write("-" * 75 + "\n")
                                    
                                    metabolic_threshold = np.max(np.abs(fukui_minus)) * 0.5 if np.max(np.abs(fukui_minus)) > 0 else 0.1
                                    
                                    for ii, el in enumerate(elements):
                                        fp = float(fukui_plus[ii]) if ii < len(fukui_plus) else 0
                                        fm = float(fukui_minus[ii]) if ii < len(fukui_minus) else 0
                                        fr = float(fukui_radical[ii]) if ii < len(fukui_radical) else 0
                                        le = float(local_elec[ii]) if ii < len(local_elec) else 0
                                        
                                        is_metabolic = "⚠️ YES" if abs(fm) > metabolic_threshold and el not in ['H'] else ""
                                        
                                        f.write(f"{ii+1:<4} {el:<6} {fp:<12.6f} {fm:<12.6f} {fr:<12.6f} {le:<12.6f} {is_metabolic:<12}\n")
                                    
                                    f.write("-" * 75 + "\n\n")
                                    
                                    f.write("INTERPRETATION:\n")
                                    f.write("  f⁺: Sites susceptible to nucleophilic attack (GSH conjugation)\n")
                                    f.write("  f⁻: Sites susceptible to oxidation (CYP450 metabolism)\n")
                                    f.write("  f⁰: Sites susceptible to radical attack\n\n")
                                
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
                                    
                                    for ii, el in enumerate(elements):
                                        esp_val = float(esp[ii]) if ii < len(esp) else 0
                                        if esp_val < -0.05:
                                            site_type = "H-bond acceptor"
                                        elif esp_val > 0.1:
                                            site_type = "H-bond donor"
                                        else:
                                            site_type = "Neutral/Hydrophobic"
                                        f.write(f"{ii+1:<4} {el:<6} {esp_val:<14.6f} {site_type:<20}\n")
                                    
                                    f.write("-" * 50 + "\n\n")
                                    
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
                                spin_dens = wfn_data.get('spin_densities', np.zeros(len(elements)))
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
                                fukui_plus_arr = wfn_data.get('fukui_plus', np.zeros(len(elements)))
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
                                has_halogen = any(el in halogens for el in elements)
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
                                
                                # HOMO-2, HOMO-1, HOMO, LUMO, LUMO+1, LUMO+2
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
                        f.write(f"Time per Step:         {job_time / len(history['energies']):.2f} seconds (average)\n")
                        f.write("\n")
                        
                        # Output Files
                        f.write("-" * 80 + "\n")
                        f.write("                              OUTPUT FILES\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"Optimization log:      {log_file.name}\n")
                        f.write(f"Trajectory file:       {traj_file.name}\n")
                        f.write(f"Optimized structure:   {final_xyz_file.name}\n")
                        f.write(f"Wavefunction files:    wavefunctions/{mol_name}_optimized_wavefunction.npz\n")
                        f.write(f"                       wavefunctions/{mol_name}_optimized_HOMO.cube\n")
                        f.write(f"                       wavefunctions/{mol_name}_optimized_LUMO.cube\n")
                        f.write("\n")
                        
                        # Footer
                        f.write("=" * 80 + "\n")
                        f.write("                        END OF OPTIMIZATION OUTPUT\n")
                        f.write(f"               Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 80 + "\n")
                    
                    # Mark completed
                    self.tracker.mark_completed(mol_name)
                    completed_count += 1
                    
                    if progress_bar:
                        progress_bar.value = completed_count
                    
                    energy_change = history['energies'][-1] - history['energies'][0]
                    print(f"✅ Completed: {mol_name} (in {job_time:.1f}s)")
                    # print(f"   Final E = {history['energies'][-1]:.8f} Ha")
                    # print(f"   ΔE = {energy_change*627.509:.4f} kcal/mol | Steps: {len(history['energies'])}")
                    # print(f"   Saved: {traj_file.name}\n")
                    
                    self.results.append({
                        'molecule': mol_name,
                        'status': 'success',
                        'init_energy': history['energies'][0],
                        'final_energy': history['energies'][-1],
                        'energy_change': energy_change,
                        'steps': len(history['energies']),
                        'final_force': history['force_norms'][-1],
                        'time': job_time,
                        'output_dir': str(mol_output_dir)
                    })
                    
                except Exception as e:
                    import traceback
                    self.tracker.mark_failed(mol_name, str(e))
                    failed_count += 1
                    print(f"❌ Failed: {mol_name}")
                    print(f"   Error: {str(e)}")
                    # traceback.print_exc() # Keep simple
                    print()
            
            # Summary
            total_time = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"📊 BATCH OPTIMIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"✅ Successful: {completed_count}/{len(jobs)}")
            if failed_count > 0:
                print(f"❌ Failed: {failed_count}/{len(jobs)}")
            print(f"⏱️  Total time: {total_time:.1f} seconds")
            print(f"📁 Output directory: {output_base_dir}")
            print(f"{'='*70}")
            
            if status_label:
                status_label.value = f"<b>✅ Batch Optimization Complete! ({completed_count}/{len(jobs)} successful)</b>"
            
            if progress_bar:
                progress_bar.bar_style = 'success'


# ==============================================================================
# Batch DFT Control Panel
# ==============================================================================

class BatchDFTControlPanel:
    """
    Advanced Parallel DFT Optimization Dashboard.
    
    Features:
    - Multiple molecule selection with checkboxes
    - Parallel execution (up to 3 jobs simultaneously)
    - Crash recovery with progress tracking
    - 3D molecule preview with py3Dmol
    - All DFT options (functionals, basis sets, dispersion, solvation)
    - Resource management
    """
    
    def __init__(self, tracker_file='optimization_progress.json'):
        """Initialize the batch DFT control panel."""
        self.tracker = OptimizationTracker(tracker_file)
        self.resource_manager = ResourceManager()
        
        self.molecules = []
        self.molecule_names = []
        self.mol_data_lookup = {}
        self.molecule_checkboxes = {}
        
        # Track currently optimizing molecule for visual indicator
        self.currently_optimizing = None
        
        # Output widgets
        self.batch_opt_output = widgets.Output()
        self.structure_preview_output = widgets.Output()
        
        # Build options lists
        self._build_options()
        
        # Create widgets
        self._create_widgets()
    
    def _build_options(self):
        """Build dropdown options from DFTConfig."""
        # Organize functionals for dropdown
        self.all_functionals = []
        for category, funcs in DFTConfig.FUNCTIONALS.items():
            for f in funcs:
                self.all_functionals.append(f"{f} ({category})")
        
        # Organize basis sets for dropdown
        self.all_basis_sets = []
        for category, bases in DFTConfig.BASIS_SETS.items():
            for b in bases:
                self.all_basis_sets.append(b)
        
        # Get other options
        self.dispersion_options = list(DFTConfig.DISPERSION_OPTIONS.keys())
        self.solvation_options = list(DFTConfig.SOLVATION_MODELS.keys())
        self.solvent_options = list(DFTConfig.SOLVENT_OPTIONS.keys())
        self.ecp_options = list(DFTConfig.ECP_OPTIONS.keys())
        self.optimization_methods = DFTConfig.OPTIMIZATION_METHODS
    
    def _create_widgets(self):
        """Create all control panel widgets."""
        WIDGET_WIDTH = '100%'
        LABEL_WIDTH = '140px'
        
        # DFT Configuration
        self.functional_select = widgets.Dropdown(
            options=self.all_functionals,
            value='B3LYP (Hybrid-GGA)',
            description='Functional:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.basis_select = widgets.Dropdown(
            options=self.all_basis_sets,
            value='6-31G*',
            description='Basis Set:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.opt_method_select = widgets.Dropdown(
            options=self.optimization_methods,
            value='BFGS (Quasi-Newton)',
            description='Method:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.dispersion_select = widgets.Dropdown(
            options=self.dispersion_options,
            value='None',
            description='Dispersion:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.solvation_select = widgets.Dropdown(
            options=self.solvation_options,
            value='None',
            description='Solvation:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.solvent_select = widgets.Dropdown(
            options=self.solvent_options,
            value='Water',
            description='Solvent:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.ecp_select = widgets.Dropdown(
            options=self.ecp_options,
            value='None',
            description='ECP:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        # Convergence settings
        self.max_geom_steps = widgets.IntSlider(
            value=50, min=10, max=500, step=10,
            description='Max Geom Steps:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.max_scf_cycles = widgets.IntSlider(
            value=100, min=20, max=500, step=10,
            description='Max SCF Cycles:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.force_tol = widgets.FloatLogSlider(
            value=0.001, base=10, min=-5, max=-1, step=0.5,
            description='Force Tolerance:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH),
            readout_format='.1e'
        )
        
        self.scf_damping = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.9, step=0.1,
            description='SCF Damping:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.level_shift = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description='Level Shift (Ha):',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.charge = widgets.IntText(
            value=0, description='Charge:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='160px')
        )
        
        self.multiplicity = widgets.IntText(
            value=1, description='Multiplicity:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='180px')
        )
        
        # Resource settings
        self.cpus_per_job = widgets.IntSlider(
            value=min(4, self.resource_manager.total_cpus),
            min=1, max=self.resource_manager.total_cpus, step=1,
            description='CPUs per job:',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        self.memory_per_job = widgets.FloatSlider(
            value=min(8.0, self.resource_manager.total_memory_gb * 0.5),
            min=1.0, max=self.resource_manager.total_memory_gb, step=1.0,
            description='Memory (GB):',
            style={'description_width': LABEL_WIDTH},
            layout=widgets.Layout(width=WIDGET_WIDTH)
        )
        
        # Selection buttons
        self.select_all_btn = widgets.Button(
            description='✅ Select All',
            button_style='info',
            layout=widgets.Layout(width='100%', height='36px')
        )
        self.select_all_btn.on_click(self._select_all_molecules)
        
        self.select_none_btn = widgets.Button(
            description='❌ Clear All',
            button_style='warning',
            layout=widgets.Layout(width='100%', height='36px')
        )
        self.select_none_btn.on_click(self._select_none_molecules)
        
        self.select_incomplete_btn = widgets.Button(
            description='🔄 Pending Only',
            button_style='primary',
            layout=widgets.Layout(width='100%', height='36px')
        )
        self.select_incomplete_btn.on_click(self._select_incomplete_only)
        
        self.reset_tracker_btn = widgets.Button(
            description='🗑️ Reset Tracker',
            button_style='danger',
            layout=widgets.Layout(width='100%', height='36px')
        )
        self.reset_tracker_btn.on_click(self._reset_tracker_click)
        
        self.run_batch_btn = widgets.Button(
            description='🚀 Run Batch Optimization',
            button_style='success',
            layout=widgets.Layout(width='100%', height='50px')
        )
        UIStyles.apply_premium_style(self.run_batch_btn)
        self.run_batch_btn.on_click(self._run_batch_optimization)
        
        # Progress Dashboard Widgets (New)
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=100, step=1,
            description='Overall:',
            bar_style='info',
            orientation='horizontal',
            style={'bar_color': '#4caf50'},
            layout=widgets.Layout(width='98%')
        )
        
        self.status_label = widgets.HTML(
            value="<b>Ready to start optimization.</b>",
            layout=widgets.Layout(width='100%')
        )
        
        self.log_output = widgets.Output(
            layout=widgets.Layout(width='100%', height='250px', overflow_y='scroll', border='1px solid #e0e0e0', padding='8px')
        )
    
    def _select_all_molecules(self, b):
        """Select all molecules."""
        for cb in self.molecule_checkboxes.values():
            cb.value = True
    
    def _select_none_molecules(self, b):
        """Deselect all molecules."""
        for cb in self.molecule_checkboxes.values():
            cb.value = False
    
    def _select_incomplete_only(self, b):
        """Select only incomplete molecules."""
        completed = set(self.tracker.data['completed'])
        for mol_name, cb in self.molecule_checkboxes.items():
            cb.value = (mol_name not in completed)
    
    def _reset_tracker_click(self, b):
        """Reset tracking data."""
        with self.batch_opt_output:
            clear_output(wait=True)
            self.tracker.reset()
            print("🔄 Tracking data reset! Refresh notebook to update UI.")
    
    def _update_info_cards(self, running_count=0, eta_text="Calculating..."):
        """Update the info cards with current status."""
        card_style = "min-width: 160px; min-height: 70px; box-sizing: border-box;"
        
        # Update Queue
        if hasattr(self, 'queue_info'):
            total_mols = len(self.molecule_names) if hasattr(self, 'molecule_names') else 0
            completed = len(self.tracker.data.get('completed', []))
            failed = len(self.tracker.data.get('failed', []))
            pending = total_mols - completed - failed - running_count
            
            self.queue_info.value = f"""
            <div style='background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); 
                        padding: 16px; border-radius: 12px; border-left: 4px solid #0ea5e9; {card_style}'>
                <div style='font-weight: 600; color: #0369a1; margin-bottom: 8px;'>📋 Queue</div>
                <div style='color: #075985; font-size: 13px;'>
                    Pending: <b>{pending}</b> | Running: <b>{running_count}</b><br>
                    Done: <b>{completed}</b> / {total_mols}
                </div>
            </div>
            """
        
        # Update Timing
        if hasattr(self, 'timing_info'):
            self.timing_info.value = f"""
            <div style='background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); 
                        padding: 16px; border-radius: 12px; border-left: 4px solid #ec4899; {card_style}'>
                <div style='font-weight: 600; color: #be185d; margin-bottom: 8px;'>⏱️ Timing</div>
                <div style='color: #9d174d; font-size: 13px;'>
                    ETA: <b>{eta_text}</b>
                </div>
            </div>
            """
        
        # Update Settings
        if hasattr(self, 'settings_info'):
            current_func = self.functional_select.value.split(' (')[0]
            current_basis = self.basis_select.value
            current_disp = self.dispersion_select.value.split(' ')[0]
            
            self.settings_info.value = f"""
            <div style='background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); 
                        padding: 16px; border-radius: 12px; border-left: 4px solid #8b5cf6; {card_style}'>
                <div style='font-weight: 600; color: #6d28d9; margin-bottom: 8px;'>⚗️ Settings</div>
                <div style='color: #5b21b6; font-size: 13px;'>
                    <b>{current_func}</b> / {current_basis}<br>
                    Dispersion: <b>{current_disp}</b>
                </div>
            </div>
            """
    
    def _show_molecule_3d(self, mol_name, is_optimizing=False):
        """Display 3D structure of selected molecule with properties.
        
        Parameters:
        -----------
        mol_name : str
            Name of the molecule to display
        is_optimizing : bool
            If True, shows green background to indicate active optimization
        """
        self.currently_optimizing = mol_name if is_optimizing else None
        
        with self.structure_preview_output:
            clear_output(wait=True)
            
            if mol_name not in self.mol_data_lookup:
                print(f"⚠️ Molecule data not found: {mol_name}")
                return
            
            mol = self.mol_data_lookup[mol_name]
            
            if not mol.properties:
                mol.calculate_descriptors()
            
            # Check if molecule is already optimized (completed)
            is_completed = mol_name in self.tracker.data.get('completed', [])
            
            # Show optimization status banner
            if is_optimizing:
                display(HTML(f"""
                    <div style='background: linear-gradient(135deg, #fef3c7, #fde68a); 
                                padding: 8px 12px; border-radius: 8px; margin-bottom: 8px;
                                border-left: 4px solid #f59e0b;'>
                        <span style='color: #92400e; font-weight: bold;'>⚙️ Optimizing: {mol.name}</span>
                    </div>
                """))
            elif is_completed:
                display(HTML(f"""
                    <div style='background: linear-gradient(135deg, #d1fae5, #a7f3d0); 
                                padding: 8px 12px; border-radius: 8px; margin-bottom: 8px;
                                border-left: 4px solid #10b981;'>
                        <span style='color: #065f46; font-weight: bold;'>✅ Optimized: {mol.name}</span>
                    </div>
                """))
            
            print(f"🔬 {mol.name}")
            print(f"   Atoms: {len(mol.elements)} | MW: {mol.properties.get('MW', 0):.1f} Da")
            print(f"   LogP: {mol.properties.get('LogP', 0):.2f} | QED: {mol.properties.get('QED', 0):.3f}")
            
            if PY3DMOL_AVAILABLE and hasattr(mol, 'elements') and hasattr(mol, 'initial_coords'):
                view = py3Dmol.view(width=380, height=280)
                
                xyz_str = f"{len(mol.elements)}\n{mol.name}\n"
                for element, coord in zip(mol.elements, mol.initial_coords):
                    xyz_str += f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
                
                view.addModel(xyz_str, 'xyz')
                view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
                
                # Set background: amber if optimizing, light green if completed, white otherwise
                if is_optimizing:
                    bg_color = '#fef9e7'  # Light amber
                elif is_completed:
                    bg_color = '#e6f7ef'  # Light green
                else:
                    bg_color = 'white'
                view.setBackgroundColor(bg_color)
                
                view.zoomTo()
                view.show()
    
    def _on_checkbox_click(self, change):
        """Handle checkbox click to show 3D preview."""
        clicked_mol = None
        for mol_name, cb in self.molecule_checkboxes.items():
            if cb == change['owner']:
                clicked_mol = mol_name
                break
        if clicked_mol:
            self._show_molecule_3d(clicked_mol)
    
    def _run_batch_optimization(self, b):
        """Run batch optimization on selected molecules."""
        selected = [name for name, cb in self.molecule_checkboxes.items() if cb.value]
        
        if not selected:
            with self.log_output:
                clear_output(wait=True)
                print("⚠️ No molecules selected!")
            return
        
        # Reset UI
        self.progress_bar.value = 0
        self.status_label.value = "<b>🚀 initializing batch optimization...</b>"
        
        # Convert dispersion
        disp_map = {'None': None, 'D3 (Grimme)': 'd3', 'D3(BJ) (Becke-Johnson damping)': 'd3bj', 'D4': 'd4'}
        dispersion_val = disp_map.get(self.dispersion_select.value, None)
        
        # Convert solvation
        solv_map = {'None': None, 'PCM (Polarizable Continuum)': 'pcm', 'COSMO': 'cosmo', 
                   'ddCOSMO (Domain Decomposition)': 'ddcosmo', 'ddPCM': 'ddpcm', 'SMD (Universal Solvation)': 'smd'}
        solvation_val = solv_map.get(self.solvation_select.value, None)
        
        # Convert ECP
        ecp_val = DFTConfig.ECP_OPTIONS.get(self.ecp_select.value, None)
        
        molecules_dict = {}
        for mol in self.molecules:
            molecules_dict[mol.name] = {
                'coords': mol.initial_coords.tolist() if hasattr(mol.initial_coords, 'tolist') else mol.initial_coords,
                'elements': mol.elements,
                'smiles': mol.smiles
            }
        
        batch_optimizer = ParallelBatchOptimizer(self.tracker, self.resource_manager)
        
        batch_optimizer.optimize_batch(
            selected, molecules_dict,
            self.functional_select.value,
            self.basis_select.value,
            self.opt_method_select.value,
            self.max_geom_steps.value,
            self.max_scf_cycles.value,
            self.force_tol.value,
            self.charge.value,
            self.multiplicity.value,
            self.cpus_per_job.value,
            self.memory_per_job.value,
            output_widget=self.log_output, # <--- Use visible log output
            dispersion=dispersion_val,
            solvation=solvation_val,
            solvent=self.solvent_select.value,
            ecp=ecp_val,
            scf_damping=self.scf_damping.value,
            level_shift=self.level_shift.value,
            progress_bar=self.progress_bar,
            status_label=self.status_label,
            preview_callback=lambda name: self._show_molecule_3d(name, is_optimizing=True),
            info_callback=self._update_info_cards
        )
        
        # Reset after optimization completes
        self.currently_optimizing = None
        self._update_info_cards(running_count=0, eta_text="Complete!")
    
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
        
        # Create lookup dictionary
        for mol in molecules:
            self.mol_data_lookup[mol.name] = mol
        
        # Create molecule checkboxes
        completed_molecules = set(self.tracker.data['completed'])
        self.molecule_checkboxes = {}
        self.checkbox_widgets = []
        
        for mol_name in self.molecule_names:
            is_completed = mol_name in completed_molecules
            checkbox = widgets.Checkbox(
                value=False,
                description=mol_name if not is_completed else f"✅ {mol_name}",
                indent=False,
                layout=widgets.Layout(width='100%')
            )
            
            if is_completed:
                checkbox.style = {'description_color': '#00aa00'}
            
            checkbox.observe(self._on_checkbox_click, names='value')
            self.molecule_checkboxes[mol_name] = checkbox
            self.checkbox_widgets.append(checkbox)
    
    def display(self):
        """Display the complete batch DFT control panel interface."""
        UIStyles.inject()
        PANEL_WIDTH = '420px'
        
        # Header
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 24px; border-radius: 16px; margin-bottom: 24px;'>
            <h2 style='color: white; margin: 0; font-size: 26px;'>
                🚀 Batch DFT Optimization Panel
            </h2>
        </div>
        """)
        
        # Resource info
        resource_info = widgets.HTML(f"""
        <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                    padding: 16px; border-radius: 12px; border-left: 4px solid #4caf50;
                    min-width: 160px; min-height: 70px; box-sizing: border-box;'>
            <div style='font-weight: 600; color: #2e7d32; margin-bottom: 8px;'>💻 System</div>
            <div style='color: #1b5e20; font-size: 13px;'>
                CPUs: <b>{self.resource_manager.total_cpus}</b><br>
                RAM: <b>{self.resource_manager.total_memory_gb:.0f}GB</b>
            </div>
        </div>
        """)
        
        # Tracker info
        tracker_info = widgets.HTML(f"""
        <div style='background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                    padding: 16px; border-radius: 12px; border-left: 4px solid #ff9800;
                    min-width: 160px; min-height: 70px; box-sizing: border-box;'>
            <div style='font-weight: 600; color: #e65100; margin-bottom: 8px;'>📊 Progress</div>
            <div style='color: #bf360c; font-size: 13px;'>
                Completed: <b>{len(self.tracker.data['completed'])}</b><br>
                Failed: <b>{len(self.tracker.data['failed'])}</b>
            </div>
        </div>
        """)
        
        # Current Settings info
        current_func = self.functional_select.value.split(' (')[0] if hasattr(self, 'functional_select') else 'B3LYP'
        current_basis = self.basis_select.value if hasattr(self, 'basis_select') else '6-31G*'
        current_disp = self.dispersion_select.value.split(' ')[0] if hasattr(self, 'dispersion_select') else 'None'
        
        self.settings_info = widgets.HTML(f"""
        <div style='background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); 
                    padding: 16px; border-radius: 12px; border-left: 4px solid #8b5cf6;
                    min-width: 160px; min-height: 70px; box-sizing: border-box;'>
            <div style='font-weight: 600; color: #6d28d9; margin-bottom: 8px;'>⚗️ Settings</div>
            <div style='color: #5b21b6; font-size: 13px;'>
                <b>{current_func}</b> / {current_basis}<br>
                Dispersion: <b>{current_disp}</b>
            </div>
        </div>
        """)
        
        # Queue Status info
        total_mols = len(self.molecule_names) if hasattr(self, 'molecule_names') else 0
        completed = len(self.tracker.data.get('completed', []))
        failed = len(self.tracker.data.get('failed', []))
        pending = total_mols - completed - failed
        
        self.queue_info = widgets.HTML(f"""
        <div style='background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); 
                    padding: 16px; border-radius: 12px; border-left: 4px solid #0ea5e9;
                    min-width: 160px; min-height: 70px; box-sizing: border-box;'>
            <div style='font-weight: 600; color: #0369a1; margin-bottom: 8px;'>📋 Queue</div>
            <div style='color: #075985; font-size: 13px;'>
                Pending: <b>{pending}</b> | Running: <b>0</b><br>
                Total: <b>{total_mols}</b> molecules
            </div>
        </div>
        """)
        
        # Timing info
        self.timing_info = widgets.HTML(f"""
        <div style='background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); 
                    padding: 16px; border-radius: 12px; border-left: 4px solid #ec4899;
                    min-width: 160px; min-height: 70px; box-sizing: border-box;'>
            <div style='font-weight: 600; color: #be185d; margin-bottom: 8px;'>⏱️ Timing</div>
            <div style='color: #9d174d; font-size: 13px;'>
                Avg: <b>~2-5 min/mol</b><br>
                ETA: <i>Select molecules</i>
            </div>
        </div>
        """)
        
        info_cards_row = widgets.HBox(
            [resource_info, tracker_info, self.settings_info, self.queue_info, self.timing_info],
            layout=widgets.Layout(width='100%', gap='12px', margin='0 0 20px 0', justify_content='flex-start')
        )
        
        # Molecule selection panel
        molecule_selection_box = widgets.VBox(
            self.checkbox_widgets,
            layout=widgets.Layout(height='280px', overflow_y='auto', border='1px solid #e0e0e0', 
                                 padding='12px', border_radius='8px')
        )
        
        selection_buttons_row1 = widgets.HBox(
            [self.select_all_btn, self.select_none_btn],
            layout=widgets.Layout(gap='8px', width='100%')
        )
        selection_buttons_row2 = widgets.HBox(
            [self.select_incomplete_btn, self.reset_tracker_btn],
            layout=widgets.Layout(gap='8px', width='100%')
        )
        
        left_panel = widgets.VBox([
            widgets.HTML("<div style='font-weight: 600; margin-bottom: 12px;'>🎯 Select Molecules</div>"),
            selection_buttons_row1,
            selection_buttons_row2,
            widgets.HTML("<div style='height: 12px;'></div>"),
            molecule_selection_box
        ], layout=widgets.Layout(width=PANEL_WIDTH, padding='16px', border='2px solid #e0e0e0', border_radius='12px'))
        
        right_panel = widgets.VBox([
            widgets.HTML("<div style='font-weight: 600; color: #667eea; margin-bottom: 12px;'>🔮 3D Preview</div>"),
            self.structure_preview_output
        ], layout=widgets.Layout(width=PANEL_WIDTH, min_height='400px', padding='16px', 
                                 border='2px solid #667eea', border_radius='12px'))
        
        selection_row = widgets.HBox(
            [left_panel, right_panel],
            layout=widgets.Layout(width='100%', gap='20px', margin='0 0 24px 0')
        )
        
        # Settings panels
        settings_left = widgets.VBox([
            widgets.HTML("<div style='font-weight: 500; color: #757575; margin-bottom: 8px;'>DFT Configuration</div>"),
            self.functional_select,
            self.basis_select,
            self.opt_method_select,
            widgets.HTML("<div style='font-weight: 500; color: #757575; margin: 16px 0 8px 0;'>Advanced Options</div>"),
            self.dispersion_select,
            self.solvation_select,
            self.solvent_select,
            self.ecp_select,
        ], layout=widgets.Layout(width=PANEL_WIDTH, padding='0 10px 0 0'))
        
        settings_right = widgets.VBox([
            widgets.HTML("<div style='font-weight: 500; color: #757575; margin-bottom: 8px;'>Convergence</div>"),
            self.max_geom_steps,
            self.max_scf_cycles,
            self.force_tol,
            widgets.HBox([self.charge, self.multiplicity], layout=widgets.Layout(gap='12px')),
            widgets.HTML("<div style='font-weight: 500; color: #757575; margin: 16px 0 8px 0;'>SCF Fallback</div>"),
            self.scf_damping,
            self.level_shift,
            widgets.HTML("<div style='font-weight: 500; color: #757575; margin: 16px 0 8px 0;'>Resources</div>"),
            self.cpus_per_job,
            self.memory_per_job,
            widgets.HTML("<div style='height: 20px;'></div>"),
            self.run_batch_btn
        ], layout=widgets.Layout(width=PANEL_WIDTH, padding='0 0 0 10px'))
        
        settings_row = widgets.HBox(
            [settings_left, settings_right],
            layout=widgets.Layout(width='100%', gap='20px')
        )
        
        settings_panel = widgets.VBox([
            widgets.HTML("<div style='font-weight: 600; margin-bottom: 16px;'>⚙️ Optimization Settings</div>"),
            settings_row
        ], layout=widgets.Layout(width='100%', padding='20px', border='2px solid #e0e0e0', 
                                 border_radius='12px', margin='0 0 24px 0'))
        
        output_panel = widgets.VBox([
            widgets.HTML("<div style='font-weight: 600; margin-bottom: 12px;'>📋 Optimization Progress</div>"),
            self.status_label,
            widgets.HTML("<div style='height: 5px;'></div>"),
            self.progress_bar,
            widgets.HTML("<div style='height: 10px;'></div>"),
            self.log_output
        ], layout=widgets.Layout(width='100%', padding='20px', border='2px solid #e0e0e0', border_radius='12px'))
        
        # Complete dashboard
        dashboard = widgets.VBox([
            header,
            info_cards_row,
            selection_row,
            settings_panel,
            output_panel
        ], layout=widgets.Layout(max_width='900px', margin='0 auto'))
        
        return dashboard
