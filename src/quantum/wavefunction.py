"""
Wavefunction management and analysis tools.
"""

import numpy as np
import pickle
from pathlib import Path
from pyscf.tools import molden, cubegen


class WavefunctionManager:
    """Manage saving and loading of wavefunction data."""
    
    def __init__(self, output_dir=None):
        """Initialize wavefunction manager."""
        self.output_dir = Path(output_dir) if output_dir else Path("wavefunctions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_wavefunction(self, result, filename_prefix, save_cube_files=False, 
                         num_mos=6, cube_resolution=0.1):
        """
        Save wavefunction data to files.
        
        Parameters:
        -----------
        result : dict
            DFT calculation result
        filename_prefix : str
            Prefix for output files
        save_cube_files : bool
            Whether to save MO cube files
        num_mos : int
            Number of MOs around HOMO-LUMO to save
        cube_resolution : float
            Grid resolution for cube files (Bohr)
        
        Returns:
        --------
        dict : Dictionary of saved file paths
        """
        if result is None:
            print("❌ No wavefunction data to save")
            return None
        
        mol = result['mol']
        mf = result['mf']
        mo_coeff = result['mo_coeff']
        mo_energy = result['mo_energy']
        mo_occ = result['mo_occ']
        
        saved_files = {}
        
        # Save NumPy archive
        npz_file = self.output_dir / f"{filename_prefix}_wavefunction.npz"
        np.savez(npz_file,
                 mo_coeff=mo_coeff,
                 mo_energy=mo_energy,
                 mo_occ=mo_occ,
                 energy=result['energy'],
                 converged=result['converged'],
                 homo_lumo_gap=result.get('homo_lumo_gap'))
        saved_files['npz'] = str(npz_file)
        print(f"   ✓ Saved wavefunction data: {npz_file.name}")
        
        # Save Molden file
        molden_file = self.output_dir / f"{filename_prefix}.molden"
        try:
            with open(molden_file, 'w') as f:
                molden.header(mol, f)
                molden.orbital_coeff(mol, f, mo_coeff, ene=mo_energy, occ=mo_occ)
            saved_files['molden'] = str(molden_file)
            print(f"   ✓ Saved Molden file: {molden_file.name}")
        except Exception as e:
            print(f"   ⚠ Could not save Molden file: {e}")
        
        # Save cube files for key MOs
        if save_cube_files:
            cube_files = self._save_mo_cube_files(
                mol, mo_coeff, mo_energy, mo_occ, filename_prefix, 
                num_mos, cube_resolution
            )
            saved_files['cube_files'] = cube_files
        
        # Save Python pickle for easy loading
        pickle_file = self.output_dir / f"{filename_prefix}_wavefunction.pkl"
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump({
                    'mo_coeff': mo_coeff,
                    'mo_energy': mo_energy, 
                    'mo_occ': mo_occ,
                    'energy': result['energy'],
                    'converged': result['converged'],
                    'dipole': result.get('dipole'),
                    'homo_lumo_gap': result.get('homo_lumo_gap')
                }, f)
            saved_files['pickle'] = str(pickle_file)
            print(f"   ✓ Saved Python pickle: {pickle_file.name}")
        except Exception as e:
            print(f"   ⚠ Could not save pickle file: {e}")
        
        return saved_files
    
    def _save_mo_cube_files(self, mol, mo_coeff, mo_energy, mo_occ, filename_prefix,
                           num_mos, resolution):
        """Save molecular orbital cube files."""
        cube_files = []
        
        # Find HOMO and LUMO indices
        occ_indices = np.where(mo_occ > 0)[0]
        homo_idx = occ_indices[-1] if len(occ_indices) > 0 else 0
        lumo_idx = homo_idx + 1 if homo_idx + 1 < len(mo_energy) else homo_idx
        
        # Determine range of MOs to save
        n_below = num_mos // 2
        n_above = num_mos - n_below
        start_mo = max(0, homo_idx - n_below + 1)
        end_mo = min(len(mo_energy), lumo_idx + n_above)
        
        for mo_idx in range(start_mo, end_mo):
            if mo_idx == homo_idx:
                mo_label = "HOMO"
            elif mo_idx == lumo_idx:
                mo_label = "LUMO"
            elif mo_idx < homo_idx:
                mo_label = f"HOMO-{homo_idx - mo_idx}"
            else:
                mo_label = f"LUMO+{mo_idx - lumo_idx}"
            
            cube_file = self.output_dir / f"{filename_prefix}_{mo_label}.cube"
            try:
                cubegen.orbital(mol, str(cube_file), mo_coeff[:, mo_idx], 
                               resolution=resolution)
                cube_files.append(str(cube_file))
                print(f"   ✓ Saved {mo_label} cube: {cube_file.name}")
            except Exception as e:
                print(f"   ⚠ Could not save {mo_label} cube: {e}")
        
        return cube_files
    
    def load_wavefunction(self, filename_prefix):
        """
        Load wavefunction data from files.
        
        Parameters:
        -----------
        filename_prefix : str
            Prefix of wavefunction files
        
        Returns:
        --------
        dict : Wavefunction data
        """
        # Try to load from pickle first
        pickle_file = self.output_dir / f"{filename_prefix}_wavefunction.pkl"
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✓ Loaded wavefunction from {pickle_file.name}")
                return data
            except Exception as e:
                print(f"⚠ Could not load pickle file: {e}")
        
        # Try to load from NumPy archive
        npz_file = self.output_dir / f"{filename_prefix}_wavefunction.npz"
        if npz_file.exists():
            try:
                data = np.load(npz_file, allow_pickle=True)
                wfn_data = {
                    'mo_coeff': data['mo_coeff'],
                    'mo_energy': data['mo_energy'],
                    'mo_occ': data['mo_occ'],
                    'energy': float(data['energy']),
                    'converged': bool(data['converged'])
                }
                if 'homo_lumo_gap' in data:
                    wfn_data['homo_lumo_gap'] = float(data['homo_lumo_gap'])
                
                print(f"✓ Loaded wavefunction from {npz_file.name}")
                return wfn_data
            except Exception as e:
                print(f"⚠ Could not load NPZ file: {e}")
        
        print(f"❌ No wavefunction files found for {filename_prefix}")
        return None
    
    def list_saved_wavefunctions(self):
        """List all saved wavefunction files."""
        wfn_files = list(self.output_dir.glob("*_wavefunction.*"))
        
        if not wfn_files:
            print("No saved wavefunctions found")
            return []
        
        # Group by prefix
        prefixes = set()
        for f in wfn_files:
            if '_wavefunction.' in f.name:
                prefix = f.name.split('_wavefunction.')[0]
                prefixes.add(prefix)
        
        print(f"Found {len(prefixes)} saved wavefunctions:")
        for prefix in sorted(prefixes):
            print(f"  - {prefix}")
            
            # Check what files exist for this prefix
            files = []
            for ext in ['npz', 'pkl', 'molden']:
                f = self.output_dir / f"{prefix}_wavefunction.{ext}"
                if f.exists():
                    files.append(ext)
            
            # Check for cube files
            cube_files = list(self.output_dir.glob(f"{prefix}_*.cube"))
            if cube_files:
                files.append(f"{len(cube_files)} cube files")
            
            print(f"    Files: {', '.join(files)}")
        
        return list(sorted(prefixes))
    
    def delete_wavefunction(self, filename_prefix):
        """Delete all files for a wavefunction."""
        deleted = []
        
        # Delete main wavefunction files
        for ext in ['npz', 'pkl', 'molden']:
            f = self.output_dir / f"{filename_prefix}_wavefunction.{ext}"
            if f.exists():
                f.unlink()
                deleted.append(f.name)
        
        # Delete cube files
        cube_files = list(self.output_dir.glob(f"{filename_prefix}_*.cube"))
        for f in cube_files:
            f.unlink()
            deleted.append(f.name)
        
        if deleted:
            print(f"✓ Deleted {len(deleted)} files for {filename_prefix}")
            for f in deleted:
                print(f"  - {f}")
        else:
            print(f"No files found for {filename_prefix}")
    
    def get_wavefunction_info(self, filename_prefix):
        """Get summary information about a saved wavefunction."""
        wfn_data = self.load_wavefunction(filename_prefix)
        
        if wfn_data is None:
            return None
        
        mo_energy = wfn_data['mo_energy']
        mo_occ = wfn_data['mo_occ']
        
        # Find HOMO-LUMO
        occ_indices = np.where(mo_occ > 0)[0]
        homo_idx = occ_indices[-1] if len(occ_indices) > 0 else None
        lumo_idx = homo_idx + 1 if homo_idx is not None and homo_idx + 1 < len(mo_energy) else None
        
        info = {
            'filename_prefix': filename_prefix,
            'energy': wfn_data['energy'],
            'converged': wfn_data['converged'],
            'num_mos': len(mo_energy),
            'num_electrons': int(np.sum(mo_occ)),
            'homo_energy': mo_energy[homo_idx] if homo_idx is not None else None,
            'lumo_energy': mo_energy[lumo_idx] if lumo_idx is not None else None,
            'homo_lumo_gap': wfn_data.get('homo_lumo_gap'),
            'dipole': wfn_data.get('dipole')
        }
        
        return info