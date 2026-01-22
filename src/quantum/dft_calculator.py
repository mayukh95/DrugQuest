"""
Enhanced DFT calculator with PySCF integration.
"""

import numpy as np
import time
from pathlib import Path
from pyscf import gto, dft, grad


class EnhancedDFTCalculator:
    """Enhanced DFT calculator with advanced features."""
    
    # Supported DFT functionals organized by category
    FUNCTIONALS = {
        'Hybrid': ['B3LYP', 'PBE0', 'M06-2X', 'wB97X-D', 'CAM-B3LYP'],
        'GGA': ['PBE', 'BLYP', 'BP86', 'PW91'],
        'Meta-GGA': ['M06-L', 'TPSS', 'SCAN'],
        'Range-Separated': ['wB97X', 'LC-wPBE', 'HSE06']
    }
    
    # Supported basis sets organized by category
    BASIS_SETS = {
        'Minimal': ['STO-3G', '3-21G'],
        'Split-Valence': ['6-31G', '6-31G*', '6-31G**', '6-31+G*', '6-31++G**'],
        'Correlation-Consistent': ['cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ'],
        'Augmented': ['aug-cc-pVDZ', 'aug-cc-pVTZ', 'aug-cc-pVQZ'],
        'Def2': ['def2-SVP', 'def2-TZVP', 'def2-QZVP']
    }
    
    # Dispersion correction options
    DISPERSION_OPTIONS = {
        'None': None,
        'D3 (Grimme)': 'D3',
        'D3BJ (Becke-Johnson)': 'D3BJ',
        'D4 (Latest)': 'D4'
    }
    
    # Solvation models
    SOLVATION_MODELS = {
        'None': None,
        'PCM (Polarizable Continuum)': 'PCM',
        'SMD (Solvation Model Density)': 'SMD',
        'CPCM (Conductor-like PCM)': 'CPCM'
    }
    
    # Solvent options
    SOLVENT_OPTIONS = {
        'Water': 'Water',
        'Methanol': 'Methanol',
        'Ethanol': 'Ethanol',
        'DMSO': 'DMSO',
        'Acetone': 'Acetone',
        'Benzene': 'Benzene',
        'Chloroform': 'Chloroform',
        'Dichloromethane': 'Dichloromethane'
    }
    
    # Effective Core Potential options
    ECP_OPTIONS = {
        'None': None,
        'LANL2DZ': 'LANL2DZ',
        'SDD': 'SDD',
        'Stuttgart': 'Stuttgart'
    }
    
    def __init__(self, functional='B3LYP', basis='6-31G*', max_scf_cycles=150, 
                 num_processors=1, max_memory=4000, dispersion=None, solvation=None, 
                 solvent='Water', ecp=None, scf_damping=0.0, level_shift=0.0, verbose=False):
        """
        Initialize DFT calculator.
        
        Parameters:
        -----------
        functional : str
            DFT functional (e.g., 'B3LYP', 'PBE0', 'M06')
        basis : str
            Basis set (e.g., '6-31G*', 'aug-cc-pVTZ')
        max_scf_cycles : int
            Maximum SCF iterations
        num_processors : int
            Number of CPU cores
        max_memory : int
            Memory limit in MB
        dispersion : str
            Dispersion correction (e.g., 'D3BJ')
        solvation : str
            Solvation model (e.g., 'PCM')
        solvent : str
            Solvent name
        ecp : str
            Effective core potential
        scf_damping : float
            SCF damping factor
        level_shift : float
            Level shift parameter
        verbose : bool
            Verbose output
        """
        self.functional = functional
        self.basis = basis
        self.max_scf_cycles = max_scf_cycles
        self.num_processors = num_processors
        self.max_memory = max_memory
        self.dispersion = dispersion
        self.solvation = solvation
        self.solvent = solvent
        self.ecp = ecp
        self.scf_damping = scf_damping
        self.level_shift = level_shift
        self.verbose = verbose
        self.scf_history = []
    
    def _build_geometry_string(self, coords, elements):
        """Build geometry string for PySCF."""
        geometry = ""
        for el, coord in zip(elements, coords):
            geometry += f"{el} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}; "
        return geometry
    
    def _apply_scf_settings(self, mf):
        """Apply SCF convergence settings."""
        mf.max_cycle = self.max_scf_cycles
        mf.conv_tol = 1e-8
        mf.conv_tol_grad = 1e-6
        
        if self.scf_damping > 0:
            mf.damp = self.scf_damping
        
        if self.level_shift > 0:
            mf.level_shift = self.level_shift
        
        return mf
    
    def _apply_dispersion(self, mf):
        """Apply dispersion correction."""
        if self.dispersion:
            try:
                if self.dispersion.upper() in ['D3', 'D3BJ']:
                    from pyscf import dftd3
                    mf = dftd3.dftd3(mf)
                    if self.verbose:
                        print(f"   Applied {self.dispersion} dispersion correction")
            except ImportError:
                if self.verbose:
                    print("   Warning: dftd3 not available, skipping dispersion correction")
        return mf
    
    def _apply_solvation(self, mf, mol):
        """Apply solvation model."""
        if self.solvation:
            try:
                if self.solvation.upper() == 'PCM':
                    from pyscf import solvent
                    mf = solvent.ddcosmo.ddcosmo_for_scf(mf, solvent=self.solvent)
                    if self.verbose:
                        print(f"   Applied PCM solvation model with {self.solvent}")
            except ImportError:
                if self.verbose:
                    print("   Warning: Solvation models not available")
        return mf
    
    def _setup_scf_callback(self, mf):
        """Setup SCF callback to capture iteration-by-iteration info."""
        scf_iterations = []
        
        def scf_callback(envs):
            """Callback function called at each SCF iteration."""
            cycle = envs.get('cycle', 0)
            e_tot = envs.get('e_tot', 0.0)
            
            # Get DIIS error if available
            diis_error = 0.0
            if 'norm_ddm' in envs:
                diis_error = envs['norm_ddm']
            elif 'norm_gorb' in envs:
                diis_error = envs['norm_gorb']
            
            # Calculate delta E
            delta_e = 0.0
            if len(scf_iterations) > 0:
                delta_e = e_tot - scf_iterations[-1]['energy']
            
            scf_iterations.append({
                'cycle': cycle,
                'energy': e_tot,
                'delta_e': delta_e,
                'diis_error': diis_error
            })
        
        mf.callback = scf_callback
        return mf, scf_iterations
    
    def _try_scf_with_fallbacks(self, mf):
        """Try SCF with fallback strategies."""
        # First attempt
        try:
            energy = mf.kernel()
            if mf.converged:
                return energy, True
        except Exception as e:
            if self.verbose:
                print(f"   SCF failed: {e}")
        
        # Fallback 1: Increase damping
        try:
            mf.damp = max(0.5, self.scf_damping)
            mf.reset()
            energy = mf.kernel()
            if mf.converged:
                return energy, True
        except:
            pass
        
        # Fallback 2: Level shift
        try:
            mf.level_shift = 0.2
            mf.reset()
            energy = mf.kernel()
            if mf.converged:
                return energy, True
        except:
            pass
        
        # Fallback 3: DIIS
        try:
            mf.diis = True
            mf.reset()
            energy = mf.kernel()
            return energy, mf.converged
        except:
            return None, False
    
    def calculate_energy_and_forces(self, coords, elements, charge=0, multiplicity=1, 
                                  step_number=None, log_file=None):
        """
        Calculate energy and forces for given geometry.
        
        Parameters:
        -----------
        coords : np.ndarray
            Atomic coordinates in Angstrom
        elements : list
            List of element symbols
        charge : int
            Molecular charge
        multiplicity : int
            Spin multiplicity
        step_number : int
            Optimization step number
        log_file : str
            Log file path
        
        Returns:
        --------
        dict : Calculation results
        """
        try:
            # Build geometry
            atom_str = self._build_geometry_string(coords, elements)
            
            # Create molecule
            mol = gto.M(
                atom=atom_str,
                charge=charge,
                spin=multiplicity - 1,
                unit='Angstrom',
                max_memory=self.max_memory,
                verbose=0 if not self.verbose else 3
            )
            
            if self.ecp:
                mol.ecp = self.ecp
            
            if log_file:
                mol.output = log_file
            
            mol.build()
            
            # DFT calculation
            if multiplicity == 1:
                mf = dft.RKS(mol)
            else:
                mf = dft.UKS(mol)
            
            mf.xc = self.functional
            mf = self._apply_scf_settings(mf)
            mf = self._apply_dispersion(mf)
            mf = self._apply_solvation(mf, mol)
            
            # Setup SCF callback to capture iteration details
            mf, scf_iterations = self._setup_scf_callback(mf)
            
            # Run SCF
            energy, converged = self._try_scf_with_fallbacks(mf)
            
            if energy is None:
                raise ValueError("SCF calculation failed")
            
            # Store SCF info with detailed iteration data
            scf_info = {
                'converged': converged,
                'cycles': getattr(mf, 'cycle', len(scf_iterations)),
                'e_tot': energy,
                'step': step_number,
                'iterations': scf_iterations  # Detailed iteration-by-iteration data
            }
            self.scf_history.append(scf_info)
            
            # Gradient calculation
            if multiplicity == 1:
                g = grad.RKS(mf)
            else:
                g = grad.UKS(mf)
            
            g.verbose = 0
            gradient = g.kernel()
            forces = -gradient
            
            # Calculate properties
            force_norm = np.linalg.norm(forces)
            max_force = np.max(np.abs(forces))
            rms_force = np.sqrt(np.mean(forces**2))
            
            # Energy conversions
            energy_eV = energy * 27.2114
            energy_kcal = energy * 627.509
            
            # Molecular properties
            n_electrons = mol.nelectron
            n_basis = mol.nao
            scf_cycles = getattr(mf, 'cycle', None)
            
            # Dipole moment
            dipole = None
            try:
                dm = mf.make_rdm1()
                charges = [mol.atom_charge(i) for i in range(mol.natm)]
                coords_bohr = mol.atom_coords()
                nuc_dipole = np.einsum('i,ix->x', charges, coords_bohr)
                el_dipole = -np.einsum('xij,ji->x', mol.intor('int1e_r'), dm)
                dipole = (nuc_dipole + el_dipole) * 2.541746  # au to Debye
            except:
                dipole = None
            
            # HOMO-LUMO gap
            homo_lumo_gap = None
            if hasattr(mf, 'mo_energy') and mf.mo_energy is not None:
                mo_occ = mf.mo_occ
                if multiplicity == 1:
                    homo_indices = np.where(mo_occ > 0)[0]
                    if len(homo_indices) > 0:
                        homo_idx = homo_indices[-1]
                        lumo_idx = homo_idx + 1 if homo_idx + 1 < len(mf.mo_energy) else None
                        if lumo_idx is not None:
                            homo_lumo_gap = (mf.mo_energy[lumo_idx] - mf.mo_energy[homo_idx]) * 27.2114
            
            return {
                'energy': energy,
                'forces': forces,
                'force_norm': force_norm,
                'max_force': max_force,
                'rms_force': rms_force,
                'converged': converged,
                'homo_lumo_gap': homo_lumo_gap,
                'mol': mol,
                'mf': mf,
                'mo_coeff': mf.mo_coeff,
                'mo_energy': mf.mo_energy,
                'mo_occ': mf.mo_occ,
                'energy_eV': energy_eV,
                'energy_kcal': energy_kcal,
                'n_electrons': n_electrons,
                'n_basis': n_basis,
                'scf_cycles': scf_cycles,
                'dipole': dipole
            }
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error in DFT calculation: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def save_wavefunction(self, coords, elements, output_dir, file_name, charge=0, multiplicity=1):
        """
        Calculate and save wavefunction data for visualization.
        
        Parameters:
        -----------
        coords : np.ndarray
            Atomic coordinates
        elements : list
            Element symbols
        output_dir : Path
            Directory to save files
        file_name : str
            Base name for files
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build molecule
            atom_str = self._build_geometry_string(coords, elements)
            
            mol = gto.M(
                atom=atom_str,
                charge=charge,
                spin=multiplicity - 1,
                unit='Angstrom',
                max_memory=self.max_memory,
                verbose=0
            )
            
            if self.ecp:
                mol.ecp = self.ecp
                
            mol.build()
            
            # Run DFT
            if multiplicity == 1:
                mf = dft.RKS(mol)
            else:
                mf = dft.UKS(mol)
            
            mf.xc = self.functional
            mf = self._apply_scf_settings(mf)
            mf = self._apply_dispersion(mf)
            mf = self._apply_solvation(mf, mol)
            mf.kernel()
            
            if not mf.converged:
                if self.verbose:
                    print(f"   Warning: Wavefunction save requested but SCF not converged")
            
            # Create wavefunctions directory
            wfn_dir = output_dir / "wavefunctions"
            wfn_dir.mkdir(parents=True, exist_ok=True)
            
            npz_file = wfn_dir / f"{file_name}_wavefunction.npz"
            
            # Calculate HOMO-LUMO gap
            mo_occ = mf.mo_occ
            mo_energy = mf.mo_energy
            occ_indices = np.where(mo_occ > 0)[0]
            homo_idx = occ_indices[-1] if len(occ_indices) > 0 else 0
            lumo_idx = homo_idx + 1 if homo_idx + 1 < len(mo_energy) else None
            homo_lumo_gap = None
            if lumo_idx is not None:
                homo_lumo_gap = (mo_energy[lumo_idx] - mo_energy[homo_idx]) * 27.2114  # eV
            
            # ============================================================
            # COMPREHENSIVE PySCF DATA EXTRACTION
            # ============================================================
            
            # Initialize all variables with defaults
            nuclear_repulsion = 0.0
            one_electron_energy = 0.0
            two_electron_energy = 0.0
            xc_energy = 0.0
            kinetic_energy = 0.0
            potential_energy = 0.0
            dipole = [0.0, 0.0, 0.0]
            dipole_magnitude = 0.0
            quadrupole = np.zeros((3, 3))
            mulliken_charges = np.zeros(len(elements))
            mulliken_pop_per_atom = np.zeros(len(elements))
            lowdin_charges = np.zeros(len(elements))
            hirshfeld_charges = np.zeros(len(elements))
            mayer_bond_orders = np.zeros((len(elements), len(elements)))
            wiberg_indices = np.zeros((len(elements), len(elements)))
            n_ao = len(mo_energy)
            n_mo = len(mo_energy)
            n_electrons_actual = 0
            smallest_overlap_ev = 1.0
            s_squared = 0.0
            s_expected = 0.0
            orbital_symmetries = []
            ao_labels = []
            basis_name = self.basis
            electron_density_at_nuclei = np.zeros(len(elements))
            
            try:
                # Basic molecular properties
                nuclear_repulsion = mol.energy_nuc()
                n_ao = mol.nao
                n_mo = len(mo_energy)
                n_electrons_actual = mol.nelectron
                
                # Get density matrix
                dm = mf.make_rdm1()
                
                # SCF energy components
                h1e = mf.get_hcore(mol)
                vhf = mf.get_veff(mol, dm)
                
                one_electron_energy = np.einsum('ij,ji->', h1e, dm).real
                two_electron_energy = 0.5 * np.einsum('ij,ji->', vhf, dm).real
                
                # Kinetic and potential energy
                T = mol.intor('int1e_kin')
                V = mol.intor('int1e_nuc')
                kinetic_energy = np.einsum('ij,ji->', T, dm).real
                potential_energy = np.einsum('ij,ji->', V, dm).real
                
                # XC energy (for DFT)
                if hasattr(mf, '_numint') and hasattr(mf, 'grids'):
                    try:
                        n, exc, vxc = mf._numint.nr_rks(mol, mf.grids, mf.xc, dm)
                        xc_energy = exc
                    except:
                        pass
                
                # Dipole moment
                dipole = np.array(mf.dip_moment(unit='Debye'))
                dipole_magnitude = np.linalg.norm(dipole)
                
                # Quadrupole moment
                try:
                    from pyscf import scf
                    with mol.with_common_orig([0, 0, 0]):
                        quad_ints = mol.intor('int1e_rr')
                    quadrupole = np.einsum('xij,ji->x', quad_ints.reshape(9, n_ao, n_ao), dm).reshape(3, 3).real
                except:
                    pass
                
                # ============================================================
                # CHARGE ANALYSIS METHODS
                # ============================================================
                
                # 1. Mulliken Population Analysis
                # 1. Mulliken Population Analysis & Spin Density
                spin_densities = np.zeros(len(elements))
                try:
                    S = mol.intor('int1e_ovlp')
                    mulliken_pop_result = mf.mulliken_pop(verbose=0)
                    mulliken_pop_per_atom = mulliken_pop_result[0]  # Population
                    mulliken_charges = mulliken_pop_result[1]       # Charges
                    
                    # Calculate Spin Density for open shell
                    if multiplicity != 1:
                        try:
                            # For UKS, make_rdm1 returns (2, nao, nao) array of [alpha, beta] density
                            dm_spin = mf.make_rdm1()
                            if isinstance(dm_spin, np.ndarray) and dm_spin.ndim == 3:
                                dm_a, dm_b = dm_spin[0], dm_spin[1]
                                # Calculate population for alpha and beta separately
                                pop_a = mf.mulliken_pop(mol=mol, dm=dm_a, s=S, verbose=0)[0]
                                pop_b = mf.mulliken_pop(mol=mol, dm=dm_b, s=S, verbose=0)[0]
                                spin_densities = pop_a - pop_b
                        except Exception as ex:
                             if self.verbose: print(f"Spin density calc failed: {ex}")
                except Exception as e:
                    pass
                
                # 2. Löwdin Population Analysis
                try:
                    from pyscf.lo import orth
                    S = mol.intor('int1e_ovlp')
                    # Symmetric orthogonalization
                    s_eigval, s_eigvec = np.linalg.eigh(S)
                    s_sqrt_inv = s_eigvec @ np.diag(1.0 / np.sqrt(s_eigval)) @ s_eigvec.T
                    dm_lowdin = s_sqrt_inv @ dm @ s_sqrt_inv
                    
                    # Get atomic orbital labels and sum populations
                    ao_slices = mol.aoslice_by_atom()
                    lowdin_pop = np.zeros(mol.natm)
                    for ia in range(mol.natm):
                        start, end = ao_slices[ia, 2], ao_slices[ia, 3]
                        lowdin_pop[ia] = np.diag(dm_lowdin)[start:end].sum()
                    
                    # Convert to charges
                    atom_charges = mol.atom_charges()
                    lowdin_charges = atom_charges - lowdin_pop
                except Exception as e:
                    pass
                
                # 3. Hirshfeld Charges (approximate using Becke weights if available)
                try:
                    from pyscf.dft import gen_grid, numint
                    # Use Becke partitioning as approximation to Hirshfeld
                    grids = gen_grid.Grids(mol)
                    grids.build()
                    
                    ao_value = numint.eval_ao(mol, grids.coords)
                    rho = numint.eval_rho(mol, ao_value, dm)
                    
                    # Becke weights give atomic contributions
                    hirshfeld_pop = np.zeros(mol.natm)
                    for ia in range(mol.natm):
                        becke_weights = grids.weights
                        atom_coords = mol.atom_coord(ia)
                        # Approximate Hirshfeld by Becke partitioning
                        dist = np.linalg.norm(grids.coords - atom_coords, axis=1)
                        # Weight by proximity (simplified Hirshfeld-like)
                        weight_factor = np.exp(-dist)
                        hirshfeld_pop[ia] = np.sum(rho * becke_weights * weight_factor) / np.sum(becke_weights * weight_factor)
                    hirshfeld_charges = mol.atom_charges() - hirshfeld_pop * mol.nelectron / hirshfeld_pop.sum()
                except Exception as e:
                    pass
                
                # ============================================================
                # BOND ORDER ANALYSIS
                # ============================================================
                
                # 4. Mayer Bond Orders
                try:
                    S = mol.intor('int1e_ovlp')
                    PS = dm @ S
                    ao_slices = mol.aoslice_by_atom()
                    
                    mayer_bond_orders = np.zeros((mol.natm, mol.natm))
                    for i in range(mol.natm):
                        i_start, i_end = ao_slices[i, 2], ao_slices[i, 3]
                        for j in range(i, mol.natm):
                            j_start, j_end = ao_slices[j, 2], ao_slices[j, 3]
                            
                            PS_ij = PS[i_start:i_end, j_start:j_end]
                            PS_ji = PS[j_start:j_end, i_start:i_end]
                            
                            bond_order = np.sum(PS_ij * PS_ji.T)
                            mayer_bond_orders[i, j] = bond_order
                            mayer_bond_orders[j, i] = bond_order
                except Exception as e:
                    pass
                
                # 5. Wiberg Bond Indices (from density matrix in NAO basis, approximated)
                try:
                    wiberg_indices = mayer_bond_orders.copy()  # Approximation
                except:
                    pass
                
                # ============================================================
                # ORBITAL ANALYSIS
                # ============================================================
                
                # Overlap matrix eigenvalues (linear dependency check)
                S = mol.intor('int1e_ovlp')
                overlap_eigenvalues = np.linalg.eigvalsh(S)
                smallest_overlap_ev = overlap_eigenvalues.min()
                
                # Spin expectation values (for open-shell)
                try:
                    if hasattr(mf, 'spin_square'):
                        ss, mult = mf.spin_square()
                        s_squared = ss
                        s_expected = (mult - 1) / 2
                except:
                    pass
                
                # AO labels
                try:
                    ao_labels = mol.ao_labels()
                except:
                    ao_labels = []
                
                # Electron density at nuclei
                try:
                    for ia in range(mol.natm):
                        coord = mol.atom_coord(ia)
                        ao_at_nuc = mol.eval_gto('GTOval', np.array([coord]))
                        electron_density_at_nuclei[ia] = np.einsum('i,ij,j->', ao_at_nuc[0], dm, ao_at_nuc[0])
                except:
                    pass
                
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Some PySCF data extraction failed: {e}")
            
            # ============================================================
            # CONCEPTUAL DFT REACTIVITY DESCRIPTORS
            # ============================================================
            
            # Initialize conceptual DFT variables
            homo_energy_eV = 0.0
            lumo_energy_eV = 0.0
            ionization_potential = 0.0
            electron_affinity = 0.0
            chemical_hardness = 0.0
            chemical_softness = 0.0
            electronegativity = 0.0
            chemical_potential = 0.0
            electrophilicity_index = 0.0
            nucleophilicity_index = 0.0
            
            try:
                if lumo_idx is not None:
                    homo_energy_eV = mo_energy[homo_idx] * 27.2114
                    lumo_energy_eV = mo_energy[lumo_idx] * 27.2114
                    
                    # Koopmans' theorem approximation
                    ionization_potential = -homo_energy_eV
                    electron_affinity = -lumo_energy_eV
                    
                    # Global reactivity descriptors
                    if ionization_potential - electron_affinity > 0:
                        chemical_hardness = (ionization_potential - electron_affinity) / 2
                        chemical_softness = 1 / (2 * chemical_hardness) if chemical_hardness > 0 else 0
                        electronegativity = (ionization_potential + electron_affinity) / 2
                        chemical_potential = -electronegativity
                        electrophilicity_index = electronegativity**2 / (2 * chemical_hardness) if chemical_hardness > 0 else 0
                        nucleophilicity_index = 1 / electrophilicity_index if electrophilicity_index > 0 else 0
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Conceptual DFT calculation failed: {e}")
            
            # ============================================================
            # FUKUI FUNCTIONS (Local Reactivity Descriptors)
            # ============================================================
            
            fukui_plus = np.zeros(len(elements))   # Nucleophilic attack sites
            fukui_minus = np.zeros(len(elements))  # Electrophilic attack sites (CYP450 oxidation)
            fukui_radical = np.zeros(len(elements))  # Radical attack sites
            local_electrophilicity = np.zeros(len(elements))
            local_nucleophilicity = np.zeros(len(elements))
            
            try:
                # Get Mulliken charges for neutral system
                S = mol.intor('int1e_ovlp')
                dm_neutral = mf.make_rdm1()
                mulliken_neutral = mf.mulliken_pop(verbose=0)[1]
                
                # N+1 system (anion) for f+
                try:
                    mol_anion = gto.M(
                        atom=atom_str,
                        charge=charge - 1,
                        spin=1,  # Doublet
                        unit='Angstrom',
                        max_memory=self.max_memory,
                        verbose=0
                    )
                    mol_anion.build()
                    mf_anion = dft.UKS(mol_anion)
                    mf_anion.xc = self.functional
                    mf_anion = self._apply_scf_settings(mf_anion)
                    mf_anion.max_cycle = 50  # Limit cycles for speed
                    mf_anion.kernel()
                    
                    if mf_anion.converged:
                        mulliken_anion = mf_anion.mulliken_pop(verbose=0)[1]
                        # f+ = q(N+1) - q(N) = population increase upon electron addition
                        # Using charges: f+ = charge(N) - charge(N+1)
                        fukui_plus = mulliken_neutral - mulliken_anion
                except Exception as e:
                    if self.verbose:
                        print(f"   Note: N+1 Fukui calculation skipped: {e}")
                
                # N-1 system (cation) for f-
                try:
                    mol_cation = gto.M(
                        atom=atom_str,
                        charge=charge + 1,
                        spin=1,  # Doublet
                        unit='Angstrom',
                        max_memory=self.max_memory,
                        verbose=0
                    )
                    mol_cation.build()
                    mf_cation = dft.UKS(mol_cation)
                    mf_cation.xc = self.functional
                    mf_cation = self._apply_scf_settings(mf_cation)
                    mf_cation.max_cycle = 50
                    mf_cation.kernel()
                    
                    if mf_cation.converged:
                        mulliken_cation = mf_cation.mulliken_pop(verbose=0)[1]
                        # f- = q(N) - q(N-1) = population decrease upon electron removal
                        fukui_minus = mulliken_cation - mulliken_neutral
                except Exception as e:
                    if self.verbose:
                        print(f"   Note: N-1 Fukui calculation skipped: {e}")
                
                # Radical Fukui function (average)
                fukui_radical = (fukui_plus + fukui_minus) / 2
                
                # Local electrophilicity and nucleophilicity
                local_electrophilicity = electrophilicity_index * fukui_plus
                local_nucleophilicity = fukui_minus / electrophilicity_index if electrophilicity_index > 0 else fukui_minus
                
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Fukui function calculation failed: {e}")
            
            # ============================================================
            # MOLECULAR PROPERTIES (Volume, PSA, Polarizability)
            # ============================================================
            
            polarizability = 0.0
            molecular_volume = 0.0
            polar_surface_area = 0.0
            
            try:
                # Isotropic polarizability approximation from <r^2>
                r2_integrals = mol.intor('int1e_r2')
                dm_diag = np.diag(dm) if dm.ndim == 2 else dm
                polarizability = np.einsum('ij,ji->', r2_integrals, dm).real / 3.0
                
                # Molecular Volume (from van der Waals radii)
                vdw_radii = {
                    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
                    'S': 1.80, 'P': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
                }
                volume_spheres = 0.0
                for el in elements:
                    r = vdw_radii.get(el, 1.70)  # Default to C radius
                    volume_spheres += (4/3) * np.pi * r**3
                # Apply overlap correction factor (~0.6 for drug-like molecules)
                molecular_volume = volume_spheres * 0.6
                
                # Polar Surface Area (approximate from charges and atom types)
                psa_contribution = {
                    'O': 20.23, 'N': 26.03, 'S': 0.0, 'P': 0.0
                }
                polar_surface_area = 0.0
                for i, el in enumerate(elements):
                    if el in ['O', 'N']:
                        # Base PSA contribution
                        polar_surface_area += psa_contribution.get(el, 0)
                        # Add hydrogen contributions (if attached)
                        for j, el2 in enumerate(elements):
                            if el2 == 'H' and i != j:
                                bo = mayer_bond_orders[i, j] if mayer_bond_orders is not None else 0
                                if bo > 0.5:
                                    polar_surface_area += 9.23  # H attached to polar atom
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Molecular properties calculation failed: {e}")
            
            # ============================================================
            # ELECTROSTATIC POTENTIAL SURFACE DATA
            # ============================================================
            
            esp_min = 0.0
            esp_max = 0.0
            esp_positive_avg = 0.0
            esp_negative_avg = 0.0
            esp_variance = 0.0
            esp_at_nuclei = np.zeros(len(elements))
            
            try:
                # Calculate ESP at atomic positions
                from pyscf.dft import numint
                
                atom_coords = mol.atom_coords()
                for i in range(mol.natm):
                    coord = atom_coords[i:i+1, :]
                    
                    # Nuclear contribution
                    nuc_esp = 0.0
                    for j in range(mol.natm):
                        if i != j:
                            rij = np.linalg.norm(coord[0] - atom_coords[j])
                            if rij > 0.1:
                                nuc_esp += mol.atom_charge(j) / rij
                    
                    # Electronic contribution (approximate using AO values)
                    ao_at_point = mol.eval_gto('GTOval', coord)
                    elec_esp = -np.einsum('i,ij,j->', ao_at_point[0], dm, ao_at_point[0])
                    
                    esp_at_nuclei[i] = nuc_esp + elec_esp
                
                # ESP statistics
                esp_min = np.min(esp_at_nuclei)
                esp_max = np.max(esp_at_nuclei)
                positive_esp = esp_at_nuclei[esp_at_nuclei > 0]
                negative_esp = esp_at_nuclei[esp_at_nuclei < 0]
                esp_positive_avg = np.mean(positive_esp) if len(positive_esp) > 0 else 0
                esp_negative_avg = np.mean(negative_esp) if len(negative_esp) > 0 else 0
                esp_variance = np.var(esp_at_nuclei)
                
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: ESP calculation failed: {e}")
            
            # ============================================================
            # SAVE ALL EXTRACTED DATA
            # ============================================================
            np.savez(
                npz_file,
                # Basic wavefunction data
                mo_coeff=mf.mo_coeff,
                mo_energy=mf.mo_energy,
                mo_occ=mf.mo_occ,
                elements=elements,
                coords=coords,
                energy=mf.e_tot,
                converged=mf.converged,
                homo_lumo_gap=homo_lumo_gap,
                homo_idx=homo_idx,
                lumo_idx=lumo_idx if lumo_idx else -1,
                
                # Energy decomposition
                nuclear_repulsion=nuclear_repulsion,
                one_electron_energy=one_electron_energy,
                two_electron_energy=two_electron_energy,
                xc_energy=xc_energy,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                
                # Multipole moments
                dipole=dipole,
                dipole_magnitude=dipole_magnitude,
                quadrupole=quadrupole,
                
                # Charge analyses
                mulliken_charges=mulliken_charges,
                mulliken_pop=mulliken_pop_per_atom,
                spin_densities=spin_densities, 
                lowdin_charges=lowdin_charges,
                hirshfeld_charges=hirshfeld_charges,
                
                # Bond orders
                mayer_bond_orders=mayer_bond_orders,
                wiberg_indices=wiberg_indices,
                
                # Orbital/basis info
                n_ao=n_ao,
                n_mo=n_mo,
                n_electrons=n_electrons_actual,
                smallest_overlap_ev=smallest_overlap_ev,
                s_squared=s_squared,
                s_expected=s_expected,
                electron_density_at_nuclei=electron_density_at_nuclei,
                
                # Calculation info
                functional=self.functional,
                basis_set=self.basis,
                charge=charge,
                multiplicity=multiplicity,
                
                # ============================================================
                # NEW: Conceptual DFT Reactivity Descriptors
                # ============================================================
                ionization_potential=ionization_potential,
                electron_affinity=electron_affinity,
                chemical_hardness=chemical_hardness,
                chemical_softness=chemical_softness,
                electronegativity=electronegativity,
                chemical_potential=chemical_potential,
                electrophilicity_index=electrophilicity_index,
                nucleophilicity_index=nucleophilicity_index,
                
                # ============================================================
                # NEW: Fukui Functions (Local Reactivity)
                # ============================================================
                fukui_plus=fukui_plus,        # Nucleophilic attack sites
                fukui_minus=fukui_minus,      # Electrophilic/CYP450 oxidation sites
                fukui_radical=fukui_radical,  # Radical attack sites
                local_electrophilicity=local_electrophilicity,
                local_nucleophilicity=local_nucleophilicity,
                
                # ============================================================
                # NEW: Molecular Properties
                # ============================================================
                polarizability=polarizability,
                molecular_volume=molecular_volume,
                polar_surface_area=polar_surface_area,
                
                # ============================================================
                # NEW: Electrostatic Potential Data
                # ============================================================
                esp_at_nuclei=esp_at_nuclei,
                esp_min=esp_min,
                esp_max=esp_max,
                esp_positive_avg=esp_positive_avg,
                esp_negative_avg=esp_negative_avg,
                esp_variance=esp_variance
            )
            
            # Generate Cube files for HOMO/LUMO
            try:
                from pyscf import tools
                
                # Identify HOMO/LUMO indices
                mo_occ = mf.mo_occ
                if multiplicity == 1:
                    homo_indices = np.where(mo_occ > 0)[0]
                    if len(homo_indices) > 0:
                        homo_idx = homo_indices[-1]
                        lumo_idx = homo_idx + 1 if homo_idx + 1 < len(mf.mo_energy) else None
                        
                        # Save HOMO
                        tools.cubegen.orbital(
                            mol, 
                            str(wfn_dir / f"{file_name}_HOMO.cube"), 
                            mf.mo_coeff[:, homo_idx]
                        )
                        
                        # Save LUMO
                        if lumo_idx is not None:
                            tools.cubegen.orbital(
                                mol, 
                                str(wfn_dir / f"{file_name}_LUMO.cube"), 
                                mf.mo_coeff[:, lumo_idx]
                            )
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Could not generate cube files: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error saving wavefunction: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_hessian(self, coords, elements, charge=0, multiplicity=1):
        """
        Calculate the Hessian matrix (second derivatives).
        
        Returns:
        --------
        np.ndarray : Hessian matrix (3N x 3N)
        """
        try:
            # Build geometry
            atom_str = self._build_geometry_string(coords, elements)
            
            mol = gto.M(
                atom=atom_str,
                charge=charge,
                spin=multiplicity - 1,
                unit='Angstrom',
                max_memory=self.max_memory,
                verbose=0
            )
            mol.build()
            
            # DFT calculation
            if multiplicity == 1:
                mf = dft.RKS(mol)
            else:
                mf = dft.UKS(mol)
            
            mf.xc = self.functional
            mf = self._apply_scf_settings(mf)
            mf.kernel()
            
            # Hessian calculation
            from pyscf import hessian
            hess = hessian.RKS(mf) if multiplicity == 1 else hessian.UKS(mf)
            hessian_matrix = hess.kernel()
            
            return hessian_matrix
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error in Hessian calculation: {e}")
            return None