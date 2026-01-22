"""
Geometry optimization algorithms for molecular structures.
"""

import numpy as np
import time
from scipy.optimize import minimize


class ClassicalGeometryOptimizer:
    """Geometry optimizer using classical optimization algorithms."""
    
    def __init__(self, dft_calculator, method='L-BFGS-B'):
        """
        Initialize geometry optimizer.
        
        Parameters:
        -----------
        dft_calculator : EnhancedDFTCalculator
            DFT calculator instance
        method : str
            Optimization method ('L-BFGS-B', 'BFGS', 'CG', 'Newton-CG')
        """
        self.dft_calculator = dft_calculator
        self.method = method
        self.history = {
            'coords': [],
            'energies': [], 
            'forces': [],
            'force_norms': [],
            'times': [],
            'converged_scf': []
        }
    
    def objective_function(self, coords_flat, elements, charge, multiplicity, verbose=False):
        """Objective function for optimization."""
        coords = coords_flat.reshape(-1, 3)
        
        result = self.dft_calculator.calculate_energy_and_forces(
            coords, elements, charge, multiplicity, 
            step_number=len(self.history['energies'])
        )
        
        if result is None:
            return 1e10, np.zeros_like(coords_flat)
        
        energy = result['energy']
        forces = result['forces']
        force_norm = result['force_norm']
        converged = result['converged']
        
        # Get SCF iteration details from calculator
        scf_details = None
        if hasattr(self.dft_calculator, 'scf_history') and len(self.dft_calculator.scf_history) > 0:
            scf_details = self.dft_calculator.scf_history[-1]
        
        # Store history
        self.history['coords'].append(coords.copy())
        self.history['energies'].append(energy)
        self.history['forces'].append(forces.copy())
        self.history['force_norms'].append(force_norm)
        self.history['times'].append(time.time())
        self.history['converged_scf'].append(converged)
        
        # Store detailed SCF info
        if 'scf_details' not in self.history:
            self.history['scf_details'] = []
        self.history['scf_details'].append(scf_details)
        
        # Store additional properties
        if 'max_force' not in self.history:
            self.history['max_force'] = []
        self.history['max_force'].append(result.get('max_force', force_norm))
        
        if 'rms_force' not in self.history:
            self.history['rms_force'] = []
        self.history['rms_force'].append(result.get('rms_force', force_norm))
        
        if verbose:
            print(f"   Step {len(self.history['energies']):3d}: "
                  f"E = {energy:12.8f} Ha, |F| = {force_norm:.6f} Ha/Bohr")
        
        # Return energy and negative gradient (forces)
        return energy, -forces.flatten()
    
    def optimize(self, initial_coords, elements, charge=0, multiplicity=1, 
                 max_iter=100, force_tol=0.0003, verbose=True):
        """
        Optimize molecular geometry.
        
        Parameters:
        -----------
        initial_coords : np.ndarray
            Initial coordinates (N x 3)
        elements : list
            Element symbols
        charge : int
            Molecular charge
        multiplicity : int
            Spin multiplicity
        max_iter : int
            Maximum optimization steps
        force_tol : float
            Force convergence tolerance (Ha/Bohr)
        verbose : bool
            Print optimization progress
        
        Returns:
        --------
        tuple : (optimized_coords, history)
        """
        # Reset history
        self.history = {
            'coords': [],
            'energies': [],
            'forces': [],
            'force_norms': [],
            'times': [],
            'converged_scf': []
        }
        
        if verbose:
            print(f"🚀 Starting {self.method} optimization")
            print(f"   Max iterations: {max_iter}")
            print(f"   Force tolerance: {force_tol:.6f} Ha/Bohr")
            print(f"   {'Step':>4} {'Energy (Ha)':>15} {'|Force|':>12} {'Status':>10}")
            print("   " + "-" * 50)
        
        start_time = time.time()
        coords_flat = initial_coords.flatten()
        
        # Define objective with additional args
        def objective_wrapper(x):
            return self.objective_function(x, elements, charge, multiplicity, verbose=verbose)
        
        # Optimization options
        options = {
            'maxiter': max_iter,
            'gtol': force_tol,
            'ftol': 1e-12,
            'disp': verbose
        }
        
        try:
            # Run optimization
            result = minimize(
                fun=objective_wrapper,
                x0=coords_flat,
                method=self.method,
                jac=True,  # We return both energy and gradient
                options=options
            )
            
            # Get final coordinates
            final_coords = result.x.reshape(-1, 3)
            
            # Final evaluation to get final properties
            final_result = self.dft_calculator.calculate_energy_and_forces(
                final_coords, elements, charge, multiplicity
            )
            
            total_time = time.time() - start_time
            
            if verbose:
                print("   " + "-" * 50)
                if result.success:
                    print(f"✅ Optimization converged in {len(self.history['energies'])} steps")
                else:
                    print(f"⚠️  Optimization did not converge ({result.message})")
                print(f"   Total time: {total_time:.2f} seconds")
                if final_result:
                    print(f"   Final energy: {final_result['energy']:.8f} Ha")
                    print(f"   Final force norm: {final_result['force_norm']:.6f} Ha/Bohr")
            
            return final_coords, self.history
            
        except Exception as e:
            if verbose:
                print(f"❌ Optimization failed: {e}")
            return initial_coords, self.history


class NewtonRaphsonOptimizer:
    """Newton-Raphson optimizer using analytical Hessian."""
    
    def __init__(self, dft_calculator):
        """Initialize Newton-Raphson optimizer."""
        self.dft_calculator = dft_calculator
        self.history = {
            'coords': [],
            'energies': [],
            'forces': [], 
            'force_norms': [],
            'eigenvalues': [],
            'step_sizes': []
        }
    
    def optimize(self, initial_coords, elements, charge=0, multiplicity=1,
                 max_iter=50, force_tol=0.0003, step_size_limit=0.3, verbose=True):
        """
        Optimize geometry using Newton-Raphson method.
        
        Parameters:
        -----------
        initial_coords : np.ndarray
            Initial coordinates
        elements : list
            Element symbols
        charge : int
            Molecular charge
        multiplicity : int
            Spin multiplicity
        max_iter : int
            Maximum iterations
        force_tol : float
            Force convergence tolerance
        step_size_limit : float
            Maximum step size (Angstrom)
        verbose : bool
            Print progress
        
        Returns:
        --------
        tuple : (optimized_coords, history)
        """
        coords = initial_coords.copy()
        
        if verbose:
            print("🚀 Starting Newton-Raphson optimization")
            print(f"   {'Step':>4} {'Energy (Ha)':>15} {'|Force|':>12} {'Step Size':>10}")
            print("   " + "-" * 50)
        
        for step in range(max_iter):
            # Calculate energy and forces
            result = self.dft_calculator.calculate_energy_and_forces(
                coords, elements, charge, multiplicity, step_number=step
            )
            
            if result is None:
                if verbose:
                    print(f"   Step {step}: DFT calculation failed")
                break
            
            energy = result['energy']
            forces = result['forces']
            force_norm = result['force_norm']
            
            # Store history
            self.history['coords'].append(coords.copy())
            self.history['energies'].append(energy)
            self.history['forces'].append(forces.copy())
            self.history['force_norms'].append(force_norm)
            
            if verbose:
                print(f"   {step:4d} {energy:15.8f} {force_norm:12.6f}")
            
            # Check convergence
            if force_norm < force_tol:
                if verbose:
                    print(f"✅ Converged at step {step}")
                break
            
            # Calculate Hessian
            hessian = self.dft_calculator.calculate_hessian(
                coords, elements, charge, multiplicity
            )
            
            if hessian is None:
                if verbose:
                    print(f"   Step {step}: Hessian calculation failed")
                break
            
            # Newton-Raphson step: x_new = x_old - H^(-1) * grad
            # forces = -gradient, so step = H^(-1) * forces
            try:
                # Flatten coordinates and forces
                coords_flat = coords.flatten()
                forces_flat = forces.flatten()
                
                # Calculate eigenvalues for monitoring
                eigenvals = np.linalg.eigvals(hessian)
                self.history['eigenvalues'].append(eigenvals)
                
                # Solve H * step = forces
                step_vector = np.linalg.solve(hessian, forces_flat)
                
                # Limit step size
                step_norm = np.linalg.norm(step_vector)
                if step_norm > step_size_limit:
                    step_vector *= step_size_limit / step_norm
                
                self.history['step_sizes'].append(np.linalg.norm(step_vector))
                
                # Update coordinates
                coords_flat_new = coords_flat + step_vector
                coords = coords_flat_new.reshape(-1, 3)
                
                if verbose:
                    print(f"        Step size: {np.linalg.norm(step_vector):.6f}")
                
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"   Step {step}: Singular Hessian matrix")
                break
        
        return coords, self.history