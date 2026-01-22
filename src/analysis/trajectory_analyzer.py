"""
Trajectory analysis and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path


class TrajectoryAnalyzer:
    """Analyze and visualize optimization trajectories."""
    
    def __init__(self, trajectory_file=None):
        """Initialize trajectory analyzer."""
        self.trajectory_file = trajectory_file
        self.trajectory_data = None
        
        if trajectory_file:
            self.load_trajectory(trajectory_file)
    
    def load_trajectory(self, trajectory_file):
        """Load trajectory from XYZ file."""
        trajectory_file = Path(trajectory_file)
        
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
        
        coords_list = []
        energies = []
        force_norms = []
        elements = None
        
        with open(trajectory_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].strip() and lines[i].strip().isdigit():
                n_atoms = int(lines[i].strip())
                
                # Parse comment line for energy and force info
                comment = lines[i + 1].strip()
                energy = None
                force_norm = None
                
                if 'E=' in comment:
                    energy_str = comment.split('E=')[1].split()[0]
                    try:
                        energy = float(energy_str)
                        energies.append(energy)
                    except:
                        pass
                
                if 'F=' in comment:
                    force_str = comment.split('F=')[1].split()[0]
                    try:
                        force_norm = float(force_str)
                        force_norms.append(force_norm)
                    except:
                        pass
                
                # Parse coordinates
                coords = []
                frame_elements = []
                
                for j in range(i + 2, i + 2 + n_atoms):
                    parts = lines[j].split()
                    element = parts[0]
                    coord = [float(parts[1]), float(parts[2]), float(parts[3])]
                    
                    frame_elements.append(element)
                    coords.append(coord)
                
                if elements is None:
                    elements = frame_elements
                
                coords_list.append(np.array(coords))
                i += 2 + n_atoms
            else:
                i += 1
        
        self.trajectory_data = {
            'coords': coords_list,
            'elements': elements,
            'energies': energies,
            'force_norms': force_norms,
            'n_frames': len(coords_list)
        }
        
        print(f"✓ Loaded trajectory with {len(coords_list)} frames")
    
    def analyze_convergence(self):
        """Analyze optimization convergence."""
        if not self.trajectory_data or not self.trajectory_data['energies']:
            raise ValueError("No trajectory data loaded")
        
        energies = self.trajectory_data['energies']
        force_norms = self.trajectory_data.get('force_norms', [])
        
        analysis = {
            'total_steps': len(energies),
            'initial_energy': energies[0],
            'final_energy': energies[-1],
            'energy_change': energies[0] - energies[-1],
            'energy_convergence': None,
            'force_convergence': None
        }
        
        # Energy convergence analysis
        if len(energies) > 5:
            # Check if energy change in last 5 steps is small
            recent_energies = energies[-5:]
            energy_range = max(recent_energies) - min(recent_energies)
            analysis['energy_convergence'] = energy_range < 1e-6
        
        # Force convergence
        if force_norms:
            analysis['final_force_norm'] = force_norms[-1]
            analysis['force_convergence'] = force_norms[-1] < 0.0003  # Default threshold
        
        return analysis
    
    def calculate_rmsd(self, reference_frame=0):
        """Calculate RMSD relative to reference frame."""
        if not self.trajectory_data:
            raise ValueError("No trajectory data loaded")
        
        coords_list = self.trajectory_data['coords']
        reference_coords = coords_list[reference_frame]
        
        rmsds = []
        for coords in coords_list:
            # Simple RMSD calculation (no alignment)
            diff = coords - reference_coords
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            rmsds.append(rmsd)
        
        return rmsds
    
    def calculate_bond_evolution(self, atom1_idx, atom2_idx):
        """Calculate evolution of a specific bond distance."""
        if not self.trajectory_data:
            raise ValueError("No trajectory data loaded")
        
        coords_list = self.trajectory_data['coords']
        bond_distances = []
        
        for coords in coords_list:
            distance = np.linalg.norm(coords[atom1_idx] - coords[atom2_idx])
            bond_distances.append(distance)
        
        return bond_distances
    
    def find_transition_states(self, energy_threshold=0.001):
        """Find potential transition states (energy maxima)."""
        if not self.trajectory_data or not self.trajectory_data['energies']:
            raise ValueError("No energy data available")
        
        energies = np.array(self.trajectory_data['energies'])
        
        # Find local maxima
        transition_states = []
        for i in range(1, len(energies) - 1):
            if (energies[i] > energies[i-1] and 
                energies[i] > energies[i+1] and
                energies[i] - min(energies[i-1], energies[i+1]) > energy_threshold):
                
                transition_states.append({
                    'frame': i,
                    'energy': energies[i],
                    'coordinates': self.trajectory_data['coords'][i]
                })
        
        return transition_states
    
    def plot_energy_profile(self, title="Energy Profile"):
        """Plot energy vs. optimization step."""
        if not self.trajectory_data or not self.trajectory_data['energies']:
            raise ValueError("No energy data available")
        
        energies = self.trajectory_data['energies']
        steps = list(range(len(energies)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=energies,
            mode='lines+markers',
            name='Energy',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Mark initial and final points
        fig.add_trace(go.Scatter(
            x=[0],
            y=[energies[0]],
            mode='markers',
            name='Initial',
            marker=dict(color='green', size=10, symbol='star')
        ))
        
        fig.add_trace(go.Scatter(
            x=[len(energies)-1],
            y=[energies[-1]],
            mode='markers',
            name='Final',
            marker=dict(color='red', size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Optimization Step',
            yaxis_title='Energy (Ha)',
            width=700,
            height=400
        )
        
        return fig
    
    def plot_rmsd_profile(self, reference_frame=0, title="RMSD Profile"):
        """Plot RMSD vs. optimization step."""
        rmsds = self.calculate_rmsd(reference_frame)
        steps = list(range(len(rmsds)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=rmsds,
            mode='lines+markers',
            name='RMSD',
            line=dict(color='purple', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Optimization Step',
            yaxis_title='RMSD (Å)',
            width=700,
            height=400
        )
        
        return fig
    
    def create_animation(self, title="Optimization Trajectory"):
        """Create animated visualization of the trajectory."""
        if not self.trajectory_data:
            raise ValueError("No trajectory data loaded")
        
        from ..visualization.molecular_viewer import MolecularViewer3D
        
        viewer = MolecularViewer3D()
        coords_list = self.trajectory_data['coords']
        elements = self.trajectory_data['elements']
        
        fig = viewer.animate_trajectory(coords_list, elements, title)
        return fig
    
    def export_analysis_report(self, output_file=None):
        """Export detailed analysis report."""
        if output_file is None:
            from ..core.config import Config
            output_file = Config.OUTPUT_DIR / 'trajectory_analysis.txt'
        
        convergence = self.analyze_convergence()
        rmsds = self.calculate_rmsd()
        
        with open(output_file, 'w') as f:
            f.write("TRAJECTORY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total steps: {convergence['total_steps']}\n")
            f.write(f"Initial energy: {convergence['initial_energy']:.8f} Ha\n")
            f.write(f"Final energy: {convergence['final_energy']:.8f} Ha\n")
            f.write(f"Energy change: {convergence['energy_change']:.8f} Ha\n")
            
            if convergence.get('final_force_norm'):
                f.write(f"Final force norm: {convergence['final_force_norm']:.6f} Ha/Bohr\n")
            
            f.write(f"\nConvergence status:\n")
            f.write(f"  Energy: {'✓' if convergence.get('energy_convergence') else '✗'}\n")
            f.write(f"  Force: {'✓' if convergence.get('force_convergence') else '✗'}\n")
            
            f.write(f"\nRMSD statistics:\n")
            f.write(f"  Final RMSD: {rmsds[-1]:.4f} Å\n")
            f.write(f"  Max RMSD: {max(rmsds):.4f} Å\n")
            f.write(f"  Mean RMSD: {np.mean(rmsds):.4f} Å\n")
        
        print(f"✓ Analysis report saved to {output_file}")