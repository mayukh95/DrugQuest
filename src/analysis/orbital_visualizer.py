"""
Molecular Orbital Visualization Tools.
Provides energy diagrams, density plots, and 3D isosurface rendering.
"""

import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class OrbitalVisualizer:
    """
    Molecular Orbital Visualization class for analyzing DFT wavefunctions.
    Provides energy diagrams, density plots, and 3D isosurface rendering.
    """
    
    def __init__(self):
        self.loaded_data = None
        self.mol = None
        self.mo_coeff = None
        self.mo_energy = None
        self.mo_occ = None
        self.homo_idx = None
        self.lumo_idx = None
        
    def load_wavefunction(self, filepath):
        """
        Load wavefunction data from .npz file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the .npz wavefunction file
            
        Returns:
        --------
        bool : True if successful, False otherwise
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        try:
            data = np.load(filepath, allow_pickle=True)
            self.mo_coeff = data['mo_coeff']
            self.mo_energy = data['mo_energy']
            self.mo_occ = data['mo_occ']
            
            # Load optional fields
            self.loaded_data = {}
            if 'energy' in data:
                self.loaded_data['energy'] = float(data['energy'])
            if 'converged' in data:
                self.loaded_data['converged'] = bool(data['converged'])
            if 'homo_lumo_gap' in data and data['homo_lumo_gap'] is not None:
                self.loaded_data['homo_lumo_gap'] = float(data['homo_lumo_gap'])
            
            # Find HOMO/LUMO indices
            occ_indices = np.where(self.mo_occ > 0)[0]
            self.homo_idx = occ_indices[-1] if len(occ_indices) > 0 else 0
            self.lumo_idx = self.homo_idx + 1 if self.homo_idx + 1 < len(self.mo_energy) else self.homo_idx
            
            print(f"✓ Loaded wavefunction from {filepath.name}")
            if 'energy' in self.loaded_data:
                print(f"  Total energy: {self.loaded_data['energy']:.6f} Ha")
            if 'converged' in self.loaded_data:
                print(f"  Converged: {self.loaded_data['converged']}")
            print(f"  Number of MOs: {len(self.mo_energy)}")
            print(f"  HOMO index: {self.homo_idx} (E = {self.mo_energy[self.homo_idx]:.4f} Ha)")
            print(f"  LUMO index: {self.lumo_idx} (E = {self.mo_energy[self.lumo_idx]:.4f} Ha)")
            if 'homo_lumo_gap' in self.loaded_data:
                print(f"  HOMO-LUMO gap: {self.loaded_data['homo_lumo_gap']:.2f} eV")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading wavefunction: {e}")
            return False
    
    def plot_mo_energy_diagram(self, n_orbitals=10, figsize=(12, 8), save_path=None):
        """
        Plot molecular orbital energy level diagram.
        
        Parameters:
        -----------
        n_orbitals : int
            Number of orbitals to show around HOMO-LUMO
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ Matplotlib not available")
            return None
            
        if self.mo_energy is None:
            print("❌ No wavefunction loaded. Use load_wavefunction() first.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate range of orbitals to show
        n_below = n_orbitals // 2
        n_above = n_orbitals - n_below
        start_idx = max(0, self.homo_idx - n_below + 1)
        end_idx = min(len(self.mo_energy), self.lumo_idx + n_above)
        
        # Plot energy levels
        energies_eV = self.mo_energy[start_idx:end_idx] * 27.2114  # Convert to eV
        x_positions = np.arange(len(energies_eV))
        
        colors = []
        labels = []
        for i, idx in enumerate(range(start_idx, end_idx)):
            if idx == self.homo_idx:
                colors.append('red')
                labels.append('HOMO')
            elif idx == self.lumo_idx:
                colors.append('blue')
                labels.append('LUMO')
            elif self.mo_occ[idx] > 0:
                colors.append('darkred')
                labels.append(f'HOMO-{self.homo_idx - idx}' if idx < self.homo_idx else f'Occ {idx}')
            else:
                colors.append('darkblue')
                labels.append(f'LUMO+{idx - self.lumo_idx}' if idx > self.lumo_idx else f'Virt {idx}')
        
        # Draw energy levels as horizontal lines
        for i, (e, c) in enumerate(zip(energies_eV, colors)):
            ax.hlines(e, i - 0.3, i + 0.3, colors=c, linewidth=3)
            
            # Add occupation arrows for occupied orbitals
            if start_idx + i <= self.homo_idx:
                ax.annotate('↑↓', (i, e), ha='center', va='bottom', fontsize=10)
        
        # Highlight HOMO-LUMO gap
        homo_e = self.mo_energy[self.homo_idx] * 27.2114
        lumo_e = self.mo_energy[self.lumo_idx] * 27.2114
        gap = lumo_e - homo_e
        
        ax.annotate('', xy=(self.homo_idx - start_idx + 0.5, lumo_e), 
                   xytext=(self.homo_idx - start_idx + 0.5, homo_e),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(self.homo_idx - start_idx + 0.7, (homo_e + lumo_e)/2, 
               f'Gap: {gap:.2f} eV', fontsize=11, color='green')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title('Molecular Orbital Energy Diagram', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Vacuum level')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='darkred', linewidth=3, label='Occupied'),
            Line2D([0], [0], color='red', linewidth=3, label='HOMO'),
            Line2D([0], [0], color='blue', linewidth=3, label='LUMO'),
            Line2D([0], [0], color='darkblue', linewidth=3, label='Virtual')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=4, frameon=True, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved MO energy diagram: {save_path}")
        
        plt.show()
        return fig
    
    def plot_mo_coefficients(self, mo_indices=None, figsize=(14, 6), save_path=None):
        """
        Plot MO coefficient bar charts showing atomic orbital contributions.
        
        Parameters:
        -----------
        mo_indices : list, optional
            List of MO indices to plot. Default: HOMO-1, HOMO, LUMO, LUMO+1
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ Matplotlib not available")
            return None
            
        if self.mo_coeff is None:
            print("❌ No wavefunction loaded.")
            return None
        
        if mo_indices is None:
            mo_indices = [self.homo_idx - 1, self.homo_idx, self.lumo_idx, self.lumo_idx + 1]
            mo_indices = [i for i in mo_indices if 0 <= i < len(self.mo_energy)]
        
        n_mos = len(mo_indices)
        fig, axes = plt.subplots(1, n_mos, figsize=figsize, sharey=True)
        if n_mos == 1:
            axes = [axes]
        
        for ax, mo_idx in zip(axes, mo_indices):
            coeffs = self.mo_coeff[:, mo_idx]
            n_ao = len(coeffs)
            
            colors = ['red' if c > 0 else 'blue' for c in coeffs]
            ax.barh(range(n_ao), coeffs, color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linewidth=0.5)
            
            # Label
            if mo_idx == self.homo_idx:
                label = 'HOMO'
            elif mo_idx == self.lumo_idx:
                label = 'LUMO'
            elif mo_idx < self.homo_idx:
                label = f'HOMO-{self.homo_idx - mo_idx}'
            else:
                label = f'LUMO+{mo_idx - self.lumo_idx}'
            
            ax.set_title(f'{label}\nE = {self.mo_energy[mo_idx]*27.2114:.2f} eV')
            ax.set_xlabel('Coefficient')
            if ax == axes[0]:
                ax.set_ylabel('Atomic Orbital Index')
        
        plt.suptitle('MO Coefficient Distribution', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved MO coefficients plot: {save_path}")
        
        plt.show()
        return fig
    
    def read_cube_file(self, filepath):
        """
        Read a Gaussian cube file and return grid data.
        
        Returns:
        --------
        dict with keys: 'origin', 'axes', 'n_points', 'data', 'atoms'
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"❌ Cube file not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            # Skip comment lines
            f.readline()
            f.readline()
            
            # Read origin and number of atoms
            line = f.readline().split()
            n_atoms = int(line[0])
            origin = np.array([float(x) for x in line[1:4]])
            
            # Read grid dimensions and vectors
            n_points = []
            axes = []
            for _ in range(3):
                line = f.readline().split()
                n_points.append(int(line[0]))
                axes.append(np.array([float(x) for x in line[1:4]]))
            
            n_points = np.array(n_points)
            axes = np.array(axes)
            
            # Read atoms
            atoms = []
            for _ in range(abs(n_atoms)):
                line = f.readline().split()
                atoms.append({
                    'Z': int(line[0]),
                    'charge': float(line[1]),
                    'coords': np.array([float(x) for x in line[2:5]])
                })
            
            # Read volumetric data
            data = []
            for line in f:
                data.extend([float(x) for x in line.split()])
            
            data = np.array(data).reshape(n_points)
        
        return {
            'origin': origin,
            'axes': axes,
            'n_points': n_points,
            'data': data,
            'atoms': atoms
        }
    
    def plot_cube_slice(self, cube_data, plane='xy', slice_idx=None, 
                       figsize=(10, 8), cmap='RdBu_r', save_path=None):
        """
        Plot a 2D slice of cube file data.
        
        Parameters:
        -----------
        cube_data : dict
            Output from read_cube_file()
        plane : str
            Slice plane: 'xy', 'xz', or 'yz'
        slice_idx : int, optional
            Index of slice. Default: middle
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ Matplotlib not available")
            return None
            
        if cube_data is None:
            return None
        
        data = cube_data['data']
        n_points = cube_data['n_points']
        origin = cube_data['origin']
        axes = cube_data['axes']
        
        # Select slice
        if plane == 'xy':
            if slice_idx is None:
                slice_idx = n_points[2] // 2
            slice_data = data[:, :, slice_idx].T
            xlabel, ylabel = 'X (Bohr)', 'Y (Bohr)'
            extent = [origin[0], origin[0] + n_points[0]*axes[0,0],
                     origin[1], origin[1] + n_points[1]*axes[1,1]]
        elif plane == 'xz':
            if slice_idx is None:
                slice_idx = n_points[1] // 2
            slice_data = data[:, slice_idx, :].T
            xlabel, ylabel = 'X (Bohr)', 'Z (Bohr)'
            extent = [origin[0], origin[0] + n_points[0]*axes[0,0],
                     origin[2], origin[2] + n_points[2]*axes[2,2]]
        else:  # yz
            if slice_idx is None:
                slice_idx = n_points[0] // 2
            slice_data = data[slice_idx, :, :].T
            xlabel, ylabel = 'Y (Bohr)', 'Z (Bohr)'
            extent = [origin[1], origin[1] + n_points[1]*axes[1,1],
                     origin[2], origin[2] + n_points[2]*axes[2,2]]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Symmetric colormap limits
        vmax = np.abs(slice_data).max()
        vmin = -vmax
        
        im = ax.imshow(slice_data, extent=extent, origin='lower', 
                      cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # Plot atoms in the slice plane
        for atom in cube_data['atoms']:
            coords = atom['coords']
            ax.scatter(coords[0 if plane[0]=='x' else 1], 
                      coords[1 if plane[1]=='y' else 2],
                      s=100, c='black', marker='o', edgecolors='white')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'MO Density - {plane.upper()} Plane (slice {slice_idx})', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Orbital Amplitude')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved cube slice plot: {save_path}")
        
        plt.show()
        return fig
    
    def plot_cube_isosurface_3d(self, cube_data, isovalue=0.02, save_path=None):
        """
        Create 3D isosurface visualization using plotly.
        
        Parameters:
        -----------
        cube_data : dict
            Output from read_cube_file()
        isovalue : float
            Isosurface value (typically 0.01-0.05)
        """
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly not available for 3D visualization")
            return None
            
        if cube_data is None:
            return None
        
        data = cube_data['data']
        n_points = cube_data['n_points']
        origin = cube_data['origin']
        axes = cube_data['axes']
        
        # Create coordinate grids
        x = np.linspace(origin[0], origin[0] + n_points[0]*axes[0,0], n_points[0])
        y = np.linspace(origin[1], origin[1] + n_points[1]*axes[1,1], n_points[1])
        z = np.linspace(origin[2], origin[2] + n_points[2]*axes[2,2], n_points[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create figure
        fig = go.Figure()
        
        # Positive lobe (blue)
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=isovalue,
            isomax=isovalue,
            surface_count=1,
            colorscale=[[0, 'blue'], [1, 'blue']],
            opacity=0.6,
            showscale=False,
            name='Positive'
        ))
        
        # Negative lobe (red)
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=-isovalue,
            isomax=-isovalue,
            surface_count=1,
            colorscale=[[0, 'red'], [1, 'red']],
            opacity=0.6,
            showscale=False,
            name='Negative'
        ))
        
        # Add atoms as spheres
        atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'}
        atom_colors = {1: 'white', 6: 'gray', 7: 'blue', 8: 'red', 9: 'green', 16: 'yellow', 17: 'green'}
        
        for atom in cube_data['atoms']:
            Z_atom = atom['Z']
            coords = atom['coords']
            fig.add_trace(go.Scatter3d(
                x=[coords[0]], y=[coords[1]], z=[coords[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=atom_colors.get(Z_atom, 'gray'),
                    line=dict(color='black', width=1)
                ),
                name=atom_symbols.get(Z_atom, f'Z={Z_atom}'),
                showlegend=False
            ))
        
        fig.update_layout(
            title='3D Molecular Orbital Isosurface',
            scene=dict(
                xaxis_title='X (Bohr)',
                yaxis_title='Y (Bohr)',
                zaxis_title='Z (Bohr)',
                aspectmode='data'
            ),
            width=800,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Saved 3D visualization: {save_path}")
        
        fig.show()
        return fig
    
    def visualize_all_saved_mos(self, wavefunction_dir, molecule_name, plane='xy'):
        """
        Visualize all saved MO cube files for a molecule.
        
        Parameters:
        -----------
        wavefunction_dir : str or Path
            Directory containing wavefunction files
        molecule_name : str
            Prefix used when saving (e.g., 'aspirin')
        plane : str
            Slice plane for 2D visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ Matplotlib not available")
            return None
            
        wavefunction_dir = Path(wavefunction_dir)
        
        # Find all cube files for this molecule
        cube_files = sorted(wavefunction_dir.glob(f"{molecule_name}_*.cube"))
        
        if not cube_files:
            print(f"❌ No cube files found for {molecule_name} in {wavefunction_dir}")
            return None
        
        print(f"Found {len(cube_files)} cube files for {molecule_name}")
        
        # Create subplot grid
        n_files = len(cube_files)
        n_cols = min(3, n_files)
        n_rows = (n_files + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_files == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, cube_file in enumerate(cube_files):
            cube_data = self.read_cube_file(cube_file)
            if cube_data is None:
                continue
            
            data = cube_data['data']
            n_points = cube_data['n_points']
            
            # Get middle slice
            if plane == 'xy':
                slice_data = data[:, :, n_points[2]//2].T
            elif plane == 'xz':
                slice_data = data[:, n_points[1]//2, :].T
            else:
                slice_data = data[n_points[0]//2, :, :].T
            
            vmax = np.abs(slice_data).max()
            axes[i].imshow(slice_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, 
                          origin='lower', aspect='auto')
            
            # Extract MO label from filename
            mo_label = cube_file.stem.replace(f"{molecule_name}_", "")
            axes[i].set_title(mo_label, fontsize=12)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # Hide unused axes
        for i in range(len(cube_files), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Molecular Orbitals - {molecule_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return fig


# Backwards compatibility alias
MOVisualizer = OrbitalVisualizer