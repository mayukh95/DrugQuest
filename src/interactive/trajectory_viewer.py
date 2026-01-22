"""
Interactive Trajectory Viewer - Load and visualize geometry optimization trajectories.
"""

import numpy as np
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_xyz_trajectory(xyz_file):
    """
    Load XYZ trajectory file and return frames.
    
    Parameters:
    -----------
    xyz_file : str or Path
        Path to XYZ trajectory file
    
    Returns:
    --------
    list : List of frame dictionaries with 'elements', 'coords', 'info'
    """
    frames = []
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Read number of atoms
        try:
            n_atoms = int(lines[i].strip())
        except:
            break
        
        # Read comment line (contains energy and force info)
        comment = lines[i+1].strip()
        
        # Read coordinates
        coords = []
        elements = []
        for j in range(n_atoms):
            parts = lines[i+2+j].split()
            elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        frames.append({
            'elements': elements,
            'coords': np.array(coords),
            'info': comment
        })
        
        i += n_atoms + 2
    
    return frames


def analyze_trajectory(frames):
    """
    Analyze trajectory and extract energies, forces, RMSDs.
    
    Parameters:
    -----------
    frames : list
        List of frame dictionaries from load_xyz_trajectory
    
    Returns:
    --------
    dict : Dictionary with 'energies', 'force_norms', 'rmsds' arrays
    """
    energies = []
    force_norms = []
    rmsds = []
    
    ref_coords = frames[0]['coords']
    
    for frame in frames:
        # Extract energy and force from info string
        info = frame['info']
        try:
            # Parse "Step X: E=... Ha, F=... Ha/Bohr"
            e_part = info.split('E=')[1].split(' ')[0]
            f_part = info.split('F=')[1].split(' ')[0]
            energies.append(float(e_part))
            force_norms.append(float(f_part))
        except:
            energies.append(0.0)
            force_norms.append(0.0)
        
        # Calculate RMSD from initial structure
        coords = frame['coords']
        rmsd = np.sqrt(np.mean((coords - ref_coords)**2))
        rmsds.append(rmsd)
    
    return {
        'energies': np.array(energies),
        'force_norms': np.array(force_norms),
        'rmsds': np.array(rmsds)
    }


def plot_trajectory_analysis(analysis, title="Optimization Trajectory"):
    """
    Plot energy, force, and RMSD evolution.
    
    Parameters:
    -----------
    analysis : dict
        Analysis dictionary from analyze_trajectory
    title : str
        Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = np.arange(len(analysis['energies']))
    
    # Energy plot
    axes[0, 0].plot(steps, analysis['energies'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Energy (Hartree)', fontsize=12)
    axes[0, 0].set_title('Energy Convergence', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axhline(y=analysis['energies'][-1], color='green', 
                       linestyle='--', alpha=0.7, label='Final')
    axes[0, 0].legend()
    
    # Energy change plot
    if len(analysis['energies']) > 1:
        energy_change = analysis['energies'] - analysis['energies'][0]
        axes[0, 1].plot(steps, energy_change * 627.509, 'r-', linewidth=2, marker='o', markersize=4)
        axes[0, 1].set_xlabel('Step', fontsize=12)
        axes[0, 1].set_ylabel('Energy Change (kcal/mol)', fontsize=12)
        axes[0, 1].set_title('Energy Lowering', fontsize=13, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Force plot (log scale)
    axes[1, 0].semilogy(steps, analysis['force_norms'], 'g-', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Force Norm (Ha/Bohr)', fontsize=12)
    axes[1, 0].set_title('Force Convergence', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, which='both')
    
    # RMSD plot
    axes[1, 1].plot(steps, analysis['rmsds'], 'm-', linewidth=2, marker='o', markersize=4)
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('RMSD from Initial (Å)', fontsize=12)
    axes[1, 1].set_title('Structural Change', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


class TrajectoryViewer:
    """
    Interactive Trajectory Viewer with py3Dmol visualization.
    
    Features:
    - Load XYZ trajectory files
    - Frame-by-frame visualization with slider
    - Multiple display styles (stick, sphere, line)
    - Trajectory analysis (energy, force, RMSD)
    - Directory scanning for trajectory files
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the trajectory viewer.
        
        Parameters:
        -----------
        base_dir : str or Path, optional
            Base directory for scanning trajectory files
        """
        self.base_dir = Path(base_dir) if base_dir else Path('.')
        self.loaded_trajectory = {}
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Create viewer interface widgets."""
        # Output areas
        self.traj_output = widgets.Output()
        self.traj_viewer_output = widgets.Output()
        
        # Directory input
        self.dir_input = widgets.Text(
            value=str(self.base_dir),
            placeholder='Enter directory path containing trajectory files',
            description='Directory:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        # Scan button
        self.scan_button = widgets.Button(
            description='🔍 Scan Directory',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.scan_button.on_click(self._scan_directory)
        
        # Trajectory file selector
        self.traj_file_select = widgets.Dropdown(
            options=[],
            description='Trajectory:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        # Load button
        self.load_traj_button = widgets.Button(
            description='📂 Load Trajectory',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.load_traj_button.on_click(self._on_load_traj)
        
        # Frame slider
        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=10, step=1,
            description='Frame:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.frame_slider.observe(self._on_frame_change, names='value')
        
        # Style selector
        self.style_select = widgets.Dropdown(
            options=['stick', 'sphere', 'line', 'ball-stick'],
            value='stick',
            description='Style:',
            layout=widgets.Layout(width='200px')
        )
        self.style_select.observe(self._on_style_change, names='value')
        
        # Plot analysis button
        self.plot_btn = widgets.Button(
            description='📊 Plot Analysis',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.plot_btn.on_click(self._plot_analysis)
    
    def _scan_directory(self, b=None):
        """Scan directory for trajectory files."""
        with self.traj_output:
            clear_output(wait=True)
            output_dir = Path(self.dir_input.value)
            if output_dir.exists():
                # Use rglob to find files in subdirectories
                traj_files = sorted(output_dir.rglob('*_trajectory.xyz'))
                traj_file_options = [str(f) for f in traj_files]
                self.traj_file_select.options = traj_file_options
                if traj_file_options:
                    print(f"✅ Found {len(traj_file_options)} trajectory file(s) in {output_dir}")
                else:
                    print(f"⚠️ No trajectory files (*_trajectory.xyz) found in {output_dir}")
                    print("   (Looked recursively in subfolders)")
            else:
                print(f"❌ Directory not found: {output_dir}")
                self.traj_file_select.options = []
    
    def load_trajectory(self, file_path):
        """
        Load trajectory from file.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to XYZ trajectory file
        """
        file_path = Path(file_path)
        
        with self.traj_output:
            clear_output(wait=True)
            
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return
            
            print(f"📂 Loading trajectory: {file_path.name}")
            frames = load_xyz_trajectory(file_path)
            
            if frames:
                self.loaded_trajectory = {
                    'frames': frames,
                    'file': str(file_path),
                    'n_frames': len(frames)
                }
                print(f"✅ Loaded {len(frames)} frames")
                print(f"   Atoms: {len(frames[0]['elements'])}")
                
                # Analyze trajectory
                analysis = analyze_trajectory(frames)
                self.loaded_trajectory['analysis'] = analysis
                
                print(f"\n📊 Trajectory Statistics:")
                print(f"   Initial Energy: {analysis['energies'][0]:.8f} Ha")
                print(f"   Final Energy:   {analysis['energies'][-1]:.8f} Ha")
                print(f"   Energy Change:  {(analysis['energies'][-1] - analysis['energies'][0])*627.509:.4f} kcal/mol")
                print(f"   Final Force:    {analysis['force_norms'][-1]:.6f} Ha/Bohr")
                print(f"   Final RMSD:     {analysis['rmsds'][-1]:.4f} Å")
                
                # Update slider
                self.frame_slider.max = len(frames) - 1
                self.frame_slider.value = 0
            else:
                print("❌ No frames loaded")
    
    def _on_load_traj(self, b):
        """Handle load trajectory button click."""
        if self.traj_file_select.value:
            self.load_trajectory(self.traj_file_select.value)
            self._show_frame(0)
    
    def _on_frame_change(self, change):
        """Handle frame slider change."""
        self._show_frame(change['new'])
    
    def _on_style_change(self, change):
        """Handle style change."""
        if self.loaded_trajectory.get('frames'):
            self._show_frame(self.frame_slider.value)
    
    def _show_frame(self, frame_idx):
        """Display specific frame in 3D viewer."""
        with self.traj_viewer_output:
            clear_output(wait=True)
            
            if 'frames' not in self.loaded_trajectory:
                print("⚠️ No trajectory loaded. Please load a trajectory file first.")
                return
            
            frames = self.loaded_trajectory['frames']
            if frame_idx >= len(frames):
                return
            
            frame = frames[frame_idx]
            style = self.style_select.value
            
            # Create XYZ string
            xyz_str = f"{len(frame['elements'])}\n{frame['info']}\n"
            for el, coord in zip(frame['elements'], frame['coords']):
                xyz_str += f"{el} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
            
            # Display info
            print(f"Frame {frame_idx}/{len(frames)-1}: {frame['info']}")
            
            if not PY3DMOL_AVAILABLE:
                print("py3Dmol not available for 3D visualization")
                return
            
            # Create viewer
            viewer = py3Dmol.view(width=700, height=500)
            viewer.addModel(xyz_str, "xyz")
            
            # Style options
            if style == 'stick':
                viewer.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
            elif style == 'sphere':
                viewer.setStyle({'sphere': {'scale': 0.4}})
            elif style == 'line':
                viewer.setStyle({'line': {}})
            else:  # ball-stick
                viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            
            viewer.setBackgroundColor('#f5f5f5')
            viewer.zoomTo()
            viewer.show()
    
    def _plot_analysis(self, b=None):
        """Plot trajectory analysis."""
        if 'analysis' in self.loaded_trajectory:
            plot_trajectory_analysis(
                self.loaded_trajectory['analysis'],
                title=f"Trajectory Analysis: {Path(self.loaded_trajectory.get('file', 'Unknown')).name}"
            )
        else:
            with self.traj_output:
                print("⚠️ No trajectory loaded. Please load a trajectory first.")
    
    def get_trajectory_data(self):
        """Get the loaded trajectory data."""
        return self.loaded_trajectory
    
    def display(self):
        """Display the trajectory viewer interface."""
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
            <h3 style='color: white; margin: 0;'>📹 Optimization Trajectory Viewer</h3>
            <p style='color: #f0f0f0; margin: 5px 0 0 0; font-size: 13px;'>
                Load and visualize geometry optimization trajectories
            </p>
        </div>
        """)
        
        controls_row1 = widgets.HBox([
            self.dir_input, 
            self.scan_button
        ])
        
        controls_row2 = widgets.HBox([
            self.traj_file_select, 
            self.load_traj_button
        ])
        
        controls_row3 = widgets.HBox([
            self.frame_slider,
            self.style_select,
            self.plot_btn
        ])
        
        interface = widgets.VBox([
            header,
            controls_row1,
            self.traj_output,
            controls_row2,
            controls_row3,
            self.traj_viewer_output
        ])
        
        return interface
