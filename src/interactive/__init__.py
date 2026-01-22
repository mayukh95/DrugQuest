"""
Interactive widgets for Jupyter notebook interface.
"""

from .property_filter import InteractivePropertyFilter
from .dft_control_panel import DFTControlPanel  
from .progress_monitor import LiveProgressMonitor, LiveProgressTracker, export_molecules_to_sdf
from .molecule_explorer import MoleculeExplorer
from .molecule_comparator import MoleculeComparator
from .py3dmol_viewer import InteractivePy3DmolViewer
from .batch_dft_panel import (
    BatchDFTControlPanel,
    OptimizationTracker,
    ResourceManager,
    ParallelBatchOptimizer
)
from .molecular_editor import MolecularEditor
from .trajectory_viewer import (
    TrajectoryViewer,
    load_xyz_trajectory,
    analyze_trajectory,
    plot_trajectory_analysis
)
from .wavefunction_viewer import WavefunctionVisualizer
from .pharmacophore_viewer import (
    create_pharmacophore_explorer, 
    compare_pharmacophores_grid,
    visualize_3d_pharmacophore
)

__all__ = [
    # Property & Comparison
    'InteractivePropertyFilter',
    'MoleculeComparator',
    'MoleculeExplorer',
    
    # Visualization
    'InteractivePy3DmolViewer',
    'TrajectoryViewer',
    'WavefunctionVisualizer',
    'create_pharmacophore_explorer',
    'compare_pharmacophores_grid',
    'visualize_3d_pharmacophore',
    
    # Editing
    'MolecularEditor',
    
    # DFT Optimization
    'DFTControlPanel',
    'BatchDFTControlPanel',
    'OptimizationTracker',
    'ResourceManager',
    'ParallelBatchOptimizer',
    
    # Progress & Export
    'LiveProgressMonitor',
    'LiveProgressTracker',
    'export_molecules_to_sdf',
    
    # Helper functions
    'load_xyz_trajectory',
    'analyze_trajectory',
    'plot_trajectory_analysis'
]