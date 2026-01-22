"""
MolecularDFT - Advanced Drug Discovery & Quantum Chemistry Platform

A comprehensive molecular analysis and DFT optimization package.

This package provides tools for:
- Molecular data handling and ADMET property prediction
- Quantum chemistry DFT calculations and geometry optimization  
- Interactive visualization and analysis widgets
- Parallel optimization and resource management
- Data import/export and utility functions

Main modules:
- core: Fundamental data structures and ADMET prediction
- quantum: DFT calculations and geometry optimization
- visualization: 3D molecular viewers and plotting tools
- interactive: Jupyter widgets for interactive analysis
- analysis: Trajectory and orbital analysis tools
- optimization: Parallel processing and resource management
- utils: File I/O and helper functions

Author: Mayukh
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Mayukh"
__license__ = "MIT"

# Import main classes and functions
from .core import DrugMolecule, Config, ADMETPredictor
from .quantum import EnhancedDFTCalculator, NewtonRaphsonOptimizer, MolecularOrbitalAnalyzer
from .visualization import MolecularViewer3D, PropertyPlotter, TrajectoryPlotter
from .interactive import (
    InteractivePropertyFilter, 
    DFTControlPanel, 
    LiveProgressMonitor,
    LiveProgressTracker,
    InteractivePy3DmolViewer,
    BatchDFTControlPanel,
    OptimizationTracker,
    ParallelBatchOptimizer,
    MolecularEditor,
    MoleculeComparator,
    export_molecules_to_sdf,
    TrajectoryViewer,
    WavefunctionVisualizer,
    load_xyz_trajectory,
    analyze_trajectory,
    plot_trajectory_analysis
)
from .analysis import TrajectoryAnalyzer, OrbitalVisualizer, GeometryAnalyzer
from .optimization import BatchOptimizer, ResourceManager
from .utils import MoleculeLoader, FileExporter, calculate_molecular_descriptors

__version__ = "0.1.0"
__author__ = "Drug Discovery Pipeline Team"
__email__ = "contact@drugdiscovery.ai"

__all__ = [
    # Core classes
    'DrugMolecule',
    'Config', 
    'ADMETPredictor',
    
    # Quantum chemistry
    'EnhancedDFTCalculator',
    'NewtonRaphsonOptimizer',
    'MolecularOrbitalAnalyzer',
    
    # Visualization
    'MolecularViewer3D',
    'PropertyPlotter',
    'TrajectoryPlotter',
    
    # Interactive widgets
    'InteractivePropertyFilter',
    'DFTControlPanel',
    'BatchDFTControlPanel',
    'MolecularEditor',
    'MoleculeComparator',
    'LiveProgressMonitor',
    'LiveProgressTracker',
    'InteractivePy3DmolViewer',
    'TrajectoryViewer',
    'WavefunctionVisualizer',
    'export_molecules_to_sdf',
    'load_xyz_trajectory',
    'analyze_trajectory',
    'plot_trajectory_analysis',
    
    # Analysis tools
    'TrajectoryAnalyzer',
    'OrbitalVisualizer',
    'GeometryAnalyzer',
    
    # Optimization
    'BatchOptimizer',
    'ResourceManager',
    'OptimizationTracker',
    'ParallelBatchOptimizer',
    
    # Utilities
    'MoleculeLoader',
    'FileExporter',
    'calculate_molecular_descriptors'
]


def get_example_molecules():
    """Get a few example drug molecules for testing."""
    examples = [
        ("Aspirin", "50-78-2", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "58-08-2", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Paracetamol", "103-90-2", "CC(=O)NC1=CC=C(C=C1)O"),
        ("Ibuprofen", "15687-27-1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("Penicillin", "61-33-6", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C")
    ]
    
    molecules = []
    for name, cas, smiles in examples:
        try:
            mol = DrugMolecule(name=name, cas=cas, smiles=smiles)
            molecules.append(mol)
        except Exception as e:
            print(f"Warning: Could not create example molecule {name}: {e}")
    
    return molecules


def create_default_config():
    """Create a default configuration object."""
    return Config()


def run_example_analysis():
    """Run a simple example analysis to test the package."""
    print("Running example drug discovery analysis...")
    
    # Load example molecules
    molecules = get_example_molecules()
    print(f"Loaded {len(molecules)} example molecules")
    
    # Calculate ADMET properties
    admet_predictor = ADMETPredictor()
    
    print("\nADMET Analysis Results:")
    for mol in molecules:
        properties = admet_predictor.predict_properties(mol.mol)
        print(f"{mol.name}:")
        for prop, value in properties.items():
            if isinstance(value, bool):
                print(f"  {prop}: {'PASS' if value else 'FAIL'}")
            elif isinstance(value, (int, float)):
                print(f"  {prop}: {value:.2f}")
    
    # Generate molecular descriptors
    descriptors_df = calculate_molecular_descriptors(molecules)
    print(f"\nCalculated {len(descriptors_df.columns)} molecular descriptors")
    
    # Create visualization
    try:
        viewer = MolecularViewer3D()
        fig = viewer.plot_molecule(molecules[0])
        print("✓ 3D molecular visualization created successfully")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\n✓ Example analysis completed successfully!")
    return molecules, descriptors_df


# Package metadata
PACKAGE_INFO = {
    'name': 'drug-discovery-pipeline',
    'version': __version__,
    'description': 'Comprehensive molecular analysis and DFT optimization package',
    'author': __author__,
    'email': __email__,
    'url': 'https://github.com/drugdiscovery/pipeline',
    'license': 'MIT',
    'keywords': ['chemistry', 'drug-discovery', 'DFT', 'molecular-modeling'],
    'dependencies': [
        'rdkit-pypi>=2022.9.1',
        'pyscf>=2.1.0',
        'plotly>=5.11.0',
        'ipywidgets>=8.0.0',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scipy>=1.9.0',
        'psutil>=5.9.0'
    ]
}


def get_package_info():
    """Get package information."""
    return PACKAGE_INFO