"""
Configuration settings for the drug discovery pipeline.
"""

from pathlib import Path


class Config:
    """Configuration class storing all pipeline settings."""
    
    # ==============================================================================
    # File and Directory Settings
    # ==============================================================================
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    WAVEFUNCTION_DIR = OUTPUT_DIR / 'wavefunctions'
    TRAJECTORY_DIR = OUTPUT_DIR / 'trajectories'
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, OUTPUT_DIR, WAVEFUNCTION_DIR, TRAJECTORY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # ==============================================================================
    # ADMET Filter Thresholds (Lipinski-like)
    # ==============================================================================
    MW_MIN = 150          # Minimum molecular weight (Da)
    MW_MAX = 500          # Maximum molecular weight (Da)
    LOGP_MIN = -0.5       # Minimum LogP
    LOGP_MAX = 5.0        # Maximum LogP 
    TPSA_MAX = 140        # Maximum topological polar surface area (Ų)
    HBD_MAX = 5           # Maximum hydrogen bond donors
    HBA_MAX = 10          # Maximum hydrogen bond acceptors
    ROTATABLE_BONDS_MAX = 10  # Maximum rotatable bonds
    
    # ==============================================================================
    # DFT Calculation Settings
    # ==============================================================================
    DEFAULT_FUNCTIONAL = 'B3LYP'
    DEFAULT_BASIS = 'aug-cc-pVTZ'
    DEFAULT_SCF_CYCLES = 150
    DEFAULT_CHARGE = 0
    DEFAULT_MULTIPLICITY = 1
    DEFAULT_FORCE_TOL = 0.0003     # Ha/Bohr
    DEFAULT_MAX_GEOM_STEPS = 100
    DEFAULT_MAX_MEMORY = 4000      # MB
    SAVE_WAVEFUNCTION = True
    SAVE_CUBE_FILES = False        # MO cube files (large)
    NUM_MO_TO_SAVE = 6            # Number of MOs around HOMO-LUMO
    MO_CUBE_RESOLUTION = 0.1      # Bohr
    
    # ==============================================================================
    # Molecular Docking Settings
    # ==============================================================================
    DEFAULT_PROTEIN_PDBQT = str(DATA_DIR / 'protein.pdbqt')
    DOCKING_EXHAUSTIVENESS = 8
    DOCKING_NUM_MODES = 9
    
    # ==============================================================================
    # Visualization Settings
    # ==============================================================================
    PLOT_DPI = 150
    FIGURE_WIDTH = 12
    FIGURE_HEIGHT = 8
    COLOR_PALETTE = 'viridis'
    
    # ==============================================================================
    # Interactive Widget Settings
    # ==============================================================================
    WIDGET_LAYOUT_WIDTH = '400px'
    BUTTON_WIDTH = '150px'
    
    # ==============================================================================
    # Parallel Processing Settings
    # ==============================================================================
    DEFAULT_CPUS_PER_JOB = 2
    DEFAULT_MEMORY_PER_JOB_GB = 8.0
    MAX_PARALLEL_JOBS = 3
    
    @classmethod
    def update_paths(cls, base_dir: str):
        """Update all paths when base directory changes."""
        cls.BASE_DIR = Path(base_dir)
        cls.DATA_DIR = cls.BASE_DIR / 'data'
        cls.OUTPUT_DIR = cls.BASE_DIR / 'outputs'
        cls.WAVEFUNCTION_DIR = cls.OUTPUT_DIR / 'wavefunctions'
        cls.TRAJECTORY_DIR = cls.OUTPUT_DIR / 'trajectories'
        
        # Create directories
        for directory in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.WAVEFUNCTION_DIR, cls.TRAJECTORY_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_settings_dict(cls):
        """Return current settings as dictionary."""
        return {
            attr: getattr(cls, attr) for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }