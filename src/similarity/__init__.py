"""
Molecular similarity analysis tools.
"""

from .shape_similarity import (
    compute_shape_similarity,
    compute_shape_matrix,
    visualize_shape_alignment,
    create_shape_alignment_widget
)
from .consensus import compute_consensus_scores
from .metrics import compute_sim_matrix, tanimoto_dict

__all__ = [
    'compute_shape_similarity',
    'compute_shape_matrix',
    'visualize_shape_alignment',
    'create_shape_alignment_widget',
    'compute_consensus_scores',
    'compute_sim_matrix',
    'tanimoto_dict'
]
