"""
Similarity metrics and matrix computation tools.
"""

import numpy as np
import pandas as pd
from rdkit import DataStructs

def tanimoto_dict(fp1, fp2):
    """
    Compute Tanimoto similarity for dictionary-based fingerprints (e.g., 3D pharmacophore binned features).
    """
    if not fp1 or not fp2:
        return 0.0
    keys = set(fp1) | set(fp2)
    intersection = sum(min(fp1.get(k, 0), fp2.get(k, 0)) for k in keys)
    union = sum(max(fp1.get(k, 0), fp2.get(k, 0)) for k in keys)
    return intersection / max(union, 1) if union > 0 else 0.0

def compute_sim_matrix(fps, names, metric_func=None):
    """
    Compute pairwise similarity matrix.
    
    Parameters:
    -----------
    fps : list
        List of fingerprints (RDKit BitVect or dicts)
    names : list
        List of molecule names
    metric_func : callable, optional
        Custom similarity function (default: TanimotoSimilarity for BitVects)
    """
    n = len(fps)
    mat = np.zeros((n, n))
    
    if metric_func is None:
        # Default to RDKit Tanimoto for BitVects
        for i in range(n):
            for j in range(i, n):
                if fps[i] and fps[j]:
                    s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    mat[i, j] = mat[j, i] = s
    else:
        # Use custom metric (e.g., for dicts)
        for i in range(n):
            for j in range(i, n):
                s = metric_func(fps[i], fps[j])
                mat[i, j] = mat[j, i] = s
                
    return pd.DataFrame(mat, index=names, columns=names)
