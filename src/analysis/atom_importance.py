# atom_importance.py
# Atom Importance Extraction from GAT Attention Weights

import torch
from torch_geometric.data import Batch


def get_atom_importance(model, graph, device):
    """
    Extract atom importance from GAT attention weights.
    
    Parameters:
    - model: Trained GAT model with return_attention capability
    - graph: PyG Data object for a single molecule
    - device: PyTorch device (cuda or cpu)
    
    Returns:
    - numpy array of normalized atom importance scores
    """
    model.eval()
    data = Batch.from_data_list([graph]).to(device)
    
    with torch.no_grad():
        pred, (edge_idx, attn) = model(data, return_attention=True)
    
    # Aggregate attention per atom
    n_atoms = graph.x.shape[0]
    atom_importance = torch.zeros(n_atoms)
    
    edge_idx_np = edge_idx.cpu().numpy()
    attn_np = attn.cpu().numpy().flatten()
    
    for i, (src, dst) in enumerate(edge_idx_np.T):
        if i < len(attn_np):
            atom_importance[dst] += attn_np[i]
    
    # Normalize
    atom_importance = atom_importance / (atom_importance.max() + 1e-6)
    return atom_importance.numpy()
