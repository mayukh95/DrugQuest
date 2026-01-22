# graph_analysis.py
# Comprehensive Molecular Graph Visualization & Analysis Module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, degree

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════
# 1. ATTENTION FLOW VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_attention_flow(model, graph, device, layer_names=['Layer 1', 'Layer 2']):
    """
    Visualize attention flow through GAT layers as a Sankey-like diagram.
    Shows how attention propagates from input atoms through layers.
    """
    model.eval()
    data = Batch.from_data_list([graph]).to(device)
    mol = Chem.MolFromSmiles(graph.smiles)
    n_atoms = graph.x.shape[0]
    
    # Get attention weights from both layers
    with torch.no_grad():
        # Forward pass through layer 1
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        x1, (edge_idx1, attn1) = model.gat1(x, edge_index, edge_attr=edge_attr, 
                                             return_attention_weights=True)
        x1 = model.bn1(x1)
        x1 = torch.nn.functional.elu(x1)
        
        # Forward pass through layer 2
        x2, (edge_idx2, attn2) = model.gat2(x1, edge_index, edge_attr=edge_attr,
                                             return_attention_weights=True)
    
    # Aggregate attention per atom for each layer
    def aggregate_attention(edge_idx, attn, n):
        importance = np.zeros(n)
        edge_idx_np = edge_idx.cpu().numpy()
        attn_np = attn.cpu().numpy().flatten()
        for i, (src, dst) in enumerate(edge_idx_np.T):
            if i < len(attn_np):
                importance[dst] += attn_np[i]
        return importance / (importance.max() + 1e-6)
    
    attn_layer1 = aggregate_attention(edge_idx1, attn1, n_atoms)
    attn_layer2 = aggregate_attention(edge_idx2, attn2, n_atoms)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Get atom labels
    atom_labels = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]
    
    # Layer 1 attention
    ax1 = axes[0]
    colors1 = plt.cm.Reds(attn_layer1)
    bars1 = ax1.barh(range(n_atoms), attn_layer1, color=colors1)
    ax1.set_yticks(range(n_atoms))
    ax1.set_yticklabels(atom_labels)
    ax1.set_xlabel('Attention Score')
    ax1.set_title(f'{layer_names[0]} Attention', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Layer 2 attention
    ax2 = axes[1]
    colors2 = plt.cm.Blues(attn_layer2)
    bars2 = ax2.barh(range(n_atoms), attn_layer2, color=colors2)
    ax2.set_yticks(range(n_atoms))
    ax2.set_yticklabels(atom_labels)
    ax2.set_xlabel('Attention Score')
    ax2.set_title(f'{layer_names[1]} Attention', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Attention change (flow)
    ax3 = axes[2]
    attn_change = attn_layer2 - attn_layer1
    colors3 = ['#2ecc71' if c > 0 else '#e74c3c' for c in attn_change]
    bars3 = ax3.barh(range(n_atoms), attn_change, color=colors3)
    ax3.set_yticks(range(n_atoms))
    ax3.set_yticklabels(atom_labels)
    ax3.set_xlabel('Attention Change')
    ax3.set_title('Attention Flow (L2 - L1)', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    
    plt.suptitle(f'Attention Flow Analysis: {graph.name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, {'layer1': attn_layer1, 'layer2': attn_layer2, 'change': attn_change}


# ═══════════════════════════════════════════════════════════════════════════
# 2. LAYER COMPARISON (ATTENTION MATRICES)
# ═══════════════════════════════════════════════════════════════════════════

def compare_attention_layers(model, graph, device):
    """
    Compare attention matrices across GAT layers.
    Shows heatmaps of atom-to-atom attention weights.
    """
    model.eval()
    data = Batch.from_data_list([graph]).to(device)
    mol = Chem.MolFromSmiles(graph.smiles)
    n_atoms = graph.x.shape[0]
    
    with torch.no_grad():
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Layer 1
        x1, (edge_idx1, attn1) = model.gat1(x, edge_index, edge_attr=edge_attr,
                                             return_attention_weights=True)
        x1 = model.bn1(x1)
        x1 = torch.nn.functional.elu(x1)
        
        # Layer 2
        x2, (edge_idx2, attn2) = model.gat2(x1, edge_index, edge_attr=edge_attr,
                                             return_attention_weights=True)
    
    # Build attention matrices
    def build_attention_matrix(edge_idx, attn, n):
        matrix = np.zeros((n, n))
        edge_idx_np = edge_idx.cpu().numpy()
        attn_np = attn.cpu().numpy()
        if len(attn_np.shape) > 1:
            attn_np = attn_np.mean(axis=1)  # Average across heads
        for i, (src, dst) in enumerate(edge_idx_np.T):
            if i < len(attn_np):
                matrix[src, dst] = attn_np[i]
        return matrix
    
    matrix1 = build_attention_matrix(edge_idx1, attn1, n_atoms)
    matrix2 = build_attention_matrix(edge_idx2, attn2, n_atoms)
    
    # Get atom labels
    atom_labels = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Layer 1 heatmap
    im1 = axes[0].imshow(matrix1, cmap='Reds', aspect='auto')
    axes[0].set_xticks(range(n_atoms))
    axes[0].set_yticks(range(n_atoms))
    axes[0].set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(atom_labels, fontsize=8)
    axes[0].set_title('Layer 1 Attention', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Target Atom')
    axes[0].set_ylabel('Source Atom')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Layer 2 heatmap
    im2 = axes[1].imshow(matrix2, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(n_atoms))
    axes[1].set_yticks(range(n_atoms))
    axes[1].set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(atom_labels, fontsize=8)
    axes[1].set_title('Layer 2 Attention', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Target Atom')
    axes[1].set_ylabel('Source Atom')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Difference heatmap
    diff = matrix2 - matrix1
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_xticks(range(n_atoms))
    axes[2].set_yticks(range(n_atoms))
    axes[2].set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels(atom_labels, fontsize=8)
    axes[2].set_title('Attention Difference (L2 - L1)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Target Atom')
    axes[2].set_ylabel('Source Atom')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.suptitle(f'Layer-wise Attention Comparison: {graph.name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, {'layer1': matrix1, 'layer2': matrix2, 'diff': diff}


# ═══════════════════════════════════════════════════════════════════════════
# 3. NODE EMBEDDING VISUALIZATION (t-SNE)
# ═══════════════════════════════════════════════════════════════════════════

def visualize_node_embeddings(model, graph, device, method='tsne', perplexity=5):
    """
    Visualize atom embeddings using t-SNE or PCA.
    Color by atom type, size by importance.
    """
    model.eval()
    data = Batch.from_data_list([graph]).to(device)
    mol = Chem.MolFromSmiles(graph.smiles)
    n_atoms = graph.x.shape[0]
    
    with torch.no_grad():
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Get embeddings after each layer
        x_input = x.cpu().numpy()
        
        x1, (_, attn1) = model.gat1(x, edge_index, edge_attr=edge_attr,
                                     return_attention_weights=True)
        x1 = model.bn1(x1)
        x1 = torch.nn.functional.elu(x1)
        x_layer1 = x1.cpu().numpy()
        
        x2, (_, attn2) = model.gat2(x1, edge_index, edge_attr=edge_attr,
                                     return_attention_weights=True)
        x2 = model.bn2(x2)
        x2 = torch.nn.functional.elu(x2)
        x_layer2 = x2.cpu().numpy()
    
    # Get atom types and importance
    atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
    unique_types = list(set(atom_types))
    color_map = {t: plt.cm.tab10(i) for i, t in enumerate(unique_types)}
    colors = [color_map[t] for t in atom_types]
    
    # Compute importance
    edge_idx_np = data.edge_index.cpu().numpy()
    attn_np = attn2.cpu().numpy().flatten()
    importance = np.zeros(n_atoms)
    for i, (src, dst) in enumerate(edge_idx_np.T):
        if i < len(attn_np):
            importance[dst] += attn_np[i]
    importance = importance / (importance.max() + 1e-6)
    sizes = 100 + 400 * importance
    
    # Dimensionality reduction
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    embeddings = [x_input, x_layer1, x_layer2]
    titles = ['Input Features (43d)', 'After Layer 1 (256d)', 'After Layer 2 (64d)']
    
    for ax, emb, title in zip(axes, embeddings, titles):
        if emb.shape[1] > 2:
            if method == 'tsne' and n_atoms > perplexity:
                reducer = TSNE(n_components=2, perplexity=min(perplexity, n_atoms-1), random_state=42)
            else:
                reducer = PCA(n_components=2)
            emb_2d = reducer.fit_transform(emb)
        else:
            emb_2d = emb[:, :2]
        
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=sizes, alpha=0.8, edgecolors='black')
        
        # Add atom labels
        for i, (x, y) in enumerate(emb_2d):
            ax.annotate(f'{atom_types[i]}{i}', (x, y), fontsize=8, ha='center', va='bottom')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=color_map[t], label=t) for t in unique_types]
    axes[2].legend(handles=legend_elements, loc='upper right', title='Atom Type')
    
    plt.suptitle(f'Node Embedding Evolution: {graph.name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. GRAPH EMBEDDING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_graph_embeddings(model, dataset, device, results_df=None, method='tsne', perplexity=15):
    """
    Visualize molecule embeddings in 2D space.
    Color by binding score if available.
    """
    model.eval()
    embeddings = []
    names = []
    scores = []
    
    with torch.no_grad():
        for graph in dataset:
            data = Batch.from_data_list([graph]).to(device)
            
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch
            
            # Forward pass
            x = model.gat1(x, edge_index, edge_attr=edge_attr)
            x = model.bn1(x)
            x = torch.nn.functional.elu(x)
            x = model.gat2(x, edge_index, edge_attr=edge_attr)
            x = model.bn2(x)
            x = torch.nn.functional.elu(x)
            
            # Global pooling
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(x, batch)
            embeddings.append(graph_emb.cpu().numpy().flatten())
            names.append(graph.name)
            
            if results_df is not None:
                score_row = results_df[results_df['name'] == graph.name]
                if len(score_row) > 0:
                    scores.append(score_row['binding_score'].values[0])
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
    
    embeddings = np.array(embeddings)
    scores = np.array(scores)
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings)-1), random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    emb_2d = reducer.fit_transform(embeddings)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Scatter plot colored by score
    scatter1 = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=scores, cmap='RdYlGn', 
                               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0].set_title('Graph Embeddings (Color = Binding Score)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'{method.upper()} Dimension 1')
    axes[0].set_ylabel(f'{method.upper()} Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Binding Score')
    
    # Highlight top molecules
    top_indices = np.argsort(scores)[-5:]
    for idx in top_indices:
        axes[0].annotate(names[idx], (emb_2d[idx, 0], emb_2d[idx, 1]), 
                        fontsize=8, fontweight='bold', color='darkgreen')
    
    # Density plot
    from scipy.stats import gaussian_kde
    xy = np.vstack([emb_2d[:, 0], emb_2d[:, 1]])
    try:
        z = gaussian_kde(xy)(xy)
        scatter2 = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=z, cmap='viridis',
                                   s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[1].set_title('Graph Embeddings (Color = Density)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter2, ax=axes[1], label='Density')
    except:
        axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c='steelblue', s=80, alpha=0.7)
        axes[1].set_title('Graph Embeddings', fontsize=14, fontweight='bold')
    
    axes[1].set_xlabel(f'{method.upper()} Dimension 1')
    axes[1].set_ylabel(f'{method.upper()} Dimension 2')
    
    plt.suptitle(f'Molecular Graph Embedding Space ({len(dataset)} molecules)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, emb_2d, names, scores


# ═══════════════════════════════════════════════════════════════════════════
# 5. SUBGRAPH IMPORTANCE (FUNCTIONAL GROUPS)
# ═══════════════════════════════════════════════════════════════════════════

def visualize_subgraph_importance(model, graph, device):
    """
    Visualize importance of functional groups/subgraphs.
    Highlights key binding motifs with their attention scores.
    """
    model.eval()
    data = Batch.from_data_list([graph]).to(device)
    mol = Chem.MolFromSmiles(graph.smiles)
    n_atoms = graph.x.shape[0]
    
    # Get attention weights
    with torch.no_grad():
        pred, (edge_idx, attn) = model(data, return_attention=True)
    
    edge_idx_np = edge_idx.cpu().numpy()
    attn_np = attn.cpu().numpy().flatten()
    importance = np.zeros(n_atoms)
    for i, (src, dst) in enumerate(edge_idx_np.T):
        if i < len(attn_np):
            importance[dst] += attn_np[i]
    importance = importance / (importance.max() + 1e-6)
    
    # Define functional groups to analyze
    functional_groups = {
        'Carboxylic Acid': 'C(=O)[OH]',
        'Sulfonamide': 'S(=O)(=O)N',
        'Hydroxyl': '[OH]',
        'Amine': '[NX3;H2,H1]',
        'Aromatic Ring': 'c1ccccc1',
        'Carbonyl': '[CX3]=[OX1]',
        'Ether': '[OD2]([#6])[#6]',
        'Halogen': '[F,Cl,Br,I]',
        'Nitro': '[N+](=O)[O-]',
        'Methyl': '[CH3]',
    }
    
    # Find functional groups and their importance
    fg_data = []
    all_fg_atoms = set()
    
    for fg_name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            atoms = set(a for m in matches for a in m)
            valid_atoms = [a for a in atoms if a < n_atoms]
            if valid_atoms:
                avg_importance = np.mean([importance[a] for a in valid_atoms])
                fg_data.append({
                    'name': fg_name,
                    'atoms': valid_atoms,
                    'importance': avg_importance,
                    'count': len(matches)
                })
                all_fg_atoms.update(valid_atoms)
    
    # Non-functional group atoms
    other_atoms = [i for i in range(n_atoms) if i not in all_fg_atoms]
    if other_atoms:
        fg_data.append({
            'name': 'Backbone/Other',
            'atoms': other_atoms,
            'importance': np.mean([importance[a] for a in other_atoms]),
            'count': 1
        })
    
    fg_df = pd.DataFrame(fg_data).sort_values('importance', ascending=False)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 8))
    
    # Subplot 1: Molecule with colored functional groups
    ax1 = fig.add_subplot(1, 3, 1)
    
    # Color atoms by importance
    atom_colors = {}
    for i in range(n_atoms):
        imp = importance[i]
        if imp > 0.7:
            atom_colors[i] = (0.8, 0.2, 0.2)  # Red - high
        elif imp > 0.4:
            atom_colors[i] = (1.0, 0.6, 0.0)  # Orange - medium
        else:
            atom_colors[i] = (0.2, 0.6, 0.8)  # Blue - low
    
    img = Draw.MolToImage(mol, size=(500, 400), highlightAtoms=list(range(n_atoms)),
                          highlightAtomColors=atom_colors)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Atom Importance Map', fontsize=14, fontweight='bold')
    
    # Legend for importance
    legend_elements = [
        mpatches.Patch(facecolor=(0.8, 0.2, 0.2), label='High (>0.7)'),
        mpatches.Patch(facecolor=(1.0, 0.6, 0.0), label='Medium (0.4-0.7)'),
        mpatches.Patch(facecolor=(0.2, 0.6, 0.8), label='Low (<0.4)')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=8)
    
    # Subplot 2: Functional group importance bar chart
    ax2 = fig.add_subplot(1, 3, 2)
    colors = plt.cm.RdYlGn(fg_df['importance'].values)
    bars = ax2.barh(range(len(fg_df)), fg_df['importance'].values, color=colors)
    ax2.set_yticks(range(len(fg_df)))
    ax2.set_yticklabels(fg_df['name'].values)
    ax2.set_xlabel('Average Importance')
    ax2.set_title('Functional Group Importance', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 1)
    
    # Add count annotations
    for i, (imp, count) in enumerate(zip(fg_df['importance'].values, fg_df['count'].values)):
        ax2.text(imp + 0.02, i, f'(×{count})', va='center', fontsize=9)
    
    # Subplot 3: Atom-level importance distribution
    ax3 = fig.add_subplot(1, 3, 3)
    atom_labels = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]
    colors3 = ['#d32f2f' if imp > 0.7 else '#ff9800' if imp > 0.4 else '#1976d2' for imp in importance]
    ax3.barh(range(n_atoms), importance, color=colors3)
    ax3.set_yticks(range(n_atoms))
    ax3.set_yticklabels(atom_labels, fontsize=8)
    ax3.set_xlabel('Importance Score')
    ax3.set_title('Per-Atom Importance', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.axvline(0.7, color='red', linestyle='--', alpha=0.3)
    ax3.axvline(0.4, color='orange', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'Subgraph Importance Analysis: {graph.name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, fg_df


# ═══════════════════════════════════════════════════════════════════════════
# 6. EDGE FEATURE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_edge_features(dataset, graph=None):
    """
    Visualize distribution of edge features (bond types, Mayer bond orders).
    """
    # Collect all edge features
    all_edge_attrs = []
    for g in dataset:
        if g.edge_attr.shape[0] > 0:
            all_edge_attrs.append(g.edge_attr.numpy())
    
    all_edge_attrs = np.vstack(all_edge_attrs)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Bond type distribution (columns 0-3)
    bond_types = ['Single', 'Double', 'Triple', 'Aromatic']
    bond_counts = all_edge_attrs[:, :4].sum(axis=0)
    
    ax1 = axes[0, 0]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    ax1.bar(bond_types, bond_counts, color=colors, edgecolor='black')
    ax1.set_title('Bond Type Distribution (Dataset)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    for i, v in enumerate(bond_counts):
        ax1.text(i, v + 50, f'{int(v)}', ha='center', fontweight='bold')
    
    # Mayer bond order histogram
    ax2 = axes[0, 1]
    mayer_orders = all_edge_attrs[:, 4]
    ax2.hist(mayer_orders, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax2.axvline(mayer_orders.mean(), color='red', linestyle='--', label=f'Mean: {mayer_orders.mean():.2f}')
    ax2.set_title('Mayer Bond Order Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mayer Bond Order')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Bond type vs Mayer order boxplot
    ax3 = axes[0, 2]
    bond_type_mayer = defaultdict(list)
    for edge in all_edge_attrs:
        bt_idx = np.argmax(edge[:4])
        bond_type_mayer[bond_types[bt_idx]].append(edge[4])
    
    data_for_box = [bond_type_mayer[bt] for bt in bond_types if len(bond_type_mayer[bt]) > 0]
    labels_for_box = [bt for bt in bond_types if len(bond_type_mayer[bt]) > 0]
    bp = ax3.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(data_for_box)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title('Mayer Order by Bond Type', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mayer Bond Order')
    
    # If specific graph provided, show its edge features
    if graph is not None:
        mol = Chem.MolFromSmiles(graph.smiles)
        n_atoms = mol.GetNumAtoms()
        edge_attr_g = graph.edge_attr.numpy()
        
        # Bond type pie chart for this molecule
        ax4 = axes[1, 0]
        bond_counts_g = edge_attr_g[:, :4].sum(axis=0) / 2  # Divide by 2 for undirected
        non_zero = [(bt, c) for bt, c in zip(bond_types, bond_counts_g) if c > 0]
        if non_zero:
            ax4.pie([c for _, c in non_zero], labels=[bt for bt, _ in non_zero],
                   autopct='%1.0f%%', colors=colors[:len(non_zero)], startangle=90)
        ax4.set_title(f'Bond Types: {graph.name}', fontsize=14, fontweight='bold')
        
        # Mayer bond order matrix
        ax5 = axes[1, 1]
        mayer_matrix = np.zeros((n_atoms, n_atoms))
        edge_index = graph.edge_index.numpy()
        for i, (src, dst) in enumerate(edge_index.T):
            if i < len(edge_attr_g):
                mayer_matrix[src, dst] = edge_attr_g[i, 4]
        
        im = ax5.imshow(mayer_matrix, cmap='YlOrRd')
        ax5.set_title(f'Mayer Bond Order Matrix: {graph.name}', fontsize=14, fontweight='bold')
        atom_labels = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]
        ax5.set_xticks(range(n_atoms))
        ax5.set_yticks(range(n_atoms))
        ax5.set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
        ax5.set_yticklabels(atom_labels, fontsize=8)
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # Edge feature histogram for this molecule
        ax6 = axes[1, 2]
        ax6.hist(edge_attr_g[:, 4], bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax6.set_title(f'Mayer Distribution: {graph.name}', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Mayer Bond Order')
        ax6.set_ylabel('Count')
    else:
        axes[1, 0].text(0.5, 0.5, 'Select a molecule\nto see details', ha='center', va='center', fontsize=12)
        axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Select a molecule\nto see details', ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
        axes[1, 2].text(0.5, 0.5, 'Select a molecule\nto see details', ha='center', va='center', fontsize=12)
        axes[1, 2].axis('off')
    
    plt.suptitle('Edge Feature Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. DEGREE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_degree_distribution(dataset, graph=None):
    """
    Visualize node degree distribution (graph topology analysis).
    """
    # Collect degrees from all graphs
    all_degrees = []
    for g in dataset:
        deg = degree(g.edge_index[0], num_nodes=g.x.shape[0]).numpy()
        all_degrees.extend(deg)
    
    all_degrees = np.array(all_degrees)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Dataset-wide degree distribution
    ax1 = axes[0]
    unique_degrees, counts = np.unique(all_degrees, return_counts=True)
    ax1.bar(unique_degrees, counts, color='#3498db', edgecolor='black')
    ax1.set_title('Degree Distribution (All Molecules)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Node Degree')
    ax1.set_ylabel('Count')
    ax1.set_xticks(unique_degrees)
    
    # Statistics
    stats_text = f'Mean: {all_degrees.mean():.2f}\nStd: {all_degrees.std():.2f}\nMax: {all_degrees.max()}'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Degree vs atom type (if graph provided)
    if graph is not None:
        mol = Chem.MolFromSmiles(graph.smiles)
        n_atoms = mol.GetNumAtoms()
        deg_g = degree(graph.edge_index[0], num_nodes=n_atoms).numpy()
        
        ax2 = axes[1]
        atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
        colors = plt.cm.tab10([hash(t) % 10 / 10 for t in atom_types])
        ax2.bar(range(n_atoms), deg_g, color=colors, edgecolor='black')
        ax2.set_xticks(range(n_atoms))
        ax2.set_xticklabels([f'{t}{i}' for i, t in enumerate(atom_types)], rotation=45, ha='right')
        ax2.set_title(f'Degree by Atom: {graph.name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Atom')
        ax2.set_ylabel('Degree')
        
        # Degree-type relationship
        ax3 = axes[2]
        type_degrees = defaultdict(list)
        for t, d in zip(atom_types, deg_g):
            type_degrees[t].append(d)
        
        types = list(type_degrees.keys())
        means = [np.mean(type_degrees[t]) for t in types]
        stds = [np.std(type_degrees[t]) for t in types]
        
        ax3.bar(types, means, yerr=stds, color='#2ecc71', edgecolor='black', capsize=5)
        ax3.set_title(f'Mean Degree by Atom Type: {graph.name}', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Atom Type')
        ax3.set_ylabel('Mean Degree')
    else:
        # Show degree by common atom types across dataset
        ax2 = axes[1]
        ax2.text(0.5, 0.5, 'Select a molecule\nto see atom-level analysis', 
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
        
        ax3 = axes[2]
        ax3.text(0.5, 0.5, 'Select a molecule\nto see type analysis',
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    plt.suptitle('Graph Topology: Degree Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. K-NEAREST NEIGHBORS GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_knn_graph(smiles, k=3, full_data=None):
    """
    Build a K-nearest neighbors graph based on 3D coordinates.
    Returns comparison with original bond-based graph.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 3D coordinates
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)
    mol_3d = Chem.RemoveHs(mol_3d)
    
    n_atoms = mol_3d.GetNumAtoms()
    
    # Get 3D coordinates
    conf = mol_3d.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])
    
    # Compute pairwise distances
    dist_matrix = squareform(pdist(coords))
    
    # Build KNN edges
    knn_edges = []
    for i in range(n_atoms):
        distances = dist_matrix[i]
        # Get k nearest neighbors (excluding self)
        nearest = np.argsort(distances)[1:k+1]
        for j in nearest:
            knn_edges.append((i, j))
            knn_edges.append((j, i))  # Undirected
    
    knn_edges = list(set(knn_edges))  # Remove duplicates
    
    # Original bond-based edges
    bond_edges = []
    for bond in mol_3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_edges.append((i, j))
        bond_edges.append((j, i))
    
    # Create NetworkX graphs for visualization
    G_bond = nx.Graph()
    G_knn = nx.Graph()
    
    for i in range(n_atoms):
        atom = mol_3d.GetAtomWithIdx(i)
        label = f"{atom.GetSymbol()}{i}"
        G_bond.add_node(i, label=label, pos=coords[i][:2])
        G_knn.add_node(i, label=label, pos=coords[i][:2])
    
    for i, j in bond_edges:
        if i < j:
            G_bond.add_edge(i, j)
    
    for i, j in knn_edges:
        if i < j:
            G_knn.add_edge(i, j, weight=dist_matrix[i, j])
    
    return {
        'mol': mol_3d,
        'coords': coords,
        'dist_matrix': dist_matrix,
        'bond_graph': G_bond,
        'knn_graph': G_knn,
        'k': k
    }


def visualize_knn_graph(smiles, k=3, full_data=None):
    """
    Visualize comparison between bond-based and KNN graphs.
    """
    result = build_knn_graph(smiles, k, full_data)
    if result is None:
        return None
    
    mol = result['mol']
    coords = result['coords']
    G_bond = result['bond_graph']
    G_knn = result['knn_graph']
    n_atoms = mol.GetNumAtoms()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get node positions (2D projection)
    pos = {i: coords[i][:2] for i in range(n_atoms)}
    labels = {i: f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)}
    node_colors = [plt.cm.tab10(hash(mol.GetAtomWithIdx(i).GetSymbol()) % 10) for i in range(n_atoms)]
    
    # Bond-based graph
    ax1 = axes[0]
    nx.draw(G_bond, pos, ax=ax1, with_labels=True, labels=labels,
            node_color=node_colors, node_size=500, font_size=8,
            edge_color='#3498db', width=2)
    ax1.set_title(f'Bond-Based Graph ({G_bond.number_of_edges()} edges)', fontsize=14, fontweight='bold')
    
    # KNN graph
    ax2 = axes[1]
    nx.draw(G_knn, pos, ax=ax2, with_labels=True, labels=labels,
            node_color=node_colors, node_size=500, font_size=8,
            edge_color='#e74c3c', width=2, style='dashed')
    ax2.set_title(f'K-NN Graph (k={k}, {G_knn.number_of_edges()} edges)', fontsize=14, fontweight='bold')
    
    # Overlay comparison
    ax3 = axes[2]
    # Draw bond edges
    bond_edge_list = list(G_bond.edges())
    knn_edge_list = list(G_knn.edges())
    
    # Common edges
    common = set(bond_edge_list) & set(knn_edge_list)
    bond_only = set(bond_edge_list) - common
    knn_only = set(knn_edge_list) - common
    
    nx.draw_networkx_nodes(G_bond, pos, ax=ax3, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G_bond, pos, labels, ax=ax3, font_size=8)
    
    if common:
        nx.draw_networkx_edges(G_bond, pos, edgelist=list(common), ax=ax3,
                              edge_color='#2ecc71', width=3, label='Common')
    if bond_only:
        nx.draw_networkx_edges(G_bond, pos, edgelist=list(bond_only), ax=ax3,
                              edge_color='#3498db', width=2, label='Bond only')
    if knn_only:
        nx.draw_networkx_edges(G_knn, pos, edgelist=list(knn_only), ax=ax3,
                              edge_color='#e74c3c', width=2, style='dashed', label='KNN only')
    
    ax3.set_title('Overlay Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    
    plt.suptitle(f'Graph Construction Comparison: Bond vs K-NN (k={k})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Print statistics
    print(f"\n📊 Graph Statistics:")
    print(f"   Bond-based: {G_bond.number_of_nodes()} nodes, {G_bond.number_of_edges()} edges")
    print(f"   K-NN (k={k}): {G_knn.number_of_nodes()} nodes, {G_knn.number_of_edges()} edges")
    print(f"   Common edges: {len(common)}")
    print(f"   Bond-only edges: {len(bond_only)}")
    print(f"   KNN-only edges: {len(knn_only)}")
    
    return fig, result


# ═══════════════════════════════════════════════════════════════════════════
# 9. COARSE-GRAINED GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_coarse_grained_graph(smiles, method='functional_groups'):
    """
    Build a coarse-grained graph where nodes are functional groups instead of atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n_atoms = mol.GetNumAtoms()
    
    # Define functional group patterns
    fg_patterns = {
        'Carboxylic': 'C(=O)[OH]',
        'Sulfonamide': 'S(=O)(=O)N',
        'Hydroxyl': '[OH]',
        'Amine': '[NX3;H2,H1]',
        'Aromatic': 'c1ccccc1',
        'Carbonyl': '[CX3]=[OX1]',
        'Ether': '[OD2]([#6])[#6]',
        'Halogen': '[F,Cl,Br,I]',
        'Methyl': '[CH3]',
        'Alkyl': '[CX4]',
    }
    
    # Assign atoms to functional groups
    atom_to_fg = {}
    fg_atoms = defaultdict(set)
    fg_count = defaultdict(int)
    
    for fg_name, smarts in fg_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            for match_idx, match in enumerate(matches):
                fg_id = f"{fg_name}_{fg_count[fg_name]}"
                fg_count[fg_name] += 1
                for atom_idx in match:
                    if atom_idx not in atom_to_fg:  # First match wins
                        atom_to_fg[atom_idx] = fg_id
                        fg_atoms[fg_id].add(atom_idx)
    
    # Assign remaining atoms to 'Backbone'
    backbone_count = 0
    for i in range(n_atoms):
        if i not in atom_to_fg:
            fg_id = f"Backbone_{backbone_count}"
            atom_to_fg[i] = fg_id
            fg_atoms[fg_id].add(i)
            backbone_count += 1
    
    # Merge small backbone fragments
    backbone_fgs = [fg for fg in fg_atoms if 'Backbone' in fg]
    if len(backbone_fgs) > 1:
        # Merge all backbone into one
        merged_atoms = set()
        for fg in backbone_fgs:
            merged_atoms.update(fg_atoms[fg])
            del fg_atoms[fg]
        fg_atoms['Backbone'] = merged_atoms
        for atom_idx in merged_atoms:
            atom_to_fg[atom_idx] = 'Backbone'
    
    # Build coarse-grained graph
    fg_list = list(fg_atoms.keys())
    n_fgs = len(fg_list)
    fg_to_idx = {fg: i for i, fg in enumerate(fg_list)}
    
    # Find edges between functional groups
    cg_edges = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        fg_i = atom_to_fg.get(i)
        fg_j = atom_to_fg.get(j)
        if fg_i and fg_j and fg_i != fg_j:
            idx_i, idx_j = fg_to_idx[fg_i], fg_to_idx[fg_j]
            if idx_i < idx_j:
                cg_edges.add((idx_i, idx_j))
            else:
                cg_edges.add((idx_j, idx_i))
    
    # Create NetworkX graph
    G_cg = nx.Graph()
    for i, fg in enumerate(fg_list):
        G_cg.add_node(i, label=fg.split('_')[0], atoms=list(fg_atoms[fg]), size=len(fg_atoms[fg]))
    
    for i, j in cg_edges:
        G_cg.add_edge(i, j)
    
    return {
        'mol': mol,
        'fg_atoms': dict(fg_atoms),
        'atom_to_fg': atom_to_fg,
        'cg_graph': G_cg,
        'fg_list': fg_list
    }


def visualize_coarse_grained_graph(smiles):
    """
    Visualize comparison between atom-level and coarse-grained graphs.
    """
    result = build_coarse_grained_graph(smiles)
    if result is None:
        return None
    
    mol = result['mol']
    G_cg = result['cg_graph']
    fg_atoms = result['fg_atoms']
    atom_to_fg = result['atom_to_fg']
    n_atoms = mol.GetNumAtoms()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original molecule
    ax1 = axes[0]
    img = Draw.MolToImage(mol, size=(500, 400))
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Original Molecule ({n_atoms} atoms)', fontsize=14, fontweight='bold')
    
    # Molecule with functional group coloring
    ax2 = axes[1]
    fg_colors = {}
    color_map = plt.cm.tab20(np.linspace(0, 1, len(fg_atoms)))
    for i, fg in enumerate(fg_atoms.keys()):
        for atom_idx in fg_atoms[fg]:
            fg_colors[atom_idx] = tuple(color_map[i][:3])
    
    img2 = Draw.MolToImage(mol, size=(500, 400), highlightAtoms=list(range(n_atoms)),
                           highlightAtomColors=fg_colors)
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('Functional Group Assignment', fontsize=14, fontweight='bold')
    
    # Create legend
    legend_elements = []
    for i, fg in enumerate(fg_atoms.keys()):
        legend_elements.append(mpatches.Patch(facecolor=color_map[i], label=fg.split('_')[0]))
    ax2.legend(handles=legend_elements[:10], loc='lower left', fontsize=7, ncol=2)
    
    # Coarse-grained graph
    ax3 = axes[2]
    pos = nx.spring_layout(G_cg, seed=42)
    
    # Node sizes based on number of atoms
    sizes = [300 + 200 * G_cg.nodes[n]['size'] for n in G_cg.nodes()]
    colors = [color_map[i] for i in G_cg.nodes()]
    labels = {n: G_cg.nodes[n]['label'] for n in G_cg.nodes()}
    
    nx.draw(G_cg, pos, ax=ax3, with_labels=True, labels=labels,
            node_color=colors, node_size=sizes, font_size=9, font_weight='bold',
            edge_color='#7f8c8d', width=2)
    
    ax3.set_title(f'Coarse-Grained Graph ({G_cg.number_of_nodes()} nodes, {G_cg.number_of_edges()} edges)',
                  fontsize=14, fontweight='bold')
    
    plt.suptitle('Coarse-Grained Graph Construction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Print statistics
    print(f"\n📊 Coarse-Graining Statistics:")
    print(f"   Original: {n_atoms} atoms, {mol.GetNumBonds()} bonds")
    print(f"   Coarse-grained: {G_cg.number_of_nodes()} nodes, {G_cg.number_of_edges()} edges")
    print(f"   Reduction: {(1 - G_cg.number_of_nodes()/n_atoms)*100:.1f}%")
    print(f"\n   Functional Groups:")
    for fg, atoms in fg_atoms.items():
        print(f"     {fg}: {len(atoms)} atoms")
    
    return fig, result


# ═══════════════════════════════════════════════════════════════════════════
# 10. INTERACTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def create_graph_analysis_dashboard(model, dataset, device, results_df=None):
    """
    Create an interactive dashboard for all graph analysis visualizations.
    """
    # Prepare molecule options
    mol_options = [(f"{g.name}", i) for i, g in enumerate(dataset)]
    
    # Widgets
    mol_selector = widgets.Dropdown(
        options=mol_options,
        value=0,
        description='Molecule:',
        layout=widgets.Layout(width='300px')
    )
    
    analysis_type = widgets.Dropdown(
        options=[
            ('Attention Flow', 'attention_flow'),
            ('Layer Comparison', 'layer_compare'),
            ('Node Embeddings', 'node_embed'),
            ('Graph Embeddings', 'graph_embed'),
            ('Subgraph Importance', 'subgraph'),
            ('Edge Features', 'edge_feat'),
            ('Degree Distribution', 'degree'),
            ('K-NN Graph', 'knn'),
            ('Coarse-Grained Graph', 'coarse'),
        ],
        value='attention_flow',
        description='Analysis:',
        layout=widgets.Layout(width='300px')
    )
    
    k_slider = widgets.IntSlider(
        value=3, min=2, max=6, step=1,
        description='K (KNN):',
        layout=widgets.Layout(width='300px')
    )
    
    run_btn = widgets.Button(
        description='Run Analysis',
        button_style='success',
        layout=widgets.Layout(width='150px')
    )
    
    output = widgets.Output()
    
    def run_analysis(btn):
        with output:
            clear_output(wait=True)
            
            mol_idx = mol_selector.value
            graph = dataset[mol_idx]
            analysis = analysis_type.value
            
            plt.close('all')
            
            if analysis == 'attention_flow':
                fig, _ = visualize_attention_flow(model, graph, device)
                plt.show()
            
            elif analysis == 'layer_compare':
                fig, _ = compare_attention_layers(model, graph, device)
                plt.show()
            
            elif analysis == 'node_embed':
                fig = visualize_node_embeddings(model, graph, device)
                plt.show()
            
            elif analysis == 'graph_embed':
                fig, _, _, _ = visualize_graph_embeddings(model, dataset, device, results_df)
                plt.show()
            
            elif analysis == 'subgraph':
                fig, fg_df = visualize_subgraph_importance(model, graph, device)
                plt.show()
            
            elif analysis == 'edge_feat':
                fig = visualize_edge_features(dataset, graph)
                plt.show()
            
            elif analysis == 'degree':
                fig = visualize_degree_distribution(dataset, graph)
                plt.show()
            
            elif analysis == 'knn':
                k = k_slider.value
                fig, _ = visualize_knn_graph(graph.smiles, k=k)
                plt.show()
            
            elif analysis == 'coarse':
                fig, _ = visualize_coarse_grained_graph(graph.smiles)
                plt.show()
    
    run_btn.on_click(run_analysis)
    
    # Layout
    header = widgets.HTML("""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:12px;color:white;margin-bottom:15px;'>
            <h2 style='margin:0;'>📊 Graph Analysis Dashboard</h2>
            <p style='margin:5px 0 0 0;opacity:0.8;'>Comprehensive Molecular Graph Visualization</p>
        </div>
    """)
    
    controls = widgets.HBox([mol_selector, analysis_type, k_slider, run_btn])
    
    layout = widgets.VBox([header, controls, output])
    display(layout)
    
    # Run initial analysis
    run_analysis(None)
