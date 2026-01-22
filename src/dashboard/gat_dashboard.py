# gat_dashboard.py
# Modular GAT Analysis Dashboard

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import networkx as nx

def create_gat_dashboard(model, dataset, device, atomwise_dft_dim, full_data=None):
    """GAT-Specific Analysis Dashboard.

    Parameters:
    - model: Trained GAT model
    - dataset: List of graph data objects
    - device: Torch device
    - atomwise_dft_dim: Dimension of DFT features
    - full_data: Optional dict with DFT data for deep dive
    """

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════

    def get_atom_importance(model, graph):
        """Calculate atom importance based on attention weights."""
        try:
            attn_data = get_attention_weights(model, graph)
            n_atoms = graph.x.shape[0]

            # Aggregate attention from layer 2 (final layer)
            edge_idx = attn_data['layer2']['edge_index']
            attention = attn_data['layer2']['attention']

            importance = np.zeros(n_atoms)
            for i in range(edge_idx.shape[1]):
                dst = edge_idx[1, i]
                if dst < n_atoms:
                    if len(attention.shape) > 1:
                        importance[dst] += attention[i].mean()  # Mean across heads
                    else:
                        importance[dst] += attention[i]

            # Normalize
            if importance.max() > 0:
                importance = importance / importance.max()

            return importance
        except:
            # Fallback: uniform importance
            return np.ones(graph.x.shape[0]) / graph.x.shape[0]

    def get_attention_weights(model, graph):
        """Extract attention weights from both GAT layers."""
        model.eval()
        graph_batch = Batch.from_data_list([graph]).to(device)

        with torch.no_grad():
            x = graph_batch.x
            edge_index = graph_batch.edge_index
            edge_attr = graph_batch.edge_attr

            # Layer 1 attention
            x1, (edge_idx1, attn1) = model.gat1(x, edge_index, edge_attr=edge_attr,
                                                 return_attention_weights=True)
            x1 = model.bn1(x1)
            x1 = F.elu(x1)

            # Layer 2 attention
            x2, (edge_idx2, attn2) = model.gat2(x1, edge_index, edge_attr=edge_attr,
                                                 return_attention_weights=True)

        return {
            'layer1': {
                'edge_index': edge_idx1.cpu().numpy(),
                'attention': attn1.cpu().numpy()
            },
            'layer2': {
                'edge_index': edge_idx2.cpu().numpy(),
                'attention': attn2.cpu().numpy()
            }
        }

    def get_node_features_analysis(graph):
        """Analyze node features (RDKit vs DFT contributions)."""
        x = graph.x.cpu().numpy()

        # Feature dimensions
        rdkit_dim = 33  # RDKit features
        dft_dim = atomwise_dft_dim  # DFT features

        rdkit_features = x[:, :rdkit_dim]
        dft_features = x[:, rdkit_dim:rdkit_dim+dft_dim]

        return {
            'rdkit': rdkit_features,
            'dft': dft_features,
            'rdkit_mean': np.mean(np.abs(rdkit_features), axis=1),
            'dft_mean': np.mean(np.abs(dft_features), axis=1)
        }

    def get_edge_features_analysis(graph, mol_name):
        """Analyze edge features (Bond type vs Mayer bond order)."""
        edge_attr = graph.edge_attr.cpu().numpy()
        edge_index = graph.edge_index.cpu().numpy()

        # Edge features: [Single, Double, Triple, Aromatic, Mayer_BO]
        bond_types = edge_attr[:, :4]
        mayer_bo = edge_attr[:, 4]

        return {
            'edge_index': edge_index,
            'bond_types': bond_types,
            'mayer_bo': mayer_bo
        }

    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT AREAS
    # ═══════════════════════════════════════════════════════════════════════

    attention_flow_output = widgets.Output()
    edge_heatmap_output = widgets.Output()
    layer_compare_output = widgets.Output()
    feature_attr_output = widgets.Output()
    dft_analysis_output = widgets.Output()
    network_output = widgets.Output()

    # Molecule selector
    mol_options = [(f"#{i+1}: {g.name}", g.name) for i, g in enumerate(dataset[:50])]
    mol_dropdown = widgets.Dropdown(
        options=mol_options,
        description='Molecule:',
        layout=widgets.Layout(width='400px')
    )

    mol_graph_lookup = {g.name: g for g in dataset}

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: ATTENTION FLOW VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    layer_select = widgets.Dropdown(
        options=[('Layer 1', 'layer1'), ('Layer 2', 'layer2'), ('Both', 'both')],
        value='layer2',
        description='Layer:',
        layout=widgets.Layout(width='200px')
    )
    head_select = widgets.IntSlider(value=0, min=0, max=3, description='Head:',
                                     layout=widgets.Layout(width='250px'))

    def update_attention_flow(change=None):
        with attention_flow_output:
            clear_output(wait=True)

            mol_name = mol_dropdown.value
            graph = mol_graph_lookup.get(mol_name)
            if not graph:
                return

            mol = Chem.MolFromSmiles(graph.smiles)
            if not mol:
                return

            n_atoms = mol.GetNumAtoms()

            # Get attention weights
            try:
                attn_data = get_attention_weights(model, graph)
            except Exception as e:
                display(HTML(f"<div style='color:red;'>Error getting attention: {e}</div>"))
                return

            layer = layer_select.value
            head = head_select.value

            fig, axes = plt.subplots(1, 2 if layer == 'both' else 1, figsize=(14 if layer == 'both' else 8, 6))
            if layer != 'both':
                axes = [axes]

            layers_to_plot = ['layer1', 'layer2'] if layer == 'both' else [layer]

            for ax, lyr in zip(axes, layers_to_plot):
                edge_idx = attn_data[lyr]['edge_index']
                attention = attn_data[lyr]['attention']

                # Create networkx graph
                G = nx.Graph()
                G.add_nodes_from(range(n_atoms))

                # Add edges with attention weights
                n_heads = attention.shape[1] if len(attention.shape) > 1 else 1

                for i in range(edge_idx.shape[1]):
                    src, dst = edge_idx[0, i], edge_idx[1, i]
                    if src < n_atoms and dst < n_atoms:
                        if len(attention.shape) > 1:
                            weight = attention[i, min(head, n_heads-1)]
                        else:
                            weight = attention[i]
                        if G.has_edge(src, dst):
                            G[src][dst]['weight'] = max(G[src][dst]['weight'], weight)
                        else:
                            G.add_edge(src, dst, weight=weight)

                # Layout
                pos = nx.spring_layout(G, seed=42, k=2)

                # Node labels
                labels = {i: f'{mol.GetAtomWithIdx(i).GetSymbol()}{i}' for i in range(n_atoms)}

                # Edge weights for coloring
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]

                if weights:
                    # Draw edges
                    edge_colors = plt.cm.Reds(np.array(weights) / (max(weights) + 1e-6))
                    edge_widths = [w * 5 + 0.5 for w in weights]

                    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax, alpha=0.8)

                # Draw nodes
                node_colors = ['#e53935' if mol.GetAtomWithIdx(i).GetSymbol() == 'O' else
                              '#1976d2' if mol.GetAtomWithIdx(i).GetSymbol() == 'N' else
                              '#43a047' if mol.GetAtomWithIdx(i).GetSymbol() == 'S' else
                              '#757575' for i in range(n_atoms)]

                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
                nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

                ax.set_title(f'Attention Flow - {lyr.replace("layer", "Layer ")} (Head {head})',
                            fontsize=12, fontweight='bold')
                ax.axis('off')

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            plt.colorbar(sm, ax=axes[-1], label='Attention Weight', shrink=0.8)

            plt.tight_layout()
            plt.show()

            # Legend
            display(HTML("""
                <div style='background:#f5f5f5;padding:15px;border-radius:10px;margin-top:10px;'>
                    <h4>🔍 Reading the Attention Flow:</h4>
                    <ul>
                        <li><b>Edge thickness/color</b>: Higher attention = thicker/redder edges</li>
                        <li><b>Node colors</b>: 🔴 O (Oxygen), 🔵 N (Nitrogen), 🟢 S (Sulfur), ⚫ C (Carbon)</li>
                        <li>Attention shows which atom-atom connections the model considers important</li>
                    </ul>
                </div>
            """))

    layer_select.observe(update_attention_flow, names='value')
    head_select.observe(update_attention_flow, names='value')

    attention_controls = widgets.VBox([
        widgets.HTML("<h3>🔀 Attention Flow Visualization</h3>"),
        widgets.HTML("<p>Visualize how information flows between atoms through attention weights.</p>"),
        widgets.HBox([layer_select, head_select]),
        attention_flow_output
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: EDGE IMPORTANCE HEATMAP (MAYER BOND ORDERS)
    # ═══════════════════════════════════════════════════════════════════════

    def update_edge_heatmap(change=None):
        with edge_heatmap_output:
            clear_output(wait=True)

            mol_name = mol_dropdown.value
            graph = mol_graph_lookup.get(mol_name)
            if not graph:
                return

            mol = Chem.MolFromSmiles(graph.smiles)
            if not mol:
                return

            n_atoms = mol.GetNumAtoms()

            # Get edge features
            edge_data = get_edge_features_analysis(graph, mol_name)
            edge_idx = edge_data['edge_index']
            mayer_bo = edge_data['mayer_bo']
            bond_types = edge_data['bond_types']

            # Create Mayer bond order matrix
            mayer_matrix = np.zeros((n_atoms, n_atoms))
            for i in range(edge_idx.shape[1]):
                src, dst = edge_idx[0, i], edge_idx[1, i]
                if src < n_atoms and dst < n_atoms:
                    mayer_matrix[src, dst] = mayer_bo[i]
                    mayer_matrix[dst, src] = mayer_bo[i]

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            # 1. Mayer Bond Order Heatmap
            atom_labels = [f'{mol.GetAtomWithIdx(i).GetSymbol()}{i}' for i in range(n_atoms)]

            im1 = ax1.imshow(mayer_matrix, cmap='YlOrRd', vmin=0, vmax=3)
            ax1.set_xticks(range(n_atoms))
            ax1.set_yticks(range(n_atoms))
            ax1.set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
            ax1.set_yticklabels(atom_labels, fontsize=8)
            plt.colorbar(im1, ax=ax1, label='Mayer Bond Order')
            ax1.set_title('Mayer Bond Order Matrix\n(DFT-computed)', fontsize=12, fontweight='bold')

            # 2. Bond Type Distribution
            bond_counts = {
                'Single': np.sum(bond_types[:, 0]) / 2,  # Divide by 2 for undirected
                'Double': np.sum(bond_types[:, 1]) / 2,
                'Triple': np.sum(bond_types[:, 2]) / 2,
                'Aromatic': np.sum(bond_types[:, 3]) / 2
            }

            colors = ['#42a5f5', '#66bb6a', '#ef5350', '#ab47bc']
            ax2.bar(bond_counts.keys(), bond_counts.values(), color=colors)
            ax2.set_ylabel('Number of Bonds')
            ax2.set_title('Bond Type Distribution', fontsize=12, fontweight='bold')

            # Annotate
            for i, (k, v) in enumerate(bond_counts.items()):
                if v > 0:
                    ax2.text(i, v + 0.1, f'{int(v)}', ha='center', fontweight='bold')

            # 3. Mayer BO vs Bond Type
            single_mask = bond_types[:, 0] == 1
            double_mask = bond_types[:, 1] == 1
            aromatic_mask = bond_types[:, 3] == 1

            bo_by_type = {
                'Single': mayer_bo[single_mask].tolist() if np.any(single_mask) else [],
                'Double': mayer_bo[double_mask].tolist() if np.any(double_mask) else [],
                'Aromatic': mayer_bo[aromatic_mask].tolist() if np.any(aromatic_mask) else []
            }

            # Box plot
            bp_data = [v for v in bo_by_type.values() if len(v) > 0]
            bp_labels = [k for k, v in bo_by_type.items() if len(v) > 0]

            if bp_data:
                bp = ax3.boxplot(bp_data, labels=bp_labels, patch_artist=True)
                colors_bp = ['#42a5f5', '#66bb6a', '#ab47bc']
                for patch, color in zip(bp['boxes'], colors_bp[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax3.set_ylabel('Mayer Bond Order')
            ax3.set_title('Bond Order by Type\n(Quantum Chemical)', fontsize=12, fontweight='bold')
            ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Single (~1.0)')
            ax3.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='Double (~2.0)')
            ax3.legend(loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.show()

            # Summary
            display(HTML(f"""
                <div style='background:#e8f5e9;padding:15px;border-radius:10px;margin-top:10px;'>
                    <h4>🔬 Bond Order Analysis: {mol_name}</h4>
                    <table style='width:100%;'>
                        <tr><td><b>Total Bonds:</b></td><td>{int(sum(bond_counts.values()))}</td></tr>
                        <tr><td><b>Mean Mayer BO:</b></td><td>{np.mean(mayer_bo):.3f}</td></tr>
                        <tr><td><b>Max Mayer BO:</b></td><td>{np.max(mayer_bo):.3f}</td></tr>
                        <tr><td><b>Strongest Bond:</b></td><td>Atoms {edge_idx[0, np.argmax(mayer_bo)]}-{edge_idx[1, np.argmax(mayer_bo)]}</td></tr>
                    </table>
                    <p style='margin-top:10px;'><b>Interpretation:</b> Mayer BO ~1.0 = single bond, ~1.5 = aromatic, ~2.0 = double bond</p>
                </div>
            """))

    edge_heatmap_controls = widgets.VBox([
        widgets.HTML("<h3>🔗 Edge Importance Heatmap</h3>"),
        widgets.HTML("<p>Visualize Mayer bond orders from DFT calculations - quantum chemical bond strength.</p>"),
        edge_heatmap_output
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: LAYER-WISE ATTENTION COMPARISON
    # ═══════════════════════════════════════════════════════════════════════

    def update_layer_compare(change=None):
        with layer_compare_output:
            clear_output(wait=True)

            mol_name = mol_dropdown.value
            graph = mol_graph_lookup.get(mol_name)
            if not graph:
                return

            mol = Chem.MolFromSmiles(graph.smiles)
            if not mol:
                return

            n_atoms = mol.GetNumAtoms()

            # Get attention from both layers
            try:
                attn_data = get_attention_weights(model, graph)
            except Exception as e:
                display(HTML(f"<div style='color:red;'>Error: {e}</div>"))
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Process each layer
            for layer_idx, layer_name in enumerate(['layer1', 'layer2']):
                edge_idx = attn_data[layer_name]['edge_index']
                attention = attn_data[layer_name]['attention']

                n_heads = attention.shape[1] if len(attention.shape) > 1 else 1

                # 1. Attention distribution per head (left column)
                ax = axes[layer_idx, 0]

                if n_heads > 1:
                    head_means = [attention[:, h].mean() for h in range(n_heads)]
                    head_stds = [attention[:, h].std() for h in range(n_heads)]

                    x = range(n_heads)
                    ax.bar(x, head_means, yerr=head_stds, color=['#1976d2', '#43a047', '#e53935', '#ab47bc'][:n_heads],
                          capsize=5, alpha=0.8)
                    ax.set_xlabel('Attention Head')
                    ax.set_ylabel('Mean Attention Weight')
                    ax.set_title(f'{layer_name.replace("layer", "Layer ")} - Head Comparison', fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'Head {i}' for i in range(n_heads)])
                else:
                    ax.hist(attention.flatten(), bins=30, color='#1976d2', alpha=0.7, edgecolor='white')
                    ax.set_xlabel('Attention Weight')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{layer_name.replace("layer", "Layer ")} - Attention Distribution', fontweight='bold')

                # 2. Node-level attention (right column)
                ax2 = axes[layer_idx, 1]

                # Aggregate attention per node (sum of incoming attention)
                node_attention = np.zeros(n_atoms)
                for i in range(edge_idx.shape[1]):
                    dst = edge_idx[1, i]
                    if dst < n_atoms:
                        if len(attention.shape) > 1:
                            node_attention[dst] += attention[i].mean()  # Mean across heads
                        else:
                            node_attention[dst] += attention[i]

                # Normalize
                if node_attention.max() > 0:
                    node_attention = node_attention / node_attention.max()

                atom_labels = [f'{mol.GetAtomWithIdx(i).GetSymbol()}{i}' for i in range(n_atoms)]
                colors = ['#e53935' if node_attention[i] > 0.7 else
                         '#ff9800' if node_attention[i] > 0.4 else '#42a5f5'
                         for i in range(n_atoms)]

                ax2.barh(range(n_atoms), node_attention, color=colors)
                ax2.set_yticks(range(n_atoms))
                ax2.set_yticklabels(atom_labels, fontsize=8)
                ax2.set_xlabel('Normalized Incoming Attention')
                ax2.set_title(f'{layer_name.replace("layer", "Layer ")} - Node Attention', fontweight='bold')
                ax2.invert_yaxis()
                ax2.axvline(0.7, color='red', linestyle='--', alpha=0.3)
                ax2.axvline(0.4, color='orange', linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Layer comparison insights
            display(HTML("""
                <div style='background:#fff3e0;padding:15px;border-radius:10px;margin-top:10px;'>
                    <h4>📊 Layer-wise Interpretation:</h4>
                    <table style='width:100%;'>
                        <tr><td><b>Layer 1:</b></td><td>Local neighborhood aggregation - learns immediate atomic environment</td></tr>
                        <tr><td><b>Layer 2:</b></td><td>Extended context - aggregates information from 2-hop neighbors</td></tr>
                    </table>
                    <p style='margin-top:10px;'>
                        <b>Key insight:</b> If attention patterns change significantly between layers,
                        the model is learning hierarchical representations (local → global).
                    </p>
                </div>
            """))

    layer_compare_controls = widgets.VBox([
        widgets.HTML("<h3>📊 Layer-wise Attention Comparison</h3>"),
        widgets.HTML("<p>Compare how attention evolves from Layer 1 to Layer 2.</p>"),
        layer_compare_output
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: FEATURE ATTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════

    def update_feature_attr(change=None):
        with feature_attr_output:
            clear_output(wait=True)

            mol_name = mol_dropdown.value
            graph = mol_graph_lookup.get(mol_name)
            if not graph:
                return

            mol = Chem.MolFromSmiles(graph.smiles)
            if not mol:
                return

            n_atoms = mol.GetNumAtoms()

            # Get feature analysis
            feat_data = get_node_features_analysis(graph)

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1. RDKit vs DFT feature magnitude per atom
            ax1 = axes[0, 0]
            x = range(n_atoms)
            width = 0.35

            ax1.bar([i - width/2 for i in x], feat_data['rdkit_mean'], width, label='RDKit', color='#1976d2')
            ax1.bar([i + width/2 for i in x], feat_data['dft_mean'], width, label='DFT', color='#43a047')

            atom_labels = [f'{mol.GetAtomWithIdx(i).GetSymbol()}{i}' for i in range(n_atoms)]
            ax1.set_xticks(x)
            ax1.set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
            ax1.set_ylabel('Mean |Feature Value|')
            ax1.set_title('RDKit vs DFT Feature Magnitude per Atom', fontweight='bold')
            ax1.legend()

            # 2. DFT Feature Heatmap
            ax2 = axes[0, 1]
            dft_features = feat_data['dft'][:n_atoms]

            # DFT feature names
            dft_names = ['Mulliken', 'Löwdin', 'Hirshfeld', 'Fukui+', 'Fukui-',
                        'Fukui0', 'Electrophil', 'Nucleophil', 'Spin', 'ESP'][:dft_features.shape[1]]

            im = ax2.imshow(dft_features.T, cmap='RdBu_r', aspect='auto')
            ax2.set_xticks(range(n_atoms))
            ax2.set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
            ax2.set_yticks(range(len(dft_names)))
            ax2.set_yticklabels(dft_names, fontsize=9)
            plt.colorbar(im, ax=ax2, label='Normalized Value')
            ax2.set_title('DFT Features per Atom (Normalized)', fontweight='bold')

            # 3. Feature importance by type
            ax3 = axes[1, 0]

            # Calculate variance of each DFT feature across atoms
            dft_variance = np.var(dft_features, axis=0)
            sorted_idx = np.argsort(dft_variance)[::-1]

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(dft_names)))
            ax3.barh([dft_names[i] for i in sorted_idx], dft_variance[sorted_idx], color=colors)
            ax3.set_xlabel('Variance Across Atoms')
            ax3.set_title('DFT Feature Variability\n(Higher = More Discriminative)', fontweight='bold')

            # 4. Correlation between DFT features and atom importance
            ax4 = axes[1, 1]

            # Get atom importance
            importance = get_atom_importance(model, graph)[:n_atoms]

            correlations = []
            for i in range(dft_features.shape[1]):
                if len(importance) == len(dft_features[:, i]):
                    corr = np.corrcoef(importance, dft_features[:, i])[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)

            colors = ['#4caf50' if c > 0 else '#f44336' for c in correlations]
            ax4.barh(dft_names, correlations, color=colors)
            ax4.set_xlabel('Correlation with Atom Importance')
            ax4.set_title('DFT Features vs Model Attention\n(Which Features Drive Predictions?)', fontweight='bold')
            ax4.axvline(0, color='black', linewidth=0.5)

            plt.tight_layout()
            plt.show()

            # Key insights
            top_corr_idx = np.argmax(np.abs(correlations))
            display(HTML(f"""
                <div style='background:#e3f2fd;padding:15px;border-radius:10px;margin-top:10px;'>
                    <h4>🔬 Feature Attribution Insights: {mol_name}</h4>
                    <ul>
                        <li><b>Most variable DFT feature:</b> {dft_names[sorted_idx[0]]} (variance: {dft_variance[sorted_idx[0]]:.4f})</li>
                        <li><b>Feature most correlated with importance:</b> {dft_names[top_corr_idx]} (r = {correlations[top_corr_idx]:.3f})</li>
                        <li><b>Interpretation:</b> Positive correlation means higher feature value → higher attention</li>
                    </ul>
                </div>
            """))

    feature_attr_controls = widgets.VBox([
        widgets.HTML("<h3>🧬 Feature Attribution Analysis</h3>"),
        widgets.HTML("<p>Understand which node features (RDKit vs DFT) drive model predictions.</p>"),
        feature_attr_output
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5: DFT FEATURE DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════

    dft_feature_select = widgets.Dropdown(
        options=['mulliken_charges', 'fukui_plus', 'fukui_minus', 'esp_at_nuclei',
                'local_electrophilicity', 'local_nucleophilicity'],
        value='fukui_plus',
        description='DFT Feature:',
        layout=widgets.Layout(width='300px')
    )

    def update_dft_analysis(change=None):
        with dft_analysis_output:
            clear_output(wait=True)

            if full_data is None:
                display(HTML("<div style='color:orange;'>⚠️ DFT data not provided for deep dive analysis</div>"))
                return

            mol_name = mol_dropdown.value
            graph = mol_graph_lookup.get(mol_name)
            if not graph:
                return

            mol = Chem.MolFromSmiles(graph.smiles)
            if not mol:
                return

            n_atoms = mol.GetNumAtoms()
            feature_name = dft_feature_select.value

            # Get DFT feature from full_data
            mol_key = mol_name + '_optimized' if mol_name + '_optimized' in full_data else mol_name
            if mol_key not in full_data:
                display(HTML(f"<div style='color:orange;'>⚠️ DFT data not found for {mol_name}</div>"))
                return

            atom_props = full_data[mol_key].get('atom_properties', {})
            feature_values = atom_props.get(feature_name, [])[:n_atoms]

            if len(feature_values) == 0:
                display(HTML(f"<div style='color:orange;'>⚠️ Feature {feature_name} not available</div>"))
                return

            # Pad if necessary
            while len(feature_values) < n_atoms:
                feature_values.append(0)
            feature_values = np.array(feature_values[:n_atoms])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 1. Bar chart of feature values
            atom_labels = [f'{mol.GetAtomWithIdx(i).GetSymbol()}{i}' for i in range(n_atoms)]

            # Color by value
            norm = plt.Normalize(feature_values.min(), feature_values.max())
            colors = plt.cm.RdBu_r(norm(feature_values))

            ax1.bar(range(n_atoms), feature_values, color=colors)
            ax1.set_xticks(range(n_atoms))
            ax1.set_xticklabels(atom_labels, rotation=45, ha='right')
            ax1.set_ylabel(feature_name.replace('_', ' ').title())
            ax1.set_title(f'{feature_name.replace("_", " ").title()} per Atom', fontweight='bold')
            ax1.axhline(0, color='black', linewidth=0.5)

            # Highlight max/min
            max_idx = np.argmax(feature_values)
            min_idx = np.argmin(feature_values)
            ax1.annotate(f'Max', (max_idx, feature_values[max_idx]),
                        xytext=(5, 10), textcoords='offset points', fontweight='bold', color='red')
            ax1.annotate(f'Min', (min_idx, feature_values[min_idx]),
                        xytext=(5, -15), textcoords='offset points', fontweight='bold', color='blue')

            # 2. Molecule with feature coloring
            # Create color map for atoms
            atom_colors = {}
            for i in range(n_atoms):
                val = feature_values[i]
                # Map to 0-1 range
                if feature_values.max() != feature_values.min():
                    normalized = (val - feature_values.min()) / (feature_values.max() - feature_values.min())
                else:
                    normalized = 0.5
                # Use RdBu colormap
                rgb = plt.cm.RdBu_r(normalized)[:3]
                atom_colors[i] = rgb

            img = Draw.MolToImage(mol, size=(500, 400), highlightAtoms=list(range(n_atoms)),
                                 highlightAtomColors=atom_colors)
            ax2.imshow(img)
            ax2.axis('off')
            ax2.set_title(f'Molecule colored by {feature_name.replace("_", " ").title()}', fontweight='bold')

            plt.tight_layout()
            plt.show()

            # Feature explanation
            explanations = {
                'mulliken_charges': "Partial atomic charges from Mulliken population analysis. Positive = electron deficient, Negative = electron rich.",
                'fukui_plus': "Reactivity descriptor for nucleophilic attack (electrophilic sites). High values = susceptible to nucleophiles.",
                'fukui_minus': "Reactivity descriptor for electrophilic attack (nucleophilic sites). High values = susceptible to electrophiles.",
                'esp_at_nuclei': "Electrostatic potential at nuclear positions. Indicates local charge environment.",
                'local_electrophilicity': "Site-specific electrophilicity index. High = good electron acceptor.",
                'local_nucleophilicity': "Site-specific nucleophilicity index. High = good electron donor."
            }

            display(HTML(f"""
                <div style='background:#fff8e1;padding:15px;border-radius:10px;margin-top:10px;'>
                    <h4>📖 {feature_name.replace('_', ' ').title()}</h4>
                    <p>{explanations.get(feature_name, 'DFT-derived atomic property.')}</p>
                    <table style='width:100%;margin-top:10px;'>
                        <tr><td><b>Max value:</b></td><td>{feature_values[max_idx]:.4f} at {atom_labels[max_idx]}</td></tr>
                        <tr><td><b>Min value:</b></td><td>{feature_values[min_idx]:.4f} at {atom_labels[min_idx]}</td></tr>
                        <tr><td><b>Mean:</b></td><td>{np.mean(feature_values):.4f}</td></tr>
                        <tr><td><b>Std:</b></td><td>{np.std(feature_values):.4f}</td></tr>
                    </table>
                </div>
            """))

    dft_feature_select.observe(update_dft_analysis, names='value')

    dft_analysis_controls = widgets.VBox([
        widgets.HTML("<h3>⚛️ DFT Feature Deep Dive</h3>"),
        widgets.HTML("<p>Explore individual DFT features (Fukui, charges, ESP) for selected molecule.</p>"),
        dft_feature_select,
        dft_analysis_output
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 6: MOLECULAR NETWORK VIEW
    # ═══════════════════════════════════════════════════════════════════════

    network_btn = widgets.Button(description='🌐 Generate Network', button_style='primary')

    def update_network(btn=None):
        with network_output:
            clear_output(wait=True)

            mol_name = mol_dropdown.value
            graph = mol_graph_lookup.get(mol_name)
            if not graph:
                return

            mol = Chem.MolFromSmiles(graph.smiles)
            if not mol:
                return

            n_atoms = mol.GetNumAtoms()

            # Get all data
            importance = get_atom_importance(model, graph)[:n_atoms]
            edge_data = get_edge_features_analysis(graph, mol_name)

            # Create comprehensive network
            fig, ax = plt.subplots(figsize=(12, 10))

            G = nx.Graph()

            # Add nodes with attributes
            for i in range(n_atoms):
                atom = mol.GetAtomWithIdx(i)
                G.add_node(i,
                          symbol=atom.GetSymbol(),
                          importance=importance[i],
                          aromatic=atom.GetIsAromatic())

            # Add edges with Mayer bond orders
            edge_idx = edge_data['edge_index']
            mayer_bo = edge_data['mayer_bo']

            added_edges = set()
            for i in range(edge_idx.shape[1]):
                src, dst = edge_idx[0, i], edge_idx[1, i]
                if src < n_atoms and dst < n_atoms:
                    edge_key = tuple(sorted([src, dst]))
                    if edge_key not in added_edges:
                        G.add_edge(src, dst, weight=mayer_bo[i])
                        added_edges.add(edge_key)

            # Layout
            pos = nx.kamada_kawai_layout(G)

            # Draw edges (width = Mayer BO, color = importance average)
            edges = G.edges(data=True)
            edge_weights = [d['weight'] for _, _, d in edges]
            edge_colors = [(importance[u] + importance[v]) / 2 for u, v, _ in edges]

            if edges:
                nx.draw_networkx_edges(G, pos,
                                      width=[w * 2 for w in edge_weights],
                                      edge_color=edge_colors,
                                      edge_cmap=plt.cm.Reds,
                                      alpha=0.8, ax=ax)

            # Draw nodes (size = importance, color = element)
            node_sizes = [300 + importance[i] * 700 for i in range(n_atoms)]

            element_colors = {
                'C': '#4a4a4a', 'N': '#3050f8', 'O': '#ff0d0d',
                'S': '#ffff30', 'F': '#90e050', 'Cl': '#1ff01f',
                'Br': '#a62929', 'I': '#940094', 'P': '#ff8000'
            }
            node_colors = [element_colors.get(mol.GetAtomWithIdx(i).GetSymbol(), '#808080') for i in range(n_atoms)]

            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                  edgecolors='black', linewidths=1.5, ax=ax)

            # Labels
            labels = {i: f'{mol.GetAtomWithIdx(i).GetSymbol()}' for i in range(n_atoms)}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold',
                                   font_color='white', ax=ax)

            ax.set_title(f'🌐 Molecular Network: {mol_name}\n(Node size = Importance, Edge width = Mayer BO)',
                        fontsize=14, fontweight='bold')
            ax.axis('off')

            # Legend
            legend_elements = [
                mpatches.Patch(color=element_colors['C'], label='Carbon'),
                mpatches.Patch(color=element_colors['N'], label='Nitrogen'),
                mpatches.Patch(color=element_colors['O'], label='Oxygen'),
                mpatches.Patch(color=element_colors['S'], label='Sulfur'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

            plt.tight_layout()
            plt.show()

            display(HTML("""
                <div style='background:#e8eaf6;padding:15px;border-radius:10px;margin-top:10px;'>
                    <h4>🔍 Network Visualization Guide:</h4>
                    <ul>
                        <li><b>Node size:</b> Larger = higher atom importance (more attention from GAT)</li>
                        <li><b>Node color:</b> Element type (C=gray, N=blue, O=red, S=yellow)</li>
                        <li><b>Edge width:</b> Thicker = stronger Mayer bond order (DFT)</li>
                        <li><b>Edge color:</b> Redder = higher combined importance of connected atoms</li>
                    </ul>
                </div>
            """))

    network_btn.on_click(update_network)

    network_controls = widgets.VBox([
        widgets.HTML("<h3>🌐 Combined Network Visualization</h3>"),
        widgets.HTML("<p>Comprehensive view combining atom importance + bond orders + molecular structure.</p>"),
        network_btn,
        network_output
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE TABS
    # ═══════════════════════════════════════════════════════════════════════

    tabs = widgets.Tab(children=[
        attention_controls,
        edge_heatmap_controls,
        layer_compare_controls,
        feature_attr_controls,
        dft_analysis_controls,
        network_controls
    ])

    tabs.set_title(0, '🔀 Attention Flow')
    tabs.set_title(1, '🔗 Edge Heatmap')
    tabs.set_title(2, '📊 Layer Compare')
    tabs.set_title(3, '🧬 Feature Attribution')
    tabs.set_title(4, '⚛️ DFT Deep Dive')
    tabs.set_title(5, '🌐 Network View')

    # Molecule selector callback
    def on_mol_change(change):
        update_attention_flow()
        update_edge_heatmap()
        update_layer_compare()
        update_feature_attr()
        update_dft_analysis()

    mol_dropdown.observe(on_mol_change, names='value')

    # Header
    header = widgets.HTML("""
        <div style='background:linear-gradient(135deg,#1a237e,#311b92);padding:20px;border-radius:12px;color:white;margin-bottom:15px;'>
            <h1 style='margin:0;'> GAT-Specific Analysis Dashboard</h1>
            <p style='margin:5px 0 0 0;opacity:0.8;'>Attention Flow • Edge Importance • Layer Analysis • Feature Attribution • DFT Integration</p>
        </div>
    """)

    # Molecule selector at top
    mol_selector = widgets.HBox([
        widgets.HTML("<b style='margin-right:10px;'>Select Molecule:</b>"),
        mol_dropdown
    ], layout=widgets.Layout(margin='0 0 15px 0'))

    layout = widgets.VBox([header, mol_selector, tabs])
    display(layout)

    # Initialize
    update_attention_flow()
    update_edge_heatmap()