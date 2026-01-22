"""
Advanced Analysis Dashboard for Molecular Data.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition as rgd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

# Try to import UMAP (optional)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

def create_advanced_dashboard(dataset, results):
    """
    Advanced analysis dashboard with statistical visualizations and molecular analysis.
    
    Parameters:
    -----------
    dataset : list
        List of molecule objects (must have .name and .smiles attributes)
    results : pd.DataFrame
        DataFrame containing 'name' and 'binding_score' columns
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # PREPARE DATA
    # ═══════════════════════════════════════════════════════════════════════
    
    if HAS_UMAP is False:
        print("⚠️ UMAP not installed. Using t-SNE fallback. Install with: pip install umap-learn")

    # Calculate molecular descriptors for all molecules
    descriptor_data = []
    valid_mols = []
    valid_names = []
    valid_scores = []
    
    for g in dataset:
        try:
            # Handle if dataset items are objects or dicts
            name = getattr(g, 'name', g.get('name') if isinstance(g, dict) else str(g))
            smiles = getattr(g, 'smiles', g.get('smiles') if isinstance(g, dict) else None)
            
            if not smiles:
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                score_row = results[results['name'] == name]
                score = score_row['binding_score'].values[0] if len(score_row) > 0 else 0.5
                
                desc = {
                    'name': name,
                    'smiles': smiles,
                    'score': score,
                    'HBA': Descriptors.NumHAcceptors(mol),
                    'HBD': Descriptors.NumHDonors(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'MW': Descriptors.MolWt(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'RotBonds': Descriptors.NumRotatableBonds(mol),
                    'AromaticRings': Descriptors.NumAromaticRings(mol),
                    'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
                    'FractionCSP3': Descriptors.FractionCSP3(mol),
                }
                descriptor_data.append(desc)
                valid_mols.append(mol)
                valid_names.append(name)
                valid_scores.append(score)
        except Exception as e:
            continue
    
    if not descriptor_data:
        return widgets.HTML("<div style='color:red'>No valid molecule data found to analyze.</div>")

    desc_df = pd.DataFrame(descriptor_data)
    feature_cols = ['HBA', 'HBD', 'LogP', 'MW', 'TPSA', 'RotBonds', 'AromaticRings', 'HeavyAtoms', 'FractionCSP3']
    
    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT AREAS
    # ═══════════════════════════════════════════════════════════════════════
    
    radar_output = widgets.Output()
    pca_output = widgets.Output()
    correlation_output = widgets.Output()
    breakdown_output = widgets.Output()
    substructure_output = widgets.Output()
    scaffold_output = widgets.Output()
    rgroup_output = widgets.Output()
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: RADAR CHART
    # ═══════════════════════════════════════════════════════════════════════
    
    mol_dropdown = widgets.Dropdown(
        options=[(f"{row['name']} ({row['score']:.2f})", row['name']) for _, row in desc_df.iterrows()],
        description='Molecule:',
        layout=widgets.Layout(width='400px')
    )
    compare_dropdown = widgets.Dropdown(
        options=[('None', None)] + [(f"{row['name']} ({row['score']:.2f})", row['name']) for _, row in desc_df.iterrows()],
        value=None,
        description='Compare:',
        layout=widgets.Layout(width='400px')
    )
    
    def update_radar(change):
        with radar_output:
            clear_output(wait=True)
            
            mol_name = mol_dropdown.value
            compare_name = compare_dropdown.value
            
            if not mol_name:
                return

            mol_row = desc_df[desc_df['name'] == mol_name].iloc[0]
            
            # Normalize values for radar
            categories = ['HBA', 'HBD', 'LogP', 'MW', 'TPSA', 'Aromatic', 'RotBonds']
            max_vals = [10, 5, 8, 600, 150, 4, 10]
            
            # Helper to safely normalize
            def get_norm_val(row, cat, max_v):
                col_map = {'Aromatic': 'AromaticRings'}
                col = col_map.get(cat, cat)
                val = row[col]
                if cat == 'LogP':
                    return (val + 2) / max_v
                return val / max_v

            values1 = [get_norm_val(mol_row, cat, max_v) for cat, max_v in zip(categories, max_vals)]
            values1 = [min(max(v, 0), 1) for v in values1]  # Clamp 0-1
            values1 += values1[:1]  # Close polygon
            
            angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            ax.plot(angles, values1, 'o-', linewidth=2, color='#1976d2', label=mol_name)
            ax.fill(angles, values1, alpha=0.25, color='#1976d2')
            
            if compare_name:
                comp_row = desc_df[desc_df['name'] == compare_name].iloc[0]
                values2 = [get_norm_val(comp_row, cat, max_v) for cat, max_v in zip(categories, max_vals)]
                values2 = [min(max(v, 0), 1) for v in values2]
                values2 += values2[:1]
                
                ax.plot(angles, values2, 'o-', linewidth=2, color='#e53935', label=compare_name)
                ax.fill(angles, values2, alpha=0.15, color='#e53935')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title('🕷️ Molecular Property Radar Chart', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.show()
            
            # Property table
            col_map = {'Aromatic': 'AromaticRings'}
            display(HTML(f"<h4>📋 Property Values</h4>"))
            table_html = "<table style='width:100%;border-collapse:collapse;'><thead><tr><th>Property</th><th>{}</th>".format(mol_name)
            if compare_name:
                table_html += f"<th>{compare_name}</th>"
            table_html += "</tr></thead><tbody>"
            
            for i, cat in enumerate(categories):
                col = col_map.get(cat, cat)
                val1 = mol_row[col]
                table_html += f"<tr><td><b>{cat}</b></td><td>{val1:.2f}</td>"
                if compare_name:
                    comp_row = desc_df[desc_df['name'] == compare_name].iloc[0]
                    val2 = comp_row[col]
                    table_html += f"<td>{val2:.2f}</td>"
                table_html += "</tr>"
            table_html += "</tbody></table>"
            display(HTML(table_html))
    
    mol_dropdown.observe(update_radar, names='value')
    compare_dropdown.observe(update_radar, names='value')
    
    radar_controls = widgets.VBox([
        widgets.HTML("<h3>🕷️ Radar Chart Comparison</h3>"),
        widgets.HBox([mol_dropdown, compare_dropdown]),
        radar_output
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: PCA / UMAP VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    dim_method = widgets.Dropdown(
        options=['PCA', 't-SNE'] + (['UMAP'] if HAS_UMAP else []),
        value='PCA',
        description='Method:',
        layout=widgets.Layout(width='200px')
    )
    color_by = widgets.Dropdown(
        options=['score', 'LogP', 'MW', 'HBA', 'HBD', 'TPSA', 'AromaticRings'],
        value='score',
        description='Color by:',
        layout=widgets.Layout(width='200px')
    )
    run_dim_btn = widgets.Button(description='🔄 Run Analysis', button_style='primary')
    
    def run_dimensionality(btn):
        with pca_output:
            clear_output(wait=True)
            display(HTML("<h3>⏳ Computing dimensionality reduction...</h3>"))
            
            # Prepare features
            X = desc_df[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            method = dim_method.value
            if method == 'PCA':
                reducer = PCA(n_components=2)
                X_2d = reducer.fit_transform(X_scaled)
                exp_var = reducer.explained_variance_ratio_
                title = f'PCA (Explained Var: {exp_var[0]:.1%}, {exp_var[1]:.1%})'
            elif method == 't-SNE' and HAS_TSNE:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, max(1, len(X_scaled)-1)))
                X_2d = reducer.fit_transform(X_scaled)
                title = 't-SNE Projection'
            elif method == 'UMAP' and HAS_UMAP:
                reducer = UMAP(n_components=2, random_state=42)
                X_2d = reducer.fit_transform(X_scaled)
                title = 'UMAP Projection'
            else:
                reducer = PCA(n_components=2)
                X_2d = reducer.fit_transform(X_scaled)
                title = 'PCA (fallback)'
            
            clear_output(wait=True)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            color_col = color_by.value
            colors = desc_df[color_col].values
            
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='viridis', 
                                s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
            
            # Highlight top 5
            try:
                top5_idx = desc_df.nlargest(5, 'score').index.tolist()
                for idx in top5_idx:
                    ax.annotate(desc_df.iloc[idx]['name'], 
                               (X_2d[idx, 0], X_2d[idx, 1]),
                               fontsize=9, fontweight='bold',
                               xytext=(5, 5), textcoords='offset points')
            except:
                pass
            
            plt.colorbar(scatter, label=color_col)
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            ax.set_title(f'📊 {title} (colored by {color_col})', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            display(HTML("<p>🔵 Top 5 molecules are labeled. Hover to see clusters.</p>"))
    
    run_dim_btn.on_click(run_dimensionality)
    
    pca_controls = widgets.VBox([
        widgets.HTML("<h3>📊 Dimensionality Reduction</h3>"),
        widgets.HBox([dim_method, color_by, run_dim_btn]),
        pca_output
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: CORRELATION HEATMAP
    # ═══════════════════════════════════════════════════════════════════════
    
    def show_correlation(btn=None):
        with correlation_output:
            clear_output(wait=True)
            
            # Compute correlation matrix
            corr_cols = ['score'] + feature_cols
            corr_matrix = desc_df[corr_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Labels
            ax.set_xticks(range(len(corr_cols)))
            ax.set_yticks(range(len(corr_cols)))
            ax.set_xticklabels(corr_cols, rotation=45, ha='right')
            ax.set_yticklabels(corr_cols)
            
            # Annotate
            for i in range(len(corr_cols)):
                for j in range(len(corr_cols)):
                    val = corr_matrix.iloc[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
            
            plt.colorbar(im, label='Correlation')
            ax.set_title('🔥 Property Correlation Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Key insights
            display(HTML("<h4>🔍 Key Correlations with Binding Score:</h4>"))
            try:
                score_corrs = corr_matrix['score'].drop('score').sort_values(key=abs, ascending=False)
                for prop, corr in score_corrs.head(5).items():
                    direction = "📈 Positive" if corr > 0 else "📉 Negative"
                    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                    display(HTML(f"<div style='padding:5px;'><b>{prop}</b>: {corr:.3f} ({direction}, {strength})</div>"))
            except:
                pass
    
    corr_btn = widgets.Button(description='📊 Show Correlation', button_style='info')
    corr_btn.on_click(show_correlation)
    
    corr_controls = widgets.VBox([
        widgets.HTML("<h3>🔥 Correlation Analysis</h3>"),
        corr_btn,
        correlation_output
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: SCORE BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════
    
    breakdown_dropdown = widgets.Dropdown(
        options=[(f"{row['name']} ({row['score']:.2f})", row['name']) for _, row in desc_df.iterrows()],
        description='Molecule:',
        layout=widgets.Layout(width='400px')
    )
    
    def update_breakdown(change):
        with breakdown_output:
            clear_output(wait=True)
            
            mol_name = breakdown_dropdown.value
            if not mol_name: return
            
            mol_row = desc_df[desc_df['name'] == mol_name].iloc[0]
            mol = Chem.MolFromSmiles(mol_row['smiles'])
            
            # Calculate individual scores (matching dashboard scoring)
            weights = {'Acid': 5.0, 'LogP': 3.0, 'HBA': 2.0, 'HBD': 2.0, 'Aromatic': 2.0, 'MW': 1.5}
            
            scores = {}
            
            # Acid
            has_acid = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[OH]')) or \
                      mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N'))
            scores['Acid'] = 1.0 if has_acid else 0.0
            
            # LogP
            logp = mol_row['LogP']
            scores['LogP'] = 1.0 if 1.5 <= logp <= 5.5 else 0.3
            
            # HBA
            scores['HBA'] = 1.0 - min(abs(mol_row['HBA'] - 3) / 4, 1.0)
            
            # HBD
            scores['HBD'] = 1.0 - min(abs(mol_row['HBD'] - 2) / 3, 1.0)
            
            # Aromatic
            scores['Aromatic'] = 1.0 if mol_row['AromaticRings'] >= 1 else 0.3
            
            # MW
            mw = mol_row['MW']
            scores['MW'] = 1.0 if 150 <= mw <= 500 else 0.7 if mw <= 600 else 0.4
            
            # Calculate contributions
            contributions = {k: scores[k] * weights[k] for k in weights}
            total = sum(contributions.values())
            normalized = sum(weights.values())
            final_score = total / normalized
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart of contributions
            props = list(contributions.keys())
            contribs = list(contributions.values())
            colors = ['#4caf50' if scores[p] >= 0.7 else '#ff9800' if scores[p] >= 0.4 else '#f44336' for p in props]
            
            ax1.barh(props, contribs, color=colors, edgecolor='black', linewidth=0.5)
            ax1.set_xlabel('Weighted Contribution')
            ax1.set_title(f'Score Breakdown: {mol_name}', fontweight='bold')
            ax1.axvline(x=sum(contribs)/len(contribs), color='blue', linestyle='--', label='Mean')
            
            # Pie chart
            ax2.pie(contribs, labels=props, autopct='%1.1f%%', colors=colors, 
                   explode=[0.05]*len(props), startangle=90)
            ax2.set_title(f'Contribution Distribution (Score: {final_score:.3f})', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Details table
            display(HTML("<h4>📋 Detailed Breakdown</h4>"))
            table = "<table style='width:100%;'><tr><th>Property</th><th>Raw Score</th><th>Weight</th><th>Contribution</th><th>Status</th></tr>"
            for p in props:
                status = "✅" if scores[p] >= 0.7 else "⚠️" if scores[p] >= 0.4 else "❌"
                table += f"<tr><td>{p}</td><td>{scores[p]:.2f}</td><td>{weights[p]}</td><td>{contributions[p]:.2f}</td><td>{status}</td></tr>"
            table += f"<tr style='background:#e3f2fd;font-weight:bold;'><td>TOTAL</td><td>-</td><td>{normalized:.1f}</td><td>{total:.2f}</td><td>{final_score:.3f}</td></tr>"
            table += "</table>"
            display(HTML(table))
    
    breakdown_dropdown.observe(update_breakdown, names='value')
    
    breakdown_controls = widgets.VBox([
        widgets.HTML("<h3>📊 Score Breakdown Analysis</h3>"),
        breakdown_dropdown,
        breakdown_output
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5: SUBSTRUCTURE SEARCH
    # ═══════════════════════════════════════════════════════════════════════
    
    smarts_input = widgets.Text(
        value='c1ccccc1',
        description='SMARTS:',
        placeholder='Enter SMARTS pattern',
        layout=widgets.Layout(width='400px')
    )
    
    preset_patterns = widgets.Dropdown(
        options=[
            ('Benzene Ring', 'c1ccccc1'),
            ('Carboxylic Acid', 'C(=O)[OH]'),
            ('Sulfonamide', 'S(=O)(=O)N'),
            ('Hydroxyl', '[OH]'),
            ('Primary Amine', '[NH2]'),
            ('Amide', 'C(=O)N'),
            ('Ester', 'C(=O)O[C,c]'),
            ('Halogen', '[F,Cl,Br,I]'),
            ('Nitro', '[N+](=O)[O-]'),
            ('Ketone', '[CX3](=O)[C]'),
        ],
        value='c1ccccc1',
        description='Presets:',
        layout=widgets.Layout(width='300px')
    )
    
    search_btn = widgets.Button(description='🔍 Search', button_style='success')
    
    def use_preset(change):
        smarts_input.value = preset_patterns.value
    
    preset_patterns.observe(use_preset, names='value')
    
    def run_substructure_search(btn):
        with substructure_output:
            clear_output(wait=True)
            
            smarts = smarts_input.value
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    display(HTML("<div style='color:red;'>❌ Invalid SMARTS pattern</div>"))
                    return
            except:
                display(HTML("<div style='color:red;'>❌ Error parsing SMARTS</div>"))
                return
            
            matches = []
            for i, mol in enumerate(valid_mols):
                if mol.HasSubstructMatch(pattern):
                    match_atoms = mol.GetSubstructMatch(pattern)
                    matches.append({
                        'name': valid_names[i],
                        'score': valid_scores[i],
                        'mol': mol,
                        'match_atoms': match_atoms
                    })
            
            display(HTML(f"<h3>🔍 Found {len(matches)} molecules with pattern: <code>{smarts}</code></h3>"))
            
            if matches:
                # Sort by score
                matches = sorted(matches, key=lambda x: x['score'], reverse=True)
                
                # Show top 10 with highlighting
                display(HTML("<h4>Top 10 Matches (sorted by binding score):</h4>"))
                
                top_matches = matches[:10]
                mols_to_draw = [m['mol'] for m in top_matches]
                legends = [f"{m['name']}\n({m['score']:.2f})" for m in top_matches]
                
                # Highlight matched atoms
                highlight_atoms = [list(m['match_atoms']) for m in top_matches]
                
                img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=5, subImgSize=(250, 200),
                                          legends=legends, highlightAtomLists=highlight_atoms)
                display(img)
                
                # Summary
                avg_score = np.mean([m['score'] for m in matches])
                display(HTML(f"<div style='background:#e8f5e9;padding:15px;border-radius:10px;margin-top:15px;'>"
                           f"<b>📊 Match Statistics:</b><br>"
                           f"• Total matches: {len(matches)} / {len(valid_mols)} molecules<br>"
                           f"• Average binding score: {avg_score:.3f}<br>"
                           f"• Top scorer: {matches[0]['name']} ({matches[0]['score']:.3f})</div>"))
            else:
                display(HTML("<div style='color:orange;'>⚠️ No molecules contain this substructure</div>"))
    
    search_btn.on_click(run_substructure_search)
    
    substructure_controls = widgets.VBox([
        widgets.HTML("<h3>🔍 Substructure Search</h3>"),
        widgets.HBox([smarts_input, preset_patterns]),
        search_btn,
        substructure_output
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 6: SCAFFOLD ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    scaffold_btn = widgets.Button(description='🏗️ Analyze Scaffolds', button_style='info')
    top_n_scaffolds = widgets.IntSlider(value=10, min=5, max=20, description='Top N:')
    
    def run_scaffold_analysis(btn):
        with scaffold_output:
            clear_output(wait=True)
            display(HTML("<h3>⏳ Analyzing scaffolds...</h3>"))
            
            # Extract Murcko scaffolds
            scaffolds = {}
            for i, mol in enumerate(valid_mols):
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    core_smiles = Chem.MolToSmiles(core)
                    
                    if core_smiles not in scaffolds:
                        scaffolds[core_smiles] = {
                            'mol': core,
                            'members': [],
                            'scores': []
                        }
                    scaffolds[core_smiles]['members'].append(valid_names[i])
                    scaffolds[core_smiles]['scores'].append(valid_scores[i])
                except:
                    pass
            
            clear_output(wait=True)
            
            # Sort by frequency
            scaffold_list = []
            for smiles, data in scaffolds.items():
                scaffold_list.append({
                    'smiles': smiles,
                    'mol': data['mol'],
                    'count': len(data['members']),
                    'avg_score': np.mean(data['scores']),
                    'max_score': max(data['scores']),
                    'members': data['members']
                })
            
            scaffold_list = sorted(scaffold_list, key=lambda x: x['count'], reverse=True)
            
            display(HTML(f"<h3>🏗️ Found {len(scaffold_list)} unique scaffolds</h3>"))
            
            # Top scaffolds visualization
            top_n = top_n_scaffolds.value
            top_scaffolds = scaffold_list[:top_n]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart - frequency
            names = [s['smiles'][:20] + '...' if len(s['smiles']) > 20 else s['smiles'] for s in top_scaffolds]
            counts = [s['count'] for s in top_scaffolds]
            ax1.barh(range(len(top_scaffolds)), counts, color='#42a5f5')
            ax1.set_yticks(range(len(top_scaffolds)))
            ax1.set_yticklabels([f"Scaffold {i+1}" for i in range(len(top_scaffolds))])
            ax1.set_xlabel('Number of Molecules')
            ax1.set_title('Scaffold Frequency', fontweight='bold')
            ax1.invert_yaxis()
            
            # Bar chart - avg score
            avg_scores = [s['avg_score'] for s in top_scaffolds]
            ax2.barh(range(len(top_scaffolds)), avg_scores, color='#66bb6a')
            ax2.set_yticks(range(len(top_scaffolds)))
            ax2.set_yticklabels([f"Scaffold {i+1}" for i in range(len(top_scaffolds))])
            ax2.set_xlabel('Average Binding Score')
            ax2.set_title('Scaffold Performance', fontweight='bold')
            ax2.invert_yaxis()
            
            plt.tight_layout()
            plt.show()
            
            # Draw top scaffolds
            display(HTML("<h4>🏗️ Top Scaffold Structures:</h4>"))
            mols_to_draw = [s['mol'] for s in top_scaffolds[:10]]
            legends = [f"#{i+1}: {s['count']} mols\navg: {s['avg_score']:.2f}" for i, s in enumerate(top_scaffolds[:10])]
            img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=5, subImgSize=(200, 150), legends=legends)
            display(img)
            
            # Details table
            display(HTML("<h4>📋 Scaffold Details:</h4>"))
            for i, s in enumerate(top_scaffolds[:10]):
                members_str = ', '.join(s['members'][:5])
                if len(s['members']) > 5:
                    members_str += f'... (+{len(s["members"])-5} more)'
                display(HTML(f"<div style='padding:8px;border-bottom:1px solid #eee;'>"
                           f"<b>Scaffold {i+1}</b>: {s['count']} molecules | "
                           f"Avg: {s['avg_score']:.3f} | Max: {s['max_score']:.3f}<br>"
                           f"<small>{members_str}</small></div>"))
    
    scaffold_btn.on_click(run_scaffold_analysis)
    
    scaffold_controls = widgets.VBox([
        widgets.HTML("<h3>🏗️ Murcko Scaffold Analysis</h3>"),
        widgets.HBox([top_n_scaffolds, scaffold_btn]),
        scaffold_output
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 7: R-GROUP DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════════
    
    core_smarts = widgets.Text(
        value='c1ccc(cc1)C(C)C(=O)O',
        description='Core:',
        placeholder='Enter core SMILES/SMARTS',
        layout=widgets.Layout(width='500px')
    )
    
    rgroup_btn = widgets.Button(description='🧬 Decompose R-Groups', button_style='warning')
    
    def run_rgroup_decomposition(btn):
        with rgroup_output:
            clear_output(wait=True)
            
            core_input = core_smarts.value
            
            # Try as SMILES first, then SMARTS
            core = Chem.MolFromSmiles(core_input)
            if core is None:
                core = Chem.MolFromSmarts(core_input)
            
            if core is None:
                display(HTML("<div style='color:red;'>❌ Invalid core structure. Enter valid SMILES or SMARTS.</div>"))
                return
            
            display(HTML("<h3>⏳ Running R-Group Decomposition...</h3>"))
            
            # Find molecules containing the core
            matching_mols = []
            matching_names = []
            matching_scores = []
            
            for i, mol in enumerate(valid_mols):
                if mol.HasSubstructMatch(core):
                    matching_mols.append(mol)
                    matching_names.append(valid_names[i])
                    matching_scores.append(valid_scores[i])
            
            clear_output(wait=True)
            
            if len(matching_mols) < 2:
                display(HTML(f"<div style='color:orange;'>⚠️ Only {len(matching_mols)} molecules contain this core. Need at least 2 for R-Group analysis.</div>"))
                
                # Show core
                display(HTML("<h4>Core Structure:</h4>"))
                display(Draw.MolToImage(core, size=(300, 200)))
                return
            
            display(HTML(f"<h3>🧬 R-Group Decomposition</h3>"))
            display(HTML(f"<p>Found <b>{len(matching_mols)}</b> molecules containing the core</p>"))
            
            # Show core
            display(HTML("<h4>Core Structure:</h4>"))
            display(Draw.MolToImage(core, size=(300, 200)))
            
            try:
                # Perform R-Group decomposition
                rgroups, unmatched = rgd.RGroupDecompose([core], matching_mols, asSmiles=True)
                
                if rgroups:
                    # Create DataFrame
                    rg_df = pd.DataFrame(rgroups)
                    rg_df['name'] = matching_names[:len(rg_df)]
                    rg_df['score'] = matching_scores[:len(rg_df)]
                    rg_df = rg_df.sort_values('score', ascending=False)
                    
                    # Count R-groups
                    r_cols = [c for c in rg_df.columns if c.startswith('R')]
                    display(HTML(f"<p>Identified <b>{len(r_cols)}</b> R-group positions: {', '.join(r_cols)}</p>"))
                    
                    # Show table
                    display(HTML("<h4>📋 R-Group Table (Top 15):</h4>"))
                    display_cols = ['name', 'score'] + r_cols
                    display(rg_df[display_cols].head(15).style.background_gradient(subset=['score'], cmap='Greens'))
                    
                    # Analyze R-group diversity
                    display(HTML("<h4>📊 R-Group Diversity:</h4>"))
                    for r in r_cols:
                        unique_rgroups = rg_df[r].nunique()
                        top_rgroup = rg_df[r].value_counts().head(1)
                        display(HTML(f"<div style='padding:5px;'><b>{r}</b>: {unique_rgroups} unique groups | "
                                   f"Most common: <code>{top_rgroup.index[0]}</code> ({top_rgroup.values[0]} occurrences)</div>"))
                else:
                    display(HTML("<div style='color:orange;'>⚠️ R-Group decomposition failed. Try a different core.</div>"))
                    
            except Exception as e:
                display(HTML(f"<div style='color:red;'>❌ Error in R-Group decomposition: {str(e)}</div>"))
                display(HTML("<p>Try using a simpler core structure or adding explicit attachment points with [*]</p>"))
    
    rgroup_btn.on_click(run_rgroup_decomposition)
    
    # Preset cores
    core_presets = widgets.Dropdown(
        options=[
            ('Phenylpropanoic (NSAID-like)', 'c1ccc(cc1)C(C)C(=O)O'),
            ('Phenylacetic', 'c1ccccc1CC(=O)O'),
            ('Benzene', 'c1ccccc1'),
            ('Pyridine', 'c1ccncc1'),
            ('Indole', 'c1ccc2[nH]ccc2c1'),
        ],
        description='Presets:',
        layout=widgets.Layout(width='350px')
    )
    
    def use_core_preset(change):
        core_smarts.value = core_presets.value
    
    core_presets.observe(use_core_preset, names='value')
    
    rgroup_controls = widgets.VBox([
        widgets.HTML("<h3>🧬 R-Group Decomposition</h3>"),
        widgets.HTML("<p>Analyze structural variations around a common core scaffold.</p>"),
        widgets.HBox([core_smarts, core_presets]),
        rgroup_btn,
        rgroup_output
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # CREATE TABS
    # ═══════════════════════════════════════════════════════════════════════
    
    tabs = widgets.Tab(children=[
        radar_controls,
        pca_controls,
        corr_controls,
        breakdown_controls,
        substructure_controls,
        scaffold_controls,
        rgroup_controls
    ])
    
    tabs.set_title(0, '🕷️ Radar')
    tabs.set_title(1, '📊 PCA/UMAP')
    tabs.set_title(2, '🔥 Correlation')
    tabs.set_title(3, '📊 Breakdown')
    tabs.set_title(4, '🔍 Substructure')
    tabs.set_title(5, '🏗️ Scaffolds')
    tabs.set_title(6, '🧬 R-Groups')
    
    # Header
    header = widgets.HTML("""
        <div style='background:linear-gradient(135deg,#00695c,#004d40);padding:20px;border-radius:12px;color:white;margin-bottom:15px;'>
            <h1 style='margin:0;'>📊 Advanced Analysis Dashboard</h1>
            <p style='margin:5px 0 0 0;opacity:0.8;'>Statistical Visualizations • Substructure Search • Scaffold Analysis • R-Group Decomposition</p>
        </div>
    """)
    
    layout = widgets.VBox([header, tabs])
    display(layout)
    
    # Initialize first tabs
    # We trigger them slightly delayed or just rely on user interaction if data is large
    # But usually pre-loading one is fine
    update_radar(None)
    show_correlation()
    
    return layout
