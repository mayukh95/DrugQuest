# drugquest_dashboard.py
# DrugQuest: Complete Drug Discovery Analysis Suite

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition as rgd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# FRAGMENT PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
FRAGMENT_PATTERNS = {
    'Carboxylic Acid': {'smarts': 'C(=O)[OH]', 'color': '#e53935', 'desc': 'Salt bridge'},
    'Aromatic Ring': {'smarts': 'c1ccccc1', 'color': '#8e24aa', 'desc': 'π-stacking'},
    'Hydroxyl': {'smarts': '[OH]', 'color': '#43a047', 'desc': 'H-bond donor'},
    'Carbonyl': {'smarts': '[CX3]=[OX1]', 'color': '#fb8c00', 'desc': 'H-bond acceptor'},
    'Amine': {'smarts': '[NX3;H2,H1,H0]', 'color': '#1e88e5', 'desc': 'H-bond'},
    'Halogen': {'smarts': '[F,Cl,Br,I]', 'color': '#7cb342', 'desc': 'Halogen bond'},
}


def create_drugquest_dashboard(model, dataset, results, device, get_atom_importance_func):
    """DrugQuest: Complete Drug Discovery Dashboard.

    Parameters:
    - model: Trained GAT model
    - dataset: List of graph data objects
    - results: DataFrame with binding scores
    - device: PyTorch device
    - get_atom_importance_func: Function to calculate atom importance
    """

    # Prepare data
    ranked_molecules = results.sort_values('binding_score', ascending=False).reset_index(drop=True)
    mol_graph_lookup = {g.name: g for g in dataset}

    # Calculate descriptors for all molecules
    descriptor_data = []
    valid_mols = []
    valid_names = []
    valid_scores = []

    for g in dataset:
        mol = Chem.MolFromSmiles(g.smiles)
        if mol:
            score_row = results[results['name'] == g.name]
            score = score_row['binding_score'].values[0] if len(score_row) > 0 else 0.5

            desc = {
                'name': g.name,
                'smiles': g.smiles,
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
            valid_names.append(g.name)
            valid_scores.append(score)

    desc_df = pd.DataFrame(descriptor_data)

    # ═══════════════════════════════════════════════════════════════════════
    # WIDGET STYLES
    # ═══════════════════════════════════════════════════════════════════════
    style = {'description_width': '120px'}
    slider_layout = widgets.Layout(width='280px')

    # ═══════════════════════════════════════════════════════════════════════
    # LEFT SIDEBAR WIDGETS
    # ═══════════════════════════════════════════════════════════════════════

    # Molecule selector
    dropdown_options = [(f"#{i+1}: {row['name']} ({row['binding_score']:.2f})", row['name'])
                        for i, row in ranked_molecules.iterrows()]
    molecule_dropdown = widgets.Dropdown(
        options=dropdown_options,
        value=dropdown_options[0][1],
        description='',
        layout=widgets.Layout(width='280px')
    )

    # Weight sliders
    w_acid = widgets.FloatSlider(value=5.0, min=0, max=10, step=0.5, description='Acid:', style=style, layout=slider_layout, continuous_update=False)
    w_logp = widgets.FloatSlider(value=3.0, min=0, max=10, step=0.5, description='LogP:', style=style, layout=slider_layout, continuous_update=False)
    w_hba = widgets.FloatSlider(value=2.0, min=0, max=10, step=0.5, description='HBA:', style=style, layout=slider_layout, continuous_update=False)
    w_hbd = widgets.FloatSlider(value=2.0, min=0, max=10, step=0.5, description='HBD:', style=style, layout=slider_layout, continuous_update=False)
    w_aromatic = widgets.FloatSlider(value=2.0, min=0, max=10, step=0.5, description='Aromatic:', style=style, layout=slider_layout, continuous_update=False)
    w_mw = widgets.FloatSlider(value=1.5, min=0, max=10, step=0.5, description='MW:', style=style, layout=slider_layout, continuous_update=False)

    # Target sliders
    t_hba = widgets.IntSlider(value=3, min=0, max=10, description='Target HBA:', style=style, layout=slider_layout, continuous_update=False)
    t_hbd = widgets.IntSlider(value=2, min=0, max=10, description='Target HBD:', style=style, layout=slider_layout, continuous_update=False)

    # Preset buttons
    btn_strict = widgets.Button(description='Strict', button_style='primary', layout=widgets.Layout(width='135px'))
    btn_balanced = widgets.Button(description='Balanced', button_style='warning', layout=widgets.Layout(width='135px'))
    recalc_btn = widgets.Button(description='Recalculate', button_style='success', layout=widgets.Layout(width='280px'))

    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT AREAS FOR ALL TABS
    # ═══════════════════════════════════════════════════════════════════════
    sidebar_output = widgets.Output()

    # Tab outputs
    data_plots_output = widgets.Output()
    overview_output = widgets.Output()
    chemistry_output = widgets.Output()
    molecular_output = widgets.Output()
    table_output = widgets.Output()
    breakdown_output = widgets.Output()
    structure_analysis_output = widgets.Output()

    # Current scores storage
    current_scores = {'df': ranked_molecules.copy()}

    # ═══════════════════════════════════════════════════════════════════════
    # SCORING FUNCTION
    # ═══════════════════════════════════════════════════════════════════════

    def score_molecule(mol):
        if mol is None: return 0.0
        weights = [w_acid.value, w_logp.value, w_hba.value, w_hbd.value, w_aromatic.value, w_mw.value]
        scores = []

        # Acid
        smiles = Chem.MolToSmiles(mol)
        has_acid = 1.0 if (mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[OH]')) or
                          mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N')) or
                          mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)[OH]'))) else 0.0
        scores.append(has_acid)

        # LogP
        logp = Descriptors.MolLogP(mol)
        scores.append(1.0 if 1.5 <= logp <= 5.5 else 0.3)

        # HBA
        hba = Descriptors.NumHAcceptors(mol)
        scores.append(1.0 - min(abs(hba - t_hba.value) / 4, 1.0))

        # HBD
        hbd = Descriptors.NumHDonors(mol)
        scores.append(1.0 - min(abs(hbd - t_hbd.value) / 3, 1.0))

        # Aromatic
        scores.append(1.0 if Descriptors.NumAromaticRings(mol) >= 1 else 0.3)

        # MW
        mw = Descriptors.MolWt(mol)
        scores.append(1.0 if 150 <= mw <= 500 else 0.7 if mw <= 600 else 0.4)

        return sum(s*w for s,w in zip(scores, weights)) / sum(weights) if sum(weights) > 0 else 0

    # ═══════════════════════════════════════════════════════════════════════
    # PRESET FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════

    def apply_preset(preset):
        if preset == 'strict':
            w_acid.value, w_logp.value, w_hba.value, w_hbd.value = 5.0, 3.0, 2.0, 2.0
            w_aromatic.value, w_mw.value = 2.0, 1.5
            t_hba.value, t_hbd.value = 3, 2
        elif preset == 'balanced':
            w_acid.value = w_logp.value = w_hba.value = w_hbd.value = w_aromatic.value = 3.0
            w_mw.value = 2.0
            t_hba.value, t_hbd.value = 3, 2
        recalculate(None)

    btn_strict.on_click(lambda b: apply_preset('strict'))
    btn_balanced.on_click(lambda b: apply_preset('balanced'))

    # ═══════════════════════════════════════════════════════════════════════
    # RECALCULATE FUNCTION
    # ═══════════════════════════════════════════════════════════════════════

    def recalculate(change):
        # Recalculate all scores
        scored = []
        for g in dataset:
            mol = Chem.MolFromSmiles(g.smiles)
            scored.append({'name': g.name, 'binding_score': score_molecule(mol), 'smiles': g.smiles})

        scored_df = pd.DataFrame(scored).sort_values('binding_score', ascending=False).reset_index(drop=True)
        current_scores['df'] = scored_df

        # Update dropdown
        new_options = [(f"#{i+1}: {row['name']} ({row['binding_score']:.2f})", row['name'])
                       for i, row in scored_df.iterrows()]
        molecule_dropdown.options = new_options

        # Update sidebar
        with sidebar_output:
            clear_output(wait=True)
            mean_score = scored_df['binding_score'].mean()
            max_score = scored_df['binding_score'].max()
            top_name = scored_df.iloc[0]['name']
            display(HTML(f"<div style='background:#1976d2;color:white;padding:10px;border-radius:8px;text-align:center;margin-top:10px;'><b>Top: {top_name}</b><br>Score: {max_score:.3f}<br><small>Mean: {mean_score:.3f}</small></div>"))

        # Update all tabs
        update_data_plots(scored_df)
        update_molecule_view(None)
        update_breakdown(None)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 0: DATA PLOTS (UPDATED: 2D molecule instead of bar chart)
    # ═══════════════════════════════════════════════════════════════════════

    def update_data_plots(scored_df):
        with data_plots_output:
            clear_output(wait=True)

            # Get selected molecule
            mol_name = molecule_dropdown.value
            graph = mol_graph_lookup.get(mol_name)
            mol = Chem.MolFromSmiles(graph.smiles) if graph else None
            mol_row = scored_df[scored_df['name'] == mol_name]
            rank = mol_row.index[0] + 1 if not mol_row.empty else 0
            score = mol_row['binding_score'].values[0] if not mol_row.empty else 0

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # LEFT: 2D Molecule Structure (no attention coloring)
            if mol:
                img = Draw.MolToImage(mol, size=(600, 500))
                ax1.imshow(img)
                ax1.set_title(f'{mol_name}\nRank #{rank} | Score: {score:.3f}', fontsize=14, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No molecule selected', ha='center', va='center', fontsize=14)
                ax1.set_title('Selected Molecule', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # RIGHT: Distribution histogram
            ax2.hist(scored_df['binding_score'], bins=25, color='#42a5f5', edgecolor='white', alpha=0.8)
            mean_score = scored_df['binding_score'].mean()
            ax2.axvline(x=mean_score, color='#ff9800', linewidth=2, linestyle='--', label=f'Mean ({mean_score:.3f})')
            # Mark selected molecule's score
            ax2.axvline(x=score, color='#e53935', linewidth=2, linestyle='-', label=f'{mol_name} ({score:.3f})')
            ax2.legend(fontsize=11)
            ax2.set_xlabel('Score', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Score Distribution', fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.show()

            # Rankings Table
            display(HTML("<h3>Rankings Table</h3>"))
            for i, row in scored_df.head(20).iterrows():
                bar = '█' * int(row['binding_score'] * 20)
                bg = '#e3f2fd' if row['name'] == mol_name else '#fff'
                display(HTML(f"<div style='display:flex;padding:6px 10px;background:{bg};border-bottom:1px solid #eee;'><span style='width:40px;font-weight:bold;'>#{i+1}</span><span style='width:180px;'>{row['name']}</span><span style='flex:1;color:#1976d2;font-family:monospace;'>{bar}</span><span style='width:60px;text-align:right;'>{row['binding_score']:.3f}</span></div>"))

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW (with Radar Chart)
    # ═══════════════════════════════════════════════════════════════════════

    compare_dropdown = widgets.Dropdown(
        options=[('None', None)] + dropdown_options,
        value=None,
        description='Compare:',
        layout=widgets.Layout(width='280px')
    )

    def update_molecule_view(change):
        mol_name = molecule_dropdown.value
        graph = mol_graph_lookup.get(mol_name)
        if not graph: return

        scored_df = current_scores['df']
        mol_row = scored_df[scored_df['name'] == mol_name]
        if mol_row.empty: return
        mol_row = mol_row.iloc[0]
        rank = scored_df[scored_df['name'] == mol_name].index[0] + 1
        importance = get_atom_importance_func(model, graph)
        mol = Chem.MolFromSmiles(graph.smiles)
        if not mol: return

        # Also update Rankings tab when molecule changes
        update_data_plots(scored_df)

        # ═══════════════════════════════════════════════════════════════════
        # OVERVIEW TAB (with Radar Chart)
        # ═══════════════════════════════════════════════════════════════════
        with overview_output:
            clear_output(wait=True)

            # Header
            display(HTML(f"<div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:12px;margin-bottom:15px;'><h2 style='color:white;margin:0;'>{mol_name}</h2><p style='color:#ffd700;margin:5px 0 0 0;'>Rank #{rank} | Score: {mol_row['binding_score']:.3f}</p></div>"))

            # Main layout: Structure + Importance + Radar
            fig = plt.figure(figsize=(18, 6))

            # 1. Structure with importance coloring
            ax1 = fig.add_subplot(1, 3, 1)
            atom_colors = {i: ('#d32f2f' if imp > 0.7 else '#ffc107' if imp > 0.4 else '#1976d2') for i, imp in enumerate(importance)}
            atom_colors_rgb = {i: tuple(int(c[j:j+2], 16)/255 for j in (1,3,5)) for i, c in atom_colors.items()}
            img = Draw.MolToImage(mol, size=(500, 400), highlightAtoms=list(range(len(importance))), highlightAtomColors=atom_colors_rgb)
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title('Structure (Red=Critical)', fontsize=12, fontweight='bold')

            # 2. Atom importance bar chart
            ax2 = fig.add_subplot(1, 3, 2)
            bar_colors = ['#d32f2f' if imp > 0.7 else '#ffc107' if imp > 0.4 else '#1976d2' for imp in importance]
            ax2.barh(range(len(importance)), importance, color=bar_colors)
            ax2.set_yticks(range(len(importance)))
            ax2.set_yticklabels([f'{i}:{mol.GetAtomWithIdx(i).GetSymbol()}' for i in range(min(len(importance), mol.GetNumAtoms()))], fontsize=8)
            ax2.set_xlim(0, 1.05)
            ax2.invert_yaxis()
            ax2.axvline(0.7, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(0.4, color='orange', linestyle='--', alpha=0.5)
            ax2.set_title('Atom Importance', fontsize=12, fontweight='bold')

            # 3. RADAR CHART
            ax3 = fig.add_subplot(1, 3, 3, polar=True)

            mol_props = desc_df[desc_df['name'] == mol_name].iloc[0] if mol_name in desc_df['name'].values else None

            if mol_props is not None:
                categories = ['HBA', 'HBD', 'LogP', 'MW', 'TPSA', 'Aromatic', 'RotBonds']
                max_vals = [10, 5, 8, 600, 150, 4, 10]

                values1 = [
                    mol_props['HBA'] / max_vals[0],
                    mol_props['HBD'] / max_vals[1],
                    (mol_props['LogP'] + 2) / max_vals[2],
                    mol_props['MW'] / max_vals[3],
                    mol_props['TPSA'] / max_vals[4],
                    mol_props['AromaticRings'] / max_vals[5],
                    mol_props['RotBonds'] / max_vals[6],
                ]
                values1 = [min(max(v, 0), 1) for v in values1]
                values1 += values1[:1]

                angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
                angles += angles[:1]

                ax3.plot(angles, values1, 'o-', linewidth=2, color='#1976d2', label=mol_name)
                ax3.fill(angles, values1, alpha=0.25, color='#1976d2')

                compare_name = compare_dropdown.value
                if compare_name and compare_name in desc_df['name'].values:
                    comp_props = desc_df[desc_df['name'] == compare_name].iloc[0]
                    values2 = [
                        comp_props['HBA'] / max_vals[0],
                        comp_props['HBD'] / max_vals[1],
                        (comp_props['LogP'] + 2) / max_vals[2],
                        comp_props['MW'] / max_vals[3],
                        comp_props['TPSA'] / max_vals[4],
                        comp_props['AromaticRings'] / max_vals[5],
                        comp_props['RotBonds'] / max_vals[6],
                    ]
                    values2 = [min(max(v, 0), 1) for v in values2]
                    values2 += values2[:1]
                    ax3.plot(angles, values2, 'o-', linewidth=2, color='#e53935', label=compare_name)
                    ax3.fill(angles, values2, alpha=0.15, color='#e53935')

                ax3.set_xticks(angles[:-1])
                ax3.set_xticklabels(categories, fontsize=10)
                ax3.set_ylim(0, 1)
                ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
                ax3.set_title('Property Radar', fontsize=12, fontweight='bold', pad=15)

            plt.tight_layout()
            plt.show()

            # Summary cards
            high = sum(1 for i in importance if i > 0.7)
            med = sum(1 for i in importance if 0.4 <= i <= 0.7)
            low = sum(1 for i in importance if i < 0.4)
            display(HTML(f"<div style='display:flex;gap:15px;margin-top:15px;'><div style='flex:1;background:#ffebee;padding:15px;border-radius:10px;text-align:center;'><div style='font-size:28px;font-weight:bold;color:#d32f2f;'>{high}</div><div>High</div></div><div style='flex:1;background:#fff8e1;padding:15px;border-radius:10px;text-align:center;'><div style='font-size:28px;font-weight:bold;color:#f57c00;'>{med}</div><div>Medium</div></div><div style='flex:1;background:#e3f2fd;padding:15px;border-radius:10px;text-align:center;'><div style='font-size:28px;font-weight:bold;color:#1976d2;'>{low}</div><div>Low</div></div></div>"))

            display(HTML("<h4 style='margin-top:20px;'>Compare with another molecule:</h4>"))
            display(compare_dropdown)

        # ═══════════════════════════════════════════════════════════════════
        # CHEMISTRY TAB (MERGED: Fragments + Functional Groups)
        # ═══════════════════════════════════════════════════════════════════
        with chemistry_output:
            clear_output(wait=True)
            display(HTML(f"<h2>Chemistry Analysis: {mol_name}</h2>"))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # LEFT: Fragment Importance
            frag_data = []
            for name, info in FRAGMENT_PATTERNS.items():
                pattern = Chem.MolFromSmarts(info['smarts'])
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    atoms = set(a for m in matches for a in m)
                    valid = [a for a in atoms if a < len(importance)]
                    if valid:
                        frag_data.append({'Fragment': name, 'Importance': np.mean([importance[a] for a in valid]),
                                         'Atoms': len(valid), 'Color': info['color'], 'Desc': info['desc']})

            if frag_data:
                frag_df = pd.DataFrame(frag_data).sort_values('Importance', ascending=False)
                ax1.barh(range(len(frag_df)), frag_df['Importance'], color=frag_df['Color'].tolist())
                ax1.set_yticks(range(len(frag_df)))
                ax1.set_yticklabels([f"{row['Fragment']} ({row['Desc']})" for _, row in frag_df.iterrows()])
                ax1.set_xlim(0, 1)
                ax1.invert_yaxis()
                ax1.axvline(0.7, color='red', linestyle='--', alpha=0.3)
                ax1.axvline(0.4, color='orange', linestyle='--', alpha=0.3)
                ax1.set_xlabel('Importance Score')
                ax1.set_title('Fragment Importance', fontsize=12, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No common fragments detected', ha='center', va='center', fontsize=12)
                ax1.set_title('Fragment Importance', fontsize=12, fontweight='bold')
                ax1.axis('off')

            # RIGHT: Functional Groups Count
            func_groups = [
                ('Carboxylic Acid', 'C(=O)[OH]', '#e53935'),
                ('Sulfonamide', 'S(=O)(=O)N', '#ff5722'),
                ('Hydroxyl', '[OH]', '#4caf50'),
                ('Primary Amine', '[NH2]', '#2196f3'),
                ('Secondary Amine', '[NH]', '#03a9f4'),
                ('Tertiary Amine', '[NX3;H0]', '#00bcd4'),
                ('Amide', 'C(=O)N', '#9c27b0'),
                ('Ester', 'C(=O)O[C,c]', '#673ab7'),
                ('Ether', 'COC', '#3f51b5'),
                ('Ketone', 'CC(=O)C', '#ffc107'),
                ('Halogen', '[F,Cl,Br,I]', '#8bc34a'),
            ]

            detected_groups = []
            for name, smarts, color in func_groups:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    count = len(mol.GetSubstructMatches(pattern))
                    detected_groups.append({'Group': name, 'Count': count, 'Color': color})

            if detected_groups:
                group_df = pd.DataFrame(detected_groups).sort_values('Count', ascending=False)
                ax2.barh(range(len(group_df)), group_df['Count'], color=group_df['Color'].tolist())
                ax2.set_yticks(range(len(group_df)))
                ax2.set_yticklabels(group_df['Group'])
                ax2.invert_yaxis()
                ax2.set_xlabel('Count')
                ax2.set_title('Functional Groups Detected', fontsize=12, fontweight='bold')
                for i, (_, row) in enumerate(group_df.iterrows()):
                    ax2.text(row['Count'] + 0.1, i, str(row['Count']), va='center', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No functional groups detected', ha='center', va='center', fontsize=12)
                ax2.set_title('Functional Groups Detected', fontsize=12, fontweight='bold')
                ax2.axis('off')

            plt.tight_layout()
            plt.show()

            # COX-2 Binding Context
            display(HTML("""
                <div style='background:#fff3e0;padding:15px;border-radius:10px;margin-top:15px;'>
                    <h4>COX-2 Binding Site Context</h4>
                    <table style='width:100%;'>
                        <tr><td style='width:30%;'><b>Arg120</b></td><td>Acidic groups form salt bridges (carboxylic acid, sulfonamide)</td></tr>
                        <tr><td><b>Tyr385, Trp387</b></td><td>Aromatic rings enable π-stacking interactions</td></tr>
                        <tr><td><b>Hydrophobic Pocket</b></td><td>Lipophilic groups (alkyl chains, halogens) improve binding</td></tr>
                        <tr><td><b>Ser530</b></td><td>H-bond donors/acceptors stabilize binding</td></tr>
                    </table>
                </div>
            """))

            # Summary
            display(HTML(f"""
                <div style='display:flex;gap:15px;margin-top:15px;'>
                    <div style='flex:1;background:#e8f5e9;padding:15px;border-radius:10px;text-align:center;'>
                        <div style='font-size:28px;font-weight:bold;color:#2e7d32;'>{len(frag_data)}</div>
                        <div>Fragments Found</div>
                    </div>
                    <div style='flex:1;background:#e3f2fd;padding:15px;border-radius:10px;text-align:center;'>
                        <div style='font-size:28px;font-weight:bold;color:#1565c0;'>{len(detected_groups)}</div>
                        <div>Functional Groups</div>
                    </div>
                    <div style='flex:1;background:#fff8e1;padding:15px;border-radius:10px;text-align:center;'>
                        <div style='font-size:28px;font-weight:bold;color:#f57f17;'>{sum(g['Count'] for g in detected_groups) if detected_groups else 0}</div>
                        <div>Total Occurrences</div>
                    </div>
                </div>
            """))

        # ═══════════════════════════════════════════════════════════════════
        # MOLECULAR PROPERTIES TAB
        # ═══════════════════════════════════════════════════════════════════
        with molecular_output:
            clear_output(wait=True)
            display(HTML(f"<h2>Properties: {mol_name}</h2>"))

            props = {
                'HBA': (Descriptors.NumHAcceptors(mol), t_hba.value),
                'HBD': (Descriptors.NumHDonors(mol), t_hbd.value),
                'LogP': (Descriptors.MolLogP(mol), 3.0),
                'Aromatic': (Descriptors.NumAromaticRings(mol), 1),
                'MW': (Descriptors.MolWt(mol), 350),
            }

            fig, ax = plt.subplots(figsize=(10, 4))
            names = list(props.keys())
            actual = [props[n][0] for n in names]
            ideal = [props[n][1] for n in names]
            x = np.arange(len(names))
            ax.bar(x - 0.2, ideal, 0.4, label='Target', color='#42a5f5')
            ax.bar(x + 0.2, actual, 0.4, label='Actual', color='#66bb6a')
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.legend()
            ax.set_title('Properties vs Target (COX-2 Pharmacophore)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

            # Property table
            display(HTML("<h4>All Properties:</h4>"))
            all_props = {
                'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} Da",
                'LogP': f"{Descriptors.MolLogP(mol):.2f}",
                'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
                'H-Bond Donors': Descriptors.NumHDonors(mol),
                'TPSA': f"{Descriptors.TPSA(mol):.2f} sq.A",
                'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                'Aromatic Rings': Descriptors.NumAromaticRings(mol),
                'Heavy Atoms': Descriptors.HeavyAtomCount(mol),
                'Fraction sp3': f"{Descriptors.FractionCSP3(mol):.2f}",
            }
            for k, v in all_props.items():
                display(HTML(f"<div style='padding:5px;border-bottom:1px solid #eee;'><b>{k}:</b> {v}</div>"))

        # ═══════════════════════════════════════════════════════════════════
        # ATOM TABLE TAB
        # ═══════════════════════════════════════════════════════════════════
        with table_output:
            clear_output(wait=True)
            display(HTML(f"<h2>Atom Table: {mol_name}</h2>"))
            display(HTML("<div style='background:linear-gradient(to right,#e3f2fd,#fff3e0,#ffebee);padding:10px;border-radius:8px;margin-bottom:15px;'><b>HIGH (>0.7)</b> Critical | <b>MED (0.4-0.7)</b> Supporting | <b>LOW (<0.4)</b> Modifiable</div>"))

            rows = ""
            for idx in range(min(len(importance), mol.GetNumAtoms())):
                atom = mol.GetAtomWithIdx(idx)
                imp = importance[idx]
                color = '#d32f2f' if imp > 0.7 else '#ffc107' if imp > 0.4 else '#1976d2'
                bar = '█' * int(imp * 12) + '░' * (12 - int(imp * 12))
                rows += f"<tr><td>{idx}</td><td>{atom.GetSymbol()}</td><td>{'Y' if atom.GetIsAromatic() else ''}</td><td style='font-family:monospace;color:{color};'>{bar}</td><td>{imp:.3f}</td></tr>"

            display(HTML(f"<table style='width:100%;border-collapse:collapse;'><thead style='background:#f5f5f5;'><tr><th>Atom</th><th>Elem</th><th>Arom</th><th>Visual</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"))

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: SCORE BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════

    def update_breakdown(change=None):
        with breakdown_output:
            clear_output(wait=True)

            mol_name = molecule_dropdown.value
            mol_row = desc_df[desc_df['name'] == mol_name]
            if mol_row.empty:
                display(HTML("<p>No data available</p>"))
                return
            mol_row = mol_row.iloc[0]
            mol = Chem.MolFromSmiles(mol_row['smiles'])
            if not mol:
                return

            display(HTML(f"<h2>Score Breakdown: {mol_name}</h2>"))

            weights = {'Acid': w_acid.value, 'LogP': w_logp.value, 'HBA': w_hba.value,
                      'HBD': w_hbd.value, 'Aromatic': w_aromatic.value, 'MW': w_mw.value}

            scores = {}
            has_acid = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)[OH]')) or \
                      mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N'))
            scores['Acid'] = 1.0 if has_acid else 0.0

            logp = mol_row['LogP']
            scores['LogP'] = 1.0 if 1.5 <= logp <= 5.5 else 0.3
            scores['HBA'] = 1.0 - min(abs(mol_row['HBA'] - t_hba.value) / 4, 1.0)
            scores['HBD'] = 1.0 - min(abs(mol_row['HBD'] - t_hbd.value) / 3, 1.0)
            scores['Aromatic'] = 1.0 if mol_row['AromaticRings'] >= 1 else 0.3

            mw = mol_row['MW']
            scores['MW'] = 1.0 if 150 <= mw <= 500 else 0.7 if mw <= 600 else 0.4

            contributions = {k: scores[k] * weights[k] for k in weights}
            total = sum(contributions.values())
            normalized = sum(weights.values())
            final_score = total / normalized if normalized > 0 else 0

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            props = list(contributions.keys())
            contribs = list(contributions.values())
            colors = ['#4caf50' if scores[p] >= 0.7 else '#ff9800' if scores[p] >= 0.4 else '#f44336' for p in props]

            ax1.barh(props, contribs, color=colors, edgecolor='black', linewidth=0.5)
            ax1.set_xlabel('Weighted Contribution')
            ax1.set_title(f'Score Breakdown (Final: {final_score:.3f})', fontweight='bold')

            ax2.pie(contribs, labels=props, autopct='%1.1f%%', colors=colors,
                   explode=[0.05]*len(props), startangle=90)
            ax2.set_title('Contribution Distribution', fontweight='bold')

            plt.tight_layout()
            plt.show()

            display(HTML("<h4>Detailed Breakdown</h4>"))
            table = "<table style='width:100%;'><tr><th>Property</th><th>Raw Score</th><th>Weight</th><th>Contribution</th><th>Status</th></tr>"
            for p in props:
                status = "PASS" if scores[p] >= 0.7 else "WARN" if scores[p] >= 0.4 else "FAIL"
                status_color = '#4caf50' if scores[p] >= 0.7 else '#ff9800' if scores[p] >= 0.4 else '#f44336'
                table += f"<tr><td>{p}</td><td>{scores[p]:.2f}</td><td>{weights[p]}</td><td>{contributions[p]:.2f}</td><td style='color:{status_color};font-weight:bold;'>{status}</td></tr>"
            table += f"<tr style='background:#e3f2fd;font-weight:bold;'><td>TOTAL</td><td>-</td><td>{normalized:.1f}</td><td>{total:.2f}</td><td>{final_score:.3f}</td></tr>"
            table += "</table>"
            display(HTML(table))

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: STRUCTURE ANALYSIS (Substructure + Scaffold + R-Group)
    # ═══════════════════════════════════════════════════════════════════════

    analysis_type = widgets.Dropdown(
        options=[
            ('Substructure Search', 'substructure'),
            ('Scaffold Analysis', 'scaffold'),
            ('R-Group Decomposition', 'rgroup')
        ],
        value='substructure',
        description='Analysis:',
        layout=widgets.Layout(width='300px')
    )

    smarts_input = widgets.Text(value='c1ccccc1', description='SMARTS:', placeholder='Enter SMARTS pattern', layout=widgets.Layout(width='400px'))

    preset_patterns = widgets.Dropdown(
        options=[('Benzene Ring', 'c1ccccc1'), ('Carboxylic Acid', 'C(=O)[OH]'), ('Sulfonamide', 'S(=O)(=O)N'),
                 ('Hydroxyl', '[OH]'), ('Primary Amine', '[NH2]'), ('Amide', 'C(=O)N'), ('Halogen', '[F,Cl,Br,I]')],
        value='c1ccccc1', description='Presets:', layout=widgets.Layout(width='250px'))

    top_n_scaffolds = widgets.IntSlider(value=10, min=5, max=20, description='Top N:', layout=widgets.Layout(width='250px'))

    core_smarts = widgets.Text(value='c1ccc(cc1)C(C)C(=O)O', description='Core:', placeholder='Enter core SMILES', layout=widgets.Layout(width='400px'))

    core_presets = widgets.Dropdown(
        options=[('Phenylpropanoic', 'c1ccc(cc1)C(C)C(=O)O'), ('Phenylacetic', 'c1ccccc1CC(=O)O'),
                 ('Benzene', 'c1ccccc1'), ('Pyridine', 'c1ccncc1')],
        description='Presets:', layout=widgets.Layout(width='250px'))

    run_analysis_btn = widgets.Button(description='Run Analysis', button_style='success', layout=widgets.Layout(width='150px'))

    substructure_controls = widgets.VBox([widgets.HBox([smarts_input, preset_patterns])])
    scaffold_controls_box = widgets.VBox([top_n_scaffolds])
    rgroup_controls = widgets.VBox([widgets.HBox([core_smarts, core_presets])])

    dynamic_controls = widgets.VBox([substructure_controls])

    def update_dynamic_controls(change):
        analysis = analysis_type.value
        if analysis == 'substructure':
            dynamic_controls.children = [substructure_controls]
        elif analysis == 'scaffold':
            dynamic_controls.children = [scaffold_controls_box]
        elif analysis == 'rgroup':
            dynamic_controls.children = [rgroup_controls]

    analysis_type.observe(update_dynamic_controls, names='value')

    def use_preset(change):
        smarts_input.value = preset_patterns.value
    preset_patterns.observe(use_preset, names='value')

    def use_core_preset(change):
        core_smarts.value = core_presets.value
    core_presets.observe(use_core_preset, names='value')

    def run_structure_analysis(btn):
        with structure_analysis_output:
            clear_output(wait=True)

            analysis = analysis_type.value

            if analysis == 'substructure':
                smarts = smarts_input.value
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern is None:
                        display(HTML("<div style='color:red;'>Invalid SMARTS pattern</div>"))
                        return
                except:
                    display(HTML("<div style='color:red;'>Error parsing SMARTS</div>"))
                    return

                matches = []
                for i, mol in enumerate(valid_mols):
                    if mol.HasSubstructMatch(pattern):
                        match_atoms = mol.GetSubstructMatch(pattern)
                        matches.append({'name': valid_names[i], 'score': valid_scores[i], 'mol': mol, 'match_atoms': match_atoms})

                display(HTML(f"<h3>Substructure Search Results</h3>"))
                display(HTML(f"<p>Found <b>{len(matches)}</b> molecules with pattern: <code>{smarts}</code></p>"))

                if matches:
                    matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:10]
                    img = Draw.MolsToGridImage([m['mol'] for m in matches], molsPerRow=5, subImgSize=(250, 200),
                                              legends=[f"{m['name']}\n({m['score']:.2f})" for m in matches],
                                              highlightAtomLists=[list(m['match_atoms']) for m in matches])
                    display(img)

            elif analysis == 'scaffold':
                display(HTML("<h3>Scaffold Analysis</h3>"))

                scaffolds = {}
                for i, mol in enumerate(valid_mols):
                    try:
                        core = MurckoScaffold.GetScaffoldForMol(mol)
                        core_smiles = Chem.MolToSmiles(core)
                        if core_smiles not in scaffolds:
                            scaffolds[core_smiles] = {'mol': core, 'members': [], 'scores': []}
                        scaffolds[core_smiles]['members'].append(valid_names[i])
                        scaffolds[core_smiles]['scores'].append(valid_scores[i])
                    except:
                        pass

                display(HTML(f"<p>Found <b>{len(scaffolds)}</b> unique scaffolds</p>"))

                scaffold_list = sorted([{'smiles': s, 'mol': d['mol'], 'count': len(d['members']),
                                         'avg_score': np.mean(d['scores'])} for s, d in scaffolds.items()],
                                       key=lambda x: x['count'], reverse=True)[:top_n_scaffolds.value]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.barh(range(len(scaffold_list)), [s['count'] for s in scaffold_list], color='#42a5f5')
                ax1.set_yticks(range(len(scaffold_list)))
                ax1.set_yticklabels([f"Scaffold {i+1}" for i in range(len(scaffold_list))])
                ax1.invert_yaxis()
                ax1.set_title('Scaffold Frequency', fontweight='bold')

                ax2.barh(range(len(scaffold_list)), [s['avg_score'] for s in scaffold_list], color='#66bb6a')
                ax2.set_yticks(range(len(scaffold_list)))
                ax2.set_yticklabels([f"Scaffold {i+1}" for i in range(len(scaffold_list))])
                ax2.invert_yaxis()
                ax2.set_title('Scaffold Performance', fontweight='bold')
                plt.tight_layout()
                plt.show()

                img = Draw.MolsToGridImage([s['mol'] for s in scaffold_list[:10]], molsPerRow=5, subImgSize=(200, 150),
                                          legends=[f"#{i+1}: {s['count']} mols" for i, s in enumerate(scaffold_list[:10])])
                display(img)

            elif analysis == 'rgroup':
                display(HTML("<h3>R-Group Decomposition</h3>"))

                core = Chem.MolFromSmiles(core_smarts.value) or Chem.MolFromSmarts(core_smarts.value)
                if not core:
                    display(HTML("<div style='color:red;'>Invalid core structure</div>"))
                    return

                display(Draw.MolToImage(core, size=(300, 200)))

                matching = [(mol, name, score) for mol, name, score in zip(valid_mols, valid_names, valid_scores) if mol.HasSubstructMatch(core)]
                display(HTML(f"<p>Found <b>{len(matching)}</b> molecules with core</p>"))

                if len(matching) >= 2:
                    try:
                        rgroups, _ = rgd.RGroupDecompose([core], [m[0] for m in matching], asSmiles=True)
                        if rgroups:
                            rg_df = pd.DataFrame(rgroups)
                            rg_df['name'] = [m[1] for m in matching[:len(rg_df)]]
                            rg_df['score'] = [m[2] for m in matching[:len(rg_df)]]
                            display(rg_df.sort_values('score', ascending=False).head(15).style.background_gradient(subset=['score'], cmap='Greens'))
                    except Exception as e:
                        display(HTML(f"<div style='color:red;'>Error: {e}</div>"))

    run_analysis_btn.on_click(run_structure_analysis)

    structure_tab_content = widgets.VBox([
        widgets.HTML("<h3>Structure Analysis</h3>"),
        widgets.HBox([analysis_type, run_analysis_btn]),
        dynamic_controls,
        structure_analysis_output
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE TABS
    # ═══════════════════════════════════════════════════════════════════════

    tabs = widgets.Tab(children=[
        data_plots_output,
        overview_output,
        chemistry_output,
        molecular_output,
        table_output,
        breakdown_output,
        structure_tab_content
    ])

    tabs.set_title(0, 'Rankings')
    tabs.set_title(1, 'Overview')
    tabs.set_title(2, 'Chemistry')
    tabs.set_title(3, 'Properties')
    tabs.set_title(4, 'Atom Table')
    tabs.set_title(5, 'Breakdown')
    tabs.set_title(6, 'Structure')

    # ═══════════════════════════════════════════════════════════════════════
    # OBSERVERS
    # ═══════════════════════════════════════════════════════════════════════

    molecule_dropdown.observe(lambda c: (update_molecule_view(c), update_breakdown(c)), names='value')
    compare_dropdown.observe(update_molecule_view, names='value')
    recalc_btn.on_click(recalculate)

    # ═══════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ═══════════════════════════════════════════════════════════════════════

    header = widgets.HTML("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:12px;color:white;margin-bottom:15px;'>
            <h1 style='margin:0;'>DrugQuest Dashboard</h1>
            <p style='margin:5px 0 0 0;opacity:0.8;'>Drug Discovery Panel</p>
        </div>
    """)

    sidebar = widgets.VBox([
        widgets.HTML("<h3 style='margin:0 0 10px 0;'>Scoring Weights</h3>"),
        widgets.HBox([btn_strict, btn_balanced]),
        w_acid, w_logp, w_hba, w_hbd, w_aromatic, w_mw,
        widgets.HTML("<hr style='margin:10px 0;'>"),
        widgets.HTML("<b>Target Values</b>"),
        t_hba, t_hbd,
        recalc_btn,
        sidebar_output,
        widgets.HTML("<h3 style='margin:15px 0 10px 0;'>Select Molecule</h3>"),
        molecule_dropdown
    ], layout=widgets.Layout(width='320px', padding='15px', border='1px solid #ddd', border_radius='10px'))

    main_content = widgets.VBox([tabs], layout=widgets.Layout(flex='1', padding='0 0 0 15px'))

    layout = widgets.VBox([header, widgets.HBox([sidebar, main_content])])
    display(layout)
    recalculate(None)
