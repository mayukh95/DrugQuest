import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem

class DrugScreenerDashboard:
    """
    Interactive Dashboard for Drug Discovery and Screening.
    Allows for reference-based screening, comparison, and analysis.
    """
    def __init__(self, molecules, sim_matrices, df_properties=None):
        """
        Initialize the dashboard.
        
        Parameters:
        -----------
        molecules : list
            List of DrugMolecule objects
        sim_matrices : dict
            Dictionary of similarity DataFrames {'Morgan': df, 'Pharm2D': df, 'Pharm3D': df}
        df_properties : pd.DataFrame, optional
            DataFrame containining full properties (including DFT) for detailed comparison
        """
        self.molecules = {m.name: m for m in molecules}
        self.molecule_names = sorted(list(self.molecules.keys()))
        self.sim_matrices = sim_matrices
        self.df_properties = df_properties
        
        # Default state
        self.current_ref = self.molecule_names[0] if self.molecule_names else None
        self.current_candidate = None
        self.weights = {'Morgan': 0.33, 'Pharm2D': 0.33, 'Pharm3D': 0.34}
        
        # Performance: Track which tabs need updating (dirty flags)
        self._dirty_tabs = {'viewer': True, 'alignment': True, 'pharmacophore': True, 'pharm_grid': True, 'heatmap': True, 'properties': True, 'data': True}
        self._current_tab_index = 0
        
        # Cache heavy objects
        self._feature_factory = None
        self._init_feature_factory()
        
        # UI Components
        self._setup_widgets()
        self._setup_layout()
        
        # Initial update (only ranking, not all tabs)
        if self.current_ref:
            self._update_ranking()
    
    def _init_feature_factory(self):
        """Initialize the pharmacophore feature factory once."""
        try:
            from rdkit.Chem import ChemicalFeatures
            from rdkit import RDConfig
            import os
            fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            self._feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
        except Exception:
            self._feature_factory = None
            
    def _setup_widgets(self):
        """Initialize all interactive widgets."""
        
        # --- Control Deck ---
        
        # Reference Selector
        self.ref_dropdown = widgets.Dropdown(
            options=self.molecule_names,
            value=self.current_ref,
            description='<b>Reference:</b>',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Refresh Button
        self.refresh_btn = widgets.Button(
            description='Refresh View',
            icon='refresh',
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        self.refresh_btn.on_click(self._on_refresh_click)
        
        # Observers
        self.ref_dropdown.observe(self._on_ref_change, names='value')
        
        # Weight Sliders
        style = {'description_width': '80px'}
        slider_layout = widgets.Layout(width='200px')
        
        self.w_morgan = widgets.FloatSlider(value=0.33, min=0, max=1, step=0.05, description='Morgan', style=style, layout=slider_layout)
        self.w_pharm2d = widgets.FloatSlider(value=0.33, min=0, max=1, step=0.05, description='Pharm2D', style=style, layout=slider_layout)
        self.w_pharm3d = widgets.FloatSlider(value=0.34, min=0, max=1, step=0.05, description='Pharm3D', style=style, layout=slider_layout)
        
        for w in [self.w_morgan, self.w_pharm2d, self.w_pharm3d]:
            w.observe(self._on_weight_change, names='value')
            
        # --- Left Panel: Ranking List ---
        self.ranking_output = widgets.Output(layout=widgets.Layout(height='600px', overflow_y='scroll'))
        
        # --- Right Panel: Comparison Tabs ---
        # Initialize 4 slots for Radar Charts with Dropdowns
        self.radar_slots = []
        for i in range(4):
            dd = widgets.Dropdown(description=f'Cand {i+1}:', layout=widgets.Layout(width='200px'))
            # Use partial or a closure wrapper to capture 'i' correctly in loop
            def make_handler(index):
                return lambda change: self._on_radar_dropdown_change(change, index)
            dd.observe(make_handler(i), names='value')
            
            out = widgets.Output(layout=widgets.Layout(height='350px'))
            self.radar_slots.append({'dropdown': dd, 'output': out})
            
        self.viewer_output = widgets.Output()
        self.props_output = widgets.Output()
        self.table_output = widgets.Output() # New Tab for Data Table
        self.alignment_output = widgets.Output() # Shape Alignment Tab
        self.pharmacophore_output = widgets.Output() # 3D Pharmacophore Tab
        self.pharm_grid_output = widgets.Output() # Pharmacophore Grid Tab
        self.heatmap_output = widgets.Output() # Similarity Heatmap Tab
        self.ref_info_output = widgets.Output(layout=widgets.Layout(margin='0 0 10px 0'))
        
    def _setup_layout(self):
        """Assemble the dashboard layout."""
        
        # Header
        header = widgets.HTML("""
        <div style='background: linear-gradient(90deg, #1e3a8a, #3b82f6); padding: 15px; border-radius: 10px; color: white;'>
            <h2 style='margin:0'>📊 Drug Discovery Dashboard</h2>
            <p style='margin:0; opacity:0.8'>Interactive screening and multi-parameter optimization</p>
        </div>
        """)
        
        # Controls
        dataset_info = widgets.HTML(f"<b>Dataset:</b> {len(self.molecules)} molecules")
        
        weights_box = widgets.HBox([
            widgets.Label("<b>Weights:</b>"),
            self.w_morgan, self.w_pharm2d, self.w_pharm3d
        ], layout=widgets.Layout(align_items='center', gap='10px'))
        
        controls = widgets.HBox([
            self.ref_dropdown,
            self.refresh_btn,
            weights_box
        ], layout=widgets.Layout(justify_content='space-between', padding='10px', border='1px solid #ddd', border_radius='8px', margin='10px 0'))
        
        # Main Split
        
        # Left: Ranking List
        left_panel = widgets.VBox([
            widgets.HTML("<h4>▶ Ranked Candidates</h4>"),
            self.ranking_output
        ], layout=widgets.Layout(width='30%', border='1px solid #eee', padding='5px'))
        
        # Right: Tabs with Radar Grid
        radar_row1 = widgets.HBox([
            widgets.VBox([self.radar_slots[0]['dropdown'], self.radar_slots[0]['output']], layout=widgets.Layout(width='50%')),
            widgets.VBox([self.radar_slots[1]['dropdown'], self.radar_slots[1]['output']], layout=widgets.Layout(width='50%'))
        ])
        radar_row2 = widgets.HBox([
            widgets.VBox([self.radar_slots[2]['dropdown'], self.radar_slots[2]['output']], layout=widgets.Layout(width='50%')),
            widgets.VBox([self.radar_slots[3]['dropdown'], self.radar_slots[3]['output']], layout=widgets.Layout(width='50%'))
        ])
        radar_tab_content = widgets.VBox([self.ref_info_output, radar_row1, radar_row2])

        self.tabs = widgets.Tab(children=[
            radar_tab_content,
            self.viewer_output,
            self.alignment_output,
            self.pharmacophore_output,
            self.pharm_grid_output,
            self.heatmap_output,
            self.props_output,
            self.table_output
        ])
        self.tabs.set_title(0, "🕸️ Radar Comparison")
        self.tabs.set_title(1, "🧊 3D Viewer")
        self.tabs.set_title(2, "🔄 Shape Alignment")
        self.tabs.set_title(3, "🧬 3D Pharmacophore")
        self.tabs.set_title(4, "🔬 Pharm Grid")
        self.tabs.set_title(5, "📈 Similarity Heatmap")
        self.tabs.set_title(6, "📊 Properties")
        self.tabs.set_title(7, "📋 Data Explorer")
        
        # Lazy loading: only update tab when it becomes visible
        self.tabs.observe(self._on_tab_change, names='selected_index')
        
        right_panel = widgets.VBox([
            self.tabs
        ], layout=widgets.Layout(width='70%', padding='0 0 0 15px'))
        
        main_body = widgets.HBox([left_panel, right_panel])
        
        self.layout = widgets.VBox([
            header,
            controls,
            main_body
        ])
        
    def _get_mol_image_html(self, mol_name, size=(200, 120)):
        """Generate base64 HTML image of molecule."""
        mol_obj = self.molecules.get(mol_name)
        if not mol_obj or not mol_obj.mol: return ""
        try:
            from rdkit.Chem import Draw
            import base64
            from io import BytesIO
            
            # Draw
            img_io = BytesIO()
            # Make sure we use a copy to avoids styling conflicts if any
            m = Chem.Mol(mol_obj.mol)
            AllChem.Compute2DCoords(m)
            img = Draw.MolToImage(m, size=size)
            img.save(img_io, format='PNG')
            img_str = base64.b64encode(img_io.getvalue()).decode("utf-8")
            return f'<img src="data:image/png;base64,{img_str}" style="border:1px solid #eee; border-radius:4px; margin-bottom:5px;">'
        except Exception as e:
            return f"<small>Img Error: {e}</small>"
        
    def _update_ranking(self):
        """Recalculate consensus scores and update ranking list."""
        if not self.current_ref:
            return
            
        # Update Reference Header Image
        with self.ref_info_output:
            clear_output(wait=True)
            img_html = self._get_mol_image_html(self.current_ref, size=(300, 150))
            display(widgets.HTML(f"""
            <div style="display:flex; align-items:center; background:#f8fafc; padding:10px; border-radius:8px; border:1px solid #e2e8f0">
                <div style="margin-right:20px;">{img_html}</div>
                <div>
                    <h3 style="margin:0; color:#1e40af">🎯 Reference: {self.current_ref}</h3>
                    <p style="margin:5px 0 0 0; color:#64748b; font-size:13px">All comparisons below are relative to this molecule.</p>
                </div>
            </div>
            """))
            
        ref = self.current_ref
        scores = []
        
        # Normalize weights
        total_w = self.w_morgan.value + self.w_pharm2d.value + self.w_pharm3d.value
        w_m = self.w_morgan.value / total_w if total_w > 0 else 0.33
        w_p2 = self.w_pharm2d.value / total_w if total_w > 0 else 0.33
        w_p3 = self.w_pharm3d.value / total_w if total_w > 0 else 0.34
        
        # Calculate scores
        for name in self.molecule_names:
            if name == ref: continue
            
            # Helper to get score safely
            def get_sim(mat_name, n1, n2):
                if mat_name not in self.sim_matrices: return 0
                mat = self.sim_matrices[mat_name]
                if n1 in mat.index and n2 in mat.columns:
                    return mat.loc[n1, n2]
                n1_opt, n2_opt = f"{n1}_optimized", f"{n2}_optimized"
                if n1_opt in mat.index and n2_opt in mat.columns:
                    return mat.loc[n1_opt, n2_opt]
                return 0

            s_m = get_sim('Morgan', ref, name)
            s_p2 = get_sim('Pharm2D', ref, name)
            s_p3 = get_sim('Pharm3D', ref, name)
            
            consensus = (s_m * w_m) + (s_p2 * w_p2) + (s_p3 * w_p3)
            
            scores.append({
                'Name': name,
                'Consensus': consensus,
                'Morgan': s_m,
                'Pharm2D': s_p2,
                'Pharm3D': s_p3
            })
            
        # Sort
        self.ranked_df = pd.DataFrame(scores).sort_values('Consensus', ascending=False)
        self.ranked_names = self.ranked_df['Name'].tolist()
        
        # Update Table
        with self.ranking_output:
            clear_output(wait=True)
            self._render_ranking_table()
            
        # Update Radar Dropdowns options and default values
        candidates = self.ranked_names[:4]
        for i, slot in enumerate(self.radar_slots):
            # Important: Set options first, then value
            slot['dropdown'].options = self.ranked_names
            if i < len(candidates):
                slot['dropdown'].value = candidates[i]
            else:
                slot['dropdown'].value = None
            # Manually trigger update for initial view
            self._update_single_radar(i, slot['dropdown'].value)
                
        # Auto-select top candidate if none selected for other tabs
        if not self.current_candidate and not self.ranked_df.empty:
            self.current_candidate = self.ranked_df.iloc[0]['Name']
            
    def _render_ranking_table(self):
        """Render the HTML ranking table."""
        html = """
        <style>
            .rank-table { width: 100%; border-collapse: collapse; font-size: 13px; }
            .rank-table th { background: #f3f4f6; padding: 8px; text-align: left; position: sticky; top: 0; }
            .rank-table td { border-bottom: 1px solid #eee; padding: 6px; cursor: pointer; }
            .rank-table tr:hover { background-color: #e0e7ff; }
            .rank-bar { height: 6px; background: #3b82f6; border-radius: 3px; }
        </style>
        <table class='rank-table'>
            <thead><tr><th>#</th><th>Molecule</th><th>Score</th><th>Breakdown</th></tr></thead>
            <tbody>
        """
        
        for i, row in enumerate(self.ranked_df.head(50).itertuples(), 1):
            name = row.Name
            score = row.Consensus
            width = int(score * 100)
            bg_style = "style='background-color: #dbeafe; font-weight:bold;'" if name == self.current_candidate else ""
            
            html += f"""
            <tr {bg_style}>
                <td>{i}</td>
                <td>{name}</td>
                <td><b>{score:.3f}</b></td>
                <td width="80px">
                    <div style="width: 100%; background: #e5e7eb; border-radius: 3px;">
                        <div class="rank-bar" style="width: {width}%"></div>
                    </div>
                </td>
            </tr>
            """
        html += "</tbody></table>"
        display(HTML(html))
        
        display(HTML("<p style='font-size:11px; color:#666; margin-top:5px;'><i>*Currently viewing top 50.*</i></p>"))
        
        # Update the selector logic for other tabs
        if not hasattr(self, 'candidate_selector'):
            self.candidate_selector = widgets.Dropdown(description='Select 3D:', style={'description_width': 'initial'})
            self.candidate_selector.observe(self._on_candidate_change, names='value')
            display(self.candidate_selector)
        
        self.candidate_selector.options = self.ranked_df['Name'].tolist()
        self.candidate_selector.value = self.current_candidate

    def _on_ref_change(self, change):
        if change['new']:
            self.current_ref = change['new']
            self._update_ranking()
            
    def _on_weight_change(self, change):
        self._update_ranking()
        
    def _on_refresh_click(self, b):
        """Force refresh of everything."""
        # Mark all tabs as needing update
        for key in self._dirty_tabs:
            self._dirty_tabs[key] = True
        self._update_ranking()
        self._update_current_tab()
            
    def _on_candidate_change(self, change):
        if change['new']:
            self._select_candidate(change['new'])
            
    def _select_candidate(self, name):
        """Update the comparison views for the selected candidate (lazy loading)."""
        self.current_candidate = name
        
        # Mark all tabs as dirty (needing update)
        for key in self._dirty_tabs:
            self._dirty_tabs[key] = True
        
        # Only update the currently visible tab
        self._update_current_tab()
    
    def _on_tab_change(self, change):
        """Handle tab change - lazy load content."""
        if change['new'] is not None:
            self._current_tab_index = change['new']
            self._update_current_tab()
    
    def _update_current_tab(self):
        """Update only the currently visible tab if it's dirty."""
        tab_idx = self._current_tab_index
        
        # Map tab index to update function and dirty key
        tab_map = {
            0: (None, None),  # Radar - always updated via dropdowns
            1: (self._update_3d_viewer, 'viewer'),
            2: (self._update_alignment, 'alignment'),
            3: (self._update_pharmacophore, 'pharmacophore'),
            4: (self._update_pharm_grid, 'pharm_grid'),
            5: (self._update_heatmap, 'heatmap'),
            6: (self._update_properties, 'properties'),
            7: (self._update_data_table, 'data')
        }
        
        if tab_idx in tab_map:
            update_fn, dirty_key = tab_map[tab_idx]
            if update_fn and dirty_key and self._dirty_tabs.get(dirty_key, False):
                update_fn()
                self._dirty_tabs[dirty_key] = False
        
    def _update_alignment(self):
        """Setup and display the Shape Alignment tab with interactive controls."""
        with self.alignment_output:
            clear_output(wait=True)
            
            try:
                from rdkit.Chem import rdMolAlign, rdShapeHelpers
            except ImportError:
                display(HTML("<p style='color:red'>❌ RDKit alignment modules not available.</p>"))
                return
            
            # Header
            display(HTML("""
            <div style="background:#fef3c7; padding:10px; border-radius:8px; margin-bottom:15px; border:1px solid #fcd34d;">
                <h4 style="margin:0; color:#92400e;">🔄 3D Shape Alignment</h4>
                <p style="margin:5px 0 0 0; color:#a16207; font-size:12px;">Select two molecules to perform O3A alignment and compare their 3D shapes</p>
            </div>
            """))
            
            # Create dropdowns
            mol_names = sorted(self.molecule_names)
            
            # Default values
            fixed_default = self.current_ref if self.current_ref in mol_names else mol_names[0]
            mobile_default = self.current_candidate if self.current_candidate and self.current_candidate in mol_names else (mol_names[1] if len(mol_names) > 1 else mol_names[0])
            
            # Molecule selectors
            fixed_dd = widgets.Dropdown(options=mol_names, value=fixed_default, description='Fixed:', style={'description_width': '50px'}, layout=widgets.Layout(width='250px'))
            mobile_dd = widgets.Dropdown(options=mol_names, value=mobile_default, description='Mobile:', style={'description_width': '50px'}, layout=widgets.Layout(width='250px'))
            
            # Style controls
            style_dd = widgets.Dropdown(
                options=['Stick', 'Sphere', 'Ball & Stick', 'Line'],
                value='Stick',
                description='Style:',
                style={'description_width': '45px'},
                layout=widgets.Layout(width='150px')
            )
            
            bg_dd = widgets.Dropdown(
                options=[('White', 'white'), ('Black', 'black'), ('Gray', '#2d2d2d'), ('Light Blue', '#e0f2fe')],
                value='white',
                description='BG:',
                style={'description_width': '25px'},
                layout=widgets.Layout(width='130px')
            )
            
            # Color schemes for molecules
            fixed_color = widgets.Dropdown(
                options=[('Cyan', 'cyanCarbon'), ('Green', 'greenCarbon'), ('Blue', 'blueCarbon'), ('Orange', 'orangeCarbon')],
                value='cyanCarbon',
                description='Fixed Color:',
                style={'description_width': '70px'},
                layout=widgets.Layout(width='170px')
            )
            
            mobile_color = widgets.Dropdown(
                options=[('Magenta', 'magentaCarbon'), ('Purple', 'purpleCarbon'), ('Red', 'redCarbon'), ('Yellow', 'yellowCarbon')],
                value='magentaCarbon',
                description='Mobile Color:',
                style={'description_width': '80px'},
                layout=widgets.Layout(width='180px')
            )
            
            # Show surface option
            show_surface = widgets.Checkbox(value=False, description='Surface', layout=widgets.Layout(width='90px'))
            
            align_btn = widgets.Button(description='🔄 Align', button_style='warning', layout=widgets.Layout(width='100px'))
            
            align_out = widgets.Output()
            
            def do_alignment(_):
                with align_out:
                    clear_output(wait=True)
                    
                    obj1 = self.molecules.get(fixed_dd.value)
                    obj2 = self.molecules.get(mobile_dd.value)
                    
                    if not obj1 or not obj2 or not obj1.mol or not obj2.mol:
                        print("❌ Could not load molecules")
                        return
                    
                    # Create copies
                    m1 = Chem.Mol(obj1.mol)
                    m2 = Chem.Mol(obj2.mol)
                    
                    try:
                        o3a = rdMolAlign.GetO3A(m2, m1)
                        rmsd = o3a.Align()
                        shape_sim = 1 - rdShapeHelpers.ShapeTanimotoDist(m1, m2)
                    except Exception as e:
                        print(f"Alignment failed: {e}")
                        return
                    
                    # Score Header
                    display(HTML(f"""
                    <div style="text-align:center; background:#f0fdf4; padding:10px; border-radius:8px; margin:10px 0; border:1px solid #bbf7d0;">
                        <b style="color:#166534;">Shape Sim: {shape_sim:.3f}</b>
                        <span style="color:#86efac; margin:0 15px;">•</span>
                        <b style="color:#0d9488;">RMSD: {rmsd:.2f} Å</b>
                    </div>
                    """))
                    
                    # 3D Viewer
                    view = py3Dmol.view(width=850, height=450)
                    view.setBackgroundColor(bg_dd.value)
                    
                    # Add Fixed molecule
                    view.addModel(Chem.MolToMolBlock(m1), 'mol')
                    
                    # Add Mobile molecule (Aligned)
                    view.addModel(Chem.MolToMolBlock(m2), 'mol')
                    
                    # Apply style based on selection
                    style_name = style_dd.value
                    if style_name == 'Stick':
                        view.setStyle({'model': 0}, {'stick': {'colorscheme': fixed_color.value, 'radius': 0.15}})
                        view.setStyle({'model': 1}, {'stick': {'colorscheme': mobile_color.value, 'radius': 0.15}})
                    elif style_name == 'Sphere':
                        view.setStyle({'model': 0}, {'sphere': {'colorscheme': fixed_color.value, 'scale': 0.3}})
                        view.setStyle({'model': 1}, {'sphere': {'colorscheme': mobile_color.value, 'scale': 0.3}})
                    elif style_name == 'Ball & Stick':
                        view.setStyle({'model': 0}, {'stick': {'colorscheme': fixed_color.value, 'radius': 0.12}, 'sphere': {'colorscheme': fixed_color.value, 'scale': 0.25}})
                        view.setStyle({'model': 1}, {'stick': {'colorscheme': mobile_color.value, 'radius': 0.12}, 'sphere': {'colorscheme': mobile_color.value, 'scale': 0.25}})
                    else:  # Line
                        view.setStyle({'model': 0}, {'line': {'colorscheme': fixed_color.value}})
                        view.setStyle({'model': 1}, {'line': {'colorscheme': mobile_color.value}})
                    
                    # Add surface if requested
                    if show_surface.value:
                        view.addSurface('VDW', {'opacity': 0.2, 'color': 'lightgray'}, {'model': 0})
                        view.addSurface('VDW', {'opacity': 0.2, 'color': 'pink'}, {'model': 1})
                    
                    view.zoomTo()
                    view.show()
                    
                    # Legend with actual colors
                    display(HTML(f"""
                    <div style='text-align:center; font-size:11px; color:#666; margin-top:5px'>
                        <span style='color:#06b6d4'>■ {fixed_dd.value[:20]}</span> (Fixed) | 
                        <span style='color:#d946ef'>■ {mobile_dd.value[:20]}</span> (Aligned)
                    </div>
                    """))
            
            align_btn.on_click(do_alignment)
            
            # Layout
            row1 = widgets.HBox([fixed_dd, mobile_dd, align_btn])
            row2 = widgets.HBox([style_dd, bg_dd, fixed_color, mobile_color, show_surface])
            
            display(row1)
            display(row2)
            display(align_out)
            
            # Auto-run alignment with defaults
            do_alignment(None)
    
    def _update_pharmacophore(self):
        """Setup and display the 3D Pharmacophore visualization tab with side-by-side comparison."""
        with self.pharmacophore_output:
            clear_output(wait=True)
            
            # Use cached feature factory
            if self._feature_factory is None:
                display(HTML("<p style='color:red'>❌ Pharmacophore feature factory not available.</p>"))
                return
            
            FEATURE_FACTORY = self._feature_factory
            
            # Color map
            color_map = {
                'Donor': 'blue',
                'Acceptor': 'red',
                'Aromatic': 'orange',
                'Hydrophobe': 'lightsteelblue',
                'LumpedHydrophobe': 'green',
                'PosIonizable': 'purple',
                'NegIonizable': 'cyan'
            }
            
            # Header
            display(HTML("""
            <div style="background:#fae8ff; padding:10px; border-radius:8px; margin-bottom:15px; border:1px solid #e879f9;">
                <h4 style="margin:0; color:#86198f;">🧬 3D Pharmacophore Comparison</h4>
                <p style="margin:5px 0 0 0; color:#a21caf; font-size:12px;">Compare pharmacophore features: Reference (Left) vs Candidate (Right)</p>
            </div>
            """))
            
            # Controls
            mol_names = sorted(self.molecule_names)
            ref_default = self.current_ref if self.current_ref in mol_names else mol_names[0]
            cand_default = self.current_candidate if self.current_candidate in mol_names else (mol_names[1] if len(mol_names) > 1 else mol_names[0])
            
            # Molecule selectors
            ref_dd = widgets.Dropdown(options=mol_names, value=ref_default, description='Reference:', style={'description_width': '70px'}, layout=widgets.Layout(width='260px'))
            cand_dd = widgets.Dropdown(options=mol_names, value=cand_default, description='Compare:', style={'description_width': '60px'}, layout=widgets.Layout(width='260px'))
            
            # Style and display options
            style_dd = widgets.Dropdown(options=['Stick', 'Sphere', 'Stick + Sphere', 'Line'], value='Stick + Sphere', description='Style:', style={'description_width': '40px'}, layout=widgets.Layout(width='160px'))
            bg_dd = widgets.Dropdown(options=[('White', 'white'), ('Black', 'black'), ('Gray', '#2d2d2d')], value='white', description='BG:', style={'description_width': '25px'}, layout=widgets.Layout(width='110px'))
            sphere_size = widgets.FloatSlider(value=0.7, min=0.3, max=1.5, step=0.1, description='Sphere:', style={'description_width': '50px'}, layout=widgets.Layout(width='180px'))
            
            # Toggle options
            show_labels = widgets.Checkbox(value=True, description='Labels', layout=widgets.Layout(width='75px'))
            show_hydrogens = widgets.Checkbox(value=False, description='Hydrogens', layout=widgets.Layout(width='95px'))
            show_surface = widgets.Checkbox(value=False, description='Surface', layout=widgets.Layout(width='85px'))
            
            pharm_btn = widgets.Button(description='🔬 Compare', button_style='success', layout=widgets.Layout(width='100px'))
            refresh_btn = widgets.Button(description='🔄 Refresh', button_style='info', layout=widgets.Layout(width='100px'))
            pharm_out = widgets.Output(layout=widgets.Layout(max_height='550px', overflow_y='auto'))
            
            def render_single_mol(view, mol_obj, viewer_idx, show_h, style_name, bg_color, sphere_sz, labels_on):
                """Render pharmacophore for a single molecule in the grid."""
                if not mol_obj or not mol_obj.mol:
                    return {}
                
                mol = Chem.Mol(mol_obj.mol)  # Copy
                
                # Ensure 3D
                if mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                
                # Handle hydrogens
                if show_h:
                    # Add hydrogens with 3D coordinates
                    mol_display = Chem.AddHs(mol, addCoords=True)
                else:
                    mol_display = Chem.RemoveHs(mol)
                
                # Get features
                features = FEATURE_FACTORY.GetFeaturesForMol(mol)
                
                mb = Chem.MolToMolBlock(mol_display)
                view.addModel(mb, 'sdf', viewer=viewer_idx)
                
                # Apply style
                if style_name == 'Stick':
                    view.setStyle({'stick': {'radius': 0.15}}, viewer=viewer_idx)
                elif style_name == 'Sphere':
                    view.setStyle({'sphere': {'scale': 0.3}}, viewer=viewer_idx)
                elif style_name == 'Stick + Sphere':
                    view.setStyle({'stick': {'radius': 0.12}, 'sphere': {'scale': 0.25}}, viewer=viewer_idx)
                else:
                    view.setStyle({'line': {'linewidth': 2}}, viewer=viewer_idx)
                
                # Add pharmacophore spheres
                feat_counts = {}
                for f in features:
                    fam = f.GetFamily()
                    pos = f.GetPos()
                    color = color_map.get(fam, 'white')
                    
                    view.addSphere({
                        'center': {'x': float(pos.x), 'y': float(pos.y), 'z': float(pos.z)},
                        'radius': sphere_sz,
                        'color': color,
                        'alpha': 0.6
                    }, viewer=viewer_idx)
                    
                    if labels_on:
                        view.addLabel(fam[:3], {
                            'position': {'x': float(pos.x), 'y': float(pos.y), 'z': float(pos.z) + sphere_sz + 0.3},
                            'fontSize': 9,
                            'fontColor': 'black' if bg_color == 'white' else 'white',
                            'backgroundColor': 'white' if bg_color != 'white' else 'lightgray',
                            'backgroundOpacity': 0.7
                        }, viewer=viewer_idx)
                    
                    feat_counts[fam] = feat_counts.get(fam, 0) + 1
                
                view.zoomTo(viewer=viewer_idx)
                return feat_counts
            
            def render_pharmacophore(_):
                with pharm_out:
                    clear_output(wait=True)
                    
                    ref_obj = self.molecules.get(ref_dd.value)
                    cand_obj = self.molecules.get(cand_dd.value)
                    
                    # Create side-by-side viewer
                    view = py3Dmol.view(width=900, height=450, viewergrid=(1, 2))
                    view.setBackgroundColor(bg_dd.value, viewer=(0, 0))
                    view.setBackgroundColor(bg_dd.value, viewer=(0, 1))
                    
                    # Render Reference (Left)
                    ref_counts = render_single_mol(view, ref_obj, (0, 0), show_hydrogens.value, style_dd.value, bg_dd.value, sphere_size.value, show_labels.value)
                    
                    # Render Candidate (Right)
                    cand_counts = render_single_mol(view, cand_obj, (0, 1), show_hydrogens.value, style_dd.value, bg_dd.value, sphere_size.value, show_labels.value)
                    
                    # Add surface if requested
                    if show_surface.value:
                        view.addSurface('VDW', {'opacity': 0.2, 'color': 'lightgray'}, viewer=(0, 0))
                        view.addSurface('VDW', {'opacity': 0.2, 'color': 'lightgray'}, viewer=(0, 1))
                    
                    # Labels for molecules
                    view.addLabel(f"Ref: {ref_dd.value[:15]}", {'position': {'x': 0, 'y': 10, 'z': 0}, 'fontSize': 12, 'fontColor': 'green', 'backgroundColor': 'white', 'backgroundOpacity': 0.8}, viewer=(0, 0))
                    view.addLabel(f"Cand: {cand_dd.value[:15]}", {'position': {'x': 0, 'y': 10, 'z': 0}, 'fontSize': 12, 'fontColor': 'blue', 'backgroundColor': 'white', 'backgroundOpacity': 0.8}, viewer=(0, 1))
                    
                    view.show()
                    
                    # Combined Legend
                    all_feats = set(ref_counts.keys()) | set(cand_counts.keys())
                    legend_html = "<div style='text-align:center; margin:10px 0; padding:8px; background:#f8f9fa; border-radius:8px'>"
                    legend_html += "<b>Feature Legend:</b> "
                    for ftype in sorted(all_feats):
                        c = color_map.get(ftype, 'gray')
                        text_c = 'white' if c in ['blue', 'red', 'purple', 'green'] else 'black'
                        legend_html += f"<span style='margin:0 4px'><span style='background:{c}; padding:2px 6px; border-radius:4px; color:{text_c}; font-size:10px'>{ftype}</span></span>"
                    legend_html += "</div>"
                    display(HTML(legend_html))
                    
                    # Summary comparison
                    print(f"📊 Reference: {sum(ref_counts.values())} features | Candidate: {sum(cand_counts.values())} features")
            
            pharm_btn.on_click(render_pharmacophore)
            refresh_btn.on_click(render_pharmacophore)
            
            # Layout with controls in a separate container
            controls_box = widgets.VBox([
                widgets.HBox([ref_dd, cand_dd]),
                widgets.HBox([style_dd, bg_dd, sphere_size]),
                widgets.HBox([show_labels, show_hydrogens, show_surface, pharm_btn, refresh_btn])
            ], layout=widgets.Layout(padding='10px', border='1px solid #e5e7eb', border_radius='8px', margin='0 0 10px 0'))
            
            display(controls_box)
            display(pharm_out)
            
            # Auto-render
            render_pharmacophore(None)
    
    def _update_pharm_grid(self):
        """Display pharmacophore grid comparison: Reference + Top N Hits."""
        with self.pharm_grid_output:
            clear_output(wait=True)
            
            if self._feature_factory is None:
                display(HTML("<p style='color:red'>❌ Pharmacophore feature factory not available.</p>"))
                return
            
            FEATURE_FACTORY = self._feature_factory
            
            # Color map
            color_map = {
                'Donor': 'blue',
                'Acceptor': 'red',
                'Aromatic': 'orange',
                'Hydrophobe': 'lightsteelblue',
                'LumpedHydrophobe': 'green',
                'PosIonizable': 'purple',
                'NegIonizable': 'cyan'
            }
            
            # Header
            display(HTML("""
            <div style="background:#ecfdf5; padding:10px; border-radius:8px; margin-bottom:15px; border:1px solid #6ee7b7;">
                <h4 style="margin:0; color:#047857;">🔬 Pharmacophore Grid Comparison</h4>
                <p style="margin:5px 0 0 0; color:#059669; font-size:12px;">Compare Reference with multiple top hits in a grid view</p>
            </div>
            """))
            
            # Controls
            n_hits_slider = widgets.IntSlider(value=5, min=2, max=8, step=1, description='# Hits:', style={'description_width': '50px'}, layout=widgets.Layout(width='180px'))
            style_dd = widgets.Dropdown(options=['Stick', 'Sphere', 'Stick + Sphere', 'Line'], value='Stick', description='Style:', style={'description_width': '40px'}, layout=widgets.Layout(width='150px'))
            bg_dd = widgets.Dropdown(options=[('White', 'white'), ('Black', 'black'), ('Gray', '#2d2d2d')], value='white', description='BG:', style={'description_width': '25px'}, layout=widgets.Layout(width='110px'))
            sphere_size = widgets.FloatSlider(value=0.7, min=0.3, max=1.2, step=0.1, description='Sphere:', style={'description_width': '50px'}, layout=widgets.Layout(width='170px'))
            
            show_labels = widgets.Checkbox(value=False, description='Labels', layout=widgets.Layout(width='75px'))
            show_hydrogens = widgets.Checkbox(value=False, description='Hydrogens', layout=widgets.Layout(width='95px'))
            
            grid_btn = widgets.Button(description='🔬 Generate Grid', button_style='success', layout=widgets.Layout(width='130px'))
            grid_out = widgets.Output(layout=widgets.Layout(max_height='600px', overflow_y='auto'))
            
            def render_grid(_):
                with grid_out:
                    clear_output(wait=True)
                    
                    # Get Reference + Top N hits
                    ref_name = self.current_ref
                    n_hits = n_hits_slider.value
                    
                    if not hasattr(self, 'ranked_names') or not self.ranked_names:
                        print("❌ No ranked candidates available. Please select a reference first.")
                        return
                    
                    hit_names = self.ranked_names[:n_hits]
                    all_mols = [ref_name] + hit_names
                    n_total = len(all_mols)
                    
                    # Calculate grid dimensions
                    n_cols = min(3, n_total)
                    n_rows = (n_total + n_cols - 1) // n_cols
                    
                    # Create grid view
                    view = py3Dmol.view(width=950, height=320 * n_rows, viewergrid=(n_rows, n_cols))
                    
                    all_feat_counts = []
                    
                    for idx, mol_name in enumerate(all_mols):
                        row_idx = idx // n_cols
                        col_idx = idx % n_cols
                        viewer_idx = (row_idx, col_idx)
                        
                        mol_obj = self.molecules.get(mol_name)
                        if not mol_obj or not mol_obj.mol:
                            continue
                        
                        mol = Chem.Mol(mol_obj.mol)
                        
                        # Ensure 3D
                        if mol.GetNumConformers() == 0:
                            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                        
                        # Handle hydrogens
                        if show_hydrogens.value:
                            mol_display = Chem.AddHs(mol, addCoords=True)
                        else:
                            mol_display = Chem.RemoveHs(mol)
                        
                        mb = Chem.MolToMolBlock(mol_display)
                        view.addModel(mb, 'mol', viewer=viewer_idx)
                        view.setBackgroundColor(bg_dd.value, viewer=viewer_idx)
                        
                        # Apply style
                        if style_dd.value == 'Stick':
                            view.setStyle({'stick': {'radius': 0.12}}, viewer=viewer_idx)
                        elif style_dd.value == 'Sphere':
                            view.setStyle({'sphere': {'scale': 0.3}}, viewer=viewer_idx)
                        elif style_dd.value == 'Stick + Sphere':
                            view.setStyle({'stick': {'radius': 0.1}, 'sphere': {'scale': 0.2}}, viewer=viewer_idx)
                        else:
                            view.setStyle({'line': {}}, viewer=viewer_idx)
                        
                        # Add pharmacophore spheres
                        features = FEATURE_FACTORY.GetFeaturesForMol(mol)
                        feat_counts = {}
                        
                        for f in features:
                            fam = f.GetFamily()
                            pos = f.GetPos()
                            color = color_map.get(fam, 'white')
                            
                            view.addSphere({
                                'center': {'x': float(pos.x), 'y': float(pos.y), 'z': float(pos.z)},
                                'radius': sphere_size.value,
                                'color': color,
                                'alpha': 0.6
                            }, viewer=viewer_idx)
                            
                            if show_labels.value:
                                view.addLabel(fam[:3], {
                                    'position': {'x': float(pos.x), 'y': float(pos.y), 'z': float(pos.z) + sphere_size.value + 0.2},
                                    'fontSize': 8,
                                    'fontColor': 'black' if bg_dd.value == 'white' else 'white',
                                    'backgroundOpacity': 0.5
                                }, viewer=viewer_idx)
                            
                            feat_counts[fam] = feat_counts.get(fam, 0) + 1
                        
                        all_feat_counts.append((mol_name, sum(feat_counts.values())))
                        
                        # Add molecule label
                        is_ref = idx == 0
                        label_text = f"🎯 {mol_name[:12]}" if is_ref else mol_name[:12]
                        label_color = 'green' if is_ref else 'blue'
                        view.addLabel(label_text, {
                            'position': {'x': 0, 'y': 8, 'z': 0},
                            'fontSize': 10,
                            'fontColor': label_color,
                            'backgroundColor': 'white',
                            'backgroundOpacity': 0.8
                        }, viewer=viewer_idx)
                        
                        view.zoomTo(viewer=viewer_idx)
                    
                    view.show()
                    
                    # Legend
                    legend_html = "<div style='text-align:center; margin:10px 0; padding:8px; background:#f8f9fa; border-radius:8px'>"
                    legend_html += "<b>Feature Legend:</b> "
                    for ftype, color in color_map.items():
                        text_c = 'white' if color in ['blue', 'red', 'purple', 'green'] else 'black'
                        legend_html += f"<span style='background:{color}; padding:2px 8px; border-radius:4px; color:{text_c}; font-size:11px; margin:0 3px'>{ftype}</span>"
                    legend_html += "</div>"
                    display(HTML(legend_html))
                    
                    # Summary
                    print(f"📊 Grid: {n_rows}x{n_cols} | Molecules: {n_total} | Reference: {ref_name}")
            
            grid_btn.on_click(render_grid)
            
            # Refresh button
            refresh_btn = widgets.Button(description='🔄 Refresh', button_style='info', layout=widgets.Layout(width='100px'))
            refresh_btn.on_click(render_grid)
            
            # Layout with controls in a separate container
            controls_box = widgets.VBox([
                widgets.HBox([n_hits_slider, style_dd, bg_dd]),
                widgets.HBox([sphere_size, show_labels, show_hydrogens]),
                widgets.HBox([grid_btn, refresh_btn])
            ], layout=widgets.Layout(padding='10px', border='1px solid #e5e7eb', border_radius='8px', margin='0 0 10px 0'))
            
            display(controls_box)
            display(grid_out)
            
            # Auto-render
            render_grid(None)
    
    def _update_heatmap(self):
        """Display interactive similarity heatmap with clustering."""
        with self.heatmap_output:
            clear_output(wait=True)
            
            # Header
            display(HTML("""
            <div style="background:#fef2f2; padding:10px; border-radius:8px; margin-bottom:15px; border:1px solid #fca5a5;">
                <h4 style="margin:0; color:#991b1b;">📈 Similarity Heatmap</h4>
                <p style="margin:5px 0 0 0; color:#dc2626; font-size:12px;">Interactive visualization of all pairwise molecular similarities</p>
            </div>
            """))
            
            # Controls
            sim_type = widgets.Dropdown(
                options=['Morgan', 'Pharm2D', 'Pharm3D', 'Consensus'],
                value='Morgan',
                description='Matrix:',
                style={'description_width': '50px'},
                layout=widgets.Layout(width='180px')
            )
            
            top_n = widgets.IntSlider(
                value=50,
                min=10,
                max=min(100, len(self.molecules)),
                step=10,
                description='Top N:',
                style={'description_width': '50px'},
                layout=widgets.Layout(width='220px')
            )
            
            use_clustering = widgets.Checkbox(
                value=True,
                description='Hierarchical Clustering',
                layout=widgets.Layout(width='180px')
            )
            
            colorscale_dd = widgets.Dropdown(
                options=['RdYlGn', 'Viridis', 'Blues', 'Hot', 'Plasma'],
                value='RdYlGn',
                description='Colors:',
                style={'description_width': '50px'},
                layout=widgets.Layout(width='160px')
            )
            
            generate_btn = widgets.Button(
                description='📊 Generate',
                button_style='danger',
                layout=widgets.Layout(width='110px')
            )
            
            heatmap_out = widgets.Output(layout=widgets.Layout(max_height='650px', overflow_y='auto'))
            
            def render_heatmap(_):
                with heatmap_out:
                    clear_output(wait=True)
                    
                    # Get the selected similarity matrix
                    matrix_name = sim_type.value
                    
                    if matrix_name == 'Consensus':
                        # Create weighted consensus
                        if not all(k in self.sim_matrices for k in ['Morgan', 'Pharm2D', 'Pharm3D']):
                            print("❌ Not all similarity matrices available for consensus")
                            return
                        
                        w_m = self.weights['Morgan']
                        w_p2d = self.weights['Pharm2D']
                        w_p3d = self.weights['Pharm3D']
                        
                        sim_matrix = (w_m * self.sim_matrices['Morgan'] + 
                                     w_p2d * self.sim_matrices['Pharm2D'] + 
                                     w_p3d * self.sim_matrices['Pharm3D'])
                    else:
                        if matrix_name not in self.sim_matrices:
                            print(f"❌ {matrix_name} matrix not available")
                            return
                        sim_matrix = self.sim_matrices[matrix_name]
                    
                    # Get top N molecules (based on consensus score with current reference)
                    available_mols = list(sim_matrix.index)
                    
                    if len(available_mols) == 0:
                        print("❌ Similarity matrix is empty")
                        return
                    
                    # Try to use ranked molecules if available
                    if hasattr(self, 'ranked_names') and self.ranked_names and self.current_ref:
                        # Check if reference is in matrix (might need "_optimized" suffix)
                        ref_candidates = [self.current_ref, f"{self.current_ref}_optimized"]
                        ref_in_matrix = next((r for r in ref_candidates if r in available_mols), None)
                        
                        # Build candidate list
                        candidate_mols = []
                        if ref_in_matrix:
                            candidate_mols.append(ref_in_matrix)
                        
                        # Add ranked candidates
                        for name in self.ranked_names[:top_n.value]:
                            name_candidates = [name, f"{name}_optimized", name.replace("_optimized", "")]
                            matched = next((n for n in name_candidates if n in available_mols), None)
                            if matched and matched not in candidate_mols:
                                candidate_mols.append(matched)
                        
                        top_mols = candidate_mols[:top_n.value]
                    else:
                        # Fallback: use first N molecules from matrix
                        top_mols = available_mols[:top_n.value]
                    
                    # Safety check
                    if len(top_mols) < 2:
                        print(f"❌ Not enough molecules. Using all {len(available_mols)} available molecules from matrix...")
                        top_mols = available_mols[:min(50, len(available_mols))]
                        if len(top_mols) < 2:
                            print(f"❌ Matrix only has {len(available_mols)} molecule(s)")
                            return
                    
                    # Filter matrix
                    sim_subset = sim_matrix.loc[top_mols, top_mols].copy()
                    
                    # Clustering
                    if use_clustering.value and len(top_mols) > 2:
                        try:
                            from scipy.cluster.hierarchy import linkage, dendrogram
                            from scipy.spatial.distance import squareform
                            
                            # Convert similarity to distance
                            dist_matrix = 1 - sim_subset.values
                            np.fill_diagonal(dist_matrix, 0)
                            
                            # Hierarchical clustering
                            condensed_dist = squareform(dist_matrix, checks=False)
                            linkage_matrix = linkage(condensed_dist, method='average')
                            
                            # Get dendrogram order
                            dend = dendrogram(linkage_matrix, no_plot=True)
                            cluster_order = dend['leaves']
                            
                            # Reorder
                            ordered_mols = [top_mols[i] for i in cluster_order]
                            sim_subset = sim_subset.loc[ordered_mols, ordered_mols]
                        except Exception as e:
                            print(f"Clustering failed: {e}, showing unclustered")
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=sim_subset.values,
                        x=[name[:15] for name in sim_subset.columns],
                        y=[name[:15] for name in sim_subset.index],
                        colorscale=colorscale_dd.value,
                        zmid=0.5,
                        zmin=0,
                        zmax=1,
                        text=sim_subset.values,
                        hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>',
                        colorbar=dict(title="Similarity", titleside="right")
                    ))
                    
                    # Highlight reference
                    ref_idx = list(sim_subset.index).index(self.current_ref) if self.current_ref in sim_subset.index else -1
                    
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{matrix_name} Similarity Heatmap</b> (Top {len(top_mols)} molecules)",
                            font=dict(size=14),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis=dict(
                            title="",
                            tickangle=-45,
                            side='bottom',
                            tickfont=dict(size=9)
                        ),
                        yaxis=dict(
                            title="",
                            tickfont=dict(size=9)
                        ),
                        width=850,
                        height=700,
                        margin=dict(l=120, r=120, t=80, b=120)
                    )
                    
                    # Add reference highlight
                    if ref_idx >= 0:
                        fig.add_hline(y=ref_idx, line_dash="dash", line_color="blue", line_width=2, opacity=0.5)
                        fig.add_vline(x=ref_idx, line_dash="dash", line_color="blue", line_width=2, opacity=0.5)
                    
                    fig.show()
                    
                    # Statistics
                    mean_sim = sim_subset.values[np.triu_indices_from(sim_subset.values, k=1)].mean()
                    max_sim = sim_subset.values[np.triu_indices_from(sim_subset.values, k=1)].max()
                    min_sim = sim_subset.values[np.triu_indices_from(sim_subset.values, k=1)].min()
                    
                    display(HTML(f"""
                    <div style='text-align:center; margin:10px 0; padding:8px; background:#f8f9fa; border-radius:8px'>
                        <b>Statistics:</b> Mean: {mean_sim:.3f} | Min: {min_sim:.3f} | Max: {max_sim:.3f}
                        {' | <span style="color:blue">Reference: ' + self.current_ref[:20] + '</span>' if ref_idx >= 0 else ''}
                    </div>
                    """))
            
            generate_btn.on_click(render_heatmap)
            
            # Layout
            controls_box = widgets.VBox([
                widgets.HBox([sim_type, top_n, colorscale_dd]),
                widgets.HBox([use_clustering, generate_btn])
            ], layout=widgets.Layout(padding='10px', border='1px solid #e5e7eb', border_radius='8px', margin='0 0 10px 0'))
            
            display(controls_box)
            display(heatmap_out)
            
            # Auto-render
            render_heatmap(None)
        
    def _update_data_table(self):
        """Render the full consensus table in the Data Explorer tab."""
        import matplotlib.pyplot as plt
        
        with self.table_output:
            clear_output(wait=True)
            if self.ranked_df.empty or not self.current_ref: return
            
            # --- 1. Data Table ---
            # Show Top 20
            df_show = self.ranked_df.head(20).copy()
            
            # Styling with Pandas Styler
            styler = df_show.style\
                .background_gradient(cmap='RdYlGn', subset=['Consensus'])\
                .format({'Consensus': '{:.3f}', 'Morgan': '{:.3f}', 'Pharm2D': '{:.3f}', 'Pharm3D': '{:.3f}'})\
                .set_properties(**{'text-align': 'center'})\
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f3f4f6'), ('color', 'black'), ('font-weight', 'bold'), ('text-align', 'center')]},
                    {'selector': 'td', 'props': [('border-bottom', '1px solid #eee')]}
                ])
                
            display(HTML(f"<h4>▶ Top 20 Candidates for {self.current_ref}</h4>"))
            display(styler)
            
            # --- 2. Similarity Breakdown Plots ---
            display(HTML("<hr style='margin: 20px 0;'>"))
            display(HTML(f"<h4>📊 Hit Breakdown by Metric (Top 15)</h4>"))
            
            # Helper to screen similar
            def screen_similar(ref, mat_name, top_n=15):
                if mat_name not in self.sim_matrices: return pd.Series()
                mat = self.sim_matrices[mat_name]
                
                # Handle keys
                if ref in mat.index:
                    sims = mat[ref].drop(ref)
                elif f"{ref}_optimized" in mat.index:
                     sims = mat[f"{ref}_optimized"]
                     if f"{ref}_optimized" in sims.index: sims = sims.drop(f"{ref}_optimized")
                else:
                    return pd.Series()
                    
                return sims.sort_values(ascending=False).head(top_n)

            hits_morgan = screen_similar(self.current_ref, 'Morgan')
            hits_pharm2d = screen_similar(self.current_ref, 'Pharm2D')
            hits_pharm3d = screen_similar(self.current_ref, 'Pharm3D')

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            
            # Helper for clean names
            def clean_names(idx):
                return [n.replace('_optimized', '')[:12] for n in idx]

            for ax, (title, hits) in zip(axes, [
                ('Morgan/ECFP', hits_morgan),
                ('2D Pharmacophore', hits_pharm2d),
                ('3D Pharmacophore', hits_pharm3d)
            ]):
                if hits.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center')
                    continue
                    
                names = clean_names(hits.index)
                # Color map
                colors = plt.cm.RdYlGn(hits.values)
                
                bars = ax.barh(range(len(hits)), hits.values, color=colors)
                ax.set_yticks(range(len(hits)))
                ax.set_yticklabels(names)
                ax.set_xlabel('Similarity')
                ax.set_title(title)
                ax.set_xlim(0, 1)
                ax.invert_yaxis()
                
                # Add labels
                for i, v in enumerate(hits.values):
                    ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)

            plt.suptitle(f'Top Hits per Metric for {self.current_ref}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
    def _on_radar_dropdown_change(self, change, idx):
        if change['new']:
            self._update_single_radar(idx, change['new'])

    def _update_radar(self):
        """Legacy method, now handled by _update_single_radar loops."""
        pass # Now handled per slot

    def _update_single_radar(self, idx, candidate_name):
        """Update just one radar chart slot."""
        slot = self.radar_slots[idx]
        with slot['output']:
            clear_output(wait=True)
            if not candidate_name: return
            
            # --- Gather Data ---
            mol = self.molecules.get(candidate_name)
            mol_ref = self.molecules.get(self.current_ref) # Get Reference Object
            
            if not mol or not mol_ref: return
            
            p = mol.properties
            p_ref = mol_ref.properties # Reference Props
            
            # Helper: get similarity
            def get_sim(mat_name, n1, n2):
                if mat_name not in self.sim_matrices: return 0
                mat = self.sim_matrices[mat_name]
                if n1 in mat.index and n2 in mat.columns: return mat.loc[n1, n2]
                n1_opt, n2_opt = f"{n1}_optimized", f"{n2}_optimized"
                if n1_opt in mat.index and n2_opt in mat.columns: return mat.loc[n1_opt, n2_opt]
                return 0
                
            morgan_sim = get_sim('Morgan', self.current_ref, candidate_name)
            pharm3d_sim = get_sim('Pharm3D', self.current_ref, candidate_name)
            
            # Props
            categories = ['Morgan Sim', 'Pharm3D Sim', 'HBD', 'HBA', 'LogP', 'TPSA', 'QED']
            
            # Helper to normalize
            def get_values(props, is_ref=False):
                # For Reference, Similarity to self is 1.0
                m_sim = 1.0 if is_ref else morgan_sim
                p3_sim = 1.0 if is_ref else pharm3d_sim
                
                return [
                    m_sim,
                    p3_sim,
                    min(props.get('HBD', 0) / 7, 1),
                    min(props.get('HBA', 0) / 12, 1),
                    min(props.get('LogP', 0) / 6, 1),
                    min(props.get('TPSA', 0) / 160, 1),
                    props.get('QED', 0)
                ]

            cand_values = get_values(p, is_ref=False)
            ref_values = get_values(p_ref, is_ref=True)
            
            # --- Get Image ---
            img_html = self._get_mol_image_html(candidate_name, size=(180, 100))
            
            # --- Plot ---
            fig = go.Figure()
            
            # Trace 1: Reference (Orange)
            fig.add_trace(go.Scatterpolar(
                r=ref_values, theta=categories, fill='toself', 
                name=f'Ref: {self.current_ref}',
                line_color='#f97316',  # Orange
                fillcolor='rgba(249, 115, 22, 0.1)',
                opacity=0.6,
                showlegend=True
            ))
            
            # Custom Colors for slots
            slot_colors = ['#2563eb', '#7c3aed', '#4ade80', '#166534']
            color = slot_colors[idx] if idx < len(slot_colors) else '#2563eb'
            
            # Trace 2: Candidate (slot specific color)
            fig.add_trace(go.Scatterpolar(
                r=cand_values, theta=categories, fill='toself', 
                name=candidate_name,
                line_color=color,
                opacity=0.7,
                showlegend=True
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                height=320,
                margin=dict(l=40, r=40, t=30, b=20),
                title=dict(text=f"<b>{candidate_name}</b>", font=dict(size=12), y=0.95)
            )
            
            # Composite Layout: Image + Chart
            # We use a VBox-like HTML structure or just display side-by-side using HBox widget?
            # Since output capture captures `display` calls sequentially, we can do:
            # display(HTML(table with image and plot))? No, Plotly is a widget.
            # Best way: Use VBox widget inside the Output? Or just display logic.
            # Let's try displaying HTML image then Figure.
            
            # Side-by-side Layout using HTML Flexbox wrapping the plot? hard with Plotly widget.
            # We'll put Image ON TOP of plot or TO THE LEFT.
            
            # Let's put Image to the left and Plot to the right in a mini HBox?
            # We are INSIDE a `with output:` context.
            
            # Simple approach: Display Image centered, then Plot.
            display(widgets.HTML(f"<div style='text-align:center'>{img_html}</div>"))
            fig.show()

    def _update_3d_viewer(self):
        """Update 3D Molecule View using InteractivePy3DmolViewer module."""
        from src.interactive.py3dmol_viewer import InteractivePy3DmolViewer
        
        with self.viewer_output:
            clear_output(wait=True)
            
            # Header with Reference Info
            ref_obj = self.molecules.get(self.current_ref)
            ref_img = self._get_mol_image_html(self.current_ref, size=(200, 100)) if ref_obj else ""
            
            display(HTML(f"""
            <div style="display:flex; align-items:center; background:#f0fdf4; padding:10px; border-radius:8px; margin-bottom:15px; border:1px solid #bbf7d0;">
                <div style="margin-right:15px;">{ref_img}</div>
                <div>
                    <h4 style="margin:0; color:#166534;">🎯 Reference: {self.current_ref}</h4>
                    <p style="margin:5px 0 0 0; color:#4ade80; font-size:12px;">Select a molecule below to explore its 3D structure</p>
                </div>
            </div>
            """))
            
            # Create the InteractivePy3DmolViewer
            viewer = InteractivePy3DmolViewer()
            
            # Load all molecules
            mol_list = list(self.molecules.values())
            viewer.load_molecules(mol_list)
            
            # Set default to Top 1 candidate if available
            if hasattr(self, 'ranked_names') and self.ranked_names:
                default_mol = self.ranked_names[0]
                if default_mol in viewer.molecule_dropdown.options:
                    viewer.molecule_dropdown.value = default_mol
            
            # Display the full interactive viewer interface
            display(viewer.display())
            
    def _update_properties(self):
        """Show comparison table for Top 10 candidates."""
        with self.props_output:
            clear_output(wait=True)
            
            if self.ranked_df.empty: return
            
            # Get Top 10
            top_10_names = self.ranked_df.head(10)['Name'].tolist()
            
            data = []
            for name in top_10_names:
                mol = self.molecules.get(name)
                if not mol: continue
                p = mol.properties
                
                # Find consensus score
                score = self.ranked_df[self.ranked_df['Name']==name]['Consensus'].values[0]
                
                data.append({
                    'Molecule': name,
                    'Score': score,
                    'MW': p.get('MW', 0),
                    'LogP': p.get('LogP', 0),
                    'TPSA': p.get('TPSA', 0),
                    'HBD': p.get('HBD', 0),
                    'HBA': p.get('HBA', 0),
                    'QED': p.get('QED', 0),
                    'Fsp3': p.get('FractionCSP3', 0)
                })
            
            df_comp = pd.DataFrame(data)
            
            # Styling
            styler = df_comp.style\
                .format({'Score': '{:.3f}', 'MW': '{:.1f}', 'LogP': '{:.2f}', 'TPSA': '{:.0f}', 
                         'QED': '{:.2f}', 'Fsp3': '{:.2f}'})\
                .background_gradient(subset=['Score'], cmap='RdYlGn')\
                .bar(subset=['LogP', 'MW'], color='#dbeafe')\
                .set_properties(**{'text-align': 'center'})\
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f8fafc'), ('color', '#334155'), ('font-weight', 'bold')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#f1f5f9')]}
                ])
                
            display(HTML(f"<h4>📊 Properties Comparison (Top 10)</h4>"))
            display(styler)

    def show(self):
        """Display the dashboard."""
        display(self.layout)
