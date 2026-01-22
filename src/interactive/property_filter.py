"""
Interactive Property Filter with histograms and sliders.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class InteractivePropertyFilter:
    """Filter molecules by properties with live histogram visualization.
    
    Supports both basic molecular properties and similarity scores from screening.
    """
    
    def __init__(self, molecules_list=None):
        """
        Initialize property filter.
        
        Parameters:
        -----------
        molecules_list : list
            List of DrugMolecule objects
        """
        self.molecules = molecules_list if molecules_list else []
        self.filtered_molecules = list(self.molecules)
        self.has_similarity_scores = False  # Track if scores are available
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Create filter widgets."""
        self.mw_range = widgets.FloatRangeSlider(
            value=[0, 1000],
            min=0, max=1500,
            step=10,
            description='MW:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='350px'),
            readout_format='.0f'
        )
        
        self.logp_range = widgets.FloatRangeSlider(
            value=[-5, 10],
            min=-10, max=15,
            step=0.5,
            description='LogP:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='350px')
        )
        
        self.tpsa_range = widgets.FloatRangeSlider(
            value=[0, 200],
            min=0, max=300,
            step=5,
            description='TPSA:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='350px')
        )
        
        self.qed_range = widgets.FloatRangeSlider(
            value=[0, 1],
            min=0, max=1,
            step=0.05,
            description='QED:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='350px')
        )
        
        self.hbd_max = widgets.IntSlider(
            value=10,
            min=0, max=20,
            description='Max HBD:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='350px')
        )
        
        self.hba_max = widgets.IntSlider(
            value=15,
            min=0, max=30,
            description='Max HBA:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='350px')
        )
        
        # Drug-likeness filter checkboxes
        self.lipinski_check = widgets.Checkbox(
            value=False,
            description='Lipinski (Ro5)',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='140px'),
            indent=False
        )
        
        self.veber_check = widgets.Checkbox(
            value=False,
            description='Veber',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100px'),
            indent=False
        )
        
        self.leadlike_check = widgets.Checkbox(
            value=False,
            description='Lead-like',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='110px'),
            indent=False
        )
        
        self.filter_btn = widgets.Button(
            description='🔍 Apply Filters',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.filter_btn.on_click(self._apply_filters)
        
        self.reset_btn = widgets.Button(
            description='🔄 Reset',
            button_style='warning',
            layout=widgets.Layout(width='100px')
        )
        self.reset_btn.on_click(self._reset_filters)
        
        self.result_output = widgets.Output()
        self.histogram_output = widgets.Output()
        self.molecule_list_output = widgets.Output(layout=widgets.Layout(
            max_height='400px', overflow_y='auto'
        ))
        
        # Similarity score sliders (hidden by default, shown after update_with_scores)
        self.morgan_range = widgets.FloatRangeSlider(
            value=[0, 1], min=0, max=1, step=0.05,
            description='Morgan:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='280px', display='none')
        )
        
        self.pharm2d_range = widgets.FloatRangeSlider(
            value=[0, 1], min=0, max=1, step=0.05,
            description='Pharm2D:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='280px', display='none')
        )
        
        self.pharm3d_range = widgets.FloatRangeSlider(
            value=[0, 1], min=0, max=1, step=0.05,
            description='Pharm3D:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='280px', display='none')
        )
        
        self.consensus_range = widgets.FloatRangeSlider(
            value=[0, 1], min=0, max=1, step=0.05,
            description='Consensus:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='280px', display='none')
        )
        
        # Container for similarity sliders (hidden by default, shown after update_with_scores)
        self.similarity_sliders_box = widgets.VBox([
            widgets.HTML("<b style='color:#667eea;'>🎯 Similarity Scores</b>"),
            widgets.HBox([self.morgan_range, self.pharm2d_range]),
            widgets.HBox([self.pharm3d_range, self.consensus_range])
        ], layout=widgets.Layout(display='none', padding='10px', 
                                 border='2px solid #667eea', border_radius='8px', margin='10px 0'))
        
        # Placeholder message shown before scores are computed
        self.scores_placeholder = widgets.HTML("""
            <div style='padding:10px 15px; border:2px dashed #cbd5e1; border-radius:8px; 
                        margin:10px 0; background:#f8fafc;'>
                <span style='color:#64748b;'>
                    💡 <b>Similarity scores not yet computed.</b><br>
                    <span style='font-size:12px;'>
                        Run similarity screening (Morgan, Pharm2D, Pharm3D, Consensus), then call:<br>
                        <code style='background:#e2e8f0; padding:2px 6px; border-radius:3px;'>
                            property_filter.update_with_scores(consensus_df)
                        </code>
                    </span>
                </span>
            </div>
        """)
    
    def load_molecules(self, molecules_list):
        """
        Load molecules into filter.
        
        Parameters:
        -----------
        molecules_list : list
            List of DrugMolecule objects
        """
        self.molecules = molecules_list
        self.filtered_molecules = list(molecules_list)
        self._apply_filters()
    
    def update_with_scores(self, scores_df, molecule_col='Molecule', score_cols=None):
        """
        Update molecules with similarity scores from screening.
        
        Parameters:
        -----------
        scores_df : pd.DataFrame
            DataFrame with similarity scores (columns: Molecule, Morgan, Pharm2D, Pharm3D, Consensus)
        molecule_col : str
            Column name containing molecule names
        score_cols : list, optional
            List of score column names. Default: ['Morgan', 'Pharm2D', 'Pharm3D', 'Consensus']
        """
        if score_cols is None:
            score_cols = ['Morgan', 'Pharm2D', 'Pharm3D', 'Consensus']
        
        # Create lookup dict from DataFrame
        score_lookup = {}
        for _, row in scores_df.iterrows():
            mol_name = row[molecule_col]
            # Handle both with and without _optimized suffix
            for suffix in ['', '_optimized']:
                key = mol_name.replace('_optimized', '') + suffix
                score_lookup[key] = {col: row.get(col, 0) for col in score_cols}
                score_lookup[mol_name.replace('_optimized', '')] = score_lookup[key]
        
        # Update molecule properties
        updated_count = 0
        for mol in self.molecules:
            mol_name = mol.name.replace('_optimized', '')
            if mol_name in score_lookup:
                scores = score_lookup[mol_name]
                for col in score_cols:
                    mol.properties[col] = scores.get(col, 0)
                updated_count += 1
        
        self.has_similarity_scores = updated_count > 0
        
        # Show similarity sliders and hide placeholder if scores available
        if self.has_similarity_scores:
            self.similarity_sliders_box.layout.display = 'block'
            self.scores_placeholder.layout.display = 'none'  # Hide placeholder
            self.morgan_range.layout.display = 'block' if 'Morgan' in score_cols else 'none'
            self.pharm2d_range.layout.display = 'block' if 'Pharm2D' in score_cols else 'none'
            self.pharm3d_range.layout.display = 'block' if 'Pharm3D' in score_cols else 'none'
            self.consensus_range.layout.display = 'block' if 'Consensus' in score_cols else 'none'
        
        # Re-apply filters to update display
        self._apply_filters()
        
        print(f"✅ Updated {updated_count} molecules with similarity scores")
        print("   📊 Similarity columns now visible in table")
        print("   🎚️ Similarity sliders now active")
        return updated_count
    
    def _apply_filters(self, b=None):
        """Apply current filter settings."""
        self.filtered_molecules = []
        
        for mol in self.molecules:
            if not mol.properties:
                mol.calculate_descriptors()
            
            props = mol.properties
            mw = props.get('MW', 0)
            logp = props.get('LogP', 0)
            tpsa = props.get('TPSA', 0)
            qed = props.get('QED', 0)
            hbd = props.get('HBD', 0)
            hba = props.get('HBA', 0)
            rotbond = props.get('RotatableBonds', 0)
            
            # Check range filter conditions
            if not (self.mw_range.value[0] <= mw <= self.mw_range.value[1]):
                continue
            if not (self.logp_range.value[0] <= logp <= self.logp_range.value[1]):
                continue
            if not (self.tpsa_range.value[0] <= tpsa <= self.tpsa_range.value[1]):
                continue
            if not (self.qed_range.value[0] <= qed <= self.qed_range.value[1]):
                continue
            if hbd > self.hbd_max.value:
                continue
            if hba > self.hba_max.value:
                continue
            
            # Check drug-likeness filters if enabled
            # Lipinski's Rule of 5
            if self.lipinski_check.value:
                lip_violations = 0
                if mw > 500: lip_violations += 1
                if logp > 5: lip_violations += 1
                if hbd > 5: lip_violations += 1
                if hba > 10: lip_violations += 1
                if lip_violations > 1:  # Allow 1 violation
                    continue
            
            # Veber's oral bioavailability rules
            if self.veber_check.value:
                if rotbond > 10 or tpsa > 140:
                    continue
            
            # Lead-likeness criteria
            if self.leadlike_check.value:
                if mw > 450 or logp > 4.5 or rotbond > 7:
                    continue
            
            # Check similarity score filters if available
            if self.has_similarity_scores:
                morgan = props.get('Morgan', 0)
                pharm2d = props.get('Pharm2D', 0)
                pharm3d = props.get('Pharm3D', 0)
                consensus = props.get('Consensus', 0)
                
                if self.morgan_range.layout.display != 'none':
                    if not (self.morgan_range.value[0] <= morgan <= self.morgan_range.value[1]):
                        continue
                if self.pharm2d_range.layout.display != 'none':
                    if not (self.pharm2d_range.value[0] <= pharm2d <= self.pharm2d_range.value[1]):
                        continue
                if self.pharm3d_range.layout.display != 'none':
                    if not (self.pharm3d_range.value[0] <= pharm3d <= self.pharm3d_range.value[1]):
                        continue
                if self.consensus_range.layout.display != 'none':
                    if not (self.consensus_range.value[0] <= consensus <= self.consensus_range.value[1]):
                        continue
            
            self.filtered_molecules.append(mol)
        
        self._show_results()
    
    def _reset_filters(self, b=None):
        """Reset all filters."""
        self.mw_range.value = [0, 1000]
        self.logp_range.value = [-5, 10]
        self.tpsa_range.value = [0, 200]
        self.qed_range.value = [0, 1]
        self.hbd_max.value = 10
        self.hba_max.value = 15
        # Reset drug-likeness checkboxes
        self.lipinski_check.value = False
        self.veber_check.value = False
        self.leadlike_check.value = False
        # Reset similarity sliders if available
        self.morgan_range.value = [0, 1]
        self.pharm2d_range.value = [0, 1]
        self.pharm3d_range.value = [0, 1]
        self.consensus_range.value = [0, 1]
        self._apply_filters()
    
    def _show_results(self):
        """Display filtered results."""
        with self.result_output:
            clear_output(wait=True)
            
            total = len(self.molecules)
            filtered = len(self.filtered_molecules)
            pct = (filtered / total * 100) if total > 0 else 0
            
            print(f"📊 Filtered Molecules: {filtered}/{total} ({pct:.1f}%)")
            print("─" * 50)
            
            for mol in self.filtered_molecules[:10]:
                props = mol.properties
                print(f"  • {mol.name}: MW={props.get('MW', 0):.1f}, "
                      f"LogP={props.get('LogP', 0):.2f}, QED={props.get('QED', 0):.3f}")
            
            if len(self.filtered_molecules) > 10:
                print(f"  ... and {len(self.filtered_molecules) - 10} more")
        
        # Populate molecule list table
        self._show_molecule_list()
        
        self._plot_histograms()
    
    def _show_molecule_list(self):
        """Display filtered molecules as a styled table."""
        from IPython.display import HTML
        
        with self.molecule_list_output:
            clear_output(wait=True)
            
            if not self.filtered_molecules:
                display(HTML("<p style='color:#888; padding:20px;'>No molecules match the current filters.</p>"))
                return
            
            # Build HTML table with conditional similarity columns
            html = """
            <style>
                .mol-table { width:100%; border-collapse:collapse; font-size:11px; }
                .mol-table th { background:linear-gradient(135deg, #667eea, #764ba2); color:white; 
                               padding:6px 4px; text-align:center; position:sticky; top:0; font-size:10px; }
                .mol-table td { padding:5px 4px; border-bottom:1px solid #eee; text-align:center; }
                .mol-table td:nth-child(2) { text-align:left; }
                .mol-table tr:hover { background:#f5f5ff; }
                .mol-table tr:nth-child(even) { background:#fafafa; }
                .pass { color:#10b981; font-weight:600; }
                .fail { color:#ef4444; }
                .qed-high { background:#d1fae5; color:#065f46; padding:2px 4px; border-radius:4px; }
                .qed-med { background:#fef3c7; color:#92400e; padding:2px 4px; border-radius:4px; }
                .qed-low { background:#fee2e2; color:#991b1b; padding:2px 4px; border-radius:4px; }
                .fsp3-high { background:#dbeafe; color:#1e40af; padding:2px 4px; border-radius:4px; }
                .fsp3-low { background:#fef3c7; color:#92400e; padding:2px 4px; border-radius:4px; }
                .sim-high { background:#d1fae5; color:#065f46; padding:2px 4px; border-radius:4px; font-weight:600; }
                .sim-med { background:#dbeafe; color:#1e40af; padding:2px 4px; border-radius:4px; }
                .sim-low { background:#fee2e2; color:#991b1b; padding:2px 4px; border-radius:4px; }
            </style>
            <table class='mol-table'>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Molecule</th>
                        <th>MW</th>
                        <th>LogP</th>
                        <th>TPSA</th>
                        <th>HBD</th>
                        <th>HBA</th>
                        <th>RotBond</th>
                        <th>Fsp3</th>
                        <th>QED</th>
                        <th>Lip</th>
                        <th>Veb</th>
                        <th>Lead</th>
            """
            
            # Add similarity score headers if available
            if self.has_similarity_scores:
                html += """
                        <th style='background:#2563eb;'>Morgan</th>
                        <th style='background:#7c3aed;'>Ph2D</th>
                        <th style='background:#db2777;'>Ph3D</th>
                        <th style='background:#059669;'>Cons</th>
                """
            
            html += """
                    </tr>
                </thead>
                <tbody>
            """
            
            for i, mol in enumerate(self.filtered_molecules, 1):
                props = mol.properties
                mw = props.get('MW', 0)
                logp = props.get('LogP', 0)
                tpsa = props.get('TPSA', 0)
                hbd = props.get('HBD', 0)
                hba = props.get('HBA', 0)
                rotbond = props.get('RotatableBonds', 0)
                fsp3 = props.get('FractionCSP3', 0)
                qed = props.get('QED', 0)
                
                # Lipinski check (Rule of 5)
                lip_violations = 0
                if mw > 500: lip_violations += 1
                if logp > 5: lip_violations += 1
                if hbd > 5: lip_violations += 1
                if hba > 10: lip_violations += 1
                lipinski = f"<span class='pass'>✓</span>" if lip_violations == 0 else f"<span class='fail'>✗{lip_violations}</span>"
                
                # Veber check (oral bioavailability)
                veber_pass = rotbond <= 10 and tpsa <= 140
                veber = f"<span class='pass'>✓</span>" if veber_pass else f"<span class='fail'>✗</span>"
                
                # Lead-likeness check
                lead_pass = mw <= 450 and logp <= 4.5 and rotbond <= 7
                lead = f"<span class='pass'>✓</span>" if lead_pass else f"<span class='fail'>✗</span>"
                
                # QED styling
                if qed >= 0.67:
                    qed_class = 'qed-high'
                elif qed >= 0.4:
                    qed_class = 'qed-med'
                else:
                    qed_class = 'qed-low'
                
                # Fsp3 styling
                fsp3_class = 'fsp3-high' if fsp3 >= 0.25 else 'fsp3-low'
                
                html += f"""
                    <tr>
                        <td>{i}</td>
                        <td><b>{mol.name}</b></td>
                        <td>{mw:.1f}</td>
                        <td>{logp:.2f}</td>
                        <td>{tpsa:.1f}</td>
                        <td>{hbd}</td>
                        <td>{hba}</td>
                        <td>{rotbond}</td>
                        <td><span class='{fsp3_class}'>{fsp3:.2f}</span></td>
                        <td><span class='{qed_class}'>{qed:.3f}</span></td>
                        <td>{lipinski}</td>
                        <td>{veber}</td>
                        <td>{lead}</td>
                """
                
                # Add similarity score columns if available
                if self.has_similarity_scores:
                    morgan = props.get('Morgan', 0)
                    pharm2d = props.get('Pharm2D', 0)
                    pharm3d = props.get('Pharm3D', 0)
                    consensus = props.get('Consensus', 0)
                    
                    def sim_class(val):
                        if val >= 0.7: return 'sim-high'
                        elif val >= 0.4: return 'sim-med'
                        else: return 'sim-low'
                    
                    html += f"""
                        <td><span class='{sim_class(morgan)}'>{morgan:.3f}</span></td>
                        <td><span class='{sim_class(pharm2d)}'>{pharm2d:.3f}</span></td>
                        <td><span class='{sim_class(pharm3d)}'>{pharm3d:.3f}</span></td>
                        <td><span class='{sim_class(consensus)}'>{consensus:.3f}</span></td>
                    """
                
                html += "</tr>"
            
            html += "</tbody></table>"
            display(HTML(html))
    
    def _plot_histograms(self):
        """Plot property distributions."""
        with self.histogram_output:
            clear_output(wait=True)
            
            if not self.filtered_molecules:
                print("No molecules match current filters")
                return
            
            if not PLOTLY_AVAILABLE:
                print("Plotly not available for histograms")
                return
            
            props_to_plot = ['MW', 'LogP', 'QED', 'TPSA']
            values = {p: [] for p in props_to_plot}
            
            for mol in self.filtered_molecules:
                for p in props_to_plot:
                    values[p].append(mol.properties.get(p, 0))
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=props_to_plot)
            
            colors = ['#667eea', '#f56565', '#38a169', '#ed8936']
            
            for i, (prop, color) in enumerate(zip(props_to_plot, colors)):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Histogram(x=values[prop], nbinsx=15, 
                                marker_color=color, opacity=0.7,
                                name=prop),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=400,
                width=600,
                showlegend=False,
                title_text="Property Distributions",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            fig.show()
    
    def get_filtered_molecules(self):
        """Get the currently filtered molecules."""
        return self.filtered_molecules
    
    def display(self):
        """Display the filter interface."""
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 16px; border-radius: 12px; margin-bottom: 16px;'>
            <h3 style='color: white; margin: 0;'>🎛️ Interactive Property Filter</h3>
            <p style='color: rgba(255,255,255,0.85); margin: 8px 0 0 0; font-size: 13px;'>
                Filter molecules by drug-like properties with live visualization
            </p>
        </div>
        """)
        
        filters_left = widgets.VBox([
            self.mw_range,
            self.logp_range,
            self.tpsa_range
        ])
        
        filters_right = widgets.VBox([
            self.qed_range,
            self.hbd_max,
            self.hba_max
        ])
        
        filters_row = widgets.HBox([filters_left, filters_right], 
                                   layout=widgets.Layout(gap='20px'))
        
        # Drug-likeness filter checkboxes row
        druglike_filters = widgets.HBox([
            widgets.HTML("<b style='margin-right:10px;'>🧪 Must Pass:</b>"),
            self.lipinski_check,
            self.veber_check,
            self.leadlike_check
        ], layout=widgets.Layout(
            padding='10px 15px',
            border='2px solid #10b981',
            border_radius='8px',
            margin='10px 0',
            align_items='center'
        ))
        
        buttons_row = widgets.HBox([self.filter_btn, self.reset_btn],
                                   layout=widgets.Layout(gap='12px', margin='16px 0'))
        
        results_row = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<b>📋 Summary</b>"),
                self.result_output
            ], layout=widgets.Layout(width='380px', padding='12px',
                                    border='1px solid #e0e0e0', border_radius='8px')),
            widgets.VBox([
                widgets.HTML("<b>📊 Distributions</b>"),
                self.histogram_output
            ], layout=widgets.Layout(width='650px', padding='12px',
                                    border='1px solid #e0e0e0', border_radius='8px'))
        ], layout=widgets.Layout(gap='16px'))
        
        # Molecule list panel
        molecule_list_panel = widgets.VBox([
            widgets.HTML("""
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:8px;'>
                    <b>📝 Filtered Molecules List</b>
                    <span style='color:#888; font-size:12px;'>(scrollable)</span>
                </div>
            """),
            self.molecule_list_output
        ], layout=widgets.Layout(
            width='100%', 
            padding='12px',
            border='2px solid #667eea', 
            border_radius='8px',
            margin='16px 0 0 0'
        ))
        
        interface = widgets.VBox([
            header,
            filters_row,
            druglike_filters,
            self.scores_placeholder,  # Shows placeholder before scores computed
            self.similarity_sliders_box,  # Shows sliders after update_with_scores()
            buttons_row,
            results_row,
            molecule_list_panel
        ], layout=widgets.Layout(max_width='1100px'))
        
        # Show initial results
        if self.molecules:
            self._apply_filters()
        
        return interface