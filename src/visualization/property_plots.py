"""
Property plotting and visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PropertyPlotter:
    """Create plots for molecular properties and ADMET data."""
    
    def __init__(self):
        """Initialize property plotter."""
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2', 
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e',
            'info': '#3182ce'
        }
    
    def plot_property_distribution(self, molecules, property_name, bins=20, 
                                  title=None, show_filters=True):
        """
        Plot distribution of a molecular property.
        
        Parameters:
        -----------
        molecules : list
            List of DrugMolecule objects
        property_name : str
            Property to plot ('MW', 'LogP', 'TPSA', etc.)
        bins : int
            Number of histogram bins
        title : str, optional
            Plot title
        show_filters : bool
            Show filter thresholds
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        # Extract property values
        values = []
        names = []
        passes_filters = []
        
        for mol in molecules:
            if not mol.properties:
                mol.calculate_descriptors()
            
            if property_name in mol.properties:
                values.append(mol.properties[property_name])
                names.append(mol.name)
                passes_filters.append(mol.passes_filters())
        
        if not values:
            raise ValueError(f"No {property_name} data found in molecules")
        
        # Create figure
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=bins,
            name='All Molecules',
            marker_color=self.colors['primary'],
            opacity=0.7,
            hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ))
        
        # Overlay for molecules that pass filters
        if show_filters:
            passing_values = [v for v, p in zip(values, passes_filters) if p]
            if passing_values:
                fig.add_trace(go.Histogram(
                    x=passing_values,
                    nbinsx=bins,
                    name='Pass Filters',
                    marker_color=self.colors['success'],
                    opacity=0.8
                ))
        
        # Add filter thresholds
        if show_filters:
            self._add_filter_lines(fig, property_name)
        
        # Update layout
        title = title or f"{property_name} Distribution"
        fig.update_layout(
            title=title,
            xaxis_title=property_name,
            yaxis_title='Count',
            barmode='overlay',
            width=600,
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _add_filter_lines(self, fig, property_name):
        """Add filter threshold lines to plot."""
        from ..core.config import Config
        
        thresholds = {
            'MW': {'min': Config.MW_MIN, 'max': Config.MW_MAX},
            'LogP': {'min': Config.LOGP_MIN, 'max': Config.LOGP_MAX},
            'TPSA': {'max': Config.TPSA_MAX},
            'HBD': {'max': Config.HBD_MAX},
            'HBA': {'max': Config.HBA_MAX},
            'RotatableBonds': {'max': Config.ROTATABLE_BONDS_MAX}
        }
        
        if property_name in thresholds:
            thresholds_prop = thresholds[property_name]
            
            # Add vertical lines for thresholds
            if 'min' in thresholds_prop:
                fig.add_vline(
                    x=thresholds_prop['min'], 
                    line_dash="dash", 
                    line_color=self.colors['warning'],
                    annotation_text=f"Min: {thresholds_prop['min']}"
                )
            
            if 'max' in thresholds_prop:
                fig.add_vline(
                    x=thresholds_prop['max'], 
                    line_dash="dash", 
                    line_color=self.colors['danger'],
                    annotation_text=f"Max: {thresholds_prop['max']}"
                )
    
    def plot_property_correlation(self, molecules, prop1, prop2, color_by=None):
        """
        Plot correlation between two properties.
        
        Parameters:
        -----------
        molecules : list
            List of DrugMolecule objects
        prop1 : str
            First property
        prop2 : str
            Second property  
        color_by : str, optional
            Property to color points by
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        # Extract data
        x_vals, y_vals, names, colors = [], [], [], []
        
        for mol in molecules:
            if not mol.properties:
                mol.calculate_descriptors()
            
            if prop1 in mol.properties and prop2 in mol.properties:
                x_vals.append(mol.properties[prop1])
                y_vals.append(mol.properties[prop2])
                names.append(mol.name)
                
                if color_by and color_by in mol.properties:
                    colors.append(mol.properties[color_by])
                else:
                    colors.append(mol.passes_filters())
        
        if not x_vals:
            raise ValueError(f"No data found for {prop1} and {prop2}")
        
        # Create scatter plot
        fig = go.Figure()
        
        if color_by:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    color=colors,
                    colorscale='viridis',
                    colorbar=dict(title=color_by),
                    size=8,
                    line=dict(width=1, color='white')
                ),
                text=names,
                hovertemplate='<b>%{text}</b><br>' +
                            f'{prop1}: %{{x}}<br>' +
                            f'{prop2}: %{{y}}<br>' +
                            f'{color_by}: %{{marker.color}}<extra></extra>'
            ))
        else:
            # Color by filter status
            colors_discrete = ['red' if not c else 'blue' for c in colors]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    color=colors_discrete,
                    size=8,
                    line=dict(width=1, color='white')
                ),
                text=names,
                hovertemplate='<b>%{text}</b><br>' +
                            f'{prop1}: %{{x}}<br>' +
                            f'{prop2}: %{{y}}<extra></extra>'
            ))
        
        # Add correlation line
        if len(x_vals) > 2:
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_vals), max(x_vals), 100)
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'R = {correlation:.3f}',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"{prop1} vs {prop2}",
            xaxis_title=prop1,
            yaxis_title=prop2,
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_admet_summary(self, molecules, max_molecules=50):
        """
        Create comprehensive ADMET summary plot.
        
        Parameters:
        -----------
        molecules : list
            List of DrugMolecule objects
        max_molecules : int
            Maximum number of molecules to include
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        # Limit number of molecules for readability
        molecules = molecules[:max_molecules]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['MW Distribution', 'LogP Distribution', 
                          'TPSA Distribution', 'Property Correlations',
                          'Filter Pass Rate', 'QED Distribution'],
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'histogram'}]]
        )
        
        # Extract all properties
        properties = {}
        for prop in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'QED']:
            properties[prop] = []
        
        pass_count = 0
        for mol in molecules:
            if not mol.properties:
                mol.calculate_descriptors()
            
            for prop in properties:
                if prop in mol.properties:
                    properties[prop].append(mol.properties[prop])
            
            if mol.passes_filters():
                pass_count += 1
        
        # MW distribution
        if properties['MW']:
            fig.add_trace(go.Histogram(
                x=properties['MW'],
                name='MW',
                marker_color=self.colors['primary']
            ), row=1, col=1)
        
        # LogP distribution
        if properties['LogP']:
            fig.add_trace(go.Histogram(
                x=properties['LogP'],
                name='LogP',
                marker_color=self.colors['secondary']
            ), row=1, col=2)
        
        # TPSA distribution
        if properties['TPSA']:
            fig.add_trace(go.Histogram(
                x=properties['TPSA'],
                name='TPSA',
                marker_color=self.colors['success']
            ), row=1, col=3)
        
        # MW vs LogP correlation
        if properties['MW'] and properties['LogP']:
            fig.add_trace(go.Scatter(
                x=properties['MW'],
                y=properties['LogP'],
                mode='markers',
                name='MW vs LogP',
                marker=dict(color=self.colors['info'])
            ), row=2, col=1)
        
        # Filter pass rate
        pass_rate = pass_count / len(molecules) * 100 if molecules else 0
        fail_rate = 100 - pass_rate
        fig.add_trace(go.Bar(
            x=['Pass', 'Fail'],
            y=[pass_rate, fail_rate],
            marker_color=[self.colors['success'], self.colors['danger']],
            name='Filter Status'
        ), row=2, col=2)
        
        # QED distribution
        if properties['QED']:
            fig.add_trace(go.Histogram(
                x=properties['QED'],
                name='QED',
                marker_color=self.colors['warning']
            ), row=2, col=3)
        
        fig.update_layout(
            title="ADMET Properties Summary",
            height=700,
            width=1200,
            showlegend=False
        )
        
        return fig
    
    def plot_radar_comparison(self, molecules, max_molecules=5):
        """
        Compare multiple molecules using radar charts.
        
        Parameters:
        -----------
        molecules : list
            List of DrugMolecule objects
        max_molecules : int
            Maximum number of molecules to compare
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        molecules = molecules[:max_molecules]
        
        # Properties for radar chart
        props = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'QED']
        max_vals = {'MW': 500, 'LogP': 5, 'TPSA': 140, 'HBD': 5, 'HBA': 10, 'QED': 1}
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, mol in enumerate(molecules):
            if not mol.properties:
                mol.calculate_descriptors()
            
            # Normalize values
            values = []
            for prop in props:
                if prop in mol.properties:
                    normalized = min(mol.properties[prop] / max_vals[prop], 1.2)
                    values.append(normalized)
                else:
                    values.append(0)
            
            values.append(values[0])  # Close the radar
            
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=props + [props[0]],
                fill='toself',
                name=mol.name[:20],  # Truncate long names
                line_color=color,
                fillcolor=f'rgba({",".join(str(int(color[i:i+2], 16)) for i in (1, 3, 5))}, 0.3)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.2]
                )
            ),
            showlegend=True,
            title="Molecular Property Comparison",
            width=600,
            height=600
        )
        
        return fig