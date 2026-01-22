"""
Visualizer for atom-centric DFT properties from DFTDataCollector output.
Provides multiple visualization types for per-atom charge and reactivity data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Dict, Union
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class AtomPropertyVisualizer:
    """
    Visualizer for atom-centric properties from DFT calculations.
    
    Supports:
    - Bar charts of per-atom charges
    - Heatmaps comparing multiple charge types
    - Reactivity site highlighting
    - Multi-molecule comparison
    - Interactive Plotly visualizations
    """
    
    # Color schemes for different property types
    CHARGE_CMAP = 'RdBu_r'  # Red (positive) - Blue (negative)
    FUKUI_CMAP = 'YlOrRd'   # Yellow to Red for reactivity
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with DataFrame from DFTDataCollector.
        
        Args:
            df: DataFrame with sequential per-atom columns (e.g., Mulliken_Charges_Seq)
        """
        self.df = df
        self._validate_columns()
    
    def _validate_columns(self):
        """Check that required sequential columns exist."""
        required = ['Atom_Index_Sequence', 'Element_Sequence']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _parse_sequence(self, seq_str: str, dtype=float) -> List:
        """Parse semicolon-separated sequence string to list."""
        if pd.isna(seq_str) or seq_str == '':
            return []
        return [dtype(x) for x in str(seq_str).split(';')]
    
    def get_molecule_atom_df(self, molecule_name: str) -> pd.DataFrame:
        """
        Get per-atom DataFrame for a specific molecule.
        
        Args:
            molecule_name: Name of the molecule
            
        Returns:
            DataFrame with one row per atom
        """
        row = self.df[self.df['Molecule_Name'] == molecule_name]
        if row.empty:
            raise ValueError(f"Molecule '{molecule_name}' not found")
        row = row.iloc[0]
        
        # Parse basic structure
        atom_df = pd.DataFrame({
            'Atom_Index': self._parse_sequence(row['Atom_Index_Sequence'], int),
            'Element': row['Element_Sequence'].split(';')
        })
        
        # Add charge columns
        charge_cols = {
            'Mulliken_Charges_Seq': 'Mulliken',
            'Lowdin_Charges_Seq': 'Lowdin',
            'Hirshfeld_Charges_Seq': 'Hirshfeld'
        }
        for seq_col, name in charge_cols.items():
            if seq_col in row.index and pd.notna(row[seq_col]):
                atom_df[name] = self._parse_sequence(row[seq_col])
        
        # Add Fukui columns
        fukui_cols = {
            'Fukui_Plus_Seq': 'Fukui_Plus',
            'Fukui_Minus_Seq': 'Fukui_Minus',
            'Fukui_Radical_Seq': 'Fukui_Radical'
        }
        for seq_col, name in fukui_cols.items():
            if seq_col in row.index and pd.notna(row[seq_col]):
                atom_df[name] = self._parse_sequence(row[seq_col])
        
        # Add local reactivity
        reactivity_cols = {
            'Local_Electrophilicity_Seq': 'Local_Electrophilicity',
            'Local_Nucleophilicity_Seq': 'Local_Nucleophilicity',
            'Spin_Density_Seq': 'Spin_Density',
            'ESP_At_Nuclei_Seq': 'ESP_At_Nuclei'
        }
        for seq_col, name in reactivity_cols.items():
            if seq_col in row.index and pd.notna(row[seq_col]):
                atom_df[name] = self._parse_sequence(row[seq_col])
        
        # Create atom label
        atom_df['Atom_Label'] = atom_df['Element'] + atom_df['Atom_Index'].astype(str)
        
        return atom_df
    
    def plot_charges_bar(self, molecule_name: str, charge_type: str = 'Mulliken',
                         figsize: tuple = (12, 5), highlight_extreme: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create bar chart of per-atom charges.
        
        Args:
            molecule_name: Name of the molecule
            charge_type: 'Mulliken', 'Lowdin', or 'Hirshfeld'
            figsize: Figure size
            highlight_extreme: Highlight most positive/negative atoms
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        atom_df = self.get_molecule_atom_df(molecule_name)
        
        if charge_type not in atom_df.columns:
            raise ValueError(f"Charge type '{charge_type}' not available")
        
        charges = atom_df[charge_type].values
        labels = atom_df['Atom_Label'].values
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color by charge sign
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in charges]
        
        # Highlight extremes
        if highlight_extreme:
            max_idx = np.argmax(charges)
            min_idx = np.argmin(charges)
            colors[max_idx] = '#c0392b'  # Dark red
            colors[min_idx] = '#2980b9'  # Dark blue
        
        bars = ax.bar(range(len(charges)), charges, color=colors, edgecolor='white', linewidth=0.5)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Atom', fontsize=11)
        ax.set_ylabel(f'{charge_type} Charge (e)', fontsize=11)
        ax.set_title(f'{charge_type} Charges for {molecule_name}', fontsize=13, fontweight='bold')
        
        # Add grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', label='Positive'),
                          Patch(facecolor='#3498db', label='Negative')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_charges_comparison(self, molecule_name: str, figsize: tuple = (14, 5),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare Mulliken, Löwdin, and Hirshfeld charges side by side.
        
        Args:
            molecule_name: Name of the molecule
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        atom_df = self.get_molecule_atom_df(molecule_name)
        labels = atom_df['Atom_Label'].values
        
        charge_types = ['Mulliken', 'Lowdin', 'Hirshfeld']
        available = [ct for ct in charge_types if ct in atom_df.columns]
        
        fig, axes = plt.subplots(1, len(available), figsize=figsize, sharey=True)
        if len(available) == 1:
            axes = [axes]
        
        colors_map = {'Mulliken': '#e74c3c', 'Lowdin': '#3498db', 'Hirshfeld': '#2ecc71'}
        
        for ax, charge_type in zip(axes, available):
            charges = atom_df[charge_type].values
            colors = [colors_map[charge_type] if c > 0 else 
                     plt.cm.Blues(0.6) if charge_type == 'Lowdin' else
                     plt.cm.Reds(0.6) if charge_type == 'Mulliken' else
                     plt.cm.Greens(0.6) for c in charges]
            
            ax.barh(range(len(charges)), charges, color=colors_map[charge_type], alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Charge (e)')
            ax.set_title(f'{charge_type}', fontweight='bold')
            ax.xaxis.grid(True, alpha=0.3)
        
        fig.suptitle(f'Atomic Charges Comparison: {molecule_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_fukui_heatmap(self, molecule_name: str, figsize: tuple = (10, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap of Fukui functions for reactive site analysis.
        
        Args:
            molecule_name: Name of the molecule
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        atom_df = self.get_molecule_atom_df(molecule_name)
        
        fukui_cols = ['Fukui_Plus', 'Fukui_Minus', 'Fukui_Radical']
        available = [c for c in fukui_cols if c in atom_df.columns]
        
        if not available:
            raise ValueError("No Fukui function data available")
        
        # Create matrix
        data = atom_df[available].values.T
        labels = atom_df['Atom_Label'].values
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(available)))
        ax.set_yticklabels(['f⁺ (Electrophilic)', 'f⁻ (Nucleophilic)', 'f⁰ (Radical)'][:len(available)])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.5)
        cbar.set_label('Fukui Function Value', fontsize=10)
        
        # Annotate values
        for i in range(len(available)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                             color='white' if data[i, j] > data.max()/2 else 'black', fontsize=7)
        
        ax.set_title(f'Fukui Functions (Reactive Sites): {molecule_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Atom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_reactivity_radar(self, molecule_name: str, top_n: int = 6,
                              figsize: tuple = (10, 8), save_path: Optional[str] = None) -> plt.Figure:
        """
        Radar chart of top reactive atoms.
        
        Args:
            molecule_name: Name of the molecule
            top_n: Number of most reactive atoms to display
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        atom_df = self.get_molecule_atom_df(molecule_name)
        
        # Calculate total reactivity score
        cols = ['Fukui_Plus', 'Fukui_Minus', 'Fukui_Radical']
        available = [c for c in cols if c in atom_df.columns]
        
        if not available:
            raise ValueError("No Fukui data available for radar chart")
        
        atom_df['Total_Reactivity'] = atom_df[available].sum(axis=1)
        top_atoms = atom_df.nlargest(top_n, 'Total_Reactivity')
        
        # Radar chart
        categories = available
        N = len(categories)
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        colors = plt.cm.Set2(np.linspace(0, 1, top_n))
        
        for i, (_, row) in enumerate(top_atoms.iterrows()):
            values = [row[c] for c in categories]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Atom_Label'], color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['f⁺ (Electrophilic)', 'f⁻ (Nucleophilic)', 'f⁰ (Radical)'][:N])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title(f'Top {top_n} Reactive Atoms: {molecule_name}', fontsize=13, fontweight='bold', y=1.08)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_charges(self, molecule_name: str, charge_type: str = 'Mulliken'):
        """
        Create interactive Plotly bar chart of charges.
        
        Args:
            molecule_name: Name of the molecule
            charge_type: 'Mulliken', 'Lowdin', or 'Hirshfeld'
            
        Returns:
            plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for interactive visualization. Install with: pip install plotly")
        
        atom_df = self.get_molecule_atom_df(molecule_name)
        
        if charge_type not in atom_df.columns:
            raise ValueError(f"Charge type '{charge_type}' not available")
        
        # Create colors
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in atom_df[charge_type]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=atom_df['Atom_Label'],
                y=atom_df[charge_type],
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>' +
                             f'{charge_type}: %{{y:.4f}} e<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'{charge_type} Charges: {molecule_name}',
            xaxis_title='Atom',
            yaxis_title=f'{charge_type} Charge (e)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def plot_interactive_all_properties(self, molecule_name: str):
        """
        Create comprehensive interactive dashboard of all atom properties.
        
        Args:
            molecule_name: Name of the molecule
            
        Returns:
            plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required. Install with: pip install plotly")
        
        atom_df = self.get_molecule_atom_df(molecule_name)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mulliken Charges', 'Löwdin Charges', 
                          'Hirshfeld Charges', 'Fukui Functions'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        labels = atom_df['Atom_Label']
        
        # Mulliken
        if 'Mulliken' in atom_df.columns:
            colors = ['#e74c3c' if c > 0 else '#3498db' for c in atom_df['Mulliken']]
            fig.add_trace(go.Bar(x=labels, y=atom_df['Mulliken'], marker_color=colors, 
                                name='Mulliken'), row=1, col=1)
        
        # Lowdin
        if 'Lowdin' in atom_df.columns:
            colors = ['#e74c3c' if c > 0 else '#3498db' for c in atom_df['Lowdin']]
            fig.add_trace(go.Bar(x=labels, y=atom_df['Lowdin'], marker_color=colors,
                                name='Löwdin'), row=1, col=2)
        
        # Hirshfeld
        if 'Hirshfeld' in atom_df.columns:
            colors = ['#e74c3c' if c > 0 else '#3498db' for c in atom_df['Hirshfeld']]
            fig.add_trace(go.Bar(x=labels, y=atom_df['Hirshfeld'], marker_color=colors,
                                name='Hirshfeld'), row=2, col=1)
        
        # Fukui
        fukui_cols = ['Fukui_Plus', 'Fukui_Minus', 'Fukui_Radical']
        for col, color in zip(fukui_cols, ['#e74c3c', '#3498db', '#2ecc71']):
            if col in atom_df.columns:
                fig.add_trace(go.Scatter(x=labels, y=atom_df[col], mode='lines+markers',
                                        name=col.replace('_', ' '), line=dict(color=color)), 
                             row=2, col=2)
        
        fig.update_layout(
            title=f'Atom Properties Dashboard: {molecule_name}',
            height=700,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def compare_molecules(self, molecule_names: List[str], property_name: str = 'Mulliken',
                         figsize: tuple = (14, 6), save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare atom charges across multiple molecules.
        
        Args:
            molecule_names: List of molecule names to compare
            property_name: Property to compare
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, len(molecule_names), figsize=figsize, sharey=True)
        if len(molecule_names) == 1:
            axes = [axes]
        
        all_values = []
        
        for ax, mol_name in zip(axes, molecule_names):
            try:
                atom_df = self.get_molecule_atom_df(mol_name)
                if property_name in atom_df.columns:
                    values = atom_df[property_name].values
                    all_values.extend(values)
                    labels = atom_df['Atom_Label'].values
                    
                    colors = ['#e74c3c' if c > 0 else '#3498db' for c in values]
                    ax.barh(range(len(values)), values, color=colors)
                    ax.set_yticks(range(len(labels)))
                    ax.set_yticklabels(labels, fontsize=7)
                    ax.axvline(x=0, color='black', linewidth=0.5)
                    ax.set_title(mol_name.replace('_optimized', ''), fontsize=10, fontweight='bold')
                    ax.set_xlabel(f'{property_name} Charge')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center')
        
        # Set consistent x-limits
        if all_values:
            max_abs = max(abs(min(all_values)), abs(max(all_values))) * 1.1
            for ax in axes:
                ax.set_xlim(-max_abs, max_abs)
        
        fig.suptitle(f'{property_name} Charge Comparison', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def demo_visualization():
    """Demo function to show visualization capabilities."""
    # This would be called from the notebook
    print("AtomPropertyVisualizer Demo")
    print("=" * 50)
    print("""
Usage in Jupyter notebook:

    from src.analysis.dft_data_collector import DFTDataCollector
    from src.analysis.atom_property_visualizer import AtomPropertyVisualizer
    
    # Load data
    collector = DFTDataCollector('./optimized_molecules')
    df = collector.collect_data()
    
    # Create visualizer
    viz = AtomPropertyVisualizer(df)
    
    # Get atom DataFrame for one molecule
    atom_df = viz.get_molecule_atom_df('Aspirin_optimized')
    display(atom_df)
    
    # Static matplotlib visualizations
    viz.plot_charges_bar('Aspirin_optimized', charge_type='Mulliken')
    viz.plot_charges_comparison('Aspirin_optimized')
    viz.plot_fukui_heatmap('Aspirin_optimized')
    viz.plot_reactivity_radar('Aspirin_optimized')
    
    # Interactive Plotly visualizations
    fig = viz.plot_interactive_charges('Aspirin_optimized')
    fig.show()
    
    fig = viz.plot_interactive_all_properties('Aspirin_optimized')
    fig.show()
    
    # Compare multiple molecules
    viz.compare_molecules(['Aspirin_optimized', 'Ibuprofen_optimized'], 'Mulliken')
    """)


if __name__ == "__main__":
    demo_visualization()
