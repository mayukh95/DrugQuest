"""
Trajectory visualization and analysis plots.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TrajectoryPlotter:
    """Create plots for optimization trajectories and energy landscapes."""
    
    def __init__(self):
        """Initialize trajectory plotter."""
        self.colors = {
            'energy': '#667eea',
            'force': '#e74c3c',
            'gradient': '#f39c12',
            'converged': '#27ae60',
            'failed': '#e74c3c'
        }
    
    def plot_optimization_history(self, history, title="Optimization History"):
        """
        Plot optimization convergence history.
        
        Parameters:
        -----------
        history : dict
            Optimization history with 'energies', 'force_norms', etc.
        title : str
            Plot title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if not history or 'energies' not in history:
            raise ValueError("No optimization history data provided")
        
        energies = history['energies']
        force_norms = history.get('force_norms', [])
        converged_scf = history.get('converged_scf', [])
        
        steps = list(range(len(energies)))
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Energy Convergence', 'Force Convergence', 'SCF Status'],
            vertical_spacing=0.1,
            specs=[[{}], [{}], [{}]]
        )
        
        # Energy plot
        fig.add_trace(go.Scatter(
            x=steps,
            y=energies,
            mode='lines+markers',
            name='Energy',
            line=dict(color=self.colors['energy'], width=2),
            marker=dict(size=4),
            hovertemplate='<b>Step:</b> %{x}<br><b>Energy:</b> %{y:.6f} Ha<extra></extra>'
        ), row=1, col=1)
        
        # Force plot
        if force_norms:
            fig.add_trace(go.Scatter(
                x=steps,
                y=force_norms,
                mode='lines+markers',
                name='Force Norm',
                line=dict(color=self.colors['force'], width=2),
                marker=dict(size=4),
                yaxis='y2',
                hovertemplate='<b>Step:</b> %{x}<br><b>Force:</b> %{y:.6f} Ha/Bohr<extra></extra>'
            ), row=2, col=1)
            
            # Add convergence threshold line
            fig.add_hline(y=0.0003, line_dash="dash", line_color="gray", 
                         row=2, col=1, annotation_text="Convergence Threshold")
        
        # SCF convergence status
        if converged_scf:
            scf_status = [1 if c else 0 for c in converged_scf]
            fig.add_trace(go.Scatter(
                x=steps,
                y=scf_status,
                mode='markers',
                name='SCF Converged',
                marker=dict(
                    color=[self.colors['converged'] if s else self.colors['failed'] for s in scf_status],
                    size=8,
                    symbol=['circle' if s else 'x' for s in scf_status]
                ),
                hovertemplate='<b>Step:</b> %{x}<br><b>SCF:</b> %{customdata}<extra></extra>',
                customdata=['Converged' if s else 'Failed' for s in scf_status]
            ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            width=800,
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Optimization Step", row=3, col=1)
        fig.update_yaxes(title_text="Energy (Ha)", row=1, col=1)
        fig.update_yaxes(title_text="Force Norm (Ha/Bohr)", row=2, col=1)
        fig.update_yaxes(title_text="SCF Status", row=3, col=1, tickvals=[0, 1], ticktext=['Failed', 'Converged'])
        
        return fig
    
    def plot_energy_surface_2d(self, coords_history, energies, title="Energy Surface"):
        """
        Plot 2D projection of energy surface from trajectory.
        
        Parameters:
        -----------
        coords_history : list
            List of coordinate arrays
        energies : list  
            List of energies
        title : str
            Plot title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if not coords_history or not energies:
            raise ValueError("No trajectory data provided")
        
        # Project coordinates to 2D using PCA
        coords_flat = [coords.flatten() for coords in coords_history]
        coords_array = np.array(coords_flat)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coords_array)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by energy
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers+lines',
            marker=dict(
                color=energies,
                colorscale='viridis',
                colorbar=dict(title='Energy (Ha)'),
                size=8,
                line=dict(width=1, color='white')
            ),
            line=dict(color='gray', width=1),
            name='Optimization Path',
            hovertemplate='<b>Step:</b> %{pointNumber}<br>' +
                         '<b>Energy:</b> %{marker.color:.6f} Ha<br>' +
                         '<b>PC1:</b> %{x:.3f}<br>' +
                         '<b>PC2:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Mark start and end
        fig.add_trace(go.Scatter(
            x=[coords_2d[0, 0]],
            y=[coords_2d[0, 1]],
            mode='markers',
            marker=dict(color='green', size=15, symbol='star'),
            name='Start',
            hovertemplate='<b>Start</b><br>Energy: %{customdata:.6f} Ha<extra></extra>',
            customdata=[energies[0]]
        ))
        
        fig.add_trace(go.Scatter(
            x=[coords_2d[-1, 0]],
            y=[coords_2d[-1, 1]],
            mode='markers',
            marker=dict(color='red', size=15, symbol='diamond'),
            name='End',
            hovertemplate='<b>End</b><br>Energy: %{customdata:.6f} Ha<extra></extra>',
            customdata=[energies[-1]]
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            width=700,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_rmsd_trajectory(self, coords_history, reference_coords=None, title="RMSD Trajectory"):
        """
        Plot RMSD from reference structure over trajectory.
        
        Parameters:
        -----------
        coords_history : list
            List of coordinate arrays
        reference_coords : np.ndarray, optional
            Reference coordinates (if None, uses initial structure)
        title : str
            Plot title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if not coords_history:
            raise ValueError("No trajectory data provided")
        
        if reference_coords is None:
            reference_coords = coords_history[0]
        
        # Calculate RMSD for each frame
        rmsds = []
        for coords in coords_history:
            # Align coordinates (simple translation to center of mass)
            coords_centered = coords - np.mean(coords, axis=0)
            ref_centered = reference_coords - np.mean(reference_coords, axis=0)
            
            # Calculate RMSD
            rmsd = np.sqrt(np.mean(np.sum((coords_centered - ref_centered)**2, axis=1)))
            rmsds.append(rmsd)
        
        steps = list(range(len(rmsds)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=rmsds,
            mode='lines+markers',
            name='RMSD',
            line=dict(color=self.colors['gradient'], width=2),
            marker=dict(size=4),
            hovertemplate='<b>Step:</b> %{x}<br><b>RMSD:</b> %{y:.4f} Å<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Optimization Step',
            yaxis_title='RMSD (Å)',
            width=600,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_parallel_optimization_summary(self, results, title="Parallel Optimization Summary"):
        """
        Plot summary of parallel optimization results.
        
        Parameters:
        -----------
        results : list
            List of optimization result dictionaries
        title : str
            Plot title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if not results:
            raise ValueError("No optimization results provided")
        
        # Separate successful and failed optimizations
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Energy Changes', 'Optimization Steps',
                'Success Rate', 'Energy Distribution'
            ],
            specs=[[{}, {}], [{'type': 'pie'}, {}]]
        )
        
        if successful:
            # Energy changes
            energy_changes = [r.get('energy_change', 0) for r in successful]
            molecule_names = [r.get('molecule', '')[:15] for r in successful]
            
            fig.add_trace(go.Bar(
                x=molecule_names,
                y=energy_changes,
                name='Energy Change',
                marker_color=self.colors['energy'],
                hovertemplate='<b>%{x}</b><br>ΔE: %{y:.6f} Ha<extra></extra>'
            ), row=1, col=1)
            
            # Optimization steps
            steps = [r.get('steps', 0) for r in successful]
            
            fig.add_trace(go.Bar(
                x=molecule_names,
                y=steps,
                name='Steps',
                marker_color=self.colors['force'],
                hovertemplate='<b>%{x}</b><br>Steps: %{y}<extra></extra>'
            ), row=1, col=2)
            
            # Final energies distribution
            final_energies = [r.get('final_energy', 0) for r in successful]
            
            fig.add_trace(go.Histogram(
                x=final_energies,
                name='Final Energies',
                marker_color=self.colors['converged'],
                opacity=0.7
            ), row=2, col=2)
        
        # Success rate pie chart
        success_count = len(successful)
        failed_count = len(failed)
        
        fig.add_trace(go.Pie(
            labels=['Success', 'Failed'],
            values=[success_count, failed_count],
            marker_colors=[self.colors['converged'], self.colors['failed']],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ), row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=700,
            width=1000,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Molecules", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="Molecules", row=1, col=2, tickangle=45)
        fig.update_xaxes(title_text="Final Energy (Ha)", row=2, col=2)
        fig.update_yaxes(title_text="Energy Change (Ha)", row=1, col=1)
        fig.update_yaxes(title_text="Steps", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig
    
    def animate_optimization_path(self, coords_history, elements, energies=None, 
                                 title="Optimization Animation"):
        """
        Create animated visualization of optimization trajectory.
        
        Parameters:
        -----------
        coords_history : list
            List of coordinate arrays
        elements : list
            Element symbols
        energies : list, optional
            Energies for each frame
        title : str
            Animation title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        from .molecular_viewer import MolecularViewer3D
        
        viewer = MolecularViewer3D()
        
        # Use the molecular viewer's animation capability
        fig = viewer.animate_trajectory(coords_history, elements, title)
        
        # Add energy information if available
        if energies:
            # Update frame titles to include energy
            for i, frame in enumerate(fig.frames):
                frame.name = f"Step {i}: E = {energies[i]:.6f} Ha"
        
        return fig