"""
3D molecular visualization using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import AllChem


class MolecularViewer3D:
    """Interactive 3D molecular visualization."""
    
    # Atomic colors (CPK coloring)
    ATOMIC_COLORS = {
        'H': '#FFFFFF',   # White
        'C': '#909090',   # Gray  
        'N': '#3050F8',   # Blue
        'O': '#FF0D0D',   # Red
        'F': '#90E050',   # Green
        'P': '#FF8000',   # Orange
        'S': '#FFFF30',   # Yellow
        'Cl': '#1FF01F',  # Green
        'Br': '#A62929',  # Brown
        'I': '#940094',   # Purple
        'Fe': '#E06633',  # Iron
        'Ca': '#3DFF00',  # Calcium
        'Mg': '#8AFF00',  # Magnesium
        'Na': '#AB5CF2',  # Sodium
        'K': '#8F40D4'    # Potassium
    }
    
    # Atomic radii (van der Waals)
    ATOMIC_RADII = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
        'Fe': 2.00, 'Ca': 2.00, 'Mg': 1.73, 'Na': 2.27, 'K': 2.75
    }
    
    def __init__(self):
        """Initialize 3D molecular viewer."""
        pass
    
    def plot_molecule(self, mol, coords=None, title="Molecule", style='ball_and_stick', 
                     show_bonds=True, show_labels=True, bond_threshold=1.8):
        """
        Create 3D plot of molecule.
        
        Parameters:
        -----------
        mol : rdkit.Chem.Mol or DrugMolecule
            Molecule to visualize
        coords : np.ndarray, optional
            Custom coordinates (N x 3)
        title : str
            Plot title
        style : str
            Visualization style ('ball_and_stick', 'spacefilling', 'wireframe')
        show_bonds : bool
            Show bonds
        show_labels : bool
            Show atom labels
        bond_threshold : float
            Distance threshold for bond detection
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        # Handle different input types
        if hasattr(mol, 'mol'):  # DrugMolecule object
            rdkit_mol = mol.mol
            elements = mol.elements
            if coords is None:
                coords = mol.initial_coords
        else:  # RDKit molecule
            rdkit_mol = mol
            elements = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
            if coords is None:
                # Generate 3D coordinates
                mol_h = Chem.AddHs(rdkit_mol)
                AllChem.EmbedMolecule(mol_h, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_h)
                conf = mol_h.GetConformer()
                coords = conf.GetPositions()
                elements = [atom.GetSymbol() for atom in mol_h.GetAtoms()]
        
        if coords is None:
            raise ValueError("No coordinates available for molecule")
        
        fig = go.Figure()
        
        # Plot atoms
        self._plot_atoms(fig, elements, coords, style, show_labels)
        
        # Plot bonds
        if show_bonds:
            self._plot_bonds(fig, elements, coords, bond_threshold, style)
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)', 
                zaxis_title='Z (Å)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        
        return fig
    
    def _plot_atoms(self, fig, elements, coords, style, show_labels):
        """Plot atoms in the molecule."""
        unique_elements = list(set(elements))
        
        for element in unique_elements:
            # Find atoms of this element
            indices = [i for i, el in enumerate(elements) if el == element]
            if not indices:
                continue
            
            element_coords = coords[indices]
            
            # Get color and size
            color = self.ATOMIC_COLORS.get(element, '#FF69B4')  # Default pink
            
            if style == 'spacefilling':
                size = self.ATOMIC_RADII.get(element, 1.5) * 8
                opacity = 0.8
            elif style == 'ball_and_stick':
                size = self.ATOMIC_RADII.get(element, 1.5) * 4
                opacity = 0.9
            else:  # wireframe
                size = 3
                opacity = 1.0
            
            # Create text labels
            if show_labels:
                text = [f"{element}{i+1}" for i in indices]
                textposition = 'middle center'
            else:
                text = None
                textposition = None
            
            # Plot atoms
            fig.add_trace(go.Scatter3d(
                x=element_coords[:, 0],
                y=element_coords[:, 1],
                z=element_coords[:, 2],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    color=color,
                    size=size,
                    opacity=opacity,
                    line=dict(width=1, color='black')
                ),
                text=text,
                textposition=textposition,
                textfont=dict(size=8, color='black'),
                name=element,
                hovertemplate=f'<b>{element}</b><br>' +
                            'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            ))
    
    def _plot_bonds(self, fig, elements, coords, bond_threshold, style):
        """Plot bonds between atoms."""
        bonds = self._detect_bonds(coords, elements, bond_threshold)
        
        if not bonds:
            return
        
        # Create bond traces
        x_bonds, y_bonds, z_bonds = [], [], []
        
        for i, j in bonds:
            x_bonds.extend([coords[i, 0], coords[j, 0], None])
            y_bonds.extend([coords[i, 1], coords[j, 1], None])
            z_bonds.extend([coords[i, 2], coords[j, 2], None])
        
        # Bond appearance
        if style == 'wireframe':
            line_width = 2
            line_color = 'gray'
        else:
            line_width = 4
            line_color = 'gray'
        
        fig.add_trace(go.Scatter3d(
            x=x_bonds,
            y=y_bonds,
            z=z_bonds,
            mode='lines',
            line=dict(
                color=line_color,
                width=line_width
            ),
            hoverinfo='skip',
            showlegend=False
        ))
    
    def _detect_bonds(self, coords, elements, threshold):
        """Detect bonds based on distance."""
        bonds = []
        n_atoms = len(coords)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = np.linalg.norm(coords[i] - coords[j])
                
                # Covalent radii-based threshold
                el_i, el_j = elements[i], elements[j]
                max_distance = self._get_bond_threshold(el_i, el_j, threshold)
                
                if distance < max_distance:
                    bonds.append((i, j))
        
        return bonds
    
    def _get_bond_threshold(self, el1, el2, default_threshold):
        """Get bond distance threshold for two elements."""
        # Simplified covalent radii
        cov_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39
        }
        
        r1 = cov_radii.get(el1, 0.8)
        r2 = cov_radii.get(el2, 0.8)
        
        return (r1 + r2) * 1.3  # 30% tolerance
    
    def plot_multiple_molecules(self, molecules, coords_list=None, titles=None, 
                               rows=1, cols=None):
        """
        Plot multiple molecules in subplots.
        
        Parameters:
        -----------
        molecules : list
            List of molecules
        coords_list : list, optional
            List of coordinate arrays
        titles : list, optional
            List of titles
        rows : int
            Number of rows
        cols : int, optional
            Number of columns (auto-calculated if None)
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        n_molecules = len(molecules)
        
        if cols is None:
            cols = min(3, n_molecules)
        if rows is None:
            rows = (n_molecules + cols - 1) // cols
        
        if titles is None:
            titles = [f"Molecule {i+1}" for i in range(n_molecules)]
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=titles[:n_molecules],
            horizontal_spacing=0.02,
            vertical_spacing=0.05
        )
        
        for idx, mol in enumerate(molecules):
            row = idx // cols + 1
            col = idx % cols + 1
            
            # Get coordinates
            coords = coords_list[idx] if coords_list else None
            
            # Create single molecule plot
            mol_fig = self.plot_molecule(mol, coords, show_labels=False)
            
            # Add traces to subplot
            for trace in mol_fig.data:
                fig.add_trace(trace, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            height=400 * rows,
            width=400 * cols,
            title="Molecular Comparison",
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def animate_trajectory(self, coords_trajectory, elements, title="Molecular Dynamics"):
        """
        Create animated trajectory visualization.
        
        Parameters:
        -----------
        coords_trajectory : list
            List of coordinate arrays for each frame
        elements : list
            Element symbols
        title : str
            Animation title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        n_frames = len(coords_trajectory)
        
        # Create base figure with first frame
        fig = self.plot_molecule(None, coords_trajectory[0], title, show_labels=False)
        fig.data = []  # Clear existing traces
        
        # Create frames
        frames = []
        for frame_idx, coords in enumerate(coords_trajectory):
            frame_data = []
            
            # Plot atoms for this frame
            unique_elements = list(set(elements))
            for element in unique_elements:
                indices = [i for i, el in enumerate(elements) if el == element]
                if not indices:
                    continue
                
                element_coords = coords[indices]
                color = self.ATOMIC_COLORS.get(element, '#FF69B4')
                size = self.ATOMIC_RADII.get(element, 1.5) * 4
                
                frame_data.append(go.Scatter3d(
                    x=element_coords[:, 0],
                    y=element_coords[:, 1],
                    z=element_coords[:, 2],
                    mode='markers',
                    marker=dict(color=color, size=size, opacity=0.9),
                    name=element
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=f"frame_{frame_idx}"
            ))
        
        # Add first frame data
        if frames:
            for trace in frames[0].data:
                fig.add_trace(trace)
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title=title,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 200, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 100}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f"frame_{k}"], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': f"Step {k}",
                        'method': 'animate'
                    } for k in range(n_frames)
                ],
                'active': 0,
                'transition': {'duration': 0},
                'x': 0,
                'len': 1
            }]
        )
        
        return fig