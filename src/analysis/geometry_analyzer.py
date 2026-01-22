"""
Geometry analysis tools for molecular structures.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform


class GeometryAnalyzer:
    """Analyze molecular geometries and structural parameters."""
    
    def __init__(self):
        """Initialize geometry analyzer."""
        pass
    
    def calculate_bond_distances(self, coords, elements, bond_threshold=2.0):
        """Calculate all bond distances in the molecule."""
        n_atoms = len(coords)
        bonds = []
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = np.linalg.norm(coords[i] - coords[j])
                
                if distance < bond_threshold:
                    bonds.append({
                        'atom1': i,
                        'atom2': j,
                        'element1': elements[i],
                        'element2': elements[j],
                        'distance': distance,
                        'bond_type': f"{elements[i]}-{elements[j]}"
                    })
        
        return bonds
    
    def calculate_bond_angles(self, coords, bonds):
        """Calculate bond angles."""
        angles = []
        
        # Group bonds by central atom
        atom_bonds = {}
        for bond in bonds:
            atom1, atom2 = bond['atom1'], bond['atom2']
            
            if atom1 not in atom_bonds:
                atom_bonds[atom1] = []
            if atom2 not in atom_bonds:
                atom_bonds[atom2] = []
            
            atom_bonds[atom1].append(atom2)
            atom_bonds[atom2].append(atom1)
        
        # Calculate angles
        for central_atom, connected_atoms in atom_bonds.items():
            if len(connected_atoms) >= 2:
                for i in range(len(connected_atoms)):
                    for j in range(i + 1, len(connected_atoms)):
                        atom1 = connected_atoms[i]
                        atom2 = connected_atoms[j]
                        
                        # Calculate angle
                        v1 = coords[atom1] - coords[central_atom]
                        v2 = coords[atom2] - coords[central_atom]
                        
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        
                        angles.append({
                            'atom1': atom1,
                            'central_atom': central_atom,
                            'atom2': atom2,
                            'angle': angle
                        })
        
        return angles
    
    def calculate_dihedral_angles(self, coords, bonds):
        """Calculate dihedral angles."""
        dihedrals = []
        
        # Find chains of 4 connected atoms
        for i, bond1 in enumerate(bonds):
            for j, bond2 in enumerate(bonds):
                if i >= j:
                    continue
                
                # Check if bonds share an atom
                shared_atoms = set([bond1['atom1'], bond1['atom2']]) & set([bond2['atom1'], bond2['atom2']])
                
                if len(shared_atoms) == 1:
                    shared_atom = list(shared_atoms)[0]
                    
                    # Find the other atoms
                    atoms1 = [bond1['atom1'], bond1['atom2']]
                    atoms2 = [bond2['atom1'], bond2['atom2']]
                    
                    atom1 = atoms1[0] if atoms1[1] == shared_atom else atoms1[1]
                    atom3 = atoms2[0] if atoms2[1] == shared_atom else atoms2[1]
                    
                    # Look for a fourth atom connected to atom3
                    for bond3 in bonds:
                        if atom3 in [bond3['atom1'], bond3['atom2']]:
                            atom4 = bond3['atom1'] if bond3['atom2'] == atom3 else bond3['atom2']
                            
                            if atom4 not in [atom1, shared_atom, atom3]:
                                # Calculate dihedral angle
                                dihedral = self._calculate_dihedral(
                                    coords[atom1], coords[shared_atom], 
                                    coords[atom3], coords[atom4]
                                )
                                
                                dihedrals.append({
                                    'atom1': atom1,
                                    'atom2': shared_atom,
                                    'atom3': atom3,
                                    'atom4': atom4,
                                    'dihedral': dihedral
                                })
        
        return dihedrals
    
    def _calculate_dihedral(self, r1, r2, r3, r4):
        """Calculate dihedral angle between four points."""
        # Vectors
        b1 = r2 - r1
        b2 = r3 - r2
        b3 = r4 - r3
        
        # Normal vectors
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Normalize
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        
        # Dihedral angle
        cos_dihedral = np.dot(n1, n2)
        cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
        
        dihedral = np.arccos(cos_dihedral) * 180 / np.pi
        
        # Check sign
        if np.dot(np.cross(n1, n2), b2) < 0:
            dihedral = -dihedral
        
        return dihedral
    
    def analyze_molecular_geometry(self, coords, elements):
        """Perform comprehensive geometry analysis."""
        bonds = self.calculate_bond_distances(coords, elements)
        angles = self.calculate_bond_angles(coords, bonds)
        dihedrals = self.calculate_dihedral_angles(coords, bonds)
        
        analysis = {
            'n_atoms': len(coords),
            'bonds': bonds,
            'angles': angles,
            'dihedrals': dihedrals,
            'bond_statistics': self._analyze_bond_statistics(bonds),
            'angle_statistics': self._analyze_angle_statistics(angles),
            'molecular_metrics': self._calculate_molecular_metrics(coords)
        }
        
        return analysis
    
    def _analyze_bond_statistics(self, bonds):
        """Analyze bond distance statistics."""
        if not bonds:
            return {}
        
        bond_types = {}
        for bond in bonds:
            bond_type = bond['bond_type']
            if bond_type not in bond_types:
                bond_types[bond_type] = []
            bond_types[bond_type].append(bond['distance'])
        
        statistics = {}
        for bond_type, distances in bond_types.items():
            statistics[bond_type] = {
                'count': len(distances),
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            }
        
        return statistics
    
    def _analyze_angle_statistics(self, angles):
        """Analyze bond angle statistics."""
        if not angles:
            return {}
        
        angle_values = [angle['angle'] for angle in angles]
        
        return {
            'count': len(angle_values),
            'mean': np.mean(angle_values),
            'std': np.std(angle_values),
            'min': np.min(angle_values),
            'max': np.max(angle_values)
        }
    
    def _calculate_molecular_metrics(self, coords):
        """Calculate overall molecular metrics."""
        center_of_mass = np.mean(coords, axis=0)
        
        # Radius of gyration
        distances_from_com = np.linalg.norm(coords - center_of_mass, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances_from_com**2))
        
        # Principal moments of inertia
        coords_centered = coords - center_of_mass
        inertia_tensor = np.zeros((3, 3))
        
        for i in range(len(coords)):
            r = coords_centered[i]
            inertia_tensor += np.outer(r, r)
        
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
        principal_moments = np.sort(eigenvalues)
        
        # Molecular dimensions
        max_distance = np.max(pdist(coords))
        
        return {
            'center_of_mass': center_of_mass.tolist(),
            'radius_of_gyration': radius_of_gyration,
            'principal_moments': principal_moments.tolist(),
            'max_distance': max_distance,
            'molecular_volume_estimate': (4/3) * np.pi * radius_of_gyration**3
        }
    
    def plot_bond_distribution(self, bonds, title="Bond Distance Distribution"):
        """Plot distribution of bond distances."""
        if not bonds:
            return None
        
        bond_types = {}
        for bond in bonds:
            bond_type = bond['bond_type']
            if bond_type not in bond_types:
                bond_types[bond_type] = []
            bond_types[bond_type].append(bond['distance'])
        
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for i, (bond_type, distances) in enumerate(bond_types.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Histogram(
                x=distances,
                name=bond_type,
                marker_color=color,
                opacity=0.7,
                nbinsx=10
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Bond Distance (Å)',
            yaxis_title='Count',
            barmode='overlay',
            width=700,
            height=400
        )
        
        return fig
    
    def plot_angle_distribution(self, angles, title="Bond Angle Distribution"):
        """Plot distribution of bond angles."""
        if not angles:
            return None
        
        angle_values = [angle['angle'] for angle in angles]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=angle_values,
            marker_color='steelblue',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Bond Angle (degrees)',
            yaxis_title='Count',
            width=600,
            height=400
        )
        
        return fig
    
    def create_geometry_report(self, coords, elements, output_file=None):
        """Create comprehensive geometry analysis report."""
        if output_file is None:
            from ..core.config import Config
            output_file = Config.OUTPUT_DIR / 'geometry_analysis.txt'
        
        analysis = self.analyze_molecular_geometry(coords, elements)
        
        with open(output_file, 'w') as f:
            f.write("MOLECULAR GEOMETRY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of atoms: {analysis['n_atoms']}\n")
            f.write(f"Number of bonds: {len(analysis['bonds'])}\n")
            f.write(f"Number of angles: {len(analysis['angles'])}\n")
            f.write(f"Number of dihedrals: {len(analysis['dihedrals'])}\n\n")
            
            # Molecular metrics
            metrics = analysis['molecular_metrics']
            f.write("MOLECULAR METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Center of mass: ({metrics['center_of_mass'][0]:.3f}, "
                   f"{metrics['center_of_mass'][1]:.3f}, "
                   f"{metrics['center_of_mass'][2]:.3f}) Å\n")
            f.write(f"Radius of gyration: {metrics['radius_of_gyration']:.3f} Å\n")
            f.write(f"Maximum distance: {metrics['max_distance']:.3f} Å\n")
            f.write(f"Volume estimate: {metrics['molecular_volume_estimate']:.3f} Ų\n\n")
            
            # Bond statistics
            bond_stats = analysis['bond_statistics']
            f.write("BOND STATISTICS\n")
            f.write("-" * 15 + "\n")
            for bond_type, stats in bond_stats.items():
                f.write(f"{bond_type:8s}: {stats['count']:3d} bonds, "
                       f"mean = {stats['mean']:.3f} ± {stats['std']:.3f} Å\n")
            f.write("\n")
            
            # Angle statistics
            angle_stats = analysis['angle_statistics']
            if angle_stats:
                f.write("ANGLE STATISTICS\n")
                f.write("-" * 16 + "\n")
                f.write(f"Total angles: {angle_stats['count']}\n")
                f.write(f"Mean angle: {angle_stats['mean']:.1f} ± {angle_stats['std']:.1f}°\n")
                f.write(f"Range: {angle_stats['min']:.1f}° - {angle_stats['max']:.1f}°\n")
        
        print(f"✓ Geometry analysis report saved to {output_file}")
        return output_file