"""
Molecular orbital analysis and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MolecularOrbitalAnalyzer:
    """Analyze and visualize molecular orbitals."""
    
    def __init__(self, wavefunction_data):
        """
        Initialize MO analyzer.
        
        Parameters:
        -----------
        wavefunction_data : dict
            Wavefunction data from DFT calculation
        """
        self.mo_coeff = wavefunction_data['mo_coeff']
        self.mo_energy = wavefunction_data['mo_energy']
        self.mo_occ = wavefunction_data['mo_occ']
        self.energy = wavefunction_data['energy']
        self.converged = wavefunction_data['converged']
        
        # Find HOMO and LUMO
        self._find_frontier_orbitals()
    
    def _find_frontier_orbitals(self):
        """Find HOMO and LUMO indices."""
        occ_indices = np.where(self.mo_occ > 0)[0]
        
        if len(occ_indices) > 0:
            self.homo_idx = occ_indices[-1]
            self.homo_energy = self.mo_energy[self.homo_idx]
        else:
            self.homo_idx = None
            self.homo_energy = None
        
        if self.homo_idx is not None and self.homo_idx + 1 < len(self.mo_energy):
            self.lumo_idx = self.homo_idx + 1
            self.lumo_energy = self.mo_energy[self.lumo_idx]
        else:
            self.lumo_idx = None
            self.lumo_energy = None
        
        # Calculate HOMO-LUMO gap
        if self.homo_energy is not None and self.lumo_energy is not None:
            self.homo_lumo_gap = (self.lumo_energy - self.homo_energy) * 27.2114  # eV
        else:
            self.homo_lumo_gap = None
    
    def plot_mo_energy_diagram(self, num_mos=10, show_occupied=True, show_virtual=True):
        """
        Plot molecular orbital energy diagram.
        
        Parameters:
        -----------
        num_mos : int
            Number of MOs around HOMO-LUMO to show
        show_occupied : bool
            Show occupied orbitals
        show_virtual : bool
            Show virtual orbitals
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        fig = go.Figure()
        
        # Determine MO range
        if self.homo_idx is not None:
            start_mo = max(0, self.homo_idx - num_mos//2)
            end_mo = min(len(self.mo_energy), self.homo_idx + num_mos//2 + 1)
        else:
            start_mo = 0
            end_mo = min(len(self.mo_energy), num_mos)
        
        # Prepare data
        mo_indices = list(range(start_mo, end_mo))
        energies_eV = [self.mo_energy[i] * 27.2114 for i in mo_indices]
        occupancies = [self.mo_occ[i] for i in mo_indices]
        
        # Create MO labels
        labels = []
        for i in mo_indices:
            if i == self.homo_idx:
                labels.append("HOMO")
            elif i == self.lumo_idx:
                labels.append("LUMO")
            elif self.homo_idx is not None and i < self.homo_idx:
                labels.append(f"HOMO-{self.homo_idx - i}")
            elif self.lumo_idx is not None and i > self.lumo_idx:
                labels.append(f"LUMO+{i - self.lumo_idx}")
            else:
                labels.append(f"MO {i+1}")
        
        # Plot occupied orbitals
        if show_occupied:
            occ_indices = [i for i, occ in enumerate(occupancies) if occ > 0]
            occ_energies = [energies_eV[i] for i in occ_indices]
            occ_labels = [labels[i] for i in occ_indices]
            
            fig.add_trace(go.Scatter(
                x=[0] * len(occ_energies),
                y=occ_energies,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='blue',
                    symbol='square',
                    line=dict(width=2, color='darkblue')
                ),
                text=occ_labels,
                textposition='middle right',
                name='Occupied',
                hovertemplate='<b>%{text}</b><br>Energy: %{y:.3f} eV<extra></extra>'
            ))
        
        # Plot virtual orbitals  
        if show_virtual:
            virt_indices = [i for i, occ in enumerate(occupancies) if occ == 0]
            virt_energies = [energies_eV[i] for i in virt_indices]
            virt_labels = [labels[i] for i in virt_indices]
            
            fig.add_trace(go.Scatter(
                x=[0] * len(virt_energies),
                y=virt_energies,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='square',
                    line=dict(width=2, color='darkred')
                ),
                text=virt_labels,
                textposition='middle right',
                name='Virtual',
                hovertemplate='<b>%{text}</b><br>Energy: %{y:.3f} eV<extra></extra>'
            ))
        
        # Add HOMO-LUMO gap annotation
        if self.homo_lumo_gap is not None:
            fig.add_annotation(
                x=0.5,
                y=(self.homo_energy + self.lumo_energy) / 2 * 27.2114,
                text=f"Gap: {self.homo_lumo_gap:.2f} eV",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                arrowwidth=2,
                font=dict(color="green", size=12)
            )
        
        fig.update_layout(
            title="Molecular Orbital Energy Diagram",
            xaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-1, 2]
            ),
            yaxis=dict(
                title="Energy (eV)",
                showgrid=True,
                zeroline=True
            ),
            showlegend=True,
            width=600,
            height=800,
            margin=dict(l=50, r=200, t=50, b=50)
        )
        
        return fig
    
    def plot_mo_coefficients(self, mo_idx, num_aos=20):
        """
        Plot molecular orbital coefficients.
        
        Parameters:
        -----------
        mo_idx : int
            Molecular orbital index
        num_aos : int
            Number of top AO contributions to show
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if mo_idx >= len(self.mo_energy):
            raise ValueError(f"MO index {mo_idx} out of range")
        
        coeffs = self.mo_coeff[:, mo_idx]
        
        # Find top contributions
        abs_coeffs = np.abs(coeffs)
        top_indices = np.argsort(abs_coeffs)[-num_aos:][::-1]
        
        top_coeffs = coeffs[top_indices]
        top_abs_coeffs = abs_coeffs[top_indices]
        
        # Create AO labels (simplified)
        ao_labels = [f"AO {i+1}" for i in top_indices]
        
        # Determine MO label
        if mo_idx == self.homo_idx:
            mo_label = "HOMO"
        elif mo_idx == self.lumo_idx:
            mo_label = "LUMO"
        elif self.homo_idx is not None and mo_idx < self.homo_idx:
            mo_label = f"HOMO-{self.homo_idx - mo_idx}"
        elif self.lumo_idx is not None and mo_idx > self.lumo_idx:
            mo_label = f"LUMO+{mo_idx - self.lumo_idx}"
        else:
            mo_label = f"MO {mo_idx+1}"
        
        # Create bar plot
        colors = ['red' if c < 0 else 'blue' for c in top_coeffs]
        
        fig = go.Figure(data=[
            go.Bar(
                x=ao_labels,
                y=top_coeffs,
                marker_color=colors,
                text=[f"{c:.3f}" for c in top_coeffs],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Coefficient: %{y:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f"{mo_label} Coefficients (Energy: {self.mo_energy[mo_idx]*27.2114:.2f} eV)",
            xaxis_title="Atomic Orbitals",
            yaxis_title="MO Coefficient",
            width=800,
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def get_mo_summary(self, num_mos=10):
        """
        Get summary of molecular orbitals.
        
        Parameters:
        -----------
        num_mos : int
            Number of MOs around HOMO-LUMO to include
        
        Returns:
        --------
        dict : MO summary
        """
        if self.homo_idx is not None:
            start_mo = max(0, self.homo_idx - num_mos//2)
            end_mo = min(len(self.mo_energy), self.homo_idx + num_mos//2 + 1)
        else:
            start_mo = 0
            end_mo = min(len(self.mo_energy), num_mos)
        
        mo_data = []
        for i in range(start_mo, end_mo):
            # Determine label
            if i == self.homo_idx:
                label = "HOMO"
            elif i == self.lumo_idx:
                label = "LUMO"
            elif self.homo_idx is not None and i < self.homo_idx:
                label = f"HOMO-{self.homo_idx - i}"
            elif self.lumo_idx is not None and i > self.lumo_idx:
                label = f"LUMO+{i - self.lumo_idx}"
            else:
                label = f"MO {i+1}"
            
            mo_data.append({
                'index': i,
                'label': label,
                'energy_hartree': self.mo_energy[i],
                'energy_eV': self.mo_energy[i] * 27.2114,
                'occupancy': self.mo_occ[i],
                'type': 'occupied' if self.mo_occ[i] > 0 else 'virtual'
            })
        
        summary = {
            'total_mos': len(self.mo_energy),
            'homo_idx': self.homo_idx,
            'lumo_idx': self.lumo_idx,
            'homo_energy_eV': self.homo_energy * 27.2114 if self.homo_energy is not None else None,
            'lumo_energy_eV': self.lumo_energy * 27.2114 if self.lumo_energy is not None else None,
            'homo_lumo_gap_eV': self.homo_lumo_gap,
            'num_electrons': int(np.sum(self.mo_occ)),
            'mo_data': mo_data
        }
        
        return summary
    
    def print_mo_summary(self, num_mos=10):
        """Print formatted MO summary."""
        summary = self.get_mo_summary(num_mos)
        
        print("=" * 60)
        print("🌐 MOLECULAR ORBITAL ANALYSIS")
        print("=" * 60)
        print(f"Total MOs: {summary['total_mos']}")
        print(f"Electrons: {summary['num_electrons']}")
        
        if summary['homo_lumo_gap_eV'] is not None:
            print(f"HOMO-LUMO Gap: {summary['homo_lumo_gap_eV']:.2f} eV")
        
        print("\n" + "-" * 60)
        print(f"{'MO':>8} {'Label':>12} {'Energy (eV)':>12} {'Occ':>6} {'Type':>10}")
        print("-" * 60)
        
        for mo in summary['mo_data']:
            print(f"{mo['index']+1:>8} {mo['label']:>12} {mo['energy_eV']:>12.3f} "
                  f"{mo['occupancy']:>6.1f} {mo['type']:>10}")
        
        print("=" * 60)