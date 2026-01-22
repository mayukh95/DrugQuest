"""
Live Progress Monitoring for optimization jobs.
"""

import time
import threading
import json
from pathlib import Path
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output


class LiveProgressTracker:
    """Real-time progress tracking for DFT optimizations with progress bars."""
    
    def __init__(self):
        self.progress_bars = {}
        self.status_outputs = {}
        self.container = widgets.VBox()
        
    def add_molecule(self, mol_name, total_steps):
        """Add a progress bar for a molecule."""
        progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=total_steps,
            description=f'{mol_name[:15]}...' if len(mol_name) > 15 else mol_name,
            bar_style='info',
            style={'description_width': '150px', 'bar_color': '#667eea'},
            layout=widgets.Layout(width='400px')
        )
        
        status = widgets.HTML(value="<span style='color: #666;'>Pending...</span>")
        
        row = widgets.HBox([progress, status], layout=widgets.Layout(margin='4px 0'))
        
        self.progress_bars[mol_name] = progress
        self.status_outputs[mol_name] = status
        
        self.container.children = tuple(list(self.container.children) + [row])
        
        return progress
    
    def update(self, mol_name, step, energy=None, force=None, status='running'):
        """Update progress for a molecule."""
        if mol_name not in self.progress_bars:
            return
        
        progress = self.progress_bars[mol_name]
        status_html = self.status_outputs[mol_name]
        
        progress.value = step
        
        if status == 'running':
            progress.bar_style = 'info'
            info = f"Step {step}"
            if energy is not None:
                info += f" | E={energy:.6f} Ha"
            if force is not None:
                info += f" | F={force:.4f}"
            status_html.value = f"<span style='color: #1976d2;'>⏳ {info}</span>"
        elif status == 'completed':
            progress.bar_style = 'success'
            progress.value = progress.max
            status_html.value = f"<span style='color: #388e3c;'>✅ Complete | E={energy:.6f} Ha</span>"
        elif status == 'failed':
            progress.bar_style = 'danger'
            status_html.value = f"<span style='color: #d32f2f;'>❌ Failed</span>"
    
    def display(self):
        """Display the progress tracker."""
        header = widgets.HTML("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 12px; border-radius: 10px; margin-bottom: 12px;'>
            <h4 style='color: white; margin: 0;'>📊 Live Optimization Progress</h4>
        </div>
        """)
        
        interface = widgets.VBox([header, self.container])
        return interface


class LiveProgressMonitor:
    """Monitor and display live progress of optimization jobs from log files."""
    
    def __init__(self, base_dir=None, tracking_file='optimization_progress.json'):
        """
        Initialize live progress monitor.
        
        Parameters:
        -----------
        base_dir : Path, optional
            Base directory for optimization outputs
        tracking_file : str
            Path to tracking JSON file
        """
        self.base_dir = Path(base_dir) if base_dir else Path('./optimized_molecules')
        self.tracking_file = Path(tracking_file)
        
        # Create output widget
        self.monitor_output = widgets.Output()
        
        # Get list of running/completed optimizations
        self._refresh_folders()
        
        # Molecule selector
        self.mol_monitor_select = widgets.Dropdown(
            options=self.mol_options,
            description='Molecule:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Number of lines to show
        self.lines_to_show = widgets.IntSlider(
            value=30,
            min=10,
            max=100,
            step=10,
            description='Lines:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='300px')
        )
        
        # Monitor button
        self.monitor_btn = widgets.Button(
            description='📊 Show Log',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.monitor_btn.on_click(self._show_log)
        
        self.refresh_btn = widgets.Button(
            description='🔄 Refresh',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.refresh_btn.on_click(self._refresh_folders_click)
    
    def _refresh_folders(self):
        """Refresh list of optimization folders."""
        if self.base_dir.exists():
            self.mol_dirs = sorted([d for d in self.base_dir.iterdir() if d.is_dir()])
            self.mol_options = [d.name for d in self.mol_dirs] if self.mol_dirs else ['No optimization folders found']
        else:
            self.mol_dirs = []
            self.mol_options = ['No optimization folders found']
    
    def _refresh_folders_click(self, b):
        """Handle refresh button click."""
        with self.monitor_output:
            clear_output(wait=True)
            
            self._refresh_folders()
            self.mol_monitor_select.options = self.mol_options
            
            print(f"🔄 Refreshed! Found {len(self.mol_dirs)} optimization folders:")
            for d in self.mol_dirs[:10]:
                log_file = d / 'optimization_output.log'
                status = "✅" if log_file.exists() else "⏳"
                print(f"   {status} {d.name}")
            if len(self.mol_dirs) > 10:
                print(f"   ... and {len(self.mol_dirs) - 10} more")
    
    def _show_log(self, b):
        """Show log file contents."""
        with self.monitor_output:
            clear_output(wait=True)
            
            if not self.mol_dirs or self.mol_monitor_select.value == 'No optimization folders found':
                print("⚠️ No optimization folders found!")
                return
            
            mol_folder = self.base_dir / self.mol_monitor_select.value
            log_file = mol_folder / 'optimization_output.log'
            
            if not log_file.exists():
                print(f"⚠️ Log file not found for {self.mol_monitor_select.value}")
                print(f"   Expected: {log_file}")
                print(f"\n   Available files:")
                for f in mol_folder.iterdir():
                    print(f"   - {f.name}")
                return
            
            # Read log file
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Show last N lines
            n_lines = self.lines_to_show.value
            print(f"📄 Log file: {log_file.name}")
            print(f"   Showing last {min(n_lines, len(lines))} of {len(lines)} lines")
            print(f"{'='*70}\n")
            
            for line in lines[-n_lines:]:
                print(line, end='')
            
            print(f"\n{'='*70}")
            print(f"✓ End of log (total {len(lines)} lines)")
    
    def set_base_dir(self, base_dir):
        """
        Set the base directory for optimization outputs.
        
        Parameters:
        -----------
        base_dir : Path or str
            Base directory path
        """
        self.base_dir = Path(base_dir)
        self._refresh_folders()
        self.mol_monitor_select.options = self.mol_options
    
    def display(self):
        """Display the progress monitor interface."""
        header = widgets.HTML("""
        <div style='background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #2196f3;'>
            <h3 style='margin: 0 0 10px 0;'>🔍 Live Progress Monitor</h3>
            <p style='margin: 5px 0;'>
                <b>Usage:</b> Select a molecule to view its optimization log in real-time.<br>
                • Click "Refresh" to update the molecule list<br>
                • Click "Show Log" to view the latest log entries<br>
                • Increase "Lines" to see more history
            </p>
        </div>
        """)
        
        controls_row1 = widgets.HBox([
            self.mol_monitor_select, 
            self.lines_to_show
        ])
        
        controls_row2 = widgets.HBox([
            self.monitor_btn, 
            self.refresh_btn
        ])
        
        interface = widgets.VBox([
            header,
            controls_row1,
            controls_row2,
            self.monitor_output
        ])
        
        return interface


def export_molecules_to_sdf(molecules_list, filename='molecules_export.sdf', 
                            include_optimized=True, output_dir=None):
    """
    Export molecules to SDF format with embedded properties.
    
    Parameters:
    -----------
    molecules_list : list
        List of DrugMolecule objects
    filename : str
        Output filename
    include_optimized : bool
        If True, use optimized coordinates if available
    output_dir : Path
        Output directory (default: current directory)
    
    Returns:
    --------
    Path : Path to exported file
    """
    try:
        from rdkit import Chem
    except ImportError:
        print("❌ RDKit not available for SDF export")
        return None
    
    output_dir = Path(output_dir) if output_dir else Path('.')
    output_path = output_dir / filename
    
    writer = Chem.SDWriter(str(output_path))
    exported_count = 0
    
    for mol in molecules_list:
        try:
            if mol.mol is None:
                continue
            
            # Create a copy of the molecule
            mol_copy = Chem.Mol(mol.mol)
            
            # Use optimized coordinates if available
            if include_optimized and hasattr(mol, 'optimized_coords') and mol.optimized_coords is not None:
                conf = mol_copy.GetConformer()
                for i, coord in enumerate(mol.optimized_coords):
                    conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
            
            # Add properties
            mol_copy.SetProp('_Name', mol.name)
            mol_copy.SetProp('SMILES', mol.smiles)
            
            if mol.properties:
                for prop, value in mol.properties.items():
                    if isinstance(value, (int, float)):
                        mol_copy.SetDoubleProp(prop, float(value))
                    else:
                        mol_copy.SetProp(prop, str(value))
            
            # Add DFT energy if available
            if hasattr(mol, 'final_energy') and mol.final_energy is not None:
                mol_copy.SetDoubleProp('DFT_Energy_Ha', mol.final_energy)
                mol_copy.SetDoubleProp('DFT_Energy_kcal', mol.final_energy * 627.509)
            
            writer.write(mol_copy)
            exported_count += 1
            
        except Exception as e:
            print(f"⚠️ Could not export {mol.name}: {e}")
    
    writer.close()
    
    print(f"✅ Exported {exported_count} molecules to: {output_path}")
    return output_path