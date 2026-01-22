"""
File export utilities for molecular data and calculation results.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

from ..core.molecule import DrugMolecule


class FileExporter:
    """Base class for file exporters."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize file exporter."""
        self.output_dir = output_dir or Path.cwd() / 'exports'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(self, data: Any, filename: str, **kwargs):
        """Export data to file."""
        raise NotImplementedError


class SDFExporter(FileExporter):
    """Export molecules to SDF format."""
    
    def export(self, molecules: List[DrugMolecule], filename: str, 
               include_properties=True):
        """Export molecules to SDF file."""
        output_path = self.output_dir / filename
        
        with Chem.SDWriter(str(output_path)) as writer:
            for mol_obj in molecules:
                mol = mol_obj.mol
                
                if include_properties:
                    # Add molecular properties as metadata
                    properties = mol_obj.get_properties()
                    for key, value in properties.items():
                        if isinstance(value, (int, float, str)):
                            mol.SetProp(key, str(value))
                
                writer.write(mol)
        
        print(f"✓ Exported {len(molecules)} molecules to {output_path}")
        return output_path


class PDBExporter(FileExporter):
    """Export molecules to PDB format."""
    
    def export(self, molecules: List[DrugMolecule], filename: str):
        """Export molecules to PDB file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for i, mol_obj in enumerate(molecules):
                mol = mol_obj.mol
                
                # Add 3D conformer if not present
                if mol.GetNumConformers() == 0:
                    from rdkit.Chem import rdDistGeom
                    rdDistGeom.EmbedMolecule(mol)
                
                # Write PDB block
                pdb_block = Chem.MolToPDBBlock(mol)
                
                # Add model header for multiple molecules
                if len(molecules) > 1:
                    f.write(f"MODEL        {i+1}\n")
                
                f.write(pdb_block)
                
                if len(molecules) > 1:
                    f.write("ENDMDL\n")
        
        print(f"✓ Exported {len(molecules)} molecules to {output_path}")
        return output_path


class ReportExporter(FileExporter):
    """Export analysis reports and results."""
    
    def export_molecular_report(self, molecules: List[DrugMolecule], 
                              filename: str, format='html'):
        """Export detailed molecular analysis report."""
        if format not in ['html', 'json', 'csv']:
            raise ValueError("Format must be 'html', 'json', or 'csv'")
        
        output_path = self.output_dir / f"{filename}.{format}"
        
        # Compile data
        data = []
        for mol_obj in molecules:
            mol_data = {
                'smiles': mol_obj.smiles,
                'name': mol_obj.name,
                'molecular_weight': Descriptors.ExactMolWt(mol_obj.mol),
                'logp': Descriptors.MolLogP(mol_obj.mol),
                'tpsa': Descriptors.TPSA(mol_obj.mol),
                'hbd': Descriptors.NumHDonors(mol_obj.mol),
                'hba': Descriptors.NumHAcceptors(mol_obj.mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol_obj.mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol_obj.mol)
            }
            
            # Add properties if available
            properties = mol_obj.get_properties()
            mol_data.update(properties)
            
            data.append(mol_data)
        
        if format == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == 'html':
            self._export_html_report(data, output_path)
        
        print(f"✓ Exported molecular report to {output_path}")
        return output_path
    
    def _export_html_report(self, data: List[Dict], output_path: Path):
        """Generate HTML report."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Molecular Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary { background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .molecular-prop { display: inline-block; margin: 10px; padding: 10px; background-color: #fff; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Molecular Analysis Report</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="molecular-prop">
            <strong>Total Molecules:</strong> {total_molecules}
        </div>
        <div class="molecular-prop">
            <strong>Average Molecular Weight:</strong> {avg_mw:.2f} g/mol
        </div>
        <div class="molecular-prop">
            <strong>Average LogP:</strong> {avg_logp:.2f}
        </div>
        <div class="molecular-prop">
            <strong>Average TPSA:</strong> {avg_tpsa:.2f} Ų
        </div>
    </div>
    
    <h2>Molecular Data</h2>
    <table>
        <thead>
            <tr>
""".format(
            total_molecules=len(data),
            avg_mw=sum(d.get('molecular_weight', 0) for d in data) / len(data) if data else 0,
            avg_logp=sum(d.get('logp', 0) for d in data) / len(data) if data else 0,
            avg_tpsa=sum(d.get('tpsa', 0) for d in data) / len(data) if data else 0
        )
        
        # Add table headers
        if data:
            headers = data[0].keys()
            for header in headers:
                html_content += f"                <th>{header.replace('_', ' ').title()}</th>\n"
        
        html_content += """            </tr>
        </thead>
        <tbody>
"""
        
        # Add table rows
        for mol_data in data:
            html_content += "            <tr>\n"
            for value in mol_data.values():
                if isinstance(value, float):
                    html_content += f"                <td>{value:.3f}</td>\n"
                else:
                    html_content += f"                <td>{value}</td>\n"
            html_content += "            </tr>\n"
        
        html_content += """        </tbody>
    </table>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def export_optimization_report(self, results: Dict, filename: str):
        """Export optimization results report."""
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Exported optimization report to {output_path}")
        return output_path
    
    def export_dft_results(self, results: Dict, filename: str, format='json'):
        """Export DFT calculation results."""
        if format == 'json':
            output_path = self.output_dir / f"{filename}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        elif format == 'csv':
            output_path = self.output_dir / f"{filename}.csv"
            
            # Flatten results for CSV
            flattened_data = []
            for job_id, job_result in results.items():
                if isinstance(job_result, dict):
                    row = {'job_id': job_id}
                    row.update(job_result)
                    flattened_data.append(row)
            
            if flattened_data:
                df = pd.DataFrame(flattened_data)
                df.to_csv(output_path, index=False)
        
        print(f"✓ Exported DFT results to {output_path}")
        return output_path


class CSVExporter(FileExporter):
    """Export data to CSV format."""
    
    def export(self, data: List[Dict], filename: str):
        """Export list of dictionaries to CSV."""
        output_path = self.output_dir / filename
        
        if not data:
            print("No data to export.")
            return None
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Exported data to {output_path}")
        return output_path


class JSONExporter(FileExporter):
    """Export data to JSON format."""
    
    def export(self, data: Any, filename: str, indent=2):
        """Export data to JSON file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)
        
        print(f"✓ Exported data to {output_path}")
        return output_path


class XYZExporter(FileExporter):
    """Export molecular coordinates to XYZ format."""
    
    def export(self, molecules: List[DrugMolecule], filename: str):
        """Export molecular coordinates to XYZ file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for i, mol_obj in enumerate(molecules):
                mol = mol_obj.mol
                
                # Get conformer
                if mol.GetNumConformers() == 0:
                    from rdkit.Chem import rdDistGeom
                    rdDistGeom.EmbedMolecule(mol)
                
                conf = mol.GetConformer()
                
                # Write XYZ format
                f.write(f"{mol.GetNumAtoms()}\n")
                f.write(f"Molecule {i+1}: {mol_obj.name}\n")
                
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx()
                    pos = conf.GetAtomPosition(idx)
                    symbol = atom.GetSymbol()
                    f.write(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")
                
                # Add separator for multiple molecules
                if i < len(molecules) - 1:
                    f.write("\n")
        
        print(f"✓ Exported coordinates to {output_path}")
        return output_path


class BatchExporter:
    """Export multiple file formats simultaneously."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize batch exporter."""
        self.output_dir = output_dir or Path.cwd() / 'exports'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.exporters = {
            'sdf': SDFExporter(self.output_dir),
            'pdb': PDBExporter(self.output_dir),
            'csv': CSVExporter(self.output_dir),
            'json': JSONExporter(self.output_dir),
            'xyz': XYZExporter(self.output_dir),
            'report': ReportExporter(self.output_dir)
        }
    
    def export_molecules(self, molecules: List[DrugMolecule], 
                        base_filename: str, formats: List[str]):
        """Export molecules in multiple formats."""
        exported_files = {}
        
        for format_name in formats:
            if format_name in self.exporters:
                try:
                    filename = f"{base_filename}.{format_name}"
                    exporter = self.exporters[format_name]
                    
                    if format_name == 'report':
                        output_path = exporter.export_molecular_report(
                            molecules, base_filename, format='html'
                        )
                    else:
                        output_path = exporter.export(molecules, filename)
                    
                    exported_files[format_name] = output_path
                    
                except Exception as e:
                    print(f"Error exporting {format_name}: {e}")
        
        return exported_files