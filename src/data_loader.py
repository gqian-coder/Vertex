"""
ExodusII data loader for CFD simulation results.
Reads velocity, pressure, and temperature fields from .exo files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import netCDF4 as nc


class ExodusDataLoader:
    """Load and process ExodusII simulation data."""
    
    def __init__(self, file_path: str):
        """
        Initialize loader with path to ExodusII file.
        
        Args:
            file_path: Path to .exo file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.dataset = None
        self.mesh_data = {}
        self.field_data = {}
        
    def load(self) -> Dict:
        """
        Load mesh and field data from ExodusII file.
        
        Returns:
            Dictionary containing mesh coordinates and field variables
        """
        self.dataset = nc.Dataset(str(self.file_path), 'r')
        
        # Load mesh coordinates
        self._load_mesh()
        
        # Load field variables (velocity, pressure, temperature)
        self._load_fields()
        
        return {
            'coordinates': self.mesh_data['coordinates'],
            'connectivity': self.mesh_data.get('connectivity', None),
            'element_blocks': self.mesh_data.get('element_blocks', None),
            'fields': self.field_data,
            'num_nodes': self.mesh_data['num_nodes'],
            'num_elements': self.mesh_data.get('num_elements', 0),
            'time_steps': self.field_data.get('time_values', [])
        }
    
    def _load_mesh(self):
        """Load mesh coordinates and connectivity."""
        # Get spatial dimensions
        num_nodes = len(self.dataset.dimensions['num_nodes'])
        num_dims = len(self.dataset.dimensions['num_dim'])
        
        # Load coordinates
        coords = np.zeros((num_nodes, num_dims))
        coord_names = ['coordx', 'coordy', 'coordz']
        
        for i in range(num_dims):
            if coord_names[i] in self.dataset.variables:
                coords[:, i] = self.dataset.variables[coord_names[i]][:]
        
        self.mesh_data['coordinates'] = coords
        self.mesh_data['num_nodes'] = num_nodes
        self.mesh_data['num_dims'] = num_dims
        
        # Load connectivity if available
        if 'num_elem' in self.dataset.dimensions:
            num_elements = len(self.dataset.dimensions['num_elem'])
            self.mesh_data['num_elements'] = num_elements
            
            # Load all element blocks (there may be multiple)
            self.mesh_data['element_blocks'] = []
            for var_name in sorted(self.dataset.variables.keys()):
                if 'connect' in var_name.lower():
                    conn = self.dataset.variables[var_name][:]
                    elem_type = getattr(self.dataset.variables[var_name], 'elem_type', 'UNKNOWN')
                    block_id = int(var_name.replace('connect', '')) if var_name != 'connect' else 1
                    self.mesh_data['element_blocks'].append({
                        'id': block_id,
                        'connectivity': conn,
                        'elem_type': elem_type,
                        'num_elements': conn.shape[0],
                        'num_nodes_per_elem': conn.shape[1]
                    })
            
            # For backward compatibility, keep first block as 'connectivity'
            if self.mesh_data['element_blocks']:
                self.mesh_data['connectivity'] = self.mesh_data['element_blocks'][0]['connectivity']
    
    def _load_fields(self):
        """Load field variables (velocity, pressure, temperature)."""
        # Get time steps
        if 'time_whole' in self.dataset.variables:
            time_values = self.dataset.variables['time_whole'][:]
            self.field_data['time_values'] = time_values
            num_time_steps = len(time_values)
        else:
            num_time_steps = 1
            self.field_data['time_values'] = [0.0]
        
        num_nodes = self.mesh_data['num_nodes']
        
        # Read actual variable names from the file
        if 'name_nod_var' in self.dataset.variables:
            from netCDF4 import chartostring
            var_names = chartostring(self.dataset.variables['name_nod_var'][:])
            
            # Create mapping from actual names to standardized names
            name_mapping = {}
            for actual_name in var_names:
                actual_lower = actual_name.lower().strip()
                if 'velocity_0' in actual_lower or 'vel_x' in actual_lower or actual_lower == 'ux':
                    name_mapping[actual_name] = 'velocity_0'
                elif 'velocity_1' in actual_lower or 'vel_y' in actual_lower or actual_lower == 'uy':
                    name_mapping[actual_name] = 'velocity_1'
                elif 'pressure' in actual_lower or actual_lower == 'p':
                    name_mapping[actual_name] = 'pressure'
                elif 'temperature' in actual_lower or actual_lower == 'temp' or actual_lower == 't':
                    name_mapping[actual_name] = 'temperature'
            
            # Load fields using actual variable names and indices
            for idx, actual_name in enumerate(var_names):
                var_name = f'vals_nod_var{idx + 1}'
                if var_name in self.dataset.variables and actual_name in name_mapping:
                    standard_name = name_mapping[actual_name]
                    var = self.dataset.variables[var_name]
                    if len(var.shape) == 2:  # (time, nodes)
                        self.field_data[standard_name] = var[:]
                    elif len(var.shape) == 1:  # (nodes,) - single time step
                        self.field_data[standard_name] = var[:].reshape(1, -1)
        else:
            # Fallback to hardcoded mappings if name_nod_var not available
            field_mappings = {
                'velocity_0': ['vals_nod_var1', 'velocity_0', 'vel_x', 'ux'],
                'velocity_1': ['vals_nod_var2', 'velocity_1', 'vel_y', 'uy'],
                'pressure': ['vals_nod_var3', 'lagrange_pressure', 'pressure', 'p'],
                'temperature': ['vals_nod_var4', 'temperature', 'temp', 'T']
            }
            
            for field_name, possible_names in field_mappings.items():
                for var_name in possible_names:
                    if var_name in self.dataset.variables:
                        var = self.dataset.variables[var_name]
                        if len(var.shape) == 2:  # (time, nodes)
                            self.field_data[field_name] = var[:]
                        elif len(var.shape) == 1:  # (nodes,) - single time step
                            self.field_data[field_name] = var[:].reshape(1, -1)
                        break
        
        print(f"Loaded {len(self.field_data)} field variables with {num_time_steps} time steps")
        print(f"Available fields: {list(self.field_data.keys())}")
        
        # Print statistics for each field variable
        self._print_field_statistics()
    
    def _print_field_statistics(self):
        """Print min, max, mean statistics for each field variable."""
        print("\nField Statistics:")
        for field_name, field_data in self.field_data.items():
            if field_name == 'time_values':
                continue  # Skip time values
            
            # Calculate statistics across all timesteps
            data_min = np.min(field_data)
            data_max = np.max(field_data)
            data_mean = np.mean(field_data)
            
            print(f"  {field_name:20s}: min={data_min:10.6f}, max={data_max:10.6f}, mean={data_mean:10.6f}")
    
    def get_timestep(self, timestep: int) -> Dict[str, np.ndarray]:
        """
        Get all fields at a specific time step.
        
        Args:
            timestep: Time step index
            
        Returns:
            Dictionary of field values at the specified time step
        """
        result = {}
        for field_name, field_values in self.field_data.items():
            if field_name != 'time_values' and len(field_values.shape) == 2:
                result[field_name] = field_values[timestep]
        return result
    
    def get_snapshot(self, timestep: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get coordinates and all fields at a specific time step.
        
        Args:
            timestep: Time step index
            
        Returns:
            Tuple of (coordinates, fields_dict)
        """
        coords = self.mesh_data['coordinates']
        fields = self.get_timestep(timestep)
        return coords, fields
    
    def close(self):
        """Close the dataset."""
        if self.dataset is not None:
            self.dataset.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_all_resolutions(base_dir: str) -> Dict[str, Dict]:
    """
    Load all mesh resolutions from the dataset directory.
    
    Args:
        base_dir: Base directory containing all resolution folders
        
    Returns:
        Dictionary mapping resolution names to loaded data
    """
    base_path = Path(base_dir)
    resolutions = {
        'coarse': '45-15',
        'medium': '90-30',
        'fine': '180-60',
        'finest': '360-120'
    }
    
    all_data = {}
    for res_name, folder_name in resolutions.items():
        exo_file = base_path / folder_name / 'cropped.e'
        if exo_file.exists():
            print(f"\nLoading {res_name} resolution from {exo_file}...")
            loader = ExodusDataLoader(str(exo_file))
            all_data[res_name] = loader.load()
            loader.close()
        else:
            print(f"Warning: {exo_file} not found, skipping {res_name}")
    
    return all_data


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        loader = ExodusDataLoader(file_path)
        data = loader.load()
        print(f"\nMesh info:")
        print(f"  Nodes: {data['num_nodes']}")
        print(f"  Elements: {data['num_elements']}")
        print(f"  Coordinates shape: {data['coordinates'].shape}")
        print(f"\nField variables:")
        for name, values in data['fields'].items():
            if name != 'time_values':
                print(f"  {name}: {values.shape}")
        loader.close()
    else:
        print("Usage: python data_loader.py <path_to_exodus_file>")
