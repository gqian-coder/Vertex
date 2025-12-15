"""
Mesh interpolation utilities for mapping between different resolutions.
Includes nearest neighbor, linear, and RBF interpolation methods.
"""

import numpy as np
from scipy.spatial import KDTree, cKDTree
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import hashlib
import netCDF4 as nc
from netCDF4 import stringtochar


class MeshInterpolator:
    """Interpolate field data from one mesh to another."""
    
    def __init__(self, source_coords: np.ndarray, target_coords: np.ndarray,
                 method: str = 'linear', normalize: bool = True):
        """
        Initialize interpolator between source and target meshes.
        
        Args:
            source_coords: Source mesh coordinates (N_source, ndim)
            target_coords: Target mesh coordinates (N_target, ndim)
            method: Interpolation method ('nearest', 'linear', 'rbf')
            normalize: Whether to normalize coordinates
        """
        self.source_coords = source_coords
        self.target_coords = target_coords
        self.method = method
        self.normalize = normalize
        
        # Check if we have a 2D mesh (z-coordinate is constant)
        # This is important for linear interpolation which uses Delaunay triangulation
        if source_coords.shape[1] >= 3:
            z_range = source_coords[:, 2].max() - source_coords[:, 2].min()
            if z_range < 1e-10:  # Essentially flat in z
                print(f"    Detected 2D mesh (z-range: {z_range:.2e}), using only x-y coordinates")
                self.source_coords = source_coords[:, :2].copy()
                self.target_coords = target_coords[:, :2].copy()
            else:
                self.source_coords = source_coords.copy()
                self.target_coords = target_coords.copy()
        else:
            self.source_coords = source_coords.copy()
            self.target_coords = target_coords.copy()
        
        # Normalize coordinates if requested
        if normalize:
            self.scaler = StandardScaler()
            self.source_coords_norm = self.scaler.fit_transform(self.source_coords)
            self.target_coords_norm = self.scaler.transform(self.target_coords)
        else:
            self.source_coords_norm = self.source_coords
            self.target_coords_norm = self.target_coords
        
        # Build spatial index for nearest neighbor queries
        self.kdtree = cKDTree(self.source_coords_norm)
        
        # Precompute nearest neighbors for efficiency
        self.distances, self.nearest_indices = self.kdtree.query(
            self.target_coords_norm, k=4
        )
    
    def interpolate(self, source_field: np.ndarray) -> np.ndarray:
        """
        Interpolate field from source to target mesh.
        
        Args:
            source_field: Field values on source mesh (N_source,) or (N_source, n_features)
            
        Returns:
            Interpolated field values on target mesh
        """
        if source_field.ndim == 1:
            source_field = source_field.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False
        
        if self.method == 'nearest':
            result = self._nearest_neighbor(source_field)
        elif self.method == 'linear':
            result = self._linear_interpolation(source_field)
        elif self.method == 'rbf':
            result = self._rbf_interpolation(source_field)
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
        
        if squeeze:
            result = result.squeeze()
        
        return result
    
    def _nearest_neighbor(self, source_field: np.ndarray) -> np.ndarray:
        """Nearest neighbor interpolation."""
        nearest_idx = self.nearest_indices[:, 0]
        return source_field[nearest_idx]
    
    def _linear_interpolation(self, source_field: np.ndarray) -> np.ndarray:
        """Linear interpolation using scipy."""
        n_features = source_field.shape[1]
        result = np.zeros((len(self.target_coords), n_features))
        
        for i in range(n_features):
            # Use LinearNDInterpolator with NaN fill_value
            interp = LinearNDInterpolator(
                self.source_coords_norm, 
                source_field[:, i],
                fill_value=np.nan  # Use NaN instead of 0 to identify points outside convex hull
            )
            result[:, i] = interp(self.target_coords_norm)
            
            # Fill NaN/invalid values with nearest neighbor
            nan_mask = np.isnan(result[:, i]) | (result[:, i] == 0.0)
            if np.any(nan_mask):
                nearest_idx = self.nearest_indices[nan_mask, 0]
                result[nan_mask, i] = source_field[nearest_idx, i]
        
        return result
    
    def _rbf_interpolation(self, source_field: np.ndarray) -> np.ndarray:
        """
        Radial basis function interpolation.
        
        WARNING: RBF is numerically unstable for large meshes (>10K points).
        Use --method linear for better results.
        """
        n_features = source_field.shape[1]
        result = np.zeros((len(self.target_coords), n_features))
        
        num_points = len(self.source_coords)
        
        # Warn user if dataset is large
        if num_points > 10000:
            print(f"    ⚠️  WARNING: RBF with {num_points:,} points is slow and numerically unstable!")
            print(f"    ⚠️  Recommend using --method linear instead.")
        
        # Subsample for large datasets to improve stability
        if num_points > 5000:
            indices = np.random.choice(num_points, size=5000, replace=False)
            source_coords_rbf = self.source_coords[indices]
            print(f"    Subsampling {num_points:,} → 5,000 points for stability")
        else:
            source_coords_rbf = self.source_coords
            indices = None
        
        target_coords_rbf = self.target_coords
        
        print(f"    RBF interpolation with {len(source_coords_rbf):,} source points")
        
        for i in range(n_features):
            print(f"    Building RBF interpolator for feature {i+1}/{n_features}...")
            
            if indices is not None:
                field_values = source_field[indices, i]
            else:
                field_values = source_field[:, i]
            
            interp = RBFInterpolator(
                source_coords_rbf,
                field_values,
                kernel='thin_plate_spline',
                smoothing=0.001  # Small smoothing for stability
            )
            print(f"    Evaluating RBF at {len(target_coords_rbf):,} target points...")
            result[:, i] = interp(target_coords_rbf)
            print(f"    ✓ Feature {i+1}/{n_features} complete")
        
        return result
    
    def interpolate_all_fields(self, source_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Interpolate multiple fields at once.
        
        Args:
            source_fields: Dictionary of field names to values
            
        Returns:
            Dictionary of interpolated fields
        """
        result = {}
        for field_name, field_values in source_fields.items():
            result[field_name] = self.interpolate(field_values)
        return result


def get_cache_path(coarse_data: Dict, fine_data: Dict, cache_dir: str = './cache') -> Path:
    """
    Generate unique cache file path based on data characteristics.
    
    Args:
        coarse_data: Coarse mesh data
        fine_data: Fine mesh data
        cache_dir: Directory to store cache files
        
    Returns:
        Path to cache file
    """
    # Create hash from mesh sizes and coordinates
    hash_str = f"{coarse_data['num_nodes']}_{fine_data['num_nodes']}"
    hash_str += f"_{coarse_data['coordinates'].shape}_{fine_data['coordinates'].shape}"
    
    # Add coordinate statistics for uniqueness
    hash_str += f"_{coarse_data['coordinates'].mean():.6f}_{fine_data['coordinates'].mean():.6f}"
    
    cache_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    cache_path = Path(cache_dir) / f"interp_cache_{cache_hash}.pkl"
    
    return cache_path


def save_interpolated_data(coarse_data: Dict, fine_data: Dict, 
                          interpolated_data: Dict, cache_dir: str = './cache'):
    """
    Save interpolated data to cache file.
    
    Args:
        coarse_data: Coarse mesh data
        fine_data: Fine mesh data
        interpolated_data: Interpolated field data to save
        cache_dir: Directory to store cache files
    """
    cache_path = get_cache_path(coarse_data, fine_data, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(interpolated_data, f)
    
    print(f"Saved interpolated data to: {cache_path}")


def load_interpolated_data(coarse_data: Dict, fine_data: Dict, 
                          cache_dir: str = './cache') -> Optional[Dict]:
    """
    Load interpolated data from cache if available.
    
    Args:
        coarse_data: Coarse mesh data
        fine_data: Fine mesh data
        cache_dir: Directory containing cache files
        
    Returns:
        Cached interpolated data or None if not found
    """
    cache_path = get_cache_path(coarse_data, fine_data, cache_dir)
    
    if cache_path.exists():
        print(f"Loading cached interpolated data from: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    return None


def create_training_pairs(coarse_data: Dict, fine_data: Dict,
                         num_samples: Optional[int] = None, 
                         use_cache: bool = True,
                         cache_dir: str = './cache') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training pairs from coarse and fine resolution data.
    
    Args:
        coarse_data: Coarse mesh data with coordinates and fields
        fine_data: Fine mesh data with coordinates and fields
        num_samples: Number of samples to use (None = all)
        use_cache: Whether to use cached interpolated data
        cache_dir: Directory for cache files
        
    Returns:
        Tuple of (input_features, target_features)
    """
    # Try to load from cache
    if use_cache:
        cached_data = load_interpolated_data(coarse_data, fine_data, cache_dir)
        if cached_data is not None:
            X = cached_data['interpolated_fields']
            y = cached_data['target_fields']
            
            # Apply sampling if requested
            if num_samples is not None and num_samples < len(X):
                indices = np.random.choice(len(X), num_samples, replace=False)
                X = X[indices]
                y = y[indices]
            
            print(f"Using cached data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
    
    # Extract coordinates
    coarse_coords = coarse_data['coordinates']
    fine_coords = fine_data['coordinates']
    
    # Create interpolator
    interpolator = MeshInterpolator(
        coarse_coords,
        fine_coords,
        method='linear',
        normalize=True
    )
    
    # Collect all field variables
    coarse_fields = []
    fine_fields = []
    field_names = ['velocity_0', 'velocity_1', 'pressure', 'temperature']
    
    for field_name in field_names:
        if field_name in coarse_data['fields'] and field_name in fine_data['fields']:
            coarse_field = coarse_data['fields'][field_name]
            fine_field = fine_data['fields'][field_name]
            
            # Handle multiple time steps
            if coarse_field.ndim == 2:
                num_timesteps = min(coarse_field.shape[0], fine_field.shape[0])
                for t in range(num_timesteps):
                    # Interpolate coarse to fine mesh
                    coarse_interp = interpolator.interpolate(coarse_field[t])
                    coarse_fields.append(coarse_interp)
                    fine_fields.append(fine_field[t])
            else:
                coarse_interp = interpolator.interpolate(coarse_field)
                coarse_fields.append(coarse_interp)
                fine_fields.append(fine_field)
    
    # Stack all fields
    if coarse_fields:
        X = np.stack(coarse_fields, axis=-1)  # (n_nodes, n_fields)
        y = np.stack(fine_fields, axis=-1)
        
        # Save to cache
        if use_cache:
            cache_data = {
                'interpolated_fields': X,
                'target_fields': y,
                'field_names': field_names
            }
            save_interpolated_data(coarse_data, fine_data, cache_data, cache_dir)
        
        # Sample if requested
        if num_samples is not None and num_samples < len(X):
            indices = np.random.choice(len(X), num_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        return X, y
    else:
        raise ValueError("No matching fields found between coarse and fine data")


def save_interpolated_to_exodus(target_coords: np.ndarray, 
                                interpolated_fields: Dict[str, np.ndarray],
                                output_file: str,
                                connectivity: Optional[np.ndarray] = None,
                                timesteps: Optional[List[float]] = None,
                                element_blocks: Optional[List[Dict]] = None):
    """
    Save interpolated field data to ExodusII file for ParaView visualization.
    
    Args:
        target_coords: Target mesh coordinates (N_nodes, ndim)
        interpolated_fields: Dictionary mapping field names to interpolated values
                           Each field can be (N_nodes,) for single timestep
                           or (N_timesteps, N_nodes) for multiple timesteps
        output_file: Path to output .exo file
        connectivity: Element connectivity (optional, for single block)
        timesteps: List of time values (optional)
        element_blocks: List of element block dictionaries with connectivity (optional, for multiple blocks)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_nodes = len(target_coords)
    num_dims = target_coords.shape[1]
    
    # Determine number of timesteps
    first_field = list(interpolated_fields.values())[0]
    if first_field.ndim == 2:
        num_timesteps = first_field.shape[0]
    else:
        num_timesteps = 1
    
    # Create NetCDF file
    print(f"Creating ExodusII file: {output_file}")
    with nc.Dataset(output_file, 'w', format='NETCDF3_64BIT_OFFSET') as exo:
        # Create dimensions
        exo.createDimension('len_name', 256)
        exo.createDimension('len_line', 81)
        exo.createDimension('four', 4)
        exo.createDimension('time_step', None)  # Unlimited
        exo.createDimension('num_dim', num_dims)
        exo.createDimension('num_nodes', num_nodes)
        
        # Global attributes
        exo.api_version = 8.11
        exo.version = 8.11
        exo.floating_point_word_size = 8
        exo.file_size = 1
        exo.title = 'Interpolated mesh data'
        exo.maximum_name_length = 256
        
        # QA records (required by IOSS reader)
        exo.createDimension('num_qa_rec', 1)
        qa_records = exo.createVariable('qa_records', 'S1', 
                                       ('num_qa_rec', 'four', 'len_name'))
        # Fill with basic info
        qa_data = np.array([['mesh_interpolation', '1.0', '2025-12-02', '00:00:00']], dtype='S256')
        qa_records[:] = stringtochar(qa_data)
        
        # Info records (optional but good practice)
        exo.createDimension('num_info', 2)
        info_records = exo.createVariable('info_records', 'S1', 
                                         ('num_info', 'len_line'))
        info_data = np.array(['Generated by mesh_interpolation.py', 
                             'Interpolated field data from coarse to fine mesh'], dtype='S81')
        info_records[:] = stringtochar(info_data)
        
        # Coordinate names variable
        coor_names_var = exo.createVariable('coor_names', 'S1', ('num_dim', 'len_name'))
        coord_name_list = ['x', 'y', 'z'][:num_dims]
        coord_names_array = np.array(coord_name_list, dtype='S256')
        coor_names_var[:] = stringtochar(coord_names_array)
        
        # Coordinate variables
        coord_names = ['coordx', 'coordy', 'coordz']
        for i in range(num_dims):
            coord_var = exo.createVariable(coord_names[i], 'f8', ('num_nodes',))
            coord_var[:] = target_coords[:, i]
        
        # Node number map (identity mapping)
        node_num_map = exo.createVariable('node_num_map', 'i4', ('num_nodes',))
        node_num_map[:] = np.arange(1, num_nodes + 1)
        
        # Time values
        time_var = exo.createVariable('time_whole', 'f8', ('time_step',))
        if timesteps is not None:
            time_var[:] = timesteps[:num_timesteps]
        else:
            time_var[:] = np.arange(num_timesteps, dtype=np.float64)
        
        # Node variable names
        exo.createDimension('num_nod_var', len(interpolated_fields))
        name_nod_var = exo.createVariable('name_nod_var', 'S1', 
                                         ('num_nod_var', 'len_name'))
        
        # Write field variables
        field_names_list = list(interpolated_fields.keys())
        field_names_array = np.array(field_names_list, dtype='S256')
        name_nod_var[:] = stringtochar(field_names_array)
        
        for idx, (field_name, field_values) in enumerate(interpolated_fields.items()):
            
            # Create variable
            var_name = f'vals_nod_var{idx + 1}'
            
            # Ensure field has shape (num_timesteps, num_nodes)
            if field_values.ndim == 1:
                field_values = field_values.reshape(1, -1)
            
            # Create and write variable
            field_var = exo.createVariable(var_name, 'f8', ('time_step', 'num_nodes'))
            for t in range(field_values.shape[0]):
                field_var[t, :] = field_values[t, :]
        
        # Add element connectivity if provided
        if element_blocks is not None and len(element_blocks) > 0:
            # Multiple element blocks
            num_blocks = len(element_blocks)
            total_elements = sum(block['num_elements'] for block in element_blocks)
            
            exo.createDimension('num_elem', total_elements)
            exo.createDimension('num_el_blk', num_blocks)
            
            # Element block properties
            eb_prop1 = exo.createVariable('eb_prop1', 'i4', ('num_el_blk',))
            eb_prop1.setncattr('name', 'ID')
            eb_prop1[:] = [block['id'] for block in element_blocks]
            
            # Element block status
            eb_status = exo.createVariable('eb_status', 'i4', ('num_el_blk',))
            eb_status[:] = [1] * num_blocks
            
            # Element block names
            eb_names = exo.createVariable('eb_names', 'S1', ('num_el_blk', 'len_name'))
            block_names = np.array([f"block_{block['id']}" for block in element_blocks], dtype='S256')
            eb_names[:] = stringtochar(block_names)
            
            # Create connectivity for each block
            elem_offset = 0
            for blk_idx, block in enumerate(element_blocks):
                block_id = block['id']
                connectivity = block['connectivity']
                num_elem_blk = block['num_elements']
                num_nodes_per_elem = block['num_nodes_per_elem']
                elem_type = block['elem_type']
                
                # Create dimensions for this block
                exo.createDimension(f'num_el_in_blk{block_id}', num_elem_blk)
                exo.createDimension(f'num_nod_per_el{block_id}', num_nodes_per_elem)
                
                # Connectivity (already 1-indexed from source Exodus file)
                connect_var = exo.createVariable(f'connect{block_id}', 'i4', 
                                                (f'num_el_in_blk{block_id}', f'num_nod_per_el{block_id}'))
                connect_var.setncattr('elem_type', elem_type)
                connect_var[:] = connectivity  # Already 1-indexed, no conversion needed
                
                elem_offset += num_elem_blk
            
            # Element map (global element ordering)
            elem_map = exo.createVariable('elem_map', 'i4', ('num_elem',))
            elem_map[:] = np.arange(1, total_elements + 1)
            
            # Element number map (identity mapping)
            elem_num_map = exo.createVariable('elem_num_map', 'i4', ('num_elem',))
            elem_num_map[:] = np.arange(1, total_elements + 1)
            
        elif connectivity is not None:
            # Single element block (backward compatibility)
            num_elem = connectivity.shape[0]
            num_nodes_per_elem = connectivity.shape[1]
            
            exo.createDimension('num_elem', num_elem)
            exo.createDimension('num_el_blk', 1)
            
            # Element block
            exo.createDimension('num_el_in_blk1', num_elem)
            exo.createDimension('num_nod_per_el1', num_nodes_per_elem)
            
            # Element block ID array
            eb_prop1 = exo.createVariable('eb_prop1', 'i4', ('num_el_blk',))
            eb_prop1.setncattr('name', 'ID')
            eb_prop1[:] = [1]
            
            # Element block status
            eb_status = exo.createVariable('eb_status', 'i4', ('num_el_blk',))
            eb_status[:] = [1]
            
            # Element block names (optional but helpful)
            eb_names = exo.createVariable('eb_names', 'S1', ('num_el_blk', 'len_name'))
            block_names = np.array(['block_1'], dtype='S256')
            eb_names[:] = stringtochar(block_names)
            
            # Connectivity (already 1-indexed from source Exodus file)
            connect_var = exo.createVariable('connect1', 'i4', 
                                            ('num_el_in_blk1', 'num_nod_per_el1'))
            elem_type_str = 'QUAD4' if num_nodes_per_elem == 4 else ('TRI3' if num_nodes_per_elem == 3 else f'SHELL{num_nodes_per_elem}')
            connect_var.setncattr('elem_type', elem_type_str)
            connect_var[:] = connectivity  # Already 1-indexed, no conversion needed
            
            # Element maps
            elem_map = exo.createVariable('elem_map', 'i4', ('num_elem',))
            elem_map[:] = np.arange(1, num_elem + 1)
            
            elem_num_map = exo.createVariable('elem_num_map', 'i4', ('num_elem',))
            elem_num_map[:] = np.arange(1, num_elem + 1)
    
    print(f"✓ Saved {len(interpolated_fields)} fields with {num_timesteps} timesteps")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Fields: {list(interpolated_fields.keys())}")
    print(f"  File: {output_path.absolute()}")
    print(f"\nTo visualize in ParaView:")
    print(f"  1. Open: {output_file}")
    print(f"  2. Apply")
    print(f"  3. Color by: {list(interpolated_fields.keys())[0]}")


def interpolate_and_save_exodus(coarse_data: Dict, fine_data: Dict,
                               output_file: str,
                               method: str = 'linear',
                               field_names: Optional[List[str]] = None,
                               max_timesteps: Optional[int] = None):
    """
    Convenience function to interpolate and save directly to ExodusII.
    
    Args:
        coarse_data: Coarse mesh data
        fine_data: Fine mesh data (provides target coordinates)
        output_file: Path to output .exo file
        method: Interpolation method
        field_names: List of field names to interpolate (None = all available)
        max_timesteps: Maximum number of timesteps to process (None = all)
    """
    coarse_coords = coarse_data['coordinates']
    fine_coords = fine_data['coordinates']
    
    print(f"Interpolating from {len(coarse_coords):,} to {len(fine_coords):,} nodes...")
    
    # Create interpolator
    interpolator = MeshInterpolator(coarse_coords, fine_coords, method=method)
    
    # Determine which fields to interpolate
    if field_names is None:
        field_names = ['velocity_0', 'velocity_1', 'pressure', 'temperature']
    
    available_fields = [f for f in field_names if f in coarse_data['fields']]
    
    if not available_fields:
        raise ValueError(f"No fields found. Available: {list(coarse_data['fields'].keys())}")
    
    # Interpolate each field in the correct order
    # Ensure output order matches: velocity_0, velocity_1, pressure, temperature
    field_order = ['velocity_0', 'velocity_1', 'pressure', 'temperature']
    ordered_fields = [f for f in field_order if f in available_fields]
    
    interpolated_fields = {}
    for field_idx, field_name in enumerate(ordered_fields):
        print(f"\n  Interpolating field {field_idx+1}/{len(ordered_fields)}: {field_name}")
        coarse_field = coarse_data['fields'][field_name]
        
        if coarse_field.ndim == 2:
            # Multiple timesteps
            num_timesteps = coarse_field.shape[0]
            if max_timesteps is not None:
                num_timesteps = min(num_timesteps, max_timesteps)
                print(f"    Processing first {num_timesteps} timesteps (limited by --max-timesteps)...")
            else:
                print(f"    Processing {num_timesteps} timesteps...")
            interp_values = []
            for t in range(num_timesteps):
                print(f"    Timestep {t+1}/{num_timesteps} (t={coarse_data['fields'].get('time_values', [t])[t] if 'time_values' in coarse_data['fields'] else t:.4f})...")
                interp_values.append(interpolator.interpolate(coarse_field[t]))
            interpolated_fields[field_name] = np.array(interp_values)
            print(f"  ✓ Completed {field_name}: {num_timesteps} timesteps interpolated")
        else:
            # Single timestep
            interpolated_fields[field_name] = interpolator.interpolate(coarse_field)
            print(f"  ✓ Completed {field_name}: single timestep interpolated")
    
    # Get timesteps if available
    timesteps = coarse_data['fields'].get('time_values', None)
    
    # Get connectivity and element blocks if available
    element_blocks = fine_data.get('element_blocks', None)
    connectivity = fine_data.get('connectivity', None)
    
    # Save to Exodus
    save_interpolated_to_exodus(
        fine_coords,
        interpolated_fields,
        output_file,
        connectivity=connectivity,
        timesteps=timesteps,
        element_blocks=element_blocks
    )


def compute_interpolation_error(coarse_data: Dict, fine_data: Dict,
                               method: str = 'linear') -> Dict[str, float]:
    """
    Compute interpolation error metrics.
    
    Args:
        coarse_data: Coarse mesh data
        fine_data: Fine mesh data
        method: Interpolation method
        
    Returns:
        Dictionary of error metrics per field
    """
    coarse_coords = coarse_data['coordinates']
    fine_coords = fine_data['coordinates']
    
    interpolator = MeshInterpolator(coarse_coords, fine_coords, method=method)
    
    errors = {}
    for field_name in ['velocity_0', 'velocity_1', 'pressure', 'temperature']:
        if field_name in coarse_data['fields'] and field_name in fine_data['fields']:
            coarse_field = coarse_data['fields'][field_name]
            fine_field = fine_data['fields'][field_name]
            
            # Use first time step for comparison
            if coarse_field.ndim == 2:
                coarse_field = coarse_field[0]
                fine_field = fine_field[0]
            
            # Interpolate and compute error
            interp_field = interpolator.interpolate(coarse_field)
            
            # Compute relative L2 error
            l2_error = np.linalg.norm(interp_field - fine_field) / np.linalg.norm(fine_field)
            errors[field_name] = l2_error
    
    return errors


if __name__ == "__main__":
    # Test interpolation
    print("Testing mesh interpolation...")
    
    # Create simple test case
    source_coords = np.random.rand(100, 2)
    target_coords = np.random.rand(500, 2)
    source_field = np.sin(source_coords[:, 0] * 2 * np.pi)
    
    interpolator = MeshInterpolator(source_coords, target_coords, method='linear')
    target_field = interpolator.interpolate(source_field)
    
    print(f"Source points: {len(source_coords)}")
    print(f"Target points: {len(target_coords)}")
    print(f"Interpolated field range: [{target_field.min():.3f}, {target_field.max():.3f}]")
    print("Interpolation test completed successfully!")
