"""
Training script for mesh super-resolution models.
Trains GNN or MLP models to map coarse mesh simulations to fine resolution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from data_loader import load_all_resolutions, ExodusDataLoader
from mesh_interpolation import MeshInterpolator, create_training_pairs
from models import MeshGNN, MeshEncoderDecoder, SimpleMLP, create_graph_data, build_knn_graph


class MeshDataset(Dataset):
    """Dataset for mesh-based super-resolution with pre-computed interpolation."""
    
    def __init__(self, coarse_data: Dict = None, fine_data: Dict = None,
                 interpolated_data: Dict = None, timestep_offset: int = 0,
                 use_graph: bool = True, k_neighbors: int = 8,
                 use_cache: bool = True, cache_dir: str = './cache',
                 target_mode: str = 'absolute',
                 residual_normalize: bool = False,
                 residual_stats: Optional[Dict] = None,
                 residual_norm_eps: float = 1e-8,
                 input_normalize: bool = False,
                 input_stats: Optional[Dict] = None,
                 input_norm_eps: float = 1e-8,
                 normalize_coords: bool = False,
                 coord_stats: Optional[Dict] = None,
                 coord_norm_eps: float = 1e-8,
                 physics_enabled: bool = False,
                 physics_boundary_file: Optional[str] = None,
                 physics_sidesets: Optional[Dict[str, bool]] = None):
        """
        Initialize dataset.
        
        Args:
            coarse_data: Coarse mesh data (optional if using interpolated_data)
            fine_data: Fine mesh data (ground truth)
            interpolated_data: Pre-interpolated data on fine mesh (optional)
            timestep_offset: Timestep offset between interpolated and fine data.
                           If interpolated[t+offset] = fine[t], set timestep_offset=offset
            use_graph: Whether to build graph structure
            k_neighbors: Number of neighbors for graph
            use_cache: Whether to cache interpolated data
            cache_dir: Directory for cache files
            target_mode: Training target type:
                        - 'absolute': target is fine fields
                        - 'residual': target is (fine - interpolated) fields
            residual_normalize: If True and target_mode=='residual', normalize residual targets by
                                (residual - mean) / std per variable.
            residual_stats: Optional dict with keys {'mean', 'std'} to reuse existing stats.
                            If not provided and residual_normalize is True, stats are computed from this dataset.
            residual_norm_eps: Minimum std clamp to avoid division by ~0.
        """
        self.coarse_data = coarse_data
        self.fine_data = fine_data
        self.interpolated_data = interpolated_data
        self.timestep_offset = timestep_offset
        self.use_graph = use_graph
        self.k_neighbors = k_neighbors
        if target_mode not in ['absolute', 'residual']:
            raise ValueError(f"Unknown target_mode: {target_mode} (expected 'absolute' or 'residual')")
        self.target_mode = target_mode

        self.residual_normalize = bool(residual_normalize)
        self.residual_norm_eps = float(residual_norm_eps)
        self.residual_stats: Optional[Dict] = None
        self._residual_mean: Optional[np.ndarray] = None
        self._residual_std: Optional[np.ndarray] = None
        if residual_stats is not None:
            # Expect lists/arrays of length n_features
            mean = np.asarray(residual_stats.get('mean', None), dtype=np.float32)
            std = np.asarray(residual_stats.get('std', None), dtype=np.float32)
            if mean.ndim != 1 or std.ndim != 1:
                raise ValueError("residual_stats['mean'] and ['std'] must be 1D")
            self._residual_mean = mean
            self._residual_std = np.maximum(std, self.residual_norm_eps).astype(np.float32)
            self.residual_stats = {
                'mean': self._residual_mean.tolist(),
                'std': self._residual_std.tolist(),
                'eps': self.residual_norm_eps,
            }

        # Optional input normalization (applied to coarse/interpolated field inputs only).
        self.input_normalize = bool(input_normalize)
        self.input_norm_eps = float(input_norm_eps)
        self.input_stats: Optional[Dict] = None
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        if input_stats is not None:
            mean = np.asarray(input_stats.get('mean', None), dtype=np.float32)
            std = np.asarray(input_stats.get('std', None), dtype=np.float32)
            if mean.ndim != 1 or std.ndim != 1:
                raise ValueError("input_stats['mean'] and ['std'] must be 1D")
            self._input_mean = mean
            self._input_std = np.maximum(std, self.input_norm_eps).astype(np.float32)
            self.input_stats = {
                'mean': self._input_mean.tolist(),
                'std': self._input_std.tolist(),
                'eps': self.input_norm_eps,
            }

        # Optional coordinate normalization (applied to coords before concatenation).
        self.normalize_coords = bool(normalize_coords)
        self.coord_norm_eps = float(coord_norm_eps)
        self.coord_stats: Optional[Dict] = None
        self._coord_mean: Optional[np.ndarray] = None
        self._coord_std: Optional[np.ndarray] = None
        if coord_stats is not None:
            mean = np.asarray(coord_stats.get('mean', None), dtype=np.float32)
            std = np.asarray(coord_stats.get('std', None), dtype=np.float32)
            if mean.ndim != 1 or std.ndim != 1:
                raise ValueError("coord_stats['mean'] and ['std'] must be 1D")
            self._coord_mean = mean
            self._coord_std = np.maximum(std, self.coord_norm_eps).astype(np.float32)
            self.coord_stats = {
                'mean': self._coord_mean.tolist(),
                'std': self._coord_std.tolist(),
                'eps': self.coord_norm_eps,
            }

        # Optional physics/BC metadata
        self.physics_enabled = bool(physics_enabled)
        self.physics_boundary_file = physics_boundary_file
        self.physics_sidesets = physics_sidesets or {}
        self.boundary_masks: Dict[str, np.ndarray] = {}
        
        # Get coordinates
        if interpolated_data is not None:
            # Use pre-interpolated data approach
            self.fine_coords = interpolated_data['coordinates']
            print("Using pre-interpolated data (correction-only mode)")
        else:
            # Traditional approach: interpolate from coarse
            if coarse_data is None:
                raise ValueError("Must provide either coarse_data or interpolated_data")
            self.coarse_coords = coarse_data['coordinates']
            self.fine_coords = fine_data['coordinates']
        
        # Check if mesh is 2D (z-range is very small)
        z_range = np.ptp(self.fine_coords[:, 2])
        if z_range < 1e-6:
            print(f"    Detected 2D mesh (z-range: {z_range:.2e}), using only x-y coordinates")
            if interpolated_data is None:
                self.coarse_coords = self.coarse_coords[:, :2]
            self.fine_coords = self.fine_coords[:, :2]

        # If coordinate normalization is enabled, compute stats early so that graph construction
        # (kNN distances) is consistent with the coordinates that will be fed into the model.
        if self.normalize_coords and self._coord_mean is None:
            self._compute_coord_stats_from_coords()

        # Precompute graph connectivity once (shared across all timesteps)
        self._edge_index = None
        if self.use_graph:
            print(f"    Precomputing kNN graph once (k={self.k_neighbors})...")
            coords_for_graph = self.fine_coords
            if self.normalize_coords:
                coords_for_graph = (coords_for_graph - self._coord_mean) / self._coord_std
            self._edge_index = build_knn_graph(coords_for_graph, k=self.k_neighbors)
        
        # Pre-compute and cache all interpolated data
        if interpolated_data is not None:
            print("Preparing dataset from pre-interpolated fields...")
        else:
            print("Preparing dataset (interpolating coarse to fine mesh)...")
        self.samples = []
        self._prepare_samples_with_cache(use_cache, cache_dir)

        # If residual learning + normalization requested, compute stats from dataset unless provided.
        if self.target_mode == 'residual' and self.residual_normalize and self._residual_mean is None:
            self._compute_residual_stats_from_samples()

        # If input normalization requested, compute stats from dataset unless provided.
        if self.input_normalize and self._input_mean is None:
            self._compute_input_stats_from_samples()

        # If coordinate normalization requested, compute stats unless provided.
        if self.normalize_coords and self._coord_mean is None:
            self._compute_coord_stats_from_coords()

        # Optional: build boundary node masks from a separate Exodus file (typically solution.exo)
        # This is intentionally best-effort: if it fails, training continues unchanged.
        if self.physics_enabled and self.physics_boundary_file:
            try:
                from model_physics import build_sideset_node_mask

                if self.physics_sidesets.get('cylinder', False):
                    self.boundary_masks['cylinder'] = build_sideset_node_mask(self.physics_boundary_file, 'cylinder')
                if self.physics_sidesets.get('inlet', False):
                    self.boundary_masks['inlet'] = build_sideset_node_mask(self.physics_boundary_file, 'inlet')
                if self.physics_sidesets.get('outlet', False):
                    self.boundary_masks['outlet'] = build_sideset_node_mask(self.physics_boundary_file, 'outlet')
                if self.boundary_masks:
                    print(f"    Physics boundary masks enabled from: {self.physics_boundary_file}")
            except Exception as e:
                print(f"[physics] Warning: failed to build boundary masks from {self.physics_boundary_file}: {e}")
        
        print(f"Dataset created with {len(self.samples)} samples")

    def _compute_residual_stats_from_samples(self) -> None:
        if not self.samples:
            raise ValueError("Cannot compute residual stats: dataset has no samples")

        # Accumulate per-feature moments over all nodes and timesteps.
        # samples[*]['coarse_interp'] and ['fine_features'] are (n_nodes, n_features)
        sum_ = None
        sumsq = None
        count = 0
        for sample in self.samples:
            coarse_interp = sample['coarse_interp'].astype(np.float64)
            fine_features = sample['fine_features'].astype(np.float64)
            residual = fine_features - coarse_interp
            if sum_ is None:
                sum_ = np.sum(residual, axis=0)
                sumsq = np.sum(residual ** 2, axis=0)
            else:
                sum_ += np.sum(residual, axis=0)
                sumsq += np.sum(residual ** 2, axis=0)
            count += residual.shape[0]

        mean = (sum_ / max(count, 1)).astype(np.float32)
        var = (sumsq / max(count, 1)) - mean.astype(np.float64) ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var).astype(np.float32)
        std = np.maximum(std, self.residual_norm_eps).astype(np.float32)

        self._residual_mean = mean
        self._residual_std = std
        self.residual_stats = {
            'mean': self._residual_mean.tolist(),
            'std': self._residual_std.tolist(),
            'eps': self.residual_norm_eps,
        }
        print("    Residual normalization enabled")
        print(f"      mean: {self._residual_mean}")
        print(f"      std : {self._residual_std}")

    def _compute_input_stats_from_samples(self) -> None:
        if not self.samples:
            raise ValueError("Cannot compute input stats: dataset has no samples")

        # Accumulate per-feature moments over all nodes and timesteps.
        sum_ = None
        sumsq = None
        count = 0
        for sample in self.samples:
            x = sample['coarse_interp'].astype(np.float64)
            if sum_ is None:
                sum_ = np.sum(x, axis=0)
                sumsq = np.sum(x ** 2, axis=0)
            else:
                sum_ += np.sum(x, axis=0)
                sumsq += np.sum(x ** 2, axis=0)
            count += x.shape[0]

        mean = (sum_ / max(count, 1)).astype(np.float32)
        var = (sumsq / max(count, 1)) - mean.astype(np.float64) ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var).astype(np.float32)
        std = np.maximum(std, self.input_norm_eps).astype(np.float32)

        self._input_mean = mean
        self._input_std = std
        self.input_stats = {
            'mean': self._input_mean.tolist(),
            'std': self._input_std.tolist(),
            'eps': self.input_norm_eps,
        }
        print("    Input normalization enabled (fields only)")
        print(f"      mean: {self._input_mean}")
        print(f"      std : {self._input_std}")

    def _compute_coord_stats_from_coords(self) -> None:
        coords = np.asarray(self.fine_coords, dtype=np.float64)
        mean = coords.mean(axis=0).astype(np.float32)
        std = coords.std(axis=0).astype(np.float32)
        std = np.maximum(std, self.coord_norm_eps).astype(np.float32)
        self._coord_mean = mean
        self._coord_std = std
        self.coord_stats = {
            'mean': self._coord_mean.tolist(),
            'std': self._coord_std.tolist(),
            'eps': self.coord_norm_eps,
        }
        print("    Coordinate normalization enabled")
        print(f"      mean: {self._coord_mean}")
        print(f"      std : {self._coord_std}")
    
    def _prepare_samples_with_cache(self, use_cache: bool, cache_dir: str):
        """Prepare all training samples with pre-computed interpolation."""
        from pathlib import Path
        import pickle
        import hashlib
        
        # Use pre-interpolated data if available
        if self.interpolated_data is not None:
            self._prepare_samples_from_interpolated(timestep_offset=self.timestep_offset)
            return
        
        # Generate cache key
        cache_key = f"{self.coarse_data['num_nodes']}_{self.fine_data['num_nodes']}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_path = Path(cache_dir) / f"dataset_cache_{cache_hash}.pkl"
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            print(f"Loading pre-computed samples from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            return
        
        # Build interpolator once
        print("Building interpolator...")
        interpolator = MeshInterpolator(
            self.coarse_coords, self.fine_coords, method='linear'
        )
        
        # Get field names (try both naming conventions)
        field_names = ['velocity_0', 'velocity_1', 'pressure', 'temperature']
        if 'velocity_x' in self.coarse_data['fields']:
            field_names = ['velocity_x', 'velocity_y', 'pressure', 'temperature']
        
        # Check which fields are available
        available_fields = [f for f in field_names 
                          if f in self.coarse_data['fields'] and f in self.fine_data['fields']]
        
        if not available_fields:
            raise ValueError("No matching fields found!")
        
        # Get number of timesteps
        first_field = available_fields[0]
        coarse_field = self.coarse_data['fields'][first_field]
        fine_field = self.fine_data['fields'][first_field]
        
        if coarse_field.ndim == 2:
            num_timesteps = min(coarse_field.shape[0], fine_field.shape[0])
        else:
            num_timesteps = 1
        
        print(f"Pre-computing interpolation for {num_timesteps} timesteps...")
        
        # Create samples for each timestep with pre-computed interpolation
        for t in range(num_timesteps):
            # Collect coarse fields
            coarse_features = []
            fine_features = []
            
            for field_name in available_fields:
                coarse_f = self.coarse_data['fields'][field_name]
                fine_f = self.fine_data['fields'][field_name]
                
                if coarse_f.ndim == 2:
                    coarse_features.append(coarse_f[t])
                    fine_features.append(fine_f[t])
                else:
                    coarse_features.append(coarse_f)
                    fine_features.append(fine_f)
            
            # Stack features
            coarse_features = np.stack(coarse_features, axis=-1)
            fine_features = np.stack(fine_features, axis=-1)
            
            # PRE-COMPUTE interpolation here (done once!)
            coarse_interp = interpolator.interpolate(coarse_features)
            
            self.samples.append({
                'timestep': t,
                'coarse_interp': coarse_interp,  # Store interpolated data
                'fine_features': fine_features
            })
        
        # Cache samples
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"Saved pre-computed samples to cache: {cache_path}")
    
    def _prepare_samples_from_interpolated(self, timestep_offset: int = 0):
        """Prepare samples using pre-interpolated data (correction-only mode).
        
        Args:
            timestep_offset: Offset between interpolated and fine timesteps.
                            If interpolated[t + offset] = fine[t], set offset=offset
        """
        # Get field names
        field_names = ['velocity_0', 'velocity_1', 'pressure', 'temperature']
        if 'velocity_x' in self.interpolated_data['fields']:
            field_names = ['velocity_x', 'velocity_y', 'pressure', 'temperature']
        
        # Check which fields are available
        available_fields = [f for f in field_names 
                          if f in self.interpolated_data['fields'] and f in self.fine_data['fields']]
        
        if not available_fields:
            raise ValueError("No matching fields found between interpolated and fine data!")
        
        # Get number of timesteps
        first_field = available_fields[0]
        interp_field = self.interpolated_data['fields'][first_field]
        fine_field = self.fine_data['fields'][first_field]
        
        if interp_field.ndim == 2:
            # Account for offset when determining number of valid timesteps
            num_interp_timesteps = interp_field.shape[0]
            num_fine_timesteps = fine_field.shape[0]
            
            # Calculate valid range: interpolated[t_interp] matches fine[t_fine]
            # where t_interp = t_fine + offset
            if timestep_offset >= 0:
                # Positive offset: interpolated is ahead
                num_timesteps = min(num_interp_timesteps - timestep_offset, num_fine_timesteps)
            else:
                # Negative offset: fine is ahead
                num_timesteps = min(num_interp_timesteps, num_fine_timesteps + timestep_offset)
            
            if num_timesteps <= 0:
                raise ValueError(f"Invalid timestep offset {timestep_offset}: no overlapping timesteps!")
        else:
            num_timesteps = 1
        
        if timestep_offset != 0:
            print(f"Using timestep offset: interpolated[t+{timestep_offset}] = fine[t]")
        print(f"Loading {num_timesteps} timesteps from pre-interpolated data...")
        
        # Create samples for each timestep
        for t_fine in range(num_timesteps):
            # Calculate corresponding interpolated timestep
            t_interp = t_fine + timestep_offset
            
            # Collect interpolated and fine features
            interp_features = []
            fine_features = []
            
            for field_name in available_fields:
                interp_f = self.interpolated_data['fields'][field_name]
                fine_f = self.fine_data['fields'][field_name]
                
                if interp_f.ndim == 2:
                    interp_features.append(interp_f[t_interp])
                    fine_features.append(fine_f[t_fine])
                else:
                    interp_features.append(interp_f)
                    fine_features.append(fine_f)
            
            # Stack features
            interp_features = np.stack(interp_features, axis=-1)
            fine_features = np.stack(fine_features, axis=-1)
            
            self.samples.append({
                'timestep': t_fine,
                'timestep_interp': t_interp,
                'coarse_interp': interp_features,  # Already interpolated!
                'fine_features': fine_features
            })
        
        print(f"Loaded {len(self.samples)} samples from pre-interpolated data")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get pre-computed interpolated features (no interpolation here!)
        coarse_interp = sample['coarse_interp']  # (n_fine_nodes, n_features)
        fine_features = sample['fine_features']  # (n_fine_nodes, n_features)

        # Preserve raw baseline and coordinates for optional physics/BC losses.
        coarse_interp_raw = torch.FloatTensor(coarse_interp)
        coords_raw = torch.FloatTensor(self.fine_coords)

        if self.target_mode == 'residual':
            target = fine_features - coarse_interp
            if self.residual_normalize:
                if self._residual_mean is None or self._residual_std is None:
                    raise RuntimeError("Residual normalization requested but stats were not computed")
                target = (target - self._residual_mean) / self._residual_std
        else:
            target = fine_features

        # Prepare (optionally normalized) model inputs.
        input_features = coarse_interp
        if self.input_normalize:
            if self._input_mean is None or self._input_std is None:
                raise RuntimeError("Input normalization requested but stats were not computed")
            input_features = (input_features - self._input_mean) / self._input_std

        coords = self.fine_coords
        if self.normalize_coords:
            if self._coord_mean is None or self._coord_std is None:
                raise RuntimeError("Coordinate normalization requested but stats were not computed")
            coords = (coords - self._coord_mean) / self._coord_std
        
        if self.use_graph:
            # Create graph data
            data = create_graph_data(
                coords,
                input_features,
                target=target,
                k=self.k_neighbors,
                edge_index=self._edge_index
            )

            # Optional attributes for physics losses
            data.coarse_interp_raw = coarse_interp_raw
            data.coords_raw = coords_raw
            if self.boundary_masks:
                if 'cylinder' in self.boundary_masks:
                    data.mask_cylinder = torch.as_tensor(self.boundary_masks['cylinder'], dtype=torch.bool)
                if 'inlet' in self.boundary_masks:
                    data.mask_inlet = torch.as_tensor(self.boundary_masks['inlet'], dtype=torch.bool)
                if 'outlet' in self.boundary_masks:
                    data.mask_outlet = torch.as_tensor(self.boundary_masks['outlet'], dtype=torch.bool)
            return data
        else:
            # Return as tensors for MLP
            x = torch.cat([
                torch.FloatTensor(coords),
                torch.FloatTensor(input_features)
            ], dim=-1)
            y = torch.FloatTensor(target)
            return x, y


def _graph_smoothness_loss(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Graph Laplacian-style smoothness loss: mean_{(i,j) in E} ||pred_i - pred_j||^2."""
    src = edge_index[0]
    dst = edge_index[1]
    diff = pred[src] - pred[dst]
    return (diff ** 2).mean()


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_graph=True,
    smoothness_lambda: float = 0.0,
    physics_cfg=None,
    residual_stats: Optional[Dict] = None,
    residual_learning: bool = False,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        if use_graph:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            if smoothness_lambda and smoothness_lambda > 0:
                loss = loss + float(smoothness_lambda) * _graph_smoothness_loss(out, batch.edge_index)

            # Optional physics/BC losses (only meaningful in residual/correction mode)
            if residual_learning and physics_cfg is not None and getattr(physics_cfg, 'enabled', False):
                from model_physics import physics_loss_from_batch

                phys_loss, _ = physics_loss_from_batch(batch, out, residual_stats=residual_stats, physics_cfg=physics_cfg)
                loss = loss + phys_loss
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(
    model,
    loader,
    criterion,
    device,
    use_graph=True,
    smoothness_lambda: float = 0.0,
    physics_cfg=None,
    residual_stats: Optional[Dict] = None,
    residual_learning: bool = False,
):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            if use_graph:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                if smoothness_lambda and smoothness_lambda > 0:
                    loss = loss + float(smoothness_lambda) * _graph_smoothness_loss(out, batch.edge_index)

                if residual_learning and physics_cfg is not None and getattr(physics_cfg, 'enabled', False):
                    from model_physics import physics_loss_from_batch

                    phys_loss, _ = physics_loss_from_batch(batch, out, residual_stats=residual_stats, physics_cfg=physics_cfg)
                    loss = loss + phys_loss
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(config: Dict):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    dataset_path = config['data']['dataset_path']
    all_data = load_all_resolutions(dataset_path)
    
    # Select source and target resolutions
    source_res = config['data']['source_resolution']  # e.g., 'coarse'
    target_res = config['data']['target_resolution']  # e.g., 'fine'
    
    if source_res not in all_data or target_res not in all_data:
        raise ValueError(f"Required resolutions not found: {source_res}, {target_res}")
    
    # Create dataset
    use_graph = config['model']['type'] in ['gnn', 'encoder_decoder']

    # Optional: residual learning on-the-fly interpolation path.
    # - If enabled, the model predicts residual = fine - interpolated
    # - If residual_normalization enabled, residual targets are normalized by dataset mean/std
    residual_learning = bool(config.get('training', {}).get('residual_learning', False))
    residual_normalization = bool(config.get('training', {}).get('residual_normalization', False))
    residual_norm_eps = float(config.get('training', {}).get('residual_norm_eps', 1e-8))
    target_mode = 'residual' if residual_learning else 'absolute'
    if residual_learning:
        print("Using residual learning: target = fine - interpolated")
        if residual_normalization:
            print("Using residual normalization: (residual - mean) / std")

    smoothness_lambda = float(config.get('training', {}).get('smoothness_lambda', 0.0))
    if smoothness_lambda and smoothness_lambda > 0:
        if not use_graph:
            print("WARNING: smoothness_lambda set but model is not graph-based; ignoring smoothness.")
            smoothness_lambda = 0.0
        elif not residual_learning:
            print("WARNING: smoothness_lambda set but residual_learning is false; ignoring smoothness.")
            smoothness_lambda = 0.0
    input_normalization = bool(config.get('training', {}).get('input_normalization', False))
    input_norm_eps = float(config.get('training', {}).get('input_norm_eps', 1e-8))
    normalize_coords = bool(config.get('training', {}).get('normalize_coords', False))
    coord_norm_eps = float(config.get('training', {}).get('coord_norm_eps', 1e-8))

    # Optional physics/BC constraints (only meaningful for residual learning).
    from model_physics import physics_config_from_training_cfg

    physics_cfg = physics_config_from_training_cfg(config.get('training', {}))
    if physics_cfg.enabled and not residual_learning:
        print("WARNING: training.physics.enabled=true but residual_learning=false; disabling physics losses.")
        physics_cfg.enabled = False

    phys_sidesets = {}
    if physics_cfg.enabled:
        phys_sidesets = (config.get('training', {}).get('physics', {}) or {}).get('sidesets', None)
        if not isinstance(phys_sidesets, dict) or not phys_sidesets:
            phys_sidesets = {'cylinder': True}

    dataset = MeshDataset(
        all_data[source_res],
        all_data[target_res],
        use_graph=use_graph,
        k_neighbors=config['model'].get('k_neighbors', 8),
        use_cache=config['data'].get('use_cache', True),
        cache_dir=config['data'].get('cache_dir', './cache'),
        target_mode=target_mode,
        residual_normalize=(residual_learning and residual_normalization),
        residual_norm_eps=residual_norm_eps,
        input_normalize=input_normalization,
        input_norm_eps=input_norm_eps,
        normalize_coords=normalize_coords,
        coord_norm_eps=coord_norm_eps,
        physics_enabled=bool(physics_cfg.enabled),
        physics_boundary_file=physics_cfg.boundary_file,
        physics_sidesets=phys_sidesets,
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    if use_graph:
        train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create model
    print("\nCreating model...")
    ndim = all_data[target_res]['coordinates'].shape[1]
    in_channels = ndim + 4  # coordinates + 4 fields
    out_channels = 4  # 4 fields
    
    model_type = config['model']['type']
    if model_type == 'gnn':
        model = MeshGNN(
            in_channels=in_channels,
            hidden_channels=config['model']['hidden_channels'],
            out_channels=out_channels,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    elif model_type == 'encoder_decoder':
        model = MeshEncoderDecoder(
            in_channels=in_channels,
            hidden_channels=config['model']['hidden_channels'],
            out_channels=out_channels,
            num_levels=config['model'].get('num_levels', 3),
            dropout=config['model']['dropout']
        )
    elif model_type == 'mlp':
        model = SimpleMLP(
            in_channels=in_channels,
            hidden_channels=config['model']['hidden_channels'],
            out_channels=out_channels,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"Model: {model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    print("\nStarting training...")
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_graph,
            smoothness_lambda=smoothness_lambda,
            physics_cfg=physics_cfg,
            residual_stats=getattr(dataset, 'residual_stats', None),
            residual_learning=residual_learning,
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            use_graph,
            smoothness_lambda=smoothness_lambda,
            physics_cfg=physics_cfg,
            residual_stats=getattr(dataset, 'residual_stats', None),
            residual_learning=residual_learning,
        )
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'residual_stats': getattr(dataset, 'residual_stats', None),
                'input_stats': getattr(dataset, 'input_stats', None),
                'coord_stats': getattr(dataset, 'coord_stats', None),
            }, output_dir / 'best_model.pt')
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'residual_stats': getattr(dataset, 'residual_stats', None),
                'input_stats': getattr(dataset, 'input_stats', None),
                'coord_stats': getattr(dataset, 'coord_stats', None),
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import sys
    
    # Load configuration
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'data': {
                'dataset_path': '/lustre/orion/csc143/proj-shared/gongq/frontier/VERTEX/dataset',
                'source_resolution': 'coarse',
                'target_resolution': 'fine'
            },
            'model': {
                'type': 'gnn',  # 'gnn', 'encoder_decoder', 'mlp'
                'hidden_channels': 128,
                'num_layers': 4,
                'dropout': 0.1,
                'k_neighbors': 8
            },
            'training': {
                'batch_size': 1,  # Usually 1 for graph data
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'num_epochs': 100,
                'save_every': 10,
                'output_dir': './outputs'
            }
        }
    
    train_model(config)
