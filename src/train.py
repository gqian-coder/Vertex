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
from models import MeshGNN, MeshEncoderDecoder, SimpleMLP, create_graph_data


class MeshDataset(Dataset):
    """Dataset for mesh-based super-resolution with pre-computed interpolation."""
    
    def __init__(self, coarse_data: Dict = None, fine_data: Dict = None,
                 interpolated_data: Dict = None, timestep_offset: int = 0,
                 use_graph: bool = True, k_neighbors: int = 8,
                 use_cache: bool = True, cache_dir: str = './cache',
                 target_mode: str = 'absolute'):
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
        
        # Pre-compute and cache all interpolated data
        if interpolated_data is not None:
            print("Preparing dataset from pre-interpolated fields...")
        else:
            print("Preparing dataset (interpolating coarse to fine mesh)...")
        self.samples = []
        self._prepare_samples_with_cache(use_cache, cache_dir)
        
        print(f"Dataset created with {len(self.samples)} samples")
    
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

        if self.target_mode == 'residual':
            target = fine_features - coarse_interp
        else:
            target = fine_features
        
        if self.use_graph:
            # Create graph data
            data = create_graph_data(
                self.fine_coords,
                coarse_interp,
                target=target,
                k=self.k_neighbors
            )
            return data
        else:
            # Return as tensors for MLP
            x = torch.cat([
                torch.FloatTensor(self.fine_coords),
                torch.FloatTensor(coarse_interp)
            ], dim=-1)
            y = torch.FloatTensor(target)
            return x, y


def train_epoch(model, loader, optimizer, criterion, device, use_graph=True):
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


def validate(model, loader, criterion, device, use_graph=True):
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
    dataset = MeshDataset(
        all_data[source_res],
        all_data[target_res],
        use_graph=use_graph,
        k_neighbors=config['model'].get('k_neighbors', 8),
        use_cache=config['data'].get('use_cache', True),
        cache_dir=config['data'].get('cache_dir', './cache')
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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_graph)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, use_graph)
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
                'config': config
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
