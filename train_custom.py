#!/usr/bin/env python3
"""
Quick training script for custom file paths.
Trains a super-resolution model on your 180-60 to 360-120 interpolation data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from data_loader import ExodusDataLoader
from train import MeshDataset, train_epoch, validate
from models import MeshGNN, SimpleMLP, MeshEncoderDecoder
import numpy as np
from model_physics import physics_config_from_training_cfg


def train_custom(config_path='config_custom.yaml'):
    """Train model with custom file paths."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if using pre-interpolated data
    use_preinterpolated = 'interpolated_file' in config['data']

    input_normalization = bool(config.get('training', {}).get('input_normalization', True))
    input_norm_eps = float(config.get('training', {}).get('input_norm_eps', 1e-8))
    normalize_coords = bool(config.get('training', {}).get('normalize_coords', False))
    coord_norm_eps = float(config.get('training', {}).get('coord_norm_eps', 1e-8))

    # Persist effective settings so they get captured in checkpoints.
    config.setdefault('training', {})
    config['training']['input_normalization'] = input_normalization
    config['training']['input_norm_eps'] = input_norm_eps
    config['training']['normalize_coords'] = normalize_coords
    config['training']['coord_norm_eps'] = coord_norm_eps

    # Default: drop temperature from model inputs (use vx, vy, p).
    input_feature_indices = config.get('training', {}).get('input_feature_indices', [0, 1, 2])
    config['training']['input_feature_indices'] = list(input_feature_indices)
    
    if use_preinterpolated:
        print(f"Loading pre-interpolated data (correction-only mode)...")
        print(f"  Interpolated: {config['data']['interpolated_file']}")
        print(f"  Ground truth: {config['data']['fine_file']}")
        
        # Load pre-interpolated data
        interp_reader = ExodusDataLoader(config['data']['interpolated_file'])
        interpolated_data = interp_reader.load()
        interp_reader.close()
        
        # Load ground truth fine data
        fine_reader = ExodusDataLoader(config['data']['fine_file'])
        fine_data = fine_reader.load()
        fine_reader.close()
        
        # Create dataset with pre-interpolated data
        use_graph = config['model']['type'] in ['gnn', 'encoder_decoder']
        timestep_offset = config['data'].get('timestep_offset', 0)

        residual_learning = bool(config.get('training', {}).get('residual_learning', False))
        target_mode = 'residual' if residual_learning else 'absolute'
        if residual_learning:
            print("Using residual learning: target = fine - interpolated")

        # Optional physics/BC constraints (only supported for residual/correction training)
        physics_cfg = physics_config_from_training_cfg(config.get('training', {}))
        if physics_cfg.enabled and not residual_learning:
            print("WARNING: training.physics.enabled=true but residual_learning=false; disabling physics losses.")
            physics_cfg.enabled = False

        # Which side sets to use for node masks
        phys_sidesets = {}
        if physics_cfg.enabled:
            phys_sidesets = (config.get('training', {}).get('physics', {}) or {}).get('sidesets', None)
            if not isinstance(phys_sidesets, dict) or not phys_sidesets:
                # Default to cylinder mask if any cylinder BC term is used.
                phys_sidesets = {'cylinder': True}

        residual_normalization = bool(config.get('training', {}).get('residual_normalization', False))
        residual_norm_eps = float(config.get('training', {}).get('residual_norm_eps', 1e-8))
        if residual_learning and residual_normalization:
            print("Using residual normalization: (residual - mean) / std")

        smoothness_lambda = float(config.get('training', {}).get('smoothness_lambda', 0.0))
        if smoothness_lambda and smoothness_lambda > 0:
            if not use_graph:
                print("WARNING: smoothness_lambda set but model is not graph-based; ignoring smoothness.")
                smoothness_lambda = 0.0
            elif not residual_learning:
                print("WARNING: smoothness_lambda set but residual_learning is false; ignoring smoothness.")
                smoothness_lambda = 0.0
        
        dataset = MeshDataset(
            interpolated_data=interpolated_data,
            fine_data=fine_data,
            timestep_offset=timestep_offset,
            use_graph=use_graph,
            k_neighbors=config['model']['k_neighbors'],
            use_cache=config['data'].get('use_cache', True),
            cache_dir=config['data'].get('cache_dir', './cache'),
            target_mode=target_mode,
            input_feature_indices=input_feature_indices,
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
    else:
        print(f"Loading data from custom files...")
        print(f"  Coarse: {config['data']['coarse_file']}")
        print(f"  Fine: {config['data']['fine_file']}")
        
        # Load data
        coarse_reader = ExodusDataLoader(config['data']['coarse_file'])
        coarse_data = coarse_reader.load()
        coarse_reader.close()
        
        fine_reader = ExodusDataLoader(config['data']['fine_file'])
        fine_data = fine_reader.load()
        fine_reader.close()
        
        # Create dataset
        use_graph = config['model']['type'] in ['gnn', 'encoder_decoder']
        dataset = MeshDataset(
            coarse_data=coarse_data,
            fine_data=fine_data,
            use_graph=use_graph,
            k_neighbors=config['model']['k_neighbors'],
            use_cache=config['data']['use_cache'],
            cache_dir=config['data']['cache_dir'],
            input_feature_indices=input_feature_indices,
            input_normalize=input_normalization,
            input_norm_eps=input_norm_eps,
            normalize_coords=normalize_coords,
            coord_norm_eps=coord_norm_eps,
        )

        # These features are currently only intended for correction/residual training.
        smoothness_lambda = 0.0
        physics_cfg = physics_config_from_training_cfg(config.get('training', {}))
        if physics_cfg.enabled:
            print("WARNING: physics losses are currently only supported for pre-interpolated residual training; disabling.")
            physics_cfg.enabled = False
    
    print(f"Dataset size: {len(dataset)} timesteps")
    
    # Split train/validation
    val_size = int(len(dataset) * config['training']['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Create data loaders
    # Use GeometricDataLoader for graph data, regular DataLoader for tensor data
    LoaderClass = GeometricDataLoader if use_graph else DataLoader
    
    train_loader = LoaderClass(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = LoaderClass(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    ndim = int(getattr(dataset, 'fine_coords').shape[1])
    n_input_fields = len(getattr(dataset, 'input_feature_indices', [0, 1, 2]))
    in_channels = ndim + n_input_fields  # coords + selected fields
    out_channels = 4
    
    model_type = config['model']['type']
    if model_type == 'gnn':
        model = MeshGNN(
            in_channels=in_channels,
            hidden_channels=config['model']['hidden_channels'],
            out_channels=out_channels,
            num_layers=config['model']['num_layers'],
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
    elif model_type == 'encoder_decoder':
        model = MeshEncoderDecoder(
            in_channels=in_channels,
            hidden_channels=config['model']['hidden_channels'],
            out_channels=out_channels,
            num_levels=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nModel: {model_type}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(config['training']['num_epochs']):
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
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'residual_stats': getattr(dataset, 'residual_stats', None),
                'input_stats': getattr(dataset, 'input_stats', None),
                'coord_stats': getattr(dataset, 'coord_stats', None),
            }
            torch.save(
                checkpoint,
                os.path.join(config['training']['output_dir'], 'best_model.pt')
            )
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        
        # Periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'residual_stats': getattr(dataset, 'residual_stats', None),
                'input_stats': getattr(dataset, 'input_stats', None),
                'coord_stats': getattr(dataset, 'coord_stats', None),
            }
            torch.save(
                checkpoint,
                os.path.join(config['training']['output_dir'], f'checkpoint_epoch_{epoch+1}.pt')
            )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {config['training']['output_dir']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_custom.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train_custom(args.config)
