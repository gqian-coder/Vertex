"""
Inference script for trained mesh super-resolution models.
Applies trained model to new coarse mesh data to generate fine resolution predictions.
"""

import torch
import numpy as np
from pathlib import Path
import yaml
import argparse
from typing import Dict
import matplotlib.pyplot as plt

from data_loader import ExodusDataLoader
from mesh_interpolation import MeshInterpolator, save_interpolated_to_exodus
from models import MeshGNN, MeshEncoderDecoder, SimpleMLP, create_graph_data


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Reconstruct model
    ndim = 2  # Assuming 2D
    in_channels = ndim + 4
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def inference(coarse_file: str, fine_coords: np.ndarray, model_path: str, 
             timestep: int = 0, output_dir: str = './predictions',
             save_exodus: bool = False, fine_connectivity: np.ndarray = None):
    """
    Run inference on coarse mesh data.
    
    Args:
        coarse_file: Path to coarse mesh ExodusII file
        fine_coords: Fine mesh coordinates for target resolution
        model_path: Path to trained model checkpoint
        timestep: Which timestep to use
        output_dir: Directory to save predictions
        save_exodus: Whether to save predictions as ExodusII for ParaView
        fine_connectivity: Element connectivity for ExodusII output (optional)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, config = load_model(model_path, device)
    use_graph = config['model']['type'] in ['gnn', 'encoder_decoder']
    
    # Load coarse data
    print(f"Loading coarse data from {coarse_file}...")
    with ExodusDataLoader(coarse_file) as loader:
        coarse_data = loader.load()
        coarse_coords, coarse_fields = loader.get_snapshot(timestep)
    
    print(f"Coarse mesh: {len(coarse_coords)} nodes")
    print(f"Fine mesh: {len(fine_coords)} nodes")
    
    # Interpolate coarse to fine mesh
    print("Interpolating coarse to fine mesh...")
    interpolator = MeshInterpolator(coarse_coords, fine_coords, method='linear')
    
    # Collect and interpolate fields
    field_names = ['velocity_x', 'velocity_y', 'pressure', 'temperature']
    coarse_features = []
    
    for field_name in field_names:
        if field_name in coarse_fields:
            coarse_features.append(coarse_fields[field_name])
    
    coarse_features = np.stack(coarse_features, axis=-1)
    coarse_interp = interpolator.interpolate(coarse_features)
    
    # Prepare input
    if use_graph:
        data = create_graph_data(
            fine_coords, coarse_interp, 
            k=config['model'].get('k_neighbors', 8)
        )
        data = data.to(device)
    else:
        x = np.concatenate([fine_coords, coarse_interp], axis=-1)
        x = torch.FloatTensor(x).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        if use_graph:
            prediction = model(data.x, data.edge_index)
        else:
            prediction = model(x)
    
    prediction = prediction.cpu().numpy()
    
    # Save predictions
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / f'predictions_timestep_{timestep}.npz',
        coordinates=fine_coords,
        coarse_interpolated=coarse_interp,
        prediction=prediction,
        field_names=field_names
    )
    
    print(f"Predictions saved to {output_dir}")
    
    # Save as ExodusII for ParaView if requested
    if save_exodus:
        print("\nSaving predictions as ExodusII for ParaView...")
        
        # Prepare fields dictionary
        exodus_fields = {}
        for i, field_name in enumerate(field_names):
            # Model prediction
            exodus_fields[f'{field_name}_predicted'] = prediction[:, i]
            # Coarse interpolation baseline
            exodus_fields[f'{field_name}_interpolated'] = coarse_interp[:, i]
        
        exodus_file = output_dir / f'predictions_timestep_{timestep}.exo'
        save_interpolated_to_exodus(
            fine_coords,
            exodus_fields,
            str(exodus_file),
            connectivity=fine_connectivity,
            timesteps=[float(timestep)]
        )
        print(f"\nExodusII file ready for ParaView: {exodus_file}")
    
    # Compute error if ground truth available
    return {
        'coordinates': fine_coords,
        'coarse_interpolated': coarse_interp,
        'prediction': prediction,
        'field_names': field_names
    }


def visualize_results(results: Dict, output_dir: str = './predictions'):
    """
    Visualize inference results.
    
    Args:
        results: Results dictionary from inference
        output_dir: Directory to save plots
    """
    coords = results['coordinates']
    coarse_interp = results['coarse_interpolated']
    prediction = results['prediction']
    field_names = results['field_names']
    
    output_dir = Path(output_dir)
    
    # Plot each field
    for i, field_name in enumerate(field_names):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Coarse interpolated
        scatter1 = axes[0].scatter(
            coords[:, 0], coords[:, 1], 
            c=coarse_interp[:, i], 
            cmap='viridis', s=1
        )
        axes[0].set_title(f'{field_name} - Coarse Interpolated')
        axes[0].set_aspect('equal')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Model prediction
        scatter2 = axes[1].scatter(
            coords[:, 0], coords[:, 1], 
            c=prediction[:, i], 
            cmap='viridis', s=1
        )
        axes[1].set_title(f'{field_name} - Model Prediction')
        axes[1].set_aspect('equal')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{field_name}_comparison.png', dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def compare_with_ground_truth(predictions: Dict, ground_truth_file: str, 
                              timestep: int = 0, output_dir: str = './predictions'):
    """
    Compare predictions with ground truth fine mesh data.
    
    Args:
        predictions: Predictions dictionary
        ground_truth_file: Path to ground truth ExodusII file
        timestep: Timestep to compare
        output_dir: Output directory
    """
    # Load ground truth
    print(f"Loading ground truth from {ground_truth_file}...")
    with ExodusDataLoader(ground_truth_file) as loader:
        gt_data = loader.load()
        gt_coords, gt_fields = loader.get_snapshot(timestep)
    
    # Extract predictions
    pred = predictions['prediction']
    coarse_interp = predictions['coarse_interpolated']
    field_names = predictions['field_names']
    
    # Collect ground truth
    gt_values = []
    for field_name in field_names:
        if field_name in gt_fields:
            gt_values.append(gt_fields[field_name])
    gt_values = np.stack(gt_values, axis=-1)
    
    # Compute errors
    print("\nError Analysis:")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    for i, field_name in enumerate(field_names):
        # Coarse interpolation error
        coarse_error = np.abs(coarse_interp[:, i] - gt_values[:, i])
        coarse_rmse = np.sqrt(np.mean(coarse_error ** 2))
        coarse_rel_error = np.linalg.norm(coarse_error) / np.linalg.norm(gt_values[:, i])
        
        # Model prediction error
        pred_error = np.abs(pred[:, i] - gt_values[:, i])
        pred_rmse = np.sqrt(np.mean(pred_error ** 2))
        pred_rel_error = np.linalg.norm(pred_error) / np.linalg.norm(gt_values[:, i])
        
        # Improvement
        improvement = (coarse_rmse - pred_rmse) / coarse_rmse * 100
        
        print(f"\n{field_name}:")
        print(f"  Coarse RMSE: {coarse_rmse:.6f}, Relative Error: {coarse_rel_error:.4f}")
        print(f"  Model RMSE:  {pred_rmse:.6f}, Relative Error: {pred_rel_error:.4f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        # Plot error comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Ground truth
        scatter0 = axes[0, 0].scatter(
            gt_coords[:, 0], gt_coords[:, 1],
            c=gt_values[:, i], cmap='viridis', s=1
        )
        axes[0, 0].set_title(f'{field_name} - Ground Truth')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(scatter0, ax=axes[0, 0])
        
        # Model prediction
        scatter1 = axes[0, 1].scatter(
            gt_coords[:, 0], gt_coords[:, 1],
            c=pred[:, i], cmap='viridis', s=1
        )
        axes[0, 1].set_title(f'{field_name} - Model Prediction')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(scatter1, ax=axes[0, 1])
        
        # Coarse interpolation error
        scatter2 = axes[1, 0].scatter(
            gt_coords[:, 0], gt_coords[:, 1],
            c=coarse_error, cmap='Reds', s=1
        )
        axes[1, 0].set_title(f'Coarse Interpolation Error (RMSE: {coarse_rmse:.4f})')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(scatter2, ax=axes[1, 0])
        
        # Model prediction error
        scatter3 = axes[1, 1].scatter(
            gt_coords[:, 0], gt_coords[:, 1],
            c=pred_error, cmap='Reds', s=1
        )
        axes[1, 1].set_title(f'Model Prediction Error (RMSE: {pred_rmse:.4f})')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(scatter3, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{field_name}_error_comparison.png', dpi=150)
        plt.close()
    
    print(f"\nError comparison plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--coarse', type=str, required=True, help='Path to coarse mesh file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--fine_coords', type=str, default=None, 
                       help='Path to fine mesh file (for coordinates)')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='Path to ground truth file for comparison')
    parser.add_argument('--timestep', type=int, default=0, help='Timestep to process')
    parser.add_argument('--output_dir', type=str, default='./predictions', 
                       help='Output directory')
    parser.add_argument('--save_exodus', action='store_true',
                       help='Save predictions as ExodusII file for ParaView')
    
    args = parser.parse_args()
    
    # Load fine mesh coordinates
    if args.fine_coords:
        with ExodusDataLoader(args.fine_coords) as loader:
            fine_data = loader.load()
            fine_coords = fine_data['coordinates']
            fine_connectivity = fine_data.get('connectivity', None)
    else:
        raise ValueError("Must provide fine mesh file for coordinates")
    
    # Run inference
    results = inference(
        args.coarse,
        fine_coords,
        args.model,
        timestep=args.timestep,
        output_dir=args.output_dir,
        save_exodus=args.save_exodus,
        fine_connectivity=fine_connectivity
    )
    
    # Visualize
    visualize_results(results, output_dir=args.output_dir)
    
    # Compare with ground truth if available
    if args.ground_truth:
        compare_with_ground_truth(
            results,
            args.ground_truth,
            timestep=args.timestep,
            output_dir=args.output_dir
        )
