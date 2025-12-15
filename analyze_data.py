"""
Utility script to examine and compare different mesh resolutions.
Provides statistics and visualization of the mesh hierarchy.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('src')
from data_loader import load_all_resolutions


def analyze_meshes(dataset_path: str):
    """Analyze all mesh resolutions."""
    print("=" * 70)
    print("MESH RESOLUTION ANALYSIS")
    print("=" * 70)
    
    # Load all resolutions
    all_data = load_all_resolutions(dataset_path)
    
    if not all_data:
        print("No data loaded! Check dataset path.")
        return
    
    # Print statistics
    print("\n" + "-" * 70)
    print("MESH STATISTICS")
    print("-" * 70)
    
    stats = {}
    for res_name, data in sorted(all_data.items()):
        num_nodes = data['num_nodes']
        num_elements = data['num_elements']
        coords = data['coordinates']
        
        # Compute mesh statistics
        stats[res_name] = {
            'num_nodes': num_nodes,
            'num_elements': num_elements,
            'spatial_extent': {
                'x_range': (coords[:, 0].min(), coords[:, 0].max()),
                'y_range': (coords[:, 1].min(), coords[:, 1].max()),
            },
            'num_timesteps': len(data['time_steps']) if 'time_steps' in data['fields'] else 0,
            'available_fields': [k for k in data['fields'].keys() if k != 'time_values']
        }
        
        print(f"\n{res_name.upper()}:")
        print(f"  Nodes: {num_nodes:,}")
        print(f"  Elements: {num_elements:,}")
        print(f"  X range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
        print(f"  Y range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
        print(f"  Timesteps: {stats[res_name]['num_timesteps']}")
        print(f"  Fields: {', '.join(stats[res_name]['available_fields'])}")
    
    # Visualize mesh hierarchy
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, (res_name, data) in enumerate(sorted(all_data.items())):
        coords = data['coordinates']
        ax = axes[idx]
        
        # Plot mesh points
        ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.5)
        ax.set_title(f'{res_name.capitalize()} ({data["num_nodes"]:,} nodes)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(dataset_path).parent / 'mesh_comparison.png'
    plt.savefig(output_file, dpi=150)
    print(f"Mesh visualization saved to: {output_file}")
    plt.close()
    
    # Visualize a sample field
    if 'velocity_x' in all_data['coarse']['fields']:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(sorted(all_data.items())):
            coords = data['coordinates']
            field = data['fields']['velocity_x']
            
            # Use first timestep
            if field.ndim == 2:
                field = field[0]
            
            ax = axes[idx]
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], 
                c=field, cmap='viridis', s=1
            )
            ax.set_title(f'{res_name.capitalize()} - Velocity X')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        output_file = Path(dataset_path).parent / 'field_comparison.png'
        plt.savefig(output_file, dpi=150)
        print(f"Field visualization saved to: {output_file}")
        plt.close()
    
    # Print resolution ratios
    print("\n" + "-" * 70)
    print("RESOLUTION RATIOS")
    print("-" * 70)
    
    resolutions = list(sorted(all_data.keys()))
    for i in range(len(resolutions) - 1):
        coarse = resolutions[i]
        fine = resolutions[i + 1]
        ratio = all_data[fine]['num_nodes'] / all_data[coarse]['num_nodes']
        print(f"{coarse} → {fine}: {ratio:.2f}x nodes")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze mesh resolutions')
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/002-Re-148_3-AC-beta-10000-Helios/',
        help='Path to dataset directory'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_meshes(args.dataset)
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
