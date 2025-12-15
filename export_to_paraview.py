#!/usr/bin/env python3
"""
Export interpolated mesh data to ExodusII format for ParaView visualization.
Interpolates coarse mesh fields to fine mesh and saves as .exo file.
"""

import argparse
import sys
from pathlib import Path

sys.path.append('src')

from data_loader import ExodusDataLoader
from mesh_interpolation import interpolate_and_save_exodus


def main():
    parser = argparse.ArgumentParser(
        description='Interpolate mesh data and export to ExodusII for ParaView',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Interpolate coarse to fine mesh
  python export_to_paraview.py \\
    --coarse dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \\
    --fine dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \\
    --output paraview/coarse_to_fine_interpolated.exo
  
  # Specify fields to interpolate
  python export_to_paraview.py \\
    --coarse dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \\
    --fine dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \\
    --output paraview/velocity_only.exo \\
    --fields velocity_x velocity_y
  
  # Use RBF interpolation
  python export_to_paraview.py \\
    --coarse dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \\
    --fine dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \\
    --output paraview/rbf_interpolated.exo \\
    --method rbf

Then open the .exo file in ParaView to visualize!
        """
    )
    
    parser.add_argument('--coarse', type=str, required=True,
                       help='Path to coarse mesh ExodusII file')
    parser.add_argument('--fine', type=str, required=True,
                       help='Path to fine mesh ExodusII file (for target coordinates)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output ExodusII file path')
    parser.add_argument('--method', type=str, default='linear',
                       choices=['nearest', 'linear', 'rbf'],
                       help='Interpolation method (default: linear). '\n                            'nearest=fast but blocky; '\n                            'linear=best accuracy/speed (RECOMMENDED); '\n                            'rbf=smooth but slow, auto-subsamples >5K points')
    parser.add_argument('--fields', nargs='+', type=str, default=None,
                       help='Specific fields to interpolate (default: all)')
    parser.add_argument('--max-timesteps', type=int, default=None,
                       help='Maximum number of timesteps to process (default: all)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MESH INTERPOLATION FOR PARAVIEW")
    print("="*70)
    
    try:
        # Load coarse mesh data
        print(f"\nLoading coarse mesh: {args.coarse}")
        with ExodusDataLoader(args.coarse) as loader:
            coarse_data = loader.load()
        
        print(f"  Nodes: {coarse_data['num_nodes']:,}")
        print(f"  Fields: {[k for k in coarse_data['fields'].keys() if k != 'time_values']}")
        
        # Load fine mesh data (for coordinates)
        print(f"\nLoading fine mesh: {args.fine}")
        with ExodusDataLoader(args.fine) as loader:
            fine_data = loader.load()
        
        print(f"  Nodes: {fine_data['num_nodes']:,}")
        
        # Perform interpolation and save
        print(f"\nInterpolation method: {args.method}")
        if args.fields:
            print(f"Fields to interpolate: {args.fields}")
        else:
            print("Fields to interpolate: all available")
        
        interpolate_and_save_exodus(
            coarse_data,
            fine_data,
            args.output,
            method=args.method,
            field_names=args.fields,
            max_timesteps=args.max_timesteps
        )
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nInterpolated data saved to: {args.output}")
        print("\nNext steps:")
        print("  1. Open ParaView")
        print(f"  2. File → Open → {args.output}")
        print("  3. Click 'Apply' in Properties panel")
        print("  4. Select field in 'Coloring' dropdown")
        print("  5. Adjust color scale and visualize!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
