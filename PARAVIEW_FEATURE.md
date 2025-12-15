# ParaView Export Feature - Summary

## What's New

Added **ExodusII export functionality** to save interpolated mesh data for visualization in ParaView!

## New Capabilities

### 1. Export Interpolated Data
Save interpolated fields from coarse to fine mesh as `.exo` files:
```bash
python export_to_paraview.py \
  --coarse dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --output paraview/interpolated.exo
```

### 2. Export Model Predictions
Save model predictions alongside baseline interpolation:
```bash
python inference.py \
  --coarse ... \
  --fine_coords ... \
  --model outputs/best_model.pt \
  --save_exodus \
  --output_dir paraview_results
```

### 3. Visualize in ParaView
- Open `.exo` files directly in ParaView
- Interactive 3D/2D visualization
- Compare interpolated vs. predicted fields
- Create animations, contours, streamlines
- Export publication-quality figures

## Files Added/Modified

### New Files
1. **`export_to_paraview.py`** - Standalone script for exporting interpolated data
2. **`PARAVIEW_GUIDE.md`** - Comprehensive visualization guide

### Modified Files
1. **`mesh_interpolation.py`** - Added:
   - `save_interpolated_to_exodus()` - Core export function
   - `interpolate_and_save_exodus()` - Convenience wrapper

2. **`inference.py`** - Added:
   - `--save_exodus` flag
   - Automatic ExodusII export of predictions

## Usage Examples

### Basic Interpolation Export
```bash
# Interpolate and export
python export_to_paraview.py \
  --coarse dataset/071*/restart/solution.exo \
  --fine dataset/073*/restart/solution.exo \
  --output paraview/basic.exo

# Open in ParaView
paraview paraview/basic.exo
```

### Export Specific Fields
```bash
# Only velocity
python export_to_paraview.py \
  --coarse dataset/071*/restart/solution.exo \
  --fine dataset/073*/restart/solution.exo \
  --output paraview/velocity.exo \
  --fields velocity_x velocity_y
```

### Different Interpolation Methods
```bash
# Linear (default)
--method linear

# RBF (smoother)
--method rbf

# Nearest neighbor (fastest)
--method nearest
```

### With Model Predictions
```bash
cd src
python inference.py \
  --coarse ../dataset/071*/restart/solution.exo \
  --fine_coords ../dataset/073*/restart/solution.exo \
  --model ../outputs/best_model.pt \
  --save_exodus \
  --output_dir ../paraview_results

# Creates: paraview_results/predictions_timestep_0.exo
# Contains: *_interpolated and *_predicted fields
```

## What Gets Exported

### Field Names
**From interpolation:**
- `velocity_x`, `velocity_y`
- `pressure`
- `temperature`

**From model predictions:**
- `velocity_x_predicted`, `velocity_x_interpolated`
- `velocity_y_predicted`, `velocity_y_interpolated`
- `pressure_predicted`, `pressure_interpolated`
- `temperature_predicted`, `temperature_interpolated`

### File Contents
- Node coordinates (x, y, z)
- Field values at all nodes
- Multiple timesteps (if available)
- Element connectivity (optional)

## ParaView Workflow

1. **Open file**: File → Open → `*.exo`
2. **Apply**: Click Apply button
3. **Color by field**: Select field from dropdown
4. **Customize**: Adjust color map, representation
5. **Compare**: Create split view for side-by-side
6. **Export**: Save screenshot or animation

## Benefits

✅ **Visual validation** - See interpolation quality  
✅ **Interactive exploration** - Zoom, rotate, inspect  
✅ **Side-by-side comparison** - Interpolated vs. predicted  
✅ **Publication figures** - High-quality screenshots  
✅ **Animations** - Timestep sequences  
✅ **Advanced analysis** - Gradients, streamlines, contours  

## Example Workflow

```bash
# 1. Train model
python src/train.py config.yaml

# 2. Generate predictions with ParaView export
python src/inference.py \
  --coarse dataset/071*/restart/solution.exo \
  --fine_coords dataset/073*/restart/solution.exo \
  --model outputs/best_model.pt \
  --save_exodus \
  --output_dir paraview_output

# 3. Visualize
paraview paraview_output/predictions_timestep_0.exo

# In ParaView:
# - Split view (horizontal)
# - Left: Color by velocity_x_interpolated
# - Right: Color by velocity_x_predicted
# - Compare the difference!
```

## Documentation

- **Quick reference**: This file
- **Detailed guide**: `PARAVIEW_GUIDE.md`
- **Command help**: `python export_to_paraview.py --help`

## Notes

- ExodusII is the standard format for CFD visualization
- ParaView is free, open-source, and widely used
- Files can also be opened in Visit, Tecplot, etc.
- Supports point-based visualization (no elements needed)
- All fields and timesteps included automatically

## Quick Command Reference

```bash
# Export interpolated data
python export_to_paraview.py --coarse COARSE.exo --fine FINE.exo --output OUT.exo

# Export with model predictions
python src/inference.py --coarse COARSE.exo --fine_coords FINE.exo --model MODEL.pt --save_exodus

# Open in ParaView
paraview OUTPUT.exo
```

Your mesh data is now ready for professional visualization! 🎨
