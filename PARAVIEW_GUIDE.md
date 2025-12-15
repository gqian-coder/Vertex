# ParaView Visualization Guide

## Overview

You can now export interpolated mesh data and model predictions to **ExodusII format** for visualization in ParaView. This allows you to:
- View interpolated fields on fine mesh
- Compare coarse vs. fine resolution visually
- Analyze model predictions interactively
- Create publication-quality figures
- Animate timestep sequences

## Quick Start

### Option 1: Export Interpolated Data (No Model Required)

```bash
# Interpolate coarse mesh to fine resolution and save for ParaView
python export_to_paraview.py \
  --coarse dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --output paraview/coarse_to_fine_interpolated.exo
```

**What this does:**
- Loads coarse mesh simulation data
- Interpolates all fields to fine mesh coordinates
- Saves as `.exo` file that ParaView can open
- Includes all timesteps and all field variables

### Option 2: Export Model Predictions

```bash
# Run inference and save predictions as ExodusII
cd src
python inference.py \
  --coarse ../dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine_coords ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --model ../outputs/best_model.pt \
  --timestep 0 \
  --output_dir ../paraview_predictions \
  --save_exodus
```

**What this does:**
- Runs trained model on coarse mesh
- Generates high-resolution predictions
- Saves BOTH interpolated baseline AND model predictions
- Creates `.exo` file with both for comparison

## Using ParaView

### Step 1: Open ExodusII File

1. Launch ParaView
2. **File → Open**
3. Navigate to your `.exo` file
4. Click **Apply** in the Properties panel

### Step 2: Visualize Fields

**In the Properties panel:**
1. **Coloring**: Select field to visualize (e.g., `velocity_x`)
2. **Representation**: Choose `Surface` or `Points`
3. **Color Map**: Adjust using color scale editor

**Common fields:**
- `velocity_x` - X-component of velocity
- `velocity_y` - Y-component of velocity  
- `pressure` - Pressure field
- `temperature` - Temperature field
- `*_predicted` - Model predictions (if using inference output)
- `*_interpolated` - Baseline interpolation (if using inference output)

### Step 3: Enhance Visualization

**Add velocity vectors:**
1. **Filters → Common → Glyph**
2. Set Glyph Type to `Arrow`
3. Scale Mode: `vector`
4. Vectors: Choose `velocity`

**Add contours:**
1. **Filters → Common → Contour**
2. Select field (e.g., `pressure`)
3. Set number of contours

**Add streamlines:**
1. **Filters → Common → Stream Tracer**
2. Vectors: `velocity`
3. Seed Type: `Point Source` or `Line`

### Step 4: Side-by-Side Comparison

**To compare interpolated vs predicted:**
1. Open the `.exo` file
2. Create a split view: **Split Horizontal** icon
3. Left view: Color by `velocity_x_interpolated`
4. Right view: Color by `velocity_x_predicted`
5. Use same color scale for both (link color maps)

## Advanced Usage

### Export Specific Fields Only

```bash
# Only velocity components
python export_to_paraview.py \
  --coarse dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --output paraview/velocity_only.exo \
  --fields velocity_x velocity_y
```

### Use Different Interpolation Methods

```bash
# RBF interpolation (smoother, slower)
python export_to_paraview.py \
  --coarse dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --output paraview/rbf_interpolated.exo \
  --method rbf

# Nearest neighbor (fastest, less smooth)
python export_to_paraview.py \
  --coarse ... --fine ... --output paraview/nearest.exo --method nearest
```

### Programmatic Export (Python API)

```python
from data_loader import load_all_resolutions
from mesh_interpolation import interpolate_and_save_exodus

# Load data
all_data = load_all_resolutions('./dataset')

# Export coarse → fine
interpolate_and_save_exodus(
    all_data['coarse'],
    all_data['fine'],
    output_file='paraview/my_visualization.exo',
    method='linear'
)
```

## Visualization Recipes

### Recipe 1: Velocity Magnitude

**In ParaView:**
1. Open `.exo` file
2. **Filters → Alphabetical → Calculator**
3. Result Array Name: `velocity_magnitude`
4. Formula: `sqrt(velocity_x^2 + velocity_y^2)`
5. Apply
6. Color by `velocity_magnitude`

### Recipe 2: Vorticity (2D)

```python
# In ParaView Python shell or Calculator:
# ω = ∂v/∂x - ∂u/∂y
1. Apply Gradient filter to velocity
2. Use Calculator: `Gradients[:,1,0] - Gradients[:,0,1]`
```

### Recipe 3: Animation

1. View → Animation View
2. Set time range (if multiple timesteps)
3. Play animation
4. **File → Save Animation** to export video

### Recipe 4: Publication Figure

1. Set white background: View → Background → White
2. Hide orientation axes
3. Adjust color map: Show advanced properties
4. **File → Save Screenshot** (high DPI)
5. Recommended: 300 DPI, transparent background

## Field Names Reference

### Original Fields (from simulation)
- `velocity_x`, `velocity_y` - Velocity components
- `pressure` or `lagrange_pressure` - Pressure
- `temperature` - Temperature

### From Model Predictions
- `velocity_x_predicted` - Model's velocity X prediction
- `velocity_x_interpolated` - Baseline interpolation
- `velocity_y_predicted` - Model's velocity Y prediction
- `velocity_y_interpolated` - Baseline interpolation
- `pressure_predicted` - Model's pressure prediction
- `pressure_interpolated` - Baseline interpolation
- `temperature_predicted` - Model's temperature prediction
- `temperature_interpolated` - Baseline interpolation

## Troubleshooting

### Issue: "Cannot open file"
**Cause:** File path incorrect or file doesn't exist
**Solution:**
```bash
# Check file exists
ls -lh paraview/*.exo

# Use absolute path
python export_to_paraview.py ... --output /full/path/to/output.exo
```

### Issue: "No fields visible"
**Cause:** Fields not properly named or saved
**Solution:**
1. Check available fields: Click "Information" tab in ParaView
2. Verify field names in properties panel
3. Re-export with explicit field names

### Issue: "Mesh looks wrong"
**Cause:** No element connectivity in file
**Solution:** Element connectivity is optional for point-based visualization. Use "Point Gaussian" representation.

### Issue: "Colors don't match between views"
**Cause:** Different color scale ranges
**Solution:**
1. Edit color map
2. Set "Rescale Range" to custom
3. Apply same range to both views
4. Or use "Separate Color Map" option

## File Size Considerations

**Typical file sizes:**
- Coarse (675 nodes, 10 timesteps): ~1-5 MB
- Fine (10,800 nodes, 10 timesteps): ~10-50 MB
- Finest (43,200 nodes, 10 timesteps): ~40-200 MB

**Tips for large files:**
- Export only needed fields (`--fields` option)
- Export single timestep if not animating
- Use compression (ParaView handles this automatically)
- Store on fast storage (SSD, not network drive)

## Integration with Workflow

### Complete Analysis Pipeline

```bash
# 1. Train model
cd src
python train.py ../config.yaml

# 2. Run inference with ParaView export
python inference.py \
  --coarse ../dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine_coords ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --model ../outputs/best_model.pt \
  --save_exodus \
  --output_dir ../paraview_results

# 3. Open in ParaView
# paraview ../paraview_results/predictions_timestep_0.exo

# 4. Compare visually
#    - Left panel: velocity_x_interpolated (baseline)
#    - Right panel: velocity_x_predicted (model)
```

### Batch Export Multiple Timesteps

```bash
# Export all timesteps
for t in {0..9}; do
  python inference.py \
    --coarse ... \
    --fine_coords ... \
    --model ... \
    --timestep $t \
    --save_exodus \
    --output_dir ../paraview/timestep_$t
done

# Then open all in ParaView and animate
```

## Best Practices

1. **Consistent naming**: Use descriptive output filenames
2. **Organize outputs**: Keep ParaView files in dedicated directory
3. **Document settings**: Note interpolation method and parameters
4. **Compare methods**: Export both `linear` and `rbf` to compare
5. **Verify data**: Check field ranges match expectations
6. **Save state**: In ParaView, save state file for reproducibility

## Example Visualizations

### Compare Resolution Levels
```bash
# Export all resolutions to same coordinates (finest)
for res in coarse medium fine; do
  python export_to_paraview.py \
    --coarse dataset/07*$res*/restart/solution.exo \
    --fine dataset/074-*/restart/solution.exo \
    --output paraview/${res}_to_finest.exo
done

# Open all in ParaView, create multi-view comparison
```

### Error Visualization
```python
# In ParaView Calculator:
# Compute prediction error
abs(velocity_x_predicted - velocity_x_interpolated)

# Color by error to see where model improves
```

## Summary

**Export commands:**
- **Interpolation only**: `python export_to_paraview.py --coarse ... --fine ... --output ...`
- **With model predictions**: `python inference.py ... --save_exodus`

**ParaView workflow:**
1. Open `.exo` file
2. Apply
3. Color by field
4. Customize visualization
5. Save screenshot/animation

**Key advantages:**
- ✅ Interactive 3D exploration
- ✅ Multiple fields simultaneously  
- ✅ Side-by-side comparison
- ✅ High-quality figures
- ✅ Animation support

Your interpolated data and model predictions are now ready for professional visualization! 🎨
