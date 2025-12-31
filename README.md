# Mesh Super-Resolution with Graph Neural Networks

This project implements AI models to map coarse mesh CFD simulation results to high-resolution predictions using Graph Neural Networks (GNNs).

## Overview

The model learns to enhance the resolution of flow field simulations (velocity, pressure, temperature) from coarse meshes to fine meshes. This is particularly useful for:
- Speeding up CFD simulations by running at coarse resolution and upsampling
- Generating high-resolution training data for machine learning
- Uncertainty quantification and error estimation

## Dataset Structure

Your dataset contains four mesh resolutions of oscillating heated laminar flow around a cylinder:
- **O-45-15** (coarsest): ~675 nodes
- **O-90-30**: ~2,700 nodes
- **O-180-60**: ~10,800 nodes
- **O-360-120** (finest): ~43,200 nodes

Each simulation contains:
- Velocity fields (velocity_x, velocity_y)
- Pressure field (lagrange_pressure)
- Temperature field
- Multiple timesteps

## Installation

### Prerequisites
```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch (for neural networks)
- PyTorch Geometric (for graph neural networks)
- netCDF4 (for reading ExodusII files)
- NumPy, SciPy, Matplotlib (for numerical computing and visualization)

## Usage

### 1. Training a Model

Train a GNN model to map coarse to fine resolution:

```bash
cd src
python train.py ../config.yaml
```

This will:
- Load coarse and fine mesh data from ExodusII files
- Interpolate coarse fields to fine mesh grid
- Train a GNN to learn the residual/correction
- Save trained model and training curves

**Training configurations** (edit `config.yaml`):
```yaml
data:
  source_resolution: 'coarse'    # coarse, medium, fine
  target_resolution: 'fine'       # medium, fine, finest

model:
  type: 'gnn'                     # gnn, encoder_decoder, mlp
  hidden_channels: 128
  num_layers: 4

training:
  batch_size: 1
  learning_rate: 0.001
  num_epochs: 100

  # Data loss configuration (optional)
  # - Choose which loss is optimized: 'mse' (L2) or 'huber' (SmoothL1)
  # - Optionally apply per-output-channel weights (e.g., emphasize pressure)
  data_loss_type: 'mse'                 # 'mse' or 'huber'
  data_loss_huber_delta: 1.0            # only used when data_loss_type='huber'
  data_loss_channel_weights: null       # set e.g. [1.0, 1.0, 5.0] for [vx, vy, p]
```

**Per-channel weighted loss (pressure emphasis)**

If you train a 3-output model (velocity_x, velocity_y, pressure), you can emphasize pressure by setting:

```yaml
training:
  data_loss_type: 'mse'
  data_loss_channel_weights: [1.0, 1.0, 10.0]   # [vx, vy, p]
```

To try Huber instead:

```yaml
training:
  data_loss_type: 'huber'
  data_loss_huber_delta: 1.0
  data_loss_channel_weights: [1.0, 1.0, 10.0]
```

During training, the code prints a small per-epoch breakdown of per-channel MSE (vx/vy/p) plus the weighted contributions to help you tune these weights.

### 2. Running Inference

Apply trained model to new coarse mesh data:

```bash
python inference.py \
  --coarse ../dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine_coords ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --model ../outputs/best_model.pt \
  --timestep 0 \
  --output_dir ../predictions
```

**With ground truth comparison:**
```bash
python inference.py \
  --coarse ../dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine_coords ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --ground_truth ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --model ../outputs/best_model.pt \
  --timestep 0 \
  --output_dir ../predictions
```

This generates:
- Predicted high-resolution fields
- Visualization plots comparing coarse interpolation vs. model prediction
- Error analysis (if ground truth provided)

### 3. Evaluating a Correction Model (per-timestep metrics)

If you trained a correction/residual model (pre-interpolated workflow), you can evaluate errors per timestep and per variable:

```bash
source ~/module_load.sh
python3.12 evaluate_correction_model.py --config config_preinterpolated_physics.yaml --device cuda
```

This writes a long-format CSV and a JSON summary under the evaluation folder (typically under an outputs directory such as `outputs_correction_physics/evaluation/`):
- `errors_by_variable_and_timestep.csv`
- `summary_mean_errors.json`

The CSV includes metrics like `rmse`, `mae`, plus min/max diagnostics and relative errors at extrema.

### 4. Plotting the Evaluation CSV

Use the included parser/plotter script to generate diagnostic plots:

```bash
source ~/module_load.sh
python parse_evaluation_csv.py --csv outputs_correction_physics/evaluation/errors_by_variable_and_timestep.csv
```

Optionally, you can also compare extrema values directly between the **original coarse** mesh and the **fine** mesh (no interpolation, no node mapping) to diagnose baseline bias:

```bash
source ~/module_load.sh
python parse_evaluation_csv.py \
  --csv outputs_correction_physics/evaluation/errors_by_variable_and_timestep.csv \
  --yaml-config config_preinterpolated_physics.yaml \
  --coarse-file dataset/002-Re-148_3-AC-beta-10000-Helios/90-30/cropped.e
```

If coarse/fine timesteps are not aligned, you can shift the coarse index with `--coarse-to-fine-offset` (uses `coarse[t + offset]` vs `fine[t]`).

This writes PNGs next to the CSV, including:
- `rmse_by_variable_over_time.png`
- `minmax_true_vs_model.png`
- `relerr_at_true_extrema_model.png`
- `relerr_of_extrema_values_model.png`
- `relerr_of_extrema_values_interpolated.png`
- `bias_by_variable_over_time.png` (if present in CSV)
- `mean_relative_error_by_variable_over_time.png` (if present in CSV)
- `relerr_of_extrema_values_coarse_vs_fine.png` (when using `--yaml-config` + `--coarse-file`)

## Model Architecture

### 1. MeshGNN (Graph Neural Network)
- Uses Graph Attention Networks (GAT) for message passing
- Learns spatial relationships between mesh nodes
- Best for irregular meshes with complex geometries

**Architecture:**
```
Input (coords + fields) → MLP Embedding → GAT Layers (with residuals) → Output MLP
```

### 2. MeshEncoderDecoder
- U-Net style encoder-decoder with skip connections
- Multi-scale feature extraction
- Good for capturing both local and global patterns

### 3. SimpleMLP (Baseline)
- Point-wise MLP without graph structure
- Useful as a baseline for comparison

## How It Works

### Training Pipeline
1. **Data Loading**: Read ExodusII files containing mesh coordinates and field variables
2. **Interpolation**: Map coarse mesh fields to fine mesh nodes using linear interpolation
3. **Graph Construction**: Build k-nearest neighbor graph from fine mesh coordinates
4. **Model Training**: GNN learns to predict high-res fields from interpolated coarse fields
5. **Loss**: Configurable MSE or Huber (SmoothL1), with optional per-channel weights

### Key Insight
The model learns the **residual correction** that standard interpolation methods miss:
```
Ground Truth = Linear Interpolation + GNN Correction
```

## File Structure

```
VERTEX/
├── src/
│   ├── data_loader.py          # ExodusII file loading utilities
│   ├── mesh_interpolation.py   # Mesh interpolation and alignment
│   ├── models.py                # GNN model architectures
│   ├── train.py                 # Training script
│   └── inference.py             # Inference and visualization
├── dataset/                     # Your simulation data
│   ├── 071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/
│   ├── 072-Re-148_3-EDAC-beta-10000-O-90-30-Helios/
│   ├── 073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/
│   └── 074-Re-148_3-EDAC-beta-10000-O-360-120-Helios/
├── config.yaml                  # Training configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Example Workflow

### Complete Training and Evaluation
```bash
# 1. Train model (coarse → fine)
cd src
python train.py ../config.yaml

# 2. Run inference on validation data
python inference.py \
  --coarse ../dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine_coords ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --ground_truth ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --model ../outputs/best_model.pt \
  --output_dir ../predictions

# 3. Check results
ls ../predictions/
# velocity_x_comparison.png, velocity_y_comparison.png, etc.
```

### Training Different Resolution Pairs

**Coarse → Medium:**
```yaml
# config.yaml
source_resolution: 'coarse'
target_resolution: 'medium'
```

**Medium → Fine:**
```yaml
source_resolution: 'medium'
target_resolution: 'fine'
```

**Coarse → Finest (most challenging):**
```yaml
source_resolution: 'coarse'
target_resolution: 'finest'
```

## Performance Tips

### For Better Results:
1. **More training data**: Use all available timesteps
2. **Larger model**: Increase `hidden_channels` to 256 or 512
3. **More layers**: Increase `num_layers` to 6-8
4. **Data augmentation**: Add noise to inputs during training
5. **Physics-informed loss**: Add conservation law constraints

### For Faster Training:
1. **Smaller batch size**: Set to 1 for graph data
2. **Fewer epochs**: Start with 50 epochs
3. **Use GPU**: Ensure CUDA is available
4. **Reduce k_neighbors**: Use 6 instead of 8

## Troubleshooting

### "No module named 'netCDF4'"
```bash
pip install netCDF4
```

### "CUDA out of memory"
- Reduce `hidden_channels` in config
- Reduce `batch_size` to 1
- Use CPU instead: model will run on CPU automatically if CUDA unavailable

### "No matching fields found"
Check that ExodusII files contain expected variable names. Modify `field_mappings` in `data_loader.py` if needed.

## Advanced Usage

### Custom Model Architecture
Edit `src/models.py` to add your own architecture:
```python
class CustomModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Your architecture
    
    def forward(self, x, edge_index):
        # Your forward pass
        return output
```

### Multi-Fidelity Training
Train on multiple resolution pairs simultaneously by modifying the dataset loader.

### Physics-Informed Training
Add physics constraints (e.g., divergence-free velocity) to the loss function.

## Citation

If you use this code for research, please cite the relevant paper on oscillating flow simulations.

## License

This code is provided for research purposes.

## Contact

For questions or issues, please check the code comments or create an issue.
