# Quick Start Guide - Mesh Super-Resolution

## What This Does

This AI model learns to enhance coarse mesh CFD simulations to high-resolution results. Instead of running expensive fine mesh simulations, you can:
1. Run simulations on a coarse mesh (fast)
2. Use the trained AI model to predict what the fine mesh results would be
3. Get high-resolution predictions in seconds

## Installation (One-Time Setup)

```bash
cd /lustre/orion/csc143/proj-shared/gongq/frontier/VERTEX

# Install dependencies
pip install --user -r requirements.txt
```

**Note:** If you don't have PyTorch or PyTorch Geometric installed, you may need to install them with CUDA support:
```bash
pip install --user torch torchvision torchaudio
pip install --user torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Quick Test

Verify everything works:
```bash
./test_setup.sh
```

Or test individual components:
```bash
cd src
python data_loader.py ../dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo
```

## Usage Scenarios

### Scenario 1: Explore Your Data

```bash
python analyze_data.py --dataset ./dataset
```

This will:
- Show statistics for each mesh resolution
- Generate visualizations comparing mesh sizes
- Display sample velocity fields at different resolutions
- Save `mesh_comparison.png` and `field_comparison.png`

### Scenario 2: See Interpolation Baseline

```bash
python example_workflow.py
```

This demonstrates:
- Loading ExodusII data
- Comparing interpolation methods (nearest, linear)
- Computing interpolation errors
- Visualizing where errors are largest
- Preparing graph data for GNN training

### Scenario 3: Train Your AI Model

**Option A: Use default settings**
```bash
cd src
python train.py ../config.yaml
```

**Option B: Customize training**
Edit `config.yaml` first:
```yaml
data:
  source_resolution: 'coarse'    # Starting resolution
  target_resolution: 'fine'       # Target resolution

model:
  type: 'gnn'                     # Options: gnn, encoder_decoder, mlp
  hidden_channels: 128            # Model capacity
  num_layers: 4                   # Depth

training:
  num_epochs: 100                 # Training iterations
  learning_rate: 0.001
  output_dir: './outputs'         # Where to save model
```

Then run:
```bash
cd src
python train.py ../config.yaml
```

**What happens during training:**
- Loads coarse and fine mesh data
- Creates training/validation split (80/20)
- Trains GNN to predict fine mesh fields
- Saves best model based on validation loss
- Generates training curve plot

**Expected output:**
```
Loading data...
Loaded 4 resolutions: ['coarse', 'medium', 'fine', 'finest']
Dataset created with X samples
Train samples: Y, Val samples: Z

Creating model...
Model: gnn, Parameters: 123,456

Starting training...
Epoch 1/100
Training: 100%|████████| Loss: 0.001234
Train Loss: 0.001234, Val Loss: 0.001456
Saved best model

...

Training completed! Best val loss: 0.000789
Model saved to ./outputs
```

### Scenario 4: Use Trained Model for Predictions

```bash
cd src
python inference.py \
  --coarse ../dataset/071-Re-148_3-EDAC-beta-10000-O-45-15-Helios/restart/solution.exo \
  --fine_coords ../dataset/073-Re-148_3-EDAC-beta-10000-O-180-60-Helios/restart/solution.exo \
  --model ../outputs/best_model.pt \
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

**Output files:**
```
predictions/
├── predictions_timestep_0.npz           # Numerical predictions
├── velocity_x_comparison.png            # Visual comparison
├── velocity_y_comparison.png
├── pressure_comparison.png
├── temperature_comparison.png
└── *_error_comparison.png               # Error analysis (if ground truth provided)
```

## Understanding the Results

### Training Metrics
- **Train Loss**: How well model fits training data
- **Val Loss**: How well model generalizes to unseen data
- **Lower is better** for both
- **Watch for overfitting**: If train loss decreases but val loss increases, model is memorizing

### Inference Metrics (with ground truth)
- **RMSE**: Root Mean Square Error (lower is better)
- **Relative Error**: Error normalized by field magnitude
- **Improvement**: How much better than linear interpolation (% improvement)

**Example output:**
```
velocity_x:
  Coarse RMSE: 0.012345, Relative Error: 0.0456
  Model RMSE:  0.003456, Relative Error: 0.0128
  Improvement: 72.0%
```

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'netCDF4'"
**Solution:**
```bash
pip install --user netCDF4
```

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install --user torch torch-geometric
```

### Problem: "CUDA out of memory"
**Solution 1:** Use smaller model
```yaml
# config.yaml
model:
  hidden_channels: 64      # Reduce from 128
  num_layers: 3            # Reduce from 4
```

**Solution 2:** Train on CPU (slower but works)
PyTorch automatically uses CPU if CUDA unavailable.

### Problem: Training loss not decreasing
**Solution 1:** Increase learning rate
```yaml
training:
  learning_rate: 0.01      # Increase from 0.001
```

**Solution 2:** Train longer
```yaml
training:
  num_epochs: 200          # Increase from 100
```

**Solution 3:** Use larger model
```yaml
model:
  hidden_channels: 256     # Increase from 128
```

### Problem: "No matching fields found"
**Solution:** Check that your ExodusII files contain the expected variables. Edit field mappings in `src/data_loader.py`:
```python
field_mappings = {
    'velocity_x': ['vals_nod_var1', 'velocity_0', 'your_custom_name'],
    ...
}
```

## Tips for Best Results

### 1. Data Quality
- Ensure all timesteps are valid (no NaN/Inf)
- Use multiple timesteps for training (more data = better model)
- Check that coarse and fine simulations represent the same physics

### 2. Model Selection
- **GNN (gnn)**: Best for irregular meshes, captures spatial relationships
- **Encoder-Decoder**: Good for multi-scale features
- **MLP**: Fast baseline, doesn't use graph structure

### 3. Hyperparameters
- Start with default settings
- If underfitting (high training loss): increase model capacity
- If overfitting (train loss low, val loss high): add regularization
- For larger meshes: increase `k_neighbors` to 10-12

### 4. Training Strategy
- Monitor both train and validation loss
- Stop if validation loss stops improving (early stopping)
- Save checkpoints regularly
- Experiment with different source→target resolution pairs

## Advanced: Multi-Resolution Training

Train on multiple resolution pairs simultaneously:

```python
# Modify src/train.py to load multiple pairs
pairs = [
    ('coarse', 'medium'),
    ('coarse', 'fine'),
    ('medium', 'fine')
]

# Create datasets for each pair and combine
```

## Next Steps

1. **Baseline**: Run `example_workflow.py` to see interpolation errors
2. **Train**: Run `train.py` with default config
3. **Evaluate**: Run `inference.py` with ground truth comparison
4. **Optimize**: Adjust hyperparameters based on results
5. **Apply**: Use trained model on new coarse simulations

## File Overview

```
VERTEX/
├── README.md              # Detailed documentation
├── QUICKSTART.md          # This file
├── requirements.txt       # Python dependencies
├── config.yaml            # Training configuration
├── test_setup.sh          # Verify installation
├── analyze_data.py        # Explore mesh data
├── example_workflow.py    # Demo interpolation
└── src/
    ├── data_loader.py     # Load ExodusII files
    ├── mesh_interpolation.py   # Interpolation utilities
    ├── models.py          # GNN architectures
    ├── train.py           # Training script
    └── inference.py       # Prediction script
```

## Questions?

Check the full `README.md` for detailed explanations, or examine the code comments for implementation details.
