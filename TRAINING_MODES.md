# Training Modes

The framework now supports two distinct training approaches:

## Mode 1: Full Mapping (Original)

**Config**: `config_custom.yaml`

**What it does**: Model learns the complete transformation from coarse mesh to fine mesh, including both interpolation and refinement.

**Data flow**:
```
Coarse mesh (17,763 nodes) 
  → Linear interpolation (during data loading)
  → Interpolated on fine mesh (69,958 nodes)
  → GNN model learns to refine
  → Output: refined fields
  → Compare with ground truth
```

**Files needed**:
- `coarse_file`: Original coarse mesh solution
- `fine_file`: Ground truth fine mesh solution

**Training command**:
```bash
python3.12 train_custom.py --config config_custom.yaml
```

**Pros**:
- End-to-end learning
- Can potentially learn better interpolation strategies

**Cons**:
- Model must learn both interpolation and correction
- More complex learning task

---

## Mode 2: Correction-Only (New)

**Config**: `config_preinterpolated.yaml`

**What it does**: Model learns ONLY to correct/refine already-interpolated fields. Focuses on learning residuals.

**Data flow**:
```
Pre-interpolated solution (69,958 nodes)
  → Already on fine mesh
  → GNN model learns corrections/residuals
  → Output: corrected fields
  → Compare with ground truth
```

**Files needed**:
- `interpolated_file`: Pre-computed interpolated solution on fine mesh
  - Example: `paraview/180-60_to_360-120_interpolated_linear.exo`
- `fine_file`: Ground truth fine mesh solution

**Training command**:
```bash
python3.12 train_custom.py --config config_preinterpolated.yaml
```

**Pros**:
- Model focuses purely on correction/refinement
- Simpler learning task (residual learning)
- Can use your carefully computed interpolation with offset correction
- Faster dataset preparation (no interpolation needed)

**Cons**:
- Requires pre-computed interpolation
- Tied to the specific interpolation method used

---

## Which Mode to Use?

### Use **Correction-Only** (Mode 2) when:
- ✓ You have carefully prepared interpolated data
- ✓ You've handled timestep alignment issues (like the +54 offset)
- ✓ You want the model to learn pure corrections
- ✓ You want faster experimentation

### Use **Full Mapping** (Mode 1) when:
- ✓ You want end-to-end learning
- ✓ You want the model to learn optimal interpolation
- ✓ You don't have pre-computed interpolation

---

## Key Differences in Configuration

### config_custom.yaml (Full Mapping):
```yaml
data:
  coarse_file: '/path/to/coarse.e'
  fine_file: '/path/to/fine.e'
  use_cache: true  # Cache interpolated results
```

### config_preinterpolated.yaml (Correction-Only):
```yaml
data:
  interpolated_file: '/path/to/interpolated.exo'
  fine_file: '/path/to/fine.e'
  use_cache: false  # No need to cache
```

---

## Technical Details

Both modes produce models with:
- **Input**: Fine mesh coordinates (2D) + 4 field values = 6 features per node
- **Output**: 4 refined field values per node
- **Architecture**: Same GNN/MLP models can be used

The difference is in what the "4 field values" represent in the input:
- **Mode 1**: Interpolated from coarse mesh (computed during training)
- **Mode 2**: Loaded from your pre-interpolated file

---

## Your Use Case

Since you have:
- Pre-computed interpolation: `paraview/180-60_to_360-120_interpolated_linear.exo`
- Handled timestep offset: Fine[N] = Interp[N+54]
- RMSE of ~0.00476 between interpolated and fine

**Recommendation**: Use **Mode 2 (Correction-Only)** with `config_preinterpolated.yaml` to:
1. Leverage your careful interpolation work
2. Focus model learning on corrections
3. Potentially achieve better results by treating this as a residual learning problem
