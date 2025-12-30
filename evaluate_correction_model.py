#!/usr/bin/env python3
"""Evaluate a saved correction-only model on pre-interpolated Exodus data.

Loads:
- interpolated_file (model input fields already on fine mesh)
- fine_file (ground-truth)

Computes per-variable, per-timestep errors across all overlapping timesteps,
accounting for timestep_offset such that fine[t] == interpolated[t + offset].

Outputs a CSV (long format): one row per (timestep, variable, source), where
source is either:
- model: model prediction vs fine ground truth
- interpolated: original interpolated input vs fine ground truth
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

# Ensure we can import from src/
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "src"))

from data_loader import ExodusDataLoader  # noqa: E402
from train import MeshDataset  # noqa: E402
from models import build_knn_graph, MeshGNN, MeshEncoderDecoder, SimpleMLP  # noqa: E402
from device_utils import select_torch_device, format_available_cuda_devices  # noqa: E402


VARIABLES_STD = ["velocity_0", "velocity_1", "pressure", "temperature"]
VARIABLES_FRIENDLY = ["velocity_x", "velocity_y", "pressure", "temperature"]


def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _reconstruct_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint missing 'config'. Expected a training checkpoint with architecture info."
        )

    config = checkpoint["config"]
    state_dict = checkpoint.get("model_state_dict", {})

    # Training scripts assume 2D coords after z-drop for 2D meshes.
    # Infer in_channels from weights for compatibility with different input feature sets.
    ndim = 2

    def _infer_in_channels(sd: Dict, model_type: str) -> int:
        if model_type == "gnn":
            w = sd.get("input_mlp.0.weight")
        elif model_type == "encoder_decoder":
            w = sd.get("encoders.0.0.weight")
        elif model_type == "mlp":
            w = sd.get("network.0.weight")
        else:
            w = None
        if w is None:
            return ndim + 4
        return int(w.shape[1])

    out_channels = 4

    model_type = config["model"]["type"]

    def _infer_encoder_decoder_num_levels(sd: Dict) -> int:
        max_idx = -1
        for k in sd.keys():
            if k.startswith("encoders."):
                # Format: encoders.<i>.<...>
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    max_idx = max(max_idx, int(parts[1]))
        return (max_idx + 1) if max_idx >= 0 else 3

    def _infer_encoder_decoder_hidden_channels(sd: Dict) -> int:
        # encoders.0.0.weight is Linear(out_features=hidden_channels, in_features=in_channels)
        w = sd.get("encoders.0.0.weight")
        if w is None:
            return int(config["model"].get("hidden_channels", 128))
        # torch tensor shape [out, in]
        return int(w.shape[0])
    in_channels = _infer_in_channels(state_dict, model_type)

    if model_type == "gnn":
        model = MeshGNN(
            in_channels=in_channels,
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=out_channels,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif model_type == "encoder_decoder":
        # Some older checkpoints/configs may not include num_levels (and/or may have drift).
        # Infer from the state_dict so we can reliably reload.
        inferred_levels = _infer_encoder_decoder_num_levels(state_dict)
        inferred_hidden = _infer_encoder_decoder_hidden_channels(state_dict)
        num_levels = int(config["model"].get("num_levels", inferred_levels))
        hidden_channels = int(config["model"].get("hidden_channels", inferred_hidden))
        if num_levels != inferred_levels or hidden_channels != inferred_hidden:
            # Prefer the weights' implied architecture over the config to avoid load failures.
            num_levels = inferred_levels
            hidden_channels = inferred_hidden

        model = MeshEncoderDecoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_levels=num_levels,
            dropout=config["model"]["dropout"],
        )
    elif model_type == "mlp":
        model = SimpleMLP(
            in_channels=in_channels,
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=out_channels,
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint config: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    residual_stats = checkpoint.get("residual_stats")
    input_stats = checkpoint.get("input_stats")
    coord_stats = checkpoint.get("coord_stats")
    return model, config, residual_stats, input_stats, coord_stats


def _ensure_required_fields(label: str, data: Dict, required: List[str]) -> None:
    missing = [k for k in required if k not in data.get("fields", {})]
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}. Available: {list(data.get('fields', {}).keys())}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate correction model errors per timestep and variable")
    parser.add_argument(
        "--config",
        default=str(REPO_DIR / "config_preinterpolated.yaml"),
        help="Path to config_preinterpolated.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_DIR / "outputs_correction_physics" / "best_model.pt"),
        help="Path to trained model checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write evaluation outputs (default: <checkpoint_dir>/evaluation)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cpu' or 'cuda'. Default: auto",
    )
    parser.add_argument(
        "--out-base",
        default=None,
        help=(
            "Base name (no extension) for outputs inside --output-dir. "
            "If set, writes <out-base>.csv and <out-base>.json instead of default filenames."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start timestep index within the overlapped evaluation set (default: 0)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every Nth timestep (default: 1)",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Maximum number of timesteps to evaluate after applying start/stride (default: all)",
    )

    args = parser.parse_args()

    config = _load_yaml(args.config)

    interpolated_file = config["data"]["interpolated_file"]
    fine_file = config["data"]["fine_file"]
    timestep_offset = int(config["data"].get("timestep_offset", 0))

    device = select_torch_device(args.device)
    if device.type == "cuda":
        print(format_available_cuda_devices())

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else checkpoint_path.parent / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model checkpoint...")
    model, ckpt_config, residual_stats, input_stats, coord_stats = _reconstruct_model_from_checkpoint(str(checkpoint_path), device)
    use_graph = ckpt_config["model"]["type"] in ["gnn", "encoder_decoder"]

    residual_learning = bool(ckpt_config.get("training", {}).get("residual_learning", False))
    if residual_learning:
        print("Checkpoint indicates residual learning: model output is a correction (fine - interpolated)")

    training_cfg = ckpt_config.get("training", {}) if isinstance(ckpt_config, dict) else {}
    if not isinstance(training_cfg, dict):
        training_cfg = {}

    residual_normalization = bool(training_cfg.get("residual_normalization", False))

    # Backward/robust behavior:
    # - Some training entrypoints historically defaulted input_normalization=True but did not persist
    #   the flag into the saved config. In that case, input_stats will be present in the checkpoint.
    if "input_normalization" in training_cfg:
        input_normalization = bool(training_cfg.get("input_normalization"))
    else:
        input_normalization = input_stats is not None

    if "normalize_coords" in training_cfg:
        normalize_coords = bool(training_cfg.get("normalize_coords"))
    else:
        normalize_coords = coord_stats is not None
    if residual_learning and residual_normalization:
        if not residual_stats or ("mean" not in residual_stats) or ("std" not in residual_stats):
            print(
                "WARNING: residual_normalization is true but checkpoint missing residual_stats; "
                "will treat model output as unnormalized correction."
            )
            residual_stats = None

    # Helpful sanity check: warn if you're evaluating on different files than the checkpoint was trained on.
    ckpt_data_cfg = ckpt_config.get("data", {}) if isinstance(ckpt_config, dict) else {}
    ckpt_interp = ckpt_data_cfg.get("interpolated_file")
    ckpt_fine = ckpt_data_cfg.get("fine_file")
    ckpt_offset = ckpt_data_cfg.get("timestep_offset")
    if ckpt_interp and ckpt_interp != interpolated_file:
        print(
            "WARNING: Checkpoint was trained with a different interpolated_file.\n"
            f"  checkpoint: {ckpt_interp}\n"
            f"  eval:       {interpolated_file}\n"
            "This can easily make the model look worse than the baseline interpolation."
        )
    if ckpt_fine and ckpt_fine != fine_file:
        print(
            "WARNING: Checkpoint was trained with a different fine_file.\n"
            f"  checkpoint: {ckpt_fine}\n"
            f"  eval:       {fine_file}\n"
            "This can easily make the model look worse than the baseline interpolation."
        )
    if ckpt_offset is not None and int(ckpt_offset) != timestep_offset:
        print(
            "WARNING: Checkpoint timestep_offset differs from evaluation timestep_offset.\n"
            f"  checkpoint: {int(ckpt_offset)}\n"
            f"  eval:       {timestep_offset}"
        )

    print("Loading Exodus data...")
    with ExodusDataLoader(interpolated_file) as interp_reader:
        interpolated_data = interp_reader.load()
    with ExodusDataLoader(fine_file) as fine_reader:
        fine_data = fine_reader.load()

    _ensure_required_fields("Interpolated data", interpolated_data, VARIABLES_STD)
    _ensure_required_fields("Fine (ground truth) data", fine_data, VARIABLES_STD)

    # Coordinate alignment check (node ordering must match for pointwise errors to be meaningful)
    interp_coords = interpolated_data["coordinates"]
    fine_coords_raw = fine_data["coordinates"]
    if interp_coords.shape != fine_coords_raw.shape:
        print(
            "WARNING: interpolated vs fine coordinate arrays have different shapes. "
            f"interp={interp_coords.shape}, fine={fine_coords_raw.shape}."
        )
    else:
        # Mirror the training-time 2D detection (drop z if effectively 2D)
        z_range = float(np.ptp(interp_coords[:, 2])) if interp_coords.shape[1] >= 3 else 0.0
        if z_range < 1e-6 and interp_coords.shape[1] >= 2:
            interp_xy = interp_coords[:, :2]
            fine_xy = fine_coords_raw[:, :2]
        else:
            interp_xy = interp_coords
            fine_xy = fine_coords_raw

        max_abs = float(np.max(np.abs(interp_xy - fine_xy)))
        mean_abs = float(np.mean(np.abs(interp_xy - fine_xy)))
        if max_abs > 1e-8:
            print(
                "WARNING: interpolated and fine coordinates differ (node ordering mismatch possible).\n"
                f"  mean|Δcoord|={mean_abs:.3e}, max|Δcoord|={max_abs:.3e}\n"
                "If ordering differs, both baseline and model errors here are not physically meaningful."
            )

    k_neighbors = int(ckpt_config["model"].get("k_neighbors", config["model"].get("k_neighbors", 8)))

    print("Building evaluation dataset (pre-interpolated / correction-only)...")
    dataset = MeshDataset(
        interpolated_data=interpolated_data,
        fine_data=fine_data,
        timestep_offset=timestep_offset,
        use_graph=use_graph,
        k_neighbors=k_neighbors,
        use_cache=False,
        cache_dir=str(REPO_DIR / "cache"),
    )

    # Precompute graph connectivity once (same for all timesteps)
    edge_index_t = None
    fine_coords = dataset.fine_coords

    fine_coords_in = fine_coords.astype(np.float32)
    if normalize_coords and coord_stats is not None and ("mean" in coord_stats) and ("std" in coord_stats):
        cmean = np.asarray(coord_stats["mean"], dtype=np.float32)
        cstd = np.asarray(coord_stats["std"], dtype=np.float32)
        fine_coords_in = (fine_coords_in - cmean) / cstd

    if use_graph:
        print(f"Precomputing kNN graph (k={k_neighbors})...")
        edge_index = build_knn_graph(fine_coords_in, k=k_neighbors)
        edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)

    records = []

    start = int(args.start)
    stride = max(int(args.stride), 1)
    if start < 0:
        raise ValueError("--start must be >= 0")
    if start >= len(dataset):
        raise ValueError(f"--start={start} is out of range for dataset length {len(dataset)}")

    indices = list(range(start, len(dataset), stride))
    if args.max_timesteps is not None:
        indices = indices[: int(args.max_timesteps)]

    print(f"Evaluating {len(indices)}/{len(dataset)} timesteps on device={device} (start={start}, stride={stride})...")
    with torch.no_grad():
        for i in indices:
            sample = dataset.samples[i]
            t_fine = int(sample.get("timestep", i))
            t_interp = int(sample.get("timestep_interp", t_fine + timestep_offset))

            coarse_interp = sample["coarse_interp"].astype(np.float32)  # (n_nodes, 4)
            fine_features = sample["fine_features"].astype(np.float32)  # (n_nodes, 4)

            # Select model input fields.
            # Prefer indices stored in the checkpoint config; otherwise, fall back to whatever
            # the model architecture expects (inferred from in_channels).
            def _model_in_channels(m) -> int:
                if hasattr(m, "input_mlp"):
                    return int(m.input_mlp[0].in_features)
                if hasattr(m, "encoders"):
                    return int(m.encoders[0][0].in_features)
                if hasattr(m, "network"):
                    return int(m.network[0].in_features)
                raise ValueError("Could not infer in_channels from model")

            inferred_in_channels = _model_in_channels(model)
            n_input_fields = max(inferred_in_channels - fine_coords_in.shape[1], 0)
            input_feature_indices = (ckpt_config.get("training", {}) or {}).get("input_feature_indices")
            if not input_feature_indices:
                input_feature_indices = list(range(n_input_fields))
            coarse_interp_in = coarse_interp[:, input_feature_indices]
            if input_normalization and input_stats is not None and ("mean" in input_stats) and ("std" in input_stats):
                mean = np.asarray(input_stats["mean"], dtype=np.float32)
                std = np.asarray(input_stats["std"], dtype=np.float32)
                coarse_interp_in = (coarse_interp_in - mean) / std

            x_np = np.concatenate([fine_coords_in, coarse_interp_in], axis=-1)
            x = torch.from_numpy(x_np).to(device)

            if use_graph:
                pred = model(x, edge_index_t)
            else:
                pred = model(x)

            pred_np = pred.detach().cpu().numpy()

            def _add_records(source: str, pred_or_interp: np.ndarray):
                diff = pred_or_interp - fine_features
                mse_per_var = np.mean(diff ** 2, axis=0)
                mae_per_var = np.mean(np.abs(diff), axis=0)
                rmse_per_var = np.sqrt(mse_per_var)

                # Mean signed error (bias) per variable
                bias_per_var = np.mean(diff, axis=0)

                # Mean relative error per variable: mean((pred-true)/true) over nodes where |true| is not tiny.
                # Keep sign; use NaN when denominator is ~0.
                rel_eps = 1e-12
                denom = fine_features
                rel = np.full_like(diff, np.nan, dtype=np.float64)
                mask = np.abs(denom) > rel_eps
                rel[mask] = (diff[mask] / denom[mask]).astype(np.float64)
                mean_rel_err_per_var = np.nanmean(rel, axis=0)

                # Extrema diagnostics per variable (computed against ground-truth locations)
                # For each variable v:
                #   k_max = argmax(true_v)
                #   rel_err_at_true_max = (pred_v[k_max] - true_max) / true_max
                # Same for min.
                eps = 1e-12
                true_min = np.min(fine_features, axis=0)
                true_max = np.max(fine_features, axis=0)
                pred_min = np.min(pred_or_interp, axis=0)
                pred_max = np.max(pred_or_interp, axis=0)
                idx_true_min = np.argmin(fine_features, axis=0)
                idx_true_max = np.argmax(fine_features, axis=0)

                for var_idx, (var_std, var_name) in enumerate(
                    zip(VARIABLES_STD, VARIABLES_FRIENDLY)
                ):
                    k_min = int(idx_true_min[var_idx])
                    k_max = int(idx_true_max[var_idx])
                    tmin = float(true_min[var_idx])
                    tmax = float(true_max[var_idx])

                    pred_at_tmin = float(pred_or_interp[k_min, var_idx])
                    pred_at_tmax = float(pred_or_interp[k_max, var_idx])

                    rel_err_tmin = (pred_at_tmin - tmin) / tmin if abs(tmin) > eps else float("nan")
                    rel_err_tmax = (pred_at_tmax - tmax) / tmax if abs(tmax) > eps else float("nan")

                    records.append(
                        {
                            "timestep_fine": t_fine,
                            "timestep_interpolated": t_interp,
                            "variable": var_name,
                            "variable_std": var_std,
                            "source": source,
                            "mse": float(mse_per_var[var_idx]),
                            "rmse": float(rmse_per_var[var_idx]),
                            "mae": float(mae_per_var[var_idx]),

                            # Bias + mean relative error
                            "bias": float(bias_per_var[var_idx]),
                            "mean_rel_err": float(mean_rel_err_per_var[var_idx])
                            if np.isfinite(mean_rel_err_per_var[var_idx])
                            else float("nan"),

                            # Diagnostics: extrema and relative error at GT extrema locations
                            "true_min": tmin,
                            "true_max": tmax,
                            "pred_min": float(pred_min[var_idx]),
                            "pred_max": float(pred_max[var_idx]),
                            "idx_true_min": k_min,
                            "idx_true_max": k_max,
                            "pred_at_true_min": pred_at_tmin,
                            "pred_at_true_max": pred_at_tmax,
                            "rel_err_at_true_min": float(rel_err_tmin),
                            "rel_err_at_true_max": float(rel_err_tmax),
                        }
                    )

            # Model prediction vs ground truth
            # If residual_learning, the model output is a correction, so compare (interpolated + correction) to fine.
            if residual_learning:
                correction = pred_np
                if residual_stats is not None and residual_normalization:
                    mean = np.asarray(residual_stats["mean"], dtype=np.float32)
                    std = np.asarray(residual_stats["std"], dtype=np.float32)
                    correction = correction * std + mean
                model_fields = coarse_interp + correction
            else:
                model_fields = pred_np

            _add_records("model", model_fields)
            # Baseline: original interpolated values vs ground truth
            _add_records("interpolated", coarse_interp)

    # Write CSV (long format)
    if args.out_base:
        csv_path = output_dir / f"{args.out_base}.csv"
    else:
        csv_path = output_dir / "errors_by_variable_and_timestep.csv"
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame.from_records(records)
        df = df.sort_values(["timestep_fine", "variable", "source"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
    except Exception:
        # Pandas is not listed in requirements; fall back to stdlib CSV.
        import csv

        fieldnames = [
            "timestep_fine",
            "timestep_interpolated",
            "variable",
            "variable_std",
            "source",
            "mse",
            "rmse",
            "mae",
            "bias",
            "mean_rel_err",
            "true_min",
            "true_max",
            "pred_min",
            "pred_max",
            "idx_true_min",
            "idx_true_max",
            "pred_at_true_min",
            "pred_at_true_max",
            "rel_err_at_true_min",
            "rel_err_at_true_max",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(
                records, key=lambda r: (r["timestep_fine"], r["variable"], r["source"])
            ):
                writer.writerow({k: row[k] for k in fieldnames})

    # Also write a compact JSON summary (mean across timesteps)
    summary = {}
    for source in ["model", "interpolated"]:
        summary[source] = {}
        for var in VARIABLES_FRIENDLY:
            rows = [r for r in records if r["source"] == source and r["variable"] == var]
            summary[source][var] = {
                "timesteps": len(rows),
                "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
                "mae_mean": float(np.mean([r["mae"] for r in rows])),
                "mse_mean": float(np.mean([r["mse"] for r in rows])),
            }

    if args.out_base:
        summary_path = output_dir / f"{args.out_base}.json"
    else:
        summary_path = output_dir / "summary_mean_errors.json"
    import json

    with open(summary_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "interpolated_file": interpolated_file,
                "fine_file": fine_file,
                "timestep_offset": timestep_offset,
                "use_graph": use_graph,
                "k_neighbors": k_neighbors,
                "residual_learning": residual_learning,
                "residual_normalization": residual_normalization,
                "residual_stats": residual_stats,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
