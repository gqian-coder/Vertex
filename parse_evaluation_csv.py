#!/usr/bin/env python3
"""Parse evaluation CSV and generate diagnostic plots.

Reads the long-format CSV produced by evaluate_correction_model.py and creates:
  1) Per-variable RMSE vs timestep (optionally comparing sources: model vs interpolated)
  2) Per-variable GT vs prediction min/max vs timestep
  3) Per-variable relative error at GT extrema (min/max) vs timestep

The script avoids pandas to keep dependencies minimal.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Row:
    timestep_fine: int
    variable: str
    source: str
    rmse: float
    mae: Optional[float] = None
    mse: Optional[float] = None

    bias: Optional[float] = None
    mean_rel_err: Optional[float] = None

    true_min: Optional[float] = None
    true_max: Optional[float] = None
    pred_min: Optional[float] = None
    pred_max: Optional[float] = None
    rel_err_at_true_min: Optional[float] = None
    rel_err_at_true_max: Optional[float] = None


def _as_int(value: str) -> int:
    return int(float(value))


def _as_float(value: str) -> float:
    return float(value)


def _as_float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if v == "" or v.lower() in {"nan", "none"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_rows(csv_path: Path) -> Tuple[List[Row], List[str]]:
    rows: List[Row] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)

        for r in reader:
            rows.append(
                Row(
                    timestep_fine=_as_int(r["timestep_fine"]),
                    variable=r["variable"],
                    source=r["source"],
                    rmse=_as_float(r["rmse"]),
                    mae=_as_float_or_none(r.get("mae")),
                    mse=_as_float_or_none(r.get("mse")),
                    bias=_as_float_or_none(r.get("bias")),
                    mean_rel_err=_as_float_or_none(r.get("mean_rel_err")),
                    true_min=_as_float_or_none(r.get("true_min")),
                    true_max=_as_float_or_none(r.get("true_max")),
                    pred_min=_as_float_or_none(r.get("pred_min")),
                    pred_max=_as_float_or_none(r.get("pred_max")),
                    rel_err_at_true_min=_as_float_or_none(r.get("rel_err_at_true_min")),
                    rel_err_at_true_max=_as_float_or_none(r.get("rel_err_at_true_max")),
                )
            )

    return rows, fieldnames


def _group_by_variable_source(rows: Sequence[Row]) -> Dict[Tuple[str, str], List[Row]]:
    grouped: Dict[Tuple[str, str], List[Row]] = {}
    for r in rows:
        grouped.setdefault((r.variable, r.source), []).append(r)
    for k in list(grouped.keys()):
        grouped[k] = sorted(grouped[k], key=lambda x: x.timestep_fine)
    return grouped


def _unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted(set(values))


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_rmse_vs_timestep(rows: Sequence[Row], out_dir: Path, sources: Optional[List[str]] = None) -> Path:
    import matplotlib.pyplot as plt

    grouped = _group_by_variable_source(rows)
    variables = sorted(set(v for (v, _s) in grouped.keys()))
    all_sources = sorted(set(s for (_v, s) in grouped.keys()))
    sources_to_plot = sources or all_sources

    n = len(variables)
    if n == 0:
        raise ValueError("No variables found in CSV")

    # Layout: up to 4 vars -> 2x2, else single column
    if n <= 4:
        nrows, ncols = (2, 2) if n > 1 else (1, 1)
    else:
        nrows, ncols = (n, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 3.6 * nrows), squeeze=False)

    for idx, var in enumerate(variables):
        ax = axes[idx // ncols][idx % ncols]
        for src in sources_to_plot:
            series = grouped.get((var, src))
            if not series:
                continue
            t = np.array([r.timestep_fine for r in series], dtype=np.int64)
            y = np.array([r.rmse for r in series], dtype=np.float64)
            ax.plot(t, y, label=src)

        ax.set_title(f"RMSE vs timestep: {var}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide any unused axes
    for j in range(len(variables), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path = out_dir / "rmse_by_variable_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_minmax(rows: Sequence[Row], out_dir: Path, source_for_pred: str = "model") -> Optional[Path]:
    import matplotlib.pyplot as plt

    # Needs the extended columns.
    if not any(r.true_min is not None for r in rows):
        return None

    grouped = _group_by_variable_source(rows)
    variables = sorted(set(v for (v, _s) in grouped.keys()))

    n = len(variables)
    if n <= 4:
        nrows, ncols = (2, 2) if n > 1 else (1, 1)
    else:
        nrows, ncols = (n, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 3.6 * nrows), squeeze=False)

    for idx, var in enumerate(variables):
        ax = axes[idx // ncols][idx % ncols]

        # Ground truth extrema are the same regardless of source; use any available series.
        any_series = None
        for (v, s), series in grouped.items():
            if v == var:
                any_series = series
                break
        if not any_series:
            ax.axis("off")
            continue

        t = np.array([r.timestep_fine for r in any_series], dtype=np.int64)
        true_min = np.array([r.true_min for r in any_series], dtype=np.float64)
        true_max = np.array([r.true_max for r in any_series], dtype=np.float64)
        ax.plot(t, true_min, label="true_min", linestyle="--")
        ax.plot(t, true_max, label="true_max", linestyle="--")

        pred_series = grouped.get((var, source_for_pred))
        if pred_series:
            tp = np.array([r.timestep_fine for r in pred_series], dtype=np.int64)
            pred_min = np.array([r.pred_min for r in pred_series], dtype=np.float64)
            pred_max = np.array([r.pred_max for r in pred_series], dtype=np.float64)
            ax.plot(tp, pred_min, label=f"{source_for_pred}_min")
            ax.plot(tp, pred_max, label=f"{source_for_pred}_max")

        ax.set_title(f"Min/Max vs timestep: {var}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(len(variables), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"minmax_true_vs_{source_for_pred}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_relerr_at_extrema(rows: Sequence[Row], out_dir: Path, source: str = "model") -> Optional[Path]:
    import matplotlib.pyplot as plt

    if not any(r.rel_err_at_true_min is not None for r in rows):
        return None

    grouped = _group_by_variable_source(rows)
    variables = sorted(set(v for (v, _s) in grouped.keys()))

    n = len(variables)
    if n <= 4:
        nrows, ncols = (2, 2) if n > 1 else (1, 1)
    else:
        nrows, ncols = (n, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 3.6 * nrows), squeeze=False)

    for idx, var in enumerate(variables):
        ax = axes[idx // ncols][idx % ncols]
        series = grouped.get((var, source))
        if not series:
            ax.axis("off")
            continue

        t = np.array([r.timestep_fine for r in series], dtype=np.int64)
        e_min = np.array([r.rel_err_at_true_min for r in series], dtype=np.float64)
        e_max = np.array([r.rel_err_at_true_max for r in series], dtype=np.float64)

        ax.plot(t, e_min, label="rel_err@true_min")
        ax.plot(t, e_max, label="rel_err@true_max")
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)

        ax.set_title(f"Relative error at GT extrema: {var} ({source})")
        ax.set_xlabel("timestep")
        ax.set_ylabel("relative error")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(len(variables), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"relerr_at_true_extrema_{source}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_relerr_of_extrema_values(rows: Sequence[Row], out_dir: Path, source: str = "model") -> Optional[Path]:
    """Plot relative error of extrema *values* per timestep.

    This differs from `plot_relerr_at_extrema`:
      - `plot_relerr_at_extrema` evaluates pred at the *GT argmin/argmax locations*.
      - This function compares the *extrema values themselves*:
          (max(pred) - max(GT)) / max(GT)
          (min(pred) - min(GT)) / min(GT)
    """
    import matplotlib.pyplot as plt

    if not any(r.pred_min is not None and r.true_min is not None for r in rows):
        return None

    grouped = _group_by_variable_source(rows)
    variables = sorted(set(v for (v, _s) in grouped.keys()))

    n = len(variables)
    if n <= 4:
        nrows, ncols = (2, 2) if n > 1 else (1, 1)
    else:
        nrows, ncols = (n, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 3.6 * nrows), squeeze=False)
    eps = 1e-12

    for idx, var in enumerate(variables):
        ax = axes[idx // ncols][idx % ncols]

        # Use GT extrema series from any available source (should match across sources).
        any_series = None
        for (v, _s), series in grouped.items():
            if v == var:
                any_series = series
                break
        pred_series = grouped.get((var, source))

        if not any_series or not pred_series:
            ax.axis("off")
            continue

        # Build a map from timestep -> row for the selected pred source
        pred_by_t = {r.timestep_fine: r for r in pred_series}
        t_list: List[int] = []
        rel_min_list: List[float] = []
        rel_max_list: List[float] = []

        for r in any_series:
            pr = pred_by_t.get(r.timestep_fine)
            if pr is None:
                continue
            if r.true_min is None or r.true_max is None or pr.pred_min is None or pr.pred_max is None:
                continue

            tmin = float(r.true_min)
            tmax = float(r.true_max)
            pmin = float(pr.pred_min)
            pmax = float(pr.pred_max)

            rel_min = (pmin - tmin) / tmin if abs(tmin) > eps else float("nan")
            rel_max = (pmax - tmax) / tmax if abs(tmax) > eps else float("nan")

            t_list.append(int(r.timestep_fine))
            rel_min_list.append(rel_min)
            rel_max_list.append(rel_max)

        if not t_list:
            ax.axis("off")
            continue

        t = np.asarray(t_list, dtype=np.int64)
        e_min = np.asarray(rel_min_list, dtype=np.float64)
        e_max = np.asarray(rel_max_list, dtype=np.float64)

        ax.plot(t, e_min, label="(min(pred)-min(GT))/min(GT)")
        ax.plot(t, e_max, label="(max(pred)-max(GT))/max(GT)")
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title(f"Relative error of extrema values: {var} ({source})")
        ax.set_xlabel("timestep")
        ax.set_ylabel("relative error")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(len(variables), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"relerr_of_extrema_values_{source}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_bias_vs_timestep(rows: Sequence[Row], out_dir: Path, sources: Optional[List[str]] = None) -> Optional[Path]:
    import matplotlib.pyplot as plt

    if not any(r.bias is not None for r in rows):
        return None

    grouped = _group_by_variable_source(rows)
    variables = sorted(set(v for (v, _s) in grouped.keys()))
    all_sources = sorted(set(s for (_v, s) in grouped.keys()))
    sources_to_plot = sources or all_sources

    n = len(variables)
    if n <= 4:
        nrows, ncols = (2, 2) if n > 1 else (1, 1)
    else:
        nrows, ncols = (n, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 3.6 * nrows), squeeze=False)

    for idx, var in enumerate(variables):
        ax = axes[idx // ncols][idx % ncols]
        for src in sources_to_plot:
            series = grouped.get((var, src))
            if not series:
                continue
            t = np.array([r.timestep_fine for r in series], dtype=np.int64)
            y = np.array([np.nan if r.bias is None else r.bias for r in series], dtype=np.float64)
            ax.plot(t, y, label=src)

        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title(f"Mean signed error (bias) vs timestep: {var}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("bias (mean(pred-true))")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(len(variables), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path = out_dir / "bias_by_variable_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_mean_relative_error(rows: Sequence[Row], out_dir: Path, sources: Optional[List[str]] = None) -> Optional[Path]:
    import matplotlib.pyplot as plt

    if not any(r.mean_rel_err is not None for r in rows):
        return None

    grouped = _group_by_variable_source(rows)
    variables = sorted(set(v for (v, _s) in grouped.keys()))
    all_sources = sorted(set(s for (_v, s) in grouped.keys()))
    sources_to_plot = sources or all_sources

    n = len(variables)
    if n <= 4:
        nrows, ncols = (2, 2) if n > 1 else (1, 1)
    else:
        nrows, ncols = (n, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 3.6 * nrows), squeeze=False)

    for idx, var in enumerate(variables):
        ax = axes[idx // ncols][idx % ncols]
        for src in sources_to_plot:
            series = grouped.get((var, src))
            if not series:
                continue
            t = np.array([r.timestep_fine for r in series], dtype=np.int64)
            y = np.array([np.nan if r.mean_rel_err is None else r.mean_rel_err for r in series], dtype=np.float64)
            ax.plot(t, y, label=src)

        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title(f"Mean relative error vs timestep: {var}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("mean((pred-true)/true)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(len(variables), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path = out_dir / "mean_relative_error_by_variable_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse evaluation CSV and plot diagnostics")
    parser.add_argument(
        "--csv",
        default=str(Path("outputs_correction_physics") / "evaluation" / "errors_by_variable_and_timestep.csv"),
        help="Path to errors_by_variable_and_timestep.csv",
    )
    parser.add_argument(
        "--yaml-config",
        default=None,
        help=(
            "Optional training/eval YAML (e.g., config_preinterpolated_physics.yaml). "
            "If provided together with --coarse-file, the script will also compute coarse-vs-fine "
            "extrema relative errors directly from the Exodus files."
        ),
    )
    parser.add_argument(
        "--coarse-file",
        default=None,
        help=(
            "Path to the original coarse Exodus mesh used for interpolation (e.g., .../90-30/cropped.e). "
            "Used only with --yaml-config to compute coarse-vs-fine extrema diagnostics."
        ),
    )
    parser.add_argument(
        "--coarse-to-fine-offset",
        type=int,
        default=None,
        help=(
            "Optional timestep offset when comparing coarse vs fine directly. "
            "Uses coarse[t + offset] vs fine[t]. If omitted, defaults to data.timestep_offset from --yaml-config (or 0 if not present)."
        ),
    )
    parser.add_argument(
        "--exodus-max-timesteps",
        type=int,
        default=None,
        help="Optional cap on number of timesteps when reading Exodus files (for faster plotting).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for plots (default: alongside CSV)",
    )
    parser.add_argument(
        "--pred-source",
        default="model",
        help="Which 'source' to treat as prediction for min/max and relerr plots (default: model)",
    )
    parser.add_argument(
        "--rmse-sources",
        default=None,
        help="Comma-separated sources to include in RMSE plot (default: all in CSV)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.outdir) if args.outdir else csv_path.parent
    _ensure_out_dir(out_dir)

    rows, fieldnames = read_rows(csv_path)
    print(f"Loaded {len(rows):,} rows from: {csv_path}")
    print(f"Columns: {fieldnames}")

    available_sources = sorted(set(r.source for r in rows))
    print(f"Sources: {available_sources}")

    rmse_sources = None
    if args.rmse_sources:
        rmse_sources = [s.strip() for s in args.rmse_sources.split(",") if s.strip()]

    p1 = plot_rmse_vs_timestep(rows, out_dir, sources=rmse_sources)
    print(f"Wrote: {p1}")

    p2 = plot_minmax(rows, out_dir, source_for_pred=args.pred_source)
    if p2:
        print(f"Wrote: {p2}")
    else:
        print("Min/max columns not found; skipping min/max plot")

    p3 = plot_relerr_at_extrema(rows, out_dir, source=args.pred_source)
    if p3:
        print(f"Wrote: {p3}")
    else:
        print("Relative-error-at-extrema columns not found; skipping relerr plot")

    p4 = plot_relerr_of_extrema_values(rows, out_dir, source=args.pred_source)
    if p4:
        print(f"Wrote: {p4}")
    else:
        print("Min/max columns not found; skipping extrema-value relerr plot")

    # Also produce the same extrema-value plot for the baseline interpolation if present.
    # This makes it easy to compare model vs interpolation without re-running with flags.
    if "interpolated" in available_sources and args.pred_source != "interpolated":
        p4b = plot_relerr_of_extrema_values(rows, out_dir, source="interpolated")
        if p4b:
            print(f"Wrote: {p4b}")

    p5 = plot_bias_vs_timestep(rows, out_dir)
    if p5:
        print(f"Wrote: {p5}")
    else:
        print("Bias column not found; skipping bias plot (rerun evaluator to add it)")

    p6 = plot_mean_relative_error(rows, out_dir)
    if p6:
        print(f"Wrote: {p6}")
    else:
        print("mean_rel_err column not found; skipping mean relative error plot (rerun evaluator to add it)")

    # Optional: compare extrema values between the *original coarse* and *fine* Exodus meshes.
    # This is useful for diagnosing baseline bias in pressure before interpolation.
    if args.yaml_config and args.coarse_file:
        try:
            # If the user didn't provide an explicit offset, default to the YAML's timestep_offset
            # (same convention as evaluate_correction_model.py: fine[t] == coarse[t + offset]).
            coarse_to_fine_offset = args.coarse_to_fine_offset
            if coarse_to_fine_offset is None:
                try:
                    cfg = _load_yaml(Path(args.yaml_config))
                    coarse_to_fine_offset = int((cfg.get("data", {}) or {}).get("timestep_offset", 0))
                except Exception:
                    coarse_to_fine_offset = 0

            exo_plot = plot_relerr_of_extrema_values_coarse_vs_fine(
                yaml_config_path=Path(args.yaml_config),
                coarse_file=Path(args.coarse_file),
                out_dir=out_dir,
                coarse_to_fine_offset=int(coarse_to_fine_offset),
                max_timesteps=args.exodus_max_timesteps,
            )
            if exo_plot:
                print(f"Wrote: {exo_plot}")
        except Exception as e:
            print(f"WARNING: coarse-vs-fine extrema plot failed: {e}")

    return 0


def _load_yaml(path: Path) -> Dict:
    import yaml

    with path.open("r") as f:
        return yaml.safe_load(f)


def _ensure_src_on_path() -> None:
    repo_dir = Path(__file__).resolve().parent
    src_dir = repo_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _exodus_num_timesteps(loader) -> int:
    # ExodusDataLoader stores standardized fields (time, nodes)
    fields = getattr(loader, "field_data", {})
    if not fields:
        return 0
    for k, v in fields.items():
        if k == "time_values":
            continue
        if hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[0])
    return 0


def plot_relerr_of_extrema_values_coarse_vs_fine(
    *,
    yaml_config_path: Path,
    coarse_file: Path,
    out_dir: Path,
    coarse_to_fine_offset: int = 0,
    max_timesteps: Optional[int] = None,
) -> Optional[Path]:
    """Compute and plot extrema-value relative errors comparing coarse vs fine Exodus meshes.

    For each timestep t and each variable v:
      rel_max(t) = (max(coarse_v[t+off]) - max(fine_v[t])) / max(fine_v[t])
      rel_min(t) = (min(coarse_v[t+off]) - min(fine_v[t])) / min(fine_v[t])

    Notes:
    - This compares extrema values on their native meshes (no node mapping).
    - Requires a YAML config to locate the fine_file.
    """
    import matplotlib.pyplot as plt

    cfg = _load_yaml(yaml_config_path)
    fine_file = Path((cfg.get("data", {}) or {}).get("fine_file"))
    if not fine_file:
        raise ValueError(f"YAML missing data.fine_file: {yaml_config_path}")
    if not coarse_file.exists():
        raise FileNotFoundError(coarse_file)
    if not fine_file.exists():
        raise FileNotFoundError(fine_file)

    _ensure_src_on_path()
    from data_loader import ExodusDataLoader  # type: ignore

    variables_std = ["velocity_0", "velocity_1", "pressure", "temperature"]
    variables_friendly = ["velocity_x", "velocity_y", "pressure", "temperature"]
    eps = 1e-12

    with ExodusDataLoader(str(coarse_file)) as coarse_loader, ExodusDataLoader(str(fine_file)) as fine_loader:
        coarse_data = coarse_loader.load()
        fine_data = fine_loader.load()

        for v in variables_std:
            if v not in coarse_data.get("fields", {}):
                raise ValueError(f"Coarse file missing field '{v}'. Available: {list(coarse_data.get('fields', {}).keys())}")
            if v not in fine_data.get("fields", {}):
                raise ValueError(f"Fine file missing field '{v}'. Available: {list(fine_data.get('fields', {}).keys())}")

        n_coarse = _exodus_num_timesteps(coarse_loader)
        n_fine = _exodus_num_timesteps(fine_loader)
        if n_coarse <= 0 or n_fine <= 0:
            raise ValueError("Could not determine time dimension from Exodus files")

        # Valid fine timesteps t satisfy:
        #   0 <= t < n_fine
        #   0 <= t + offset < n_coarse
        # => max(0, -offset) <= t < min(n_fine, n_coarse - offset)
        t_start = max(0, -int(coarse_to_fine_offset))
        t_end = min(int(n_fine), int(n_coarse) - int(coarse_to_fine_offset))
        n = int(t_end - t_start)
        if n <= 0:
            raise ValueError(
                f"No overlapping timesteps for offset={coarse_to_fine_offset} "
                f"(coarse={n_coarse}, fine={n_fine}, fine_range=[{t_start},{t_end}))"
            )
        if max_timesteps is not None:
            n = min(n, int(max_timesteps))
            t_end = t_start + n

        print(
            f"Coarse-vs-fine extrema comparison: offset={coarse_to_fine_offset}, "
            f"fine_timesteps=[{t_start},{t_end}) count={n}"
        )

        t_idx = np.arange(t_start, t_end, dtype=np.int64)

        # Compute time series
        rel_min = {name: np.full((n,), np.nan, dtype=np.float64) for name in variables_friendly}
        rel_max = {name: np.full((n,), np.nan, dtype=np.float64) for name in variables_friendly}

        for i, t_fine in enumerate(t_idx.tolist()):
            t_coarse = int(t_fine + int(coarse_to_fine_offset))
            _, c_fields = coarse_loader.get_snapshot(int(t_coarse))
            _, f_fields = fine_loader.get_snapshot(int(t_fine))

            for std_name, friendly in zip(variables_std, variables_friendly):
                c = np.asarray(c_fields[std_name], dtype=np.float64)
                f = np.asarray(f_fields[std_name], dtype=np.float64)

                cmin = float(np.min(c))
                cmax = float(np.max(c))
                fmin = float(np.min(f))
                fmax = float(np.max(f))

                rel_min[friendly][i] = (cmin - fmin) / fmin if abs(fmin) > eps else np.nan
                rel_max[friendly][i] = (cmax - fmax) / fmax if abs(fmax) > eps else np.nan

        # Plot
        variables = list(variables_friendly)
        nvars = len(variables)
        nrows, ncols = (2, 2) if nvars <= 4 else (nvars, 1)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 3.6 * nrows), squeeze=False)

        for idx, var in enumerate(variables):
            ax = axes[idx // ncols][idx % ncols]
            ax.plot(t_idx, rel_min[var], label="(min(coarse)-min(fine))/min(fine)")
            ax.plot(t_idx, rel_max[var], label="(max(coarse)-max(fine))/max(fine)")
            ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
            ax.set_title(f"Coarse vs fine extrema relerr: {var} (offset={coarse_to_fine_offset})")
            ax.set_xlabel("timestep (fine index)")
            ax.set_ylabel("relative error")
            ax.grid(True, alpha=0.3)
            ax.legend()

        for j in range(nvars, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout()
        out_path = out_dir / "relerr_of_extrema_values_coarse_vs_fine.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path


if __name__ == "__main__":
    raise SystemExit(main())
