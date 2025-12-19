"""Physics and boundary-condition losses for mesh CFD models.

This module is intentionally optional: training behaves exactly as before unless you
explicitly enable physics losses via config and provide required metadata.

It supports deriving boundary node masks from ExodusII side sets (e.g. inlet/outlet/cylinder)
for QUAD4 meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class PhysicsConfig:
    enabled: bool = False

    # Boundary metadata
    boundary_file: Optional[str] = None

    # Loss weights
    divergence_lambda: float = 0.0
    cylinder_no_slip_lambda: float = 0.0
    cylinder_temperature_lambda: float = 0.0

    # Boundary values (optional)
    cylinder_temperature_value: Optional[float] = None

    # Numerics
    eps: float = 1e-12


def physics_config_from_training_cfg(training_cfg: Dict) -> PhysicsConfig:
    """Parse PhysicsConfig from config['training'].

    Supports either nested:
      training:
        physics:
          enabled: true
          ...

    Or flat keys for convenience.
    """
    phys = training_cfg.get("physics", {}) if isinstance(training_cfg, dict) else {}
    if not isinstance(phys, dict):
        phys = {}

    def _get(key: str, default):
        if key in phys:
            return phys.get(key, default)
        return training_cfg.get(key, default)

    return PhysicsConfig(
        enabled=bool(_get("enabled", False)),
        boundary_file=_get("boundary_file", None),
        divergence_lambda=float(_get("divergence_lambda", 0.0)),
        cylinder_no_slip_lambda=float(_get("cylinder_no_slip_lambda", 0.0)),
        cylinder_temperature_lambda=float(_get("cylinder_temperature_lambda", 0.0)),
        cylinder_temperature_value=(
            None if _get("cylinder_temperature_value", None) is None else float(_get("cylinder_temperature_value", None))
        ),
        eps=float(_get("eps", 1e-12)),
    )


def build_sideset_node_mask(exo_path: str, sideset_name: str) -> np.ndarray:
    """Return a boolean node mask for a named ExodusII side set.

    Assumes a single element block with QUAD4 connectivity.

    Args:
        exo_path: Path to .exo
        sideset_name: One of the names found in ss_names (e.g., 'cylinder')

    Returns:
        mask: (num_nodes,) boolean mask
    """
    import netCDF4 as nc
    from netCDF4 import chartostring

    ds = nc.Dataset(str(exo_path), "r")
    try:
        num_nodes = len(ds.dimensions["num_nodes"])

        if "ss_names" not in ds.variables:
            raise ValueError("Exodus file missing ss_names; cannot build sideset mask")

        ss_names = [str(x).strip() for x in chartostring(ds.variables["ss_names"][:])]
        if sideset_name not in ss_names:
            raise ValueError(f"Side set '{sideset_name}' not found. Available: {ss_names}")

        ss_idx = ss_names.index(sideset_name) + 1  # 1-based in variable naming

        elem_var = f"elem_ss{ss_idx}"
        side_var = f"side_ss{ss_idx}"
        if elem_var not in ds.variables or side_var not in ds.variables:
            raise ValueError(f"Missing sideset variables {elem_var}/{side_var} in Exodus file")

        elem_ids = np.asarray(ds.variables[elem_var][:], dtype=np.int64)
        side_ids = np.asarray(ds.variables[side_var][:], dtype=np.int64)

        # Connectivity: Exodus is typically 1-based.
        # This repo's mesh writer keeps connectivity 1-indexed.
        connect_var = None
        for v in ds.variables.keys():
            if "connect" in v.lower():
                connect_var = v
                break
        if connect_var is None:
            raise ValueError("No connectivity variable found (connect*)")

        connectivity = np.asarray(ds.variables[connect_var][:], dtype=np.int64)
        elem_type = getattr(ds.variables[connect_var], "elem_type", "").upper()
        if "QUAD" not in elem_type and connectivity.shape[1] != 4:
            raise ValueError(f"Only QUAD4 supported for sideset masks (elem_type={elem_type}, nnpe={connectivity.shape[1]})")

        # Map QUAD4 side number -> local node indices (0-based)
        # Exodus QUAD4 convention: side 1=(1,2), 2=(2,3), 3=(3,4), 4=(4,1)
        side_to_edge = {
            1: (0, 1),
            2: (1, 2),
            3: (2, 3),
            4: (3, 0),
        }

        boundary_nodes = set()
        for e_id, s_id in zip(elem_ids, side_ids):
            if int(s_id) not in side_to_edge:
                continue
            e = int(e_id) - 1  # to 0-based element index
            if e < 0 or e >= connectivity.shape[0]:
                continue
            n0, n1 = side_to_edge[int(s_id)]
            # Node ids from connectivity are 1-based
            node_a = int(connectivity[e, n0]) - 1
            node_b = int(connectivity[e, n1]) - 1
            if 0 <= node_a < num_nodes:
                boundary_nodes.add(node_a)
            if 0 <= node_b < num_nodes:
                boundary_nodes.add(node_b)

        mask = np.zeros((num_nodes,), dtype=bool)
        if boundary_nodes:
            mask[list(boundary_nodes)] = True
        return mask
    finally:
        ds.close()


def read_exodus_coords(exo_path: str) -> np.ndarray:
    """Read coordinates from an ExodusII netCDF file.

    Returns an array of shape (num_nodes, dim) where dim is 2 or 3.
    If z exists and has near-zero range, the returned coords are (x,y).
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(exo_path), "r")
    try:
        x = np.asarray(ds.variables["coordx"][:], dtype=np.float64)
        y = np.asarray(ds.variables["coordy"][:], dtype=np.float64)
        if "coordz" in ds.variables:
            z = np.asarray(ds.variables["coordz"][:], dtype=np.float64)
            coords = np.stack([x, y, z], axis=-1)
            if float(np.ptp(coords[:, 2])) < 1e-6:
                coords = coords[:, :2]
        else:
            coords = np.stack([x, y], axis=-1)
        return coords.astype(np.float32)
    finally:
        ds.close()


def transfer_mask_by_coords(
    src_coords: np.ndarray,
    src_mask: np.ndarray,
    dst_coords: np.ndarray,
    *,
    decimals: int = 8,
    min_match_fraction: float = 0.999,
) -> Tuple[np.ndarray, float]:
    """Transfer a node mask from one mesh to another by coordinate matching.

    This is useful when your fine/training mesh is a cropped subset of a larger mesh
    that contains the boundary side sets.

    Matching strategy: round coordinates to `decimals` and do exact hash lookups.

    Returns:
        dst_mask: boolean mask aligned to dst_coords
        match_fraction: fraction of dst nodes that were matched
    """
    src_coords = np.asarray(src_coords)
    dst_coords = np.asarray(dst_coords)
    src_mask = np.asarray(src_mask).astype(bool)

    if src_coords.ndim != 2 or dst_coords.ndim != 2:
        raise ValueError("src_coords and dst_coords must be 2D arrays")
    if src_coords.shape[0] != src_mask.shape[0]:
        raise ValueError("src_coords and src_mask length mismatch")

    # Ensure same dimension (2D)
    d = min(src_coords.shape[1], dst_coords.shape[1])
    src_xy = src_coords[:, :d]
    dst_xy = dst_coords[:, :d]

    def _key(arr: np.ndarray, dec: int):
        r = np.round(arr.astype(np.float64), dec)
        return [tuple(row.tolist()) for row in r]

    # Build source lookup
    src_keys = _key(src_xy, decimals)
    lookup = {}
    for i, k in enumerate(src_keys):
        # If duplicates exist, keep first; duplicates should be rare for node coords.
        if k not in lookup:
            lookup[k] = i

    dst_keys = _key(dst_xy, decimals)
    dst_mask = np.zeros((dst_xy.shape[0],), dtype=bool)
    matched = 0
    for j, k in enumerate(dst_keys):
        i = lookup.get(k)
        if i is None:
            continue
        matched += 1
        dst_mask[j] = bool(src_mask[i])

    match_fraction = matched / max(dst_xy.shape[0], 1)
    if match_fraction < min_match_fraction:
        raise ValueError(
            f"Mask transfer match too low: {match_fraction:.4f} (needed >= {min_match_fraction}). "
            f"Try different boundary_file or adjust rounding/mesh alignment."
        )
    return dst_mask, float(match_fraction)


def transfer_mask_by_coords_best_effort(
    src_coords: np.ndarray,
    src_mask: np.ndarray,
    dst_coords: np.ndarray,
    *,
    decimals_list=(8, 7, 6, 5),
    min_match_fraction: float = 0.98,
) -> Tuple[np.ndarray, float, int]:
    """Try several coordinate roundings and return the best transfer.

    Returns:
        dst_mask, match_fraction, decimals_used
    """
    best = None
    best_frac = -1.0
    best_dec = None
    last_err = None
    for dec in decimals_list:
        try:
            m, frac = transfer_mask_by_coords(
                src_coords,
                src_mask,
                dst_coords,
                decimals=int(dec),
                min_match_fraction=min_match_fraction,
            )
            if frac > best_frac:
                best = m
                best_frac = frac
                best_dec = int(dec)
        except Exception as e:
            last_err = e
            continue

    if best is None:
        raise ValueError(f"Mask transfer failed for all decimals {tuple(decimals_list)}: {last_err}")
    return best, float(best_frac), int(best_dec)


def _approx_divergence(u: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, eps: float) -> torch.Tensor:
    """Cheap graph-based divergence approximation.

    For each directed edge (i->j), compute directional derivative along r_ij:
      d_ij = ((u_j - u_i) · r_ij) / (||r_ij||^2 + eps)

    Then average outgoing contributions per source node.

    Args:
        u: (N, 2)
        coords: (N, 2)
        edge_index: (2, E)

    Returns:
        div: (N,) approx divergence
    """
    src = edge_index[0]
    dst = edge_index[1]
    r = coords[dst] - coords[src]  # (E,2)
    du = u[dst] - u[src]
    denom = (r * r).sum(dim=-1) + eps
    d = (du * r).sum(dim=-1) / denom  # (E,)

    # scatter mean onto src
    div = torch.zeros((coords.shape[0],), device=coords.device, dtype=coords.dtype)
    cnt = torch.zeros((coords.shape[0],), device=coords.device, dtype=coords.dtype)
    div.scatter_add_(0, src, d)
    cnt.scatter_add_(0, src, torch.ones_like(d))
    div = div / torch.clamp(cnt, min=1.0)
    return div


def physics_loss_from_batch(
    batch,
    model_out: torch.Tensor,
    residual_stats: Optional[Dict] = None,
    physics_cfg: Optional[PhysicsConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute optional physics/BC loss terms.

    Requires the dataset to have attached:
      - batch.coarse_interp_raw  (N,4) unnormalized interpolated baseline
      - batch.coords_raw         (N,2) raw coordinates
      - optionally batch.mask_cylinder (N,) boolean

    The model_out is assumed to be either:
      - absolute fields (N,4), or
      - residual targets (N,4) possibly normalized (if residual_stats provided)

    We infer residual mode by presence of batch.coarse_interp_raw (always present for our datasets)
    and by the caller deciding what residual_stats to pass.
    """
    if physics_cfg is None or not physics_cfg.enabled:
        zero = model_out.new_tensor(0.0)
        return zero, {}

    loss = model_out.new_tensor(0.0)
    logs: Dict[str, float] = {}

    if not hasattr(batch, "coarse_interp_raw") or not hasattr(batch, "coords_raw"):
        # Not enough info; silently do nothing to preserve backward compatibility.
        return loss, logs

    coarse_interp_raw = batch.coarse_interp_raw  # (N,4)
    coords_raw = batch.coords_raw  # (N,2)

    # Convert model output into physical correction/fields.
    # If residual_stats exist, model_out is normalized correction; de-normalize.
    correction = model_out
    if residual_stats is not None and ("mean" in residual_stats) and ("std" in residual_stats):
        mean = torch.as_tensor(residual_stats["mean"], device=model_out.device, dtype=model_out.dtype)
        std = torch.as_tensor(residual_stats["std"], device=model_out.device, dtype=model_out.dtype)
        correction = correction * std + mean

    # Assume correction-only mode: fields = baseline + correction
    fields_pred = coarse_interp_raw + correction

    # Divergence-free penalty (incompressible)
    if physics_cfg.divergence_lambda and physics_cfg.divergence_lambda > 0:
        u = fields_pred[:, 0:2]
        div = _approx_divergence(u, coords_raw, batch.edge_index, eps=physics_cfg.eps)
        l_div = (div ** 2).mean()
        loss = loss + float(physics_cfg.divergence_lambda) * l_div
        logs["div"] = float(l_div.detach().cpu())

    # Cylinder BCs if mask present
    if hasattr(batch, "mask_cylinder"):
        mask = batch.mask_cylinder
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        if physics_cfg.cylinder_no_slip_lambda and physics_cfg.cylinder_no_slip_lambda > 0:
            u_wall = fields_pred[mask, 0:2]
            if u_wall.numel() > 0:
                l_ns = (u_wall ** 2).mean()
                loss = loss + float(physics_cfg.cylinder_no_slip_lambda) * l_ns
                logs["cylinder_no_slip"] = float(l_ns.detach().cpu())

        if physics_cfg.cylinder_temperature_lambda and physics_cfg.cylinder_temperature_lambda > 0:
            if physics_cfg.cylinder_temperature_value is None:
                raise ValueError(
                    "cylinder_temperature_lambda > 0 but cylinder_temperature_value is None. "
                    "Set training.physics.cylinder_temperature_value to enforce a wall temperature."
                )
            t_wall = fields_pred[mask, 3]
            if t_wall.numel() > 0:
                target = model_out.new_tensor(float(physics_cfg.cylinder_temperature_value))
                l_t = ((t_wall - target) ** 2).mean()
                loss = loss + float(physics_cfg.cylinder_temperature_lambda) * l_t
                logs["cylinder_temperature"] = float(l_t.detach().cpu())

    return loss, logs
