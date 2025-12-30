"""Device selection helpers.

Centralizes logic for choosing a torch device from user input / config.
"""

from typing import Optional, Union

import torch


def format_available_cuda_devices() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"

    count = torch.cuda.device_count()
    parts = [f"CUDA devices: {count}"]
    for idx in range(count):
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "<unknown>"
        parts.append(f"  - cuda:{idx}  {name}")
    return "\n".join(parts)


def select_torch_device(requested: Optional[Union[str, int]] = None) -> torch.device:
    """Select a torch device.

    Supported values:
    - None: auto-select (cuda:0 if available else cpu)
    - "cpu"
    - "cuda" (alias for cuda:0)
    - "cuda:N" (e.g. cuda:1)
    - "N" or int N (alias for cuda:N)

    Raises:
        ValueError: if the requested device is invalid or unavailable.
    """

    if requested is None or (isinstance(requested, str) and requested.strip() == ""):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(requested, int):
        requested_str = str(requested)
    else:
        requested_str = str(requested).strip().lower()

    if requested_str == "cpu":
        return torch.device("cpu")

    if requested_str == "cuda":
        requested_str = "cuda:0"

    # Allow passing just an index: "0" -> "cuda:0"
    if requested_str.isdigit():
        requested_str = f"cuda:{requested_str}"

    if requested_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "Requested a CUDA device but CUDA is not available. "
                "Use device='cpu' or install/configure CUDA."
            )

        # Validate cuda index if specified
        if requested_str == "cuda":
            requested_str = "cuda:0"

        if ":" in requested_str:
            try:
                _, idx_str = requested_str.split(":", 1)
                idx = int(idx_str)
            except Exception as exc:
                raise ValueError(f"Invalid device string: {requested!r}") from exc

            count = torch.cuda.device_count()
            if idx < 0 or idx >= count:
                raise ValueError(
                    f"Requested CUDA device cuda:{idx} but only {count} CUDA device(s) are visible.\n"
                    + format_available_cuda_devices()
                )

            # Make the selected device current for any implicit allocations.
            try:
                torch.cuda.set_device(idx)
            except Exception:
                # Not fatal; .to(device) will still target the correct device.
                pass

        return torch.device(requested_str)

    # Fall back: allow any torch.device-parsable string (e.g. "mps" on mac)
    try:
        return torch.device(requested_str)
    except Exception as exc:
        raise ValueError(
            f"Invalid device {requested!r}. Expected 'cpu', 'cuda', 'cuda:N', or an index like '0'."
        ) from exc
