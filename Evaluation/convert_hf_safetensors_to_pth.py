#!/usr/bin/env python3
"""Convert a Hugging Face safetensors checkpoint into a torch checkpoint.

The output format is compatible with this repository's training scripts, which
expect either a raw state_dict or a checkpoint dict with a "model" entry.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "safetensors is not installed. Install it with `pip install safetensors` "
        "in the Python environment you will use for conversion."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face safetensors weights to a torch .pth checkpoint."
    )
    parser.add_argument(
        "input",
        help="Path to the input model.safetensors file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .pth path. Defaults to <input_stem>.pth next to the source file.",
    )
    parser.add_argument(
        "--raw-state-dict",
        action="store_true",
        help="Save only the raw state_dict instead of wrapping it in {'model': state_dict}.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return input_path.with_suffix(".pth").resolve()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input safetensors file not found: {input_path}")

    output_path = resolve_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading safetensors checkpoint: {input_path}")
    state_dict = load_file(str(input_path), device="cpu")
    tensor_count = len(state_dict)
    param_count = sum(t.numel() for t in state_dict.values())

    if args.raw_state_dict:
        payload = state_dict
    else:
        payload = {
            "model": state_dict,
            "source": {
                "format": "safetensors",
                "path": str(input_path),
            },
        }

    torch.save(payload, output_path)

    size_bytes = os.path.getsize(output_path)
    print(f"Saved torch checkpoint: {output_path}")
    print(f"Tensor entries: {tensor_count}")
    print(f"Total parameters: {param_count}")
    print(f"Output size: {size_bytes / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()
