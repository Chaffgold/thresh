#!/usr/bin/env python3
"""Generate a small .safetensors file for unit tests.

Tensors created:
  - encoder.layer.0.weight  shape [256, 4]   (f32)
  - encoder.layer.0.bias    shape [256]       (f32)
  - decoder.head.weight     shape [7, 256]    (f32)

Output: test-data/models/test_weights.safetensors
"""

import os
import numpy as np

try:
    from safetensors.numpy import save_file
except ImportError:
    raise SystemExit(
        "safetensors Python package required: pip install safetensors numpy"
    )


def main():
    rng = np.random.default_rng(42)

    tensors = {
        "encoder.layer.0.weight": rng.standard_normal((256, 4)).astype(np.float32),
        "encoder.layer.0.bias": rng.standard_normal((256,)).astype(np.float32),
        "decoder.head.weight": rng.standard_normal((7, 256)).astype(np.float32),
    }

    # Resolve output path relative to the repo root (parent of scripts/).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    out_dir = os.path.join(repo_root, "test-data", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_weights.safetensors")

    save_file(tensors, out_path)
    print(f"Wrote {out_path}")
    for name, arr in tensors.items():
        print(f"  {name:40s} shape={arr.shape}  dtype={arr.dtype}")


if __name__ == "__main__":
    main()
