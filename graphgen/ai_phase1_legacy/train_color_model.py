"""CLI for training the phase-1 color prototype model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .color_model import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = sorted(Path(args.images_dir).glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {args.images_dir}")

    images = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in image_paths]
    model, counts = train_model(images)
    model.save(args.out)

    print(f"[OK] Saved model to {args.out}")
    for name in sorted(counts):
        print(f"{name}: {counts[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
