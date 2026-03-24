from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import CLASS_ORDER, load_image_tensor
from .model import TinyUNet


COLORS = np.array(
    [
        [255, 255, 255],
        [0, 170, 0],
        [220, 50, 50],
        [180, 50, 220],
        [50, 80, 220],
        [20, 20, 20],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run minimal vision inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True, help="Output PNG path for predicted segmentation image")
    p.add_argument("--image_size", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = TinyUNet(num_classes=len(CLASS_ORDER)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = load_image_tensor(args.image, image_size=args.image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    color_pred = COLORS[pred]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_pred).save(out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
