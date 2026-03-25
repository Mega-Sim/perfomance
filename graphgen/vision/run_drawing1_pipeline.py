from __future__ import annotations

import argparse
from pathlib import Path

from .infer import infer_image_to_graph
from .render_dxf import render_dxf_to_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render DXF then run image->graph inference")
    p.add_argument("--dxf", default="examples/Drawing1.dxf")
    p.add_argument("--tmp_image", default="tmp/drawing1.png")
    p.add_argument("--out", default="outputs/drawing1_graph.json")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--min_component", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    render_dxf_to_image(Path(args.dxf), Path(args.tmp_image), size=args.image_size)
    infer_image_to_graph(
        image=args.tmp_image,
        out=args.out,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        min_component=args.min_component,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
