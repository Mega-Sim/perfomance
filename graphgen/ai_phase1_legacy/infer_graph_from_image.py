"""Infer directed graph from standard-style image using phase-1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .arrow_direction import assign_arrows_to_edges, detect_arrow_components, finalize_edge_directions
from .color_model import CLASS_NAMES, ColorModel, heuristic_masks, load_thresholds_from_spec
from .skeleton import zhang_suen_thinning

NODE_COLORS = {"split": (255, 0, 0), "merge": (160, 0, 200), "station": (0, 0, 255)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--debug_dir", default="")
    return p.parse_args()


def _neighbors(p: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = p
    return [
        (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
        (x - 1, y),                 (x + 1, y),
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
    ]


def _extract_graph_from_skeleton(sk: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Extract nodes/edges from skeleton pixels (very simplified)."""
    ys, xs = np.where(sk > 0)
    pts = set(zip(xs.tolist(), ys.tolist()))
    deg = {}
    for p in pts:
        c = 0
        for q in _neighbors(p):
            if q in pts:
                c += 1
        deg[p] = c

    nodes = [p for p, d in deg.items() if d != 2]
    # edges: connect nodes by BFS along skeleton
    edges = []
    visited = set()

    node_set = set(nodes)

    for n in nodes:
        for q in _neighbors(n):
            if q not in pts:
                continue
            if (n, q) in visited or (q, n) in visited:
                continue
            # follow path until another node
            path = [n, q]
            prev = n
            cur = q
            while cur not in node_set:
                nxts = [t for t in _neighbors(cur) if t in pts and t != prev]
                if not nxts:
                    break
                # if branch, stop
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                path.append(cur)
            if cur in node_set and cur != n:
                edges.append((n, cur))
            for i in range(len(path) - 1):
                visited.add((path[i], path[i + 1]))

    return nodes, edges


def main() -> int:
    args = parse_args()
    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)

    model = ColorModel.load(args.model)
    # NOTE: prototype model unused in this simplified legacy inference; keep for compatibility
    spec = {"version": model.meta.get("spec_version", "unknown"), "phase1": {}}
    thresholds = load_thresholds_from_spec(spec)

    masks = heuristic_masks(img, thresholds)

    # skeletonize track
    track = masks.get("track_green", np.zeros(img.shape[:2], dtype=bool)).astype(np.uint8)
    sk = zhang_suen_thinning(track)

    nodes, edges = _extract_graph_from_skeleton(sk)

    # arrows
    arrow_mask = masks.get("arrow_black", np.zeros(img.shape[:2], dtype=bool))
    arrows = detect_arrow_components(arrow_mask)
    votes = assign_arrows_to_edges(arrows, edges)
    bits = finalize_edge_directions(edges, votes)

    # dump json
    out = {
        "nodes": [{"id": i, "x": int(p[0]), "y": int(p[1])} for i, p in enumerate(nodes)],
        "edges": [{"id": i, "u": int(u), "v": int(v), "bit": int(bits[i])} for i, ((u, v),) in enumerate(zip(edges))],
        "meta": {"legacy": True},
    }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # debug overlay
    if args.debug_dir:
        dbg = Path(args.debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for p in nodes:
            dr.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), fill=(255, 0, 0))
        for (a, b) in edges:
            dr.line((a[0], a[1], b[0], b[1]), fill=(255, 255, 0), width=1)
        im.save(dbg / "overlay.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
