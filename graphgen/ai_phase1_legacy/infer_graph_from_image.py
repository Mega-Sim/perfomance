"""Infer directed graph from standard-style image using phase-1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from graphgen.spec import load_standard_spec

from .arrow_direction import (
    assign_arrows_to_edges,
    detect_arrow_components,
    estimate_track_scale,
    finalize_edge_directions,
)
from .color_model import ColorModel, heuristic_masks, load_thresholds_from_spec
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


def _merge_nearby_nodes(
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    radius: float,
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    if not nodes:
        return nodes, edges
    r2 = float(radius * radius)

    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (x0, y0) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            x1, y1 = nodes[j]
            if (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) <= r2:
                union(i, j)

    clusters: dict[int, list[tuple[int, int]]] = {}
    for idx, p in enumerate(nodes):
        clusters.setdefault(find(idx), []).append(p)

    merged_nodes: list[tuple[int, int]] = []
    rep_to_new: dict[int, int] = {}
    for rep, pts in clusters.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        rep_to_new[rep] = len(merged_nodes)
        merged_nodes.append((int(round(float(np.mean(xs)))), int(round(float(np.mean(ys))))))

    old_to_new = {i: rep_to_new[find(i)] for i in range(len(nodes))}
    point_to_old = {p: i for i, p in enumerate(nodes)}

    merged_edges_set: set[tuple[int, int]] = set()
    for a, b in edges:
        ia = point_to_old.get(a)
        ib = point_to_old.get(b)
        if ia is None or ib is None:
            continue
        na = old_to_new[ia]
        nb = old_to_new[ib]
        if na == nb:
            continue
        e = (na, nb) if na < nb else (nb, na)
        merged_edges_set.add(e)

    merged_edges = [(merged_nodes[u], merged_nodes[v]) for (u, v) in sorted(merged_edges_set)]
    return merged_nodes, merged_edges


def main() -> int:
    args = parse_args()
    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)

    model = ColorModel.load(args.model)
    # NOTE: prototype model unused in this simplified legacy inference; keep for compatibility
    spec = load_standard_spec()
    thresholds = load_thresholds_from_spec(spec)

    masks = heuristic_masks(img, thresholds)

    # Skeletonize track only so marker pixels do not pollute topology.
    track_mask = masks.get("track_green", np.zeros(img.shape[:2], dtype=bool))
    track_scale = estimate_track_scale(track_mask)
    sk = zhang_suen_thinning(track_mask.astype(np.uint8))

    nodes, edges = _extract_graph_from_skeleton(sk)
    nodes, edges = _merge_nearby_nodes(nodes, edges, radius=max(2.0, 0.8 * track_scale))

    # arrows
    arrow_mask = masks.get("arrow_black", np.zeros(img.shape[:2], dtype=bool))
    min_arrow_area = max(16, int(1.5 * track_scale * track_scale))
    arrows = detect_arrow_components(arrow_mask, min_area=min_arrow_area)
    votes = assign_arrows_to_edges(arrows, edges, max_distance=max(8.0, 3.5 * track_scale))
    bits = finalize_edge_directions(edges, votes)

    # dump json
    node_index = {p: i for i, p in enumerate(nodes)}
    out = {
        "nodes": [{"id": i, "x": int(p[0]), "y": int(p[1])} for i, p in enumerate(nodes)],
        "edges": [
            {"id": i, "u": int(node_index[u]), "v": int(node_index[v]), "bit": int(bits[i])}
            for i, (u, v) in enumerate(edges)
            if u in node_index and v in node_index
        ],
        "meta": {
            "legacy": True,
            "threshold_source": "spec.phase1.color_thresholds_or_image_fallback",
            "track_scale_px": float(track_scale),
        },
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
