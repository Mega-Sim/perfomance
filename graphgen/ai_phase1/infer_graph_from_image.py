"""Infer directed graph from standard-style image using phase-1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from graphgen.ai_phase1.arrow_direction import assign_arrows_to_edges, detect_arrow_components, finalize_edge_directions
from graphgen.ai_phase1.color_model import CLASS_NAMES, ColorModel, heuristic_masks, load_thresholds_from_spec
from graphgen.ai_phase1.skeleton import zhang_suen_thinning

NODE_COLORS = {"split": (255, 0, 0), "merge": (160, 0, 200), "station": (0, 0, 255)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_area_px", type=int, default=30)
    p.add_argument("--snap_radius", type=float, default=20.0)
    p.add_argument("--arrow_min_area", type=int, default=50)
    p.add_argument("--arrow_max_dist", type=float, default=30.0)
    return p.parse_args()


def _segment(arr: np.ndarray, model: ColorModel) -> dict[str, np.ndarray]:
    pix = arr.astype(np.int16)
    masks = heuristic_masks(pix, load_thresholds_from_spec())
    covered = np.zeros(arr.shape[:2], dtype=bool)
    for name in CLASS_NAMES:
        covered |= masks[name]

    if np.any(~covered):
        unknown_idx = np.where(~covered)
        pixels = arr[unknown_idx].astype(np.float32)
        cls_order = list(model.classes.keys())
        means = np.stack([model.classes[n]["mean"] for n in cls_order], axis=0)
        stds = np.stack([model.classes[n]["std"] for n in cls_order], axis=0)
        z = (pixels[:, None, :] - means[None, :, :]) / stds[None, :, :]
        scores = np.sum(z * z, axis=2)
        winners = np.argmin(scores, axis=1)
        for i, cls_idx in enumerate(winners):
            y = int(unknown_idx[0][i])
            x = int(unknown_idx[1][i])
            masks[cls_order[int(cls_idx)]][y, x] = True

    return {
        "track": masks["track_green"],
        "split": masks["split_red"],
        "merge": masks["merge_purple"],
        "station": masks["station_blue"],
        "arrow": masks["arrow_black"],
    }


def _detect_nodes(mask: np.ndarray, node_type: str, min_area: int, nodes: list[dict]) -> None:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    for idx in range(1, n + 1):
        ys, xs = np.where(labels == idx)
        if xs.size < min_area:
            continue
        nodes.append(
            {
                "id": len(nodes),
                "type": node_type,
                "x": float(xs.mean()),
                "y": float(ys.mean()),
            }
        )


def _snap_nodes_to_skeleton(nodes: list[dict], skel: np.ndarray, radius: float) -> dict[int, tuple[int, int]]:
    skel_ys, skel_xs = np.where(skel)
    skel_coords = np.stack([skel_xs, skel_ys], axis=1) if skel_xs.size else np.zeros((0, 2))
    anchors: dict[int, tuple[int, int]] = {}
    for node in nodes:
        p = np.array([node["x"], node["y"]], dtype=np.float32)
        if skel_coords.size == 0:
            node["snap_failed"] = True
            continue
        d = np.linalg.norm(skel_coords - p[None, :], axis=1)
        k = int(np.argmin(d))
        if float(d[k]) <= radius:
            anchors[node["id"]] = (int(skel_coords[k][0]), int(skel_coords[k][1]))
            node["snap_failed"] = False
        else:
            node["snap_failed"] = True
    return anchors


def _neighbors8(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h:
                out.append((nx, ny))
    return out


def _extract_edges_from_skeleton(skel: np.ndarray, anchors: dict[int, tuple[int, int]]) -> list[dict]:
    h, w = skel.shape
    skel_set = {(int(x), int(y)) for y, x in np.argwhere(skel)}
    anchor_to_id = {v: k for k, v in anchors.items()}
    visited_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pair_best: dict[tuple[int, int], dict] = {}

    def seg_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    for aid, ap in anchors.items():
        for nb in _neighbors8(ap[0], ap[1], w, h):
            if nb not in skel_set:
                continue
            if seg_key(ap, nb) in visited_segments:
                continue

            path = [ap, nb]
            prev = ap
            cur = nb
            visited_segments.add(seg_key(ap, nb))
            target_id = None

            while True:
                if cur in anchor_to_id and cur != ap:
                    target_id = anchor_to_id[cur]
                    break
                nxts = [p for p in _neighbors8(cur[0], cur[1], w, h) if p in skel_set and p != prev]
                if not nxts:
                    break
                if len(nxts) > 1:
                    anchor_nxt = [p for p in nxts if p in anchor_to_id]
                    if anchor_nxt:
                        nxt = anchor_nxt[0]
                    else:
                        prev_vec = np.array([cur[0] - prev[0], cur[1] - prev[1]], dtype=np.float32)
                        best_dot = -1e9
                        nxt = nxts[0]
                        for cand in nxts:
                            cand_vec = np.array([cand[0] - cur[0], cand[1] - cur[1]], dtype=np.float32)
                            d = float(np.dot(prev_vec, cand_vec))
                            if d > best_dot:
                                best_dot = d
                                nxt = cand
                else:
                    nxt = nxts[0]
                if nxt in path:
                    break
                visited_segments.add(seg_key(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt

            if target_id is None or target_id == aid:
                continue
            edge = {
                "u": aid,
                "v": target_id,
                "path": [[int(x), int(y)] for x, y in path],
                "votes": {"uv": 0, "vu": 0},
            }
            key = tuple(sorted((aid, target_id)))
            if key not in pair_best or len(path) > len(pair_best[key]["path"]):
                pair_best[key] = edge

    return list(pair_best.values())


def _apply_topology_fill(nodes: list[dict], edges: list[dict]) -> None:
    by_node: dict[int, list[int]] = {}
    for i, e in enumerate(edges):
        by_node.setdefault(e["u"], []).append(i)
        by_node.setdefault(e["v"], []).append(i)

    node_types = {n["id"]: n["type"] for n in nodes}

    changed = True
    while changed:
        changed = False
        for nid, incident in by_node.items():
            ntype = node_types.get(nid)
            if ntype not in {"split", "merge"}:
                continue

            in_cnt = 0
            out_cnt = 0
            unknown: list[int] = []
            for ei in incident:
                e = edges[ei]
                if e["dir"] is None:
                    unknown.append(ei)
                    continue
                is_u = e["u"] == nid
                if (e["dir"] == "u->v" and is_u) or (e["dir"] == "v->u" and not is_u):
                    out_cnt += 1
                else:
                    in_cnt += 1

            if len(unknown) != 1:
                continue

            e = edges[unknown[0]]
            if ntype == "split":
                forced = in_cnt >= 1 and out_cnt == 0
                if forced and (e["confidence"] < 0.2 or e["dir"] is None):
                    e["dir"] = "u->v" if e["u"] == nid else "v->u"
                    changed = True
            elif ntype == "merge":
                forced = out_cnt >= 1 and in_cnt == 0
                if forced and (e["confidence"] < 0.2 or e["dir"] is None):
                    e["dir"] = "v->u" if e["u"] == nid else "u->v"
                    changed = True


def _save_preview_svg(
    image: np.ndarray,
    nodes: list[dict],
    edges: list[dict],
    arrows: list,
    out_path: Path,
) -> None:
    h, w = image.shape[:2]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="rgb(255,255,255)"/>',
    ]

    for edge in edges:
        if len(edge["path"]) >= 2:
            pts = " ".join(f"{p[0]},{p[1]}" for p in edge["path"])
            lines.append(f'<polyline points="{pts}" fill="none" stroke="rgb(20,20,20)" stroke-width="2"/>')
            if edge["dir"] is not None:
                mid = len(edge["path"]) // 2
                p1 = np.array(edge["path"][max(mid - 2, 0)], dtype=np.float32)
                p2 = np.array(edge["path"][min(mid + 2, len(edge["path"]) - 1)], dtype=np.float32)
                if edge["dir"] == "v->u":
                    p1, p2 = p2, p1
                d = p2 - p1
                n = np.linalg.norm(d)
                if n > 1e-6:
                    d /= n
                    tip = p2
                    left = tip - 8 * d + 4 * np.array([-d[1], d[0]])
                    right = tip - 8 * d - 4 * np.array([-d[1], d[0]])
                    lines.append(
                        f'<polygon points="{tip[0]},{tip[1]} {left[0]},{left[1]} {right[0]},{right[1]}" fill="rgb(255,140,0)"/>'
                    )

    for node in nodes:
        x, y = node["x"], node["y"]
        c = NODE_COLORS.get(node["type"], (255, 255, 0))
        lines.append(
            f'<circle cx="{x}" cy="{y}" r="4" fill="rgb({c[0]},{c[1]},{c[2]})" stroke="rgb(0,0,0)" stroke-width="1"/>'
        )

    for arrow in arrows:
        c = np.array(arrow.center)
        d = np.array(arrow.direction)
        p2 = c + 20 * d
        lines.append(
            f'<line x1="{c[0]}" y1="{c[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="rgb(0,0,0)" stroke-width="2"/>'
        )

    lines.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _save_preview_png(
    image: np.ndarray,
    nodes: list[dict],
    edges: list[dict],
    arrows: list,
    out_path: Path,
) -> None:
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    for edge in edges:
        if len(edge["path"]) >= 2:
            draw.line([tuple(p) for p in edge["path"]], fill=(20, 20, 20), width=2)
            if edge["dir"] is not None:
                mid = len(edge["path"]) // 2
                p1 = np.array(edge["path"][max(mid - 2, 0)], dtype=np.float32)
                p2 = np.array(edge["path"][min(mid + 2, len(edge["path"]) - 1)], dtype=np.float32)
                if edge["dir"] == "v->u":
                    p1, p2 = p2, p1
                d = p2 - p1
                n = np.linalg.norm(d)
                if n > 1e-6:
                    d /= n
                    tip = p2
                    left = tip - 8 * d + 4 * np.array([-d[1], d[0]])
                    right = tip - 8 * d - 4 * np.array([-d[1], d[0]])
                    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=(255, 140, 0))

    for node in nodes:
        x, y = node["x"], node["y"]
        c = NODE_COLORS.get(node["type"], (255, 255, 0))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=c, outline=(0, 0, 0))

    for arrow in arrows:
        c = np.array(arrow.center)
        d = np.array(arrow.direction)
        p2 = c + 20 * d
        draw.line((float(c[0]), float(c[1]), float(p2[0]), float(p2[1])), fill=(0, 0, 0), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    model = ColorModel.load(args.model)
    seg = _segment(arr, model)

    nodes: list[dict] = []
    _detect_nodes(seg["split"], "split", args.min_area_px, nodes)
    _detect_nodes(seg["merge"], "merge", args.min_area_px, nodes)
    _detect_nodes(seg["station"], "station", args.min_area_px, nodes)

    base = seg["track"] | seg["split"] | seg["merge"]
    skel = zhang_suen_thinning(base)
    anchors = _snap_nodes_to_skeleton(nodes, skel, args.snap_radius)
    edges = _extract_edges_from_skeleton(skel, anchors)

    arrows = detect_arrow_components(seg["arrow"], min_area=args.arrow_min_area)
    assign_arrows_to_edges(arrows, edges, max_distance=args.arrow_max_dist)
    finalize_edge_directions(edges)
    _apply_topology_fill(nodes, edges)

    graph = {
        "source_image": str(args.image),
        "nodes": [{"id": n["id"], "type": n["type"], "x": n["x"], "y": n["y"]} for n in nodes],
        "edges": [
            {
                "u": e["u"],
                "v": e["v"],
                "dir": e["dir"],
                "confidence": float(e.get("confidence", 0.0)),
                "votes": {"uv": int(e["votes"]["uv"]), "vu": int(e["votes"]["vu"])},
            }
            for e in edges
        ],
    }

    with (out_dir / "graph.json").open("w", encoding="utf-8") as fp:
        json.dump(graph, fp, ensure_ascii=False, indent=2)

    _save_preview_svg(arr, nodes, edges, arrows, out_dir / "preview.svg")
    _save_preview_png(arr, nodes, edges, arrows, out_dir / "preview.png")
    print(f"[OK] graph.json -> {out_dir / 'graph.json'}")
    print(f"[OK] preview.svg -> {out_dir / 'preview.svg'}")
    print(f"[OK] preview.png(local) -> {out_dir / 'preview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
