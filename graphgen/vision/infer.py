from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .dataset import CLASS_ORDER, load_image_tensor
from .model import TinyUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run image-based inference and emit a simple graph JSON")
    p.add_argument("--image", required=True, help="Input raster image")
    p.add_argument("--out", required=True, help="Output graph JSON path")
    p.add_argument("--checkpoint", default=None, help="Optional model checkpoint (.pt)")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--min_component", type=int, default=20, help="Ignore tiny mask components")
    return p.parse_args()


def _resolve_checkpoint(path: str | None) -> str | None:
    if path:
        return path
    default = Path("outputs/vision/tiny_unet.pt")
    if default.exists():
        print(f"[vision.infer] using default checkpoint: {default}")
        return str(default)
    return None


def _predict_classes(model: TinyUNet, image_path: str, image_size: int, device: torch.device) -> np.ndarray:
    x = load_image_tensor(image_path, image_size=image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def _pick_farthest_pair(coords: np.ndarray, max_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) == 1:
        return coords[0], coords[0]
    if len(coords) > max_samples:
        idx = np.linspace(0, len(coords) - 1, max_samples).astype(np.int64)
        coords = coords[idx]

    best_i, best_j, best_d = 0, 1, -1.0
    for i in range(len(coords)):
        d2 = np.sum((coords[i + 1 :] - coords[i]) ** 2, axis=1)
        if len(d2) == 0:
            continue
        j_rel = int(np.argmax(d2))
        if d2[j_rel] > best_d:
            best_d = float(d2[j_rel])
            best_i = i
            best_j = i + 1 + j_rel
    return coords[best_i], coords[best_j]


def mask_to_graph(mask: np.ndarray, min_component: int = 20) -> dict:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndi.label(mask.astype(np.uint8), structure=structure)

    nodes: list[dict] = []
    edges: list[dict] = []
    node_id = 0
    edge_id = 0

    for comp_id in range(1, count + 1):
        comp = labels == comp_id
        area = int(comp.sum())
        if area < min_component:
            continue

        neighbor_count = ndi.convolve(comp.astype(np.uint8), structure, mode="constant", cval=0) - comp.astype(np.uint8)
        endpoints = np.argwhere(comp & (neighbor_count <= 1))
        junctions = np.argwhere(comp & (neighbor_count >= 3))
        if len(endpoints) < 2:
            endpoints = np.argwhere(comp)

        p0, p1 = _pick_farthest_pair(endpoints.astype(np.float64))
        n0 = {"id": node_id, "x": float(p0[1]), "y": float(p0[0]), "kind": "endpoint"}
        n1 = {"id": node_id + 1, "x": float(p1[1]), "y": float(p1[0]), "kind": "endpoint"}
        nodes.extend([n0, n1])
        node_id += 2

        if len(junctions) > 0:
            step = max(1, len(junctions) // 10)
            for jp in junctions[::step]:
                nodes.append({"id": node_id, "x": float(jp[1]), "y": float(jp[0]), "kind": "junction"})
                node_id += 1

        edges.append(
            {
                "id": edge_id,
                "u": n0["id"],
                "v": n1["id"],
                "component": comp_id,
                "pixels": area,
            }
        )
        edge_id += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "components_total": int(count),
            "components_used": int(len(edges)),
            "mask_pixels": int(mask.sum()),
        },
    }


def main() -> int:
    args = parse_args()
    infer_image_to_graph(
        image=args.image,
        out=args.out,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        min_component=args.min_component,
    )
    return 0


def infer_image_to_graph(
    image: str,
    out: str,
    checkpoint: str | None = None,
    image_size: int = 256,
    min_component: int = 20,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(num_classes=len(CLASS_ORDER)).to(device)
    ckpt_path = _resolve_checkpoint(checkpoint)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "image_size" in ckpt and image_size != int(ckpt["image_size"]):
            print(
                f"[vision.infer] checkpoint image_size={ckpt['image_size']} "
                f"but using --image_size={image_size}",
            )
    else:
        print("[vision.infer] no checkpoint provided; using random-initialized model")
    model.eval()

    pred = _predict_classes(model, image, image_size, device)

    track_idx = CLASS_ORDER.index("track")
    track_mask = pred == track_idx
    if not track_mask.any():
        track_mask = pred != 0

    graph = mask_to_graph(track_mask, min_component=min_component)
    graph["meta"] = {
        "image": str(image),
        "checkpoint": ckpt_path,
        "image_size": image_size,
        "class_order": CLASS_ORDER,
        "mask_mode": "track" if (pred == track_idx).any() else "non_background_fallback",
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(str(out_path))
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
