"""Arrow component direction detection and edge vote assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class ArrowItem:
    center: tuple[float, float]
    direction: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return v / n


def detect_arrow_components(arrow_mask: np.ndarray, min_area: int = 50) -> list[ArrowItem]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n = ndimage.label(arrow_mask, structure=structure)
    arrows: list[ArrowItem] = []

    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        area = int(xs.size)
        if area < min_area:
            continue
        coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        center = coords.mean(axis=0)
        centered = coords - center
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        v = _normalize(vecs[:, np.argmax(vals)])
        vp = np.array([-v[1], v[0]], dtype=np.float32)

        t = centered @ v
        span = float(t.max() - t.min())
        if span < 1e-6:
            continue
        w = max(1e-3, 0.15 * span)
        near_max = centered[t >= (t.max() - w)]
        near_min = centered[t <= (t.min() + w)]
        spread_max = float(np.std(near_max @ vp)) if near_max.size else 0.0
        spread_min = float(np.std(near_min @ vp)) if near_min.size else 0.0

        if spread_max >= spread_min:
            direction = v
        else:
            direction = -v

        arrows.append(
            ArrowItem(
                center=(float(center[0]), float(center[1])),
                direction=(float(direction[0]), float(direction[1])),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                area=area,
            )
        )
    return arrows


def assign_arrows_to_edges(
    arrows: list[ArrowItem],
    edges: list[dict],
    max_distance: float = 30.0,
) -> None:
    for edge in edges:
        edge.setdefault("votes", {"uv": 0, "vu": 0})

    for arrow in arrows:
        center = np.array(arrow.center, dtype=np.float32)
        best_idx = -1
        best_dist = float("inf")
        best_k = 0

        for i, edge in enumerate(edges):
            path = np.asarray(edge["path"], dtype=np.float32)
            if path.size == 0:
                continue
            d = np.linalg.norm(path - center, axis=1)
            k = int(np.argmin(d))
            if float(d[k]) < best_dist:
                best_dist = float(d[k])
                best_idx = i
                best_k = k

        if best_idx < 0 or best_dist > max_distance:
            continue

        edge = edges[best_idx]
        path = np.asarray(edge["path"], dtype=np.float32)
        lo = max(best_k - 5, 0)
        hi = min(best_k + 5, len(path) - 1)
        tangent = _normalize(path[hi] - path[lo])
        ad = _normalize(np.asarray(arrow.direction, dtype=np.float32))
        if float(np.dot(tangent, ad)) >= 0.0:
            edge["votes"]["uv"] += 1
        else:
            edge["votes"]["vu"] += 1


def finalize_edge_directions(edges: list[dict]) -> None:
    for edge in edges:
        uv = int(edge["votes"]["uv"])
        vu = int(edge["votes"]["vu"])
        total = uv + vu
        if total == 0:
            edge["dir"] = None
            edge["confidence"] = 0.0
        elif uv > vu:
            edge["dir"] = "u->v"
            edge["confidence"] = abs(uv - vu) / total
        else:
            edge["dir"] = "v->u"
            edge["confidence"] = abs(uv - vu) / total
