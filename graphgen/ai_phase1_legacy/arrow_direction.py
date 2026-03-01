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
    if n == 0:
        return v
    return v / n


def detect_arrow_components(mask: np.ndarray, min_area: int = 40) -> list[ArrowItem]:
    """Detect arrow components in a binary mask."""
    lab, n = ndimage.label(mask.astype(np.uint8))
    items: list[ArrowItem] = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        area = len(xs)
        if area < min_area:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = float(xs.mean())
        cy = float(ys.mean())

        # PCA on pixels to estimate major axis direction
        pts = np.stack([xs - cx, ys - cy], axis=1).astype(np.float32)
        cov = pts.T @ pts
        w, v = np.linalg.eigh(cov)
        major = v[:, np.argmax(w)]
        major = _normalize(major)

        # Use skewness along major axis to decide head direction
        proj = pts @ major
        # compare mean of top 10% vs bottom 10%
        hi = np.percentile(proj, 90)
        lo = np.percentile(proj, 10)
        hi_mean = proj[proj >= hi].mean() if np.any(proj >= hi) else 0.0
        lo_mean = proj[proj <= lo].mean() if np.any(proj <= lo) else 0.0
        direction = major if hi_mean > -lo_mean else -major

        items.append(
            ArrowItem(
                center=(cx, cy),
                direction=(float(direction[0]), float(direction[1])),
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                area=int(area),
            )
        )
    return items


def assign_arrows_to_edges(arrows: list[ArrowItem], edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> dict[int, list[tuple[float, float]]]:
    """Assign arrows to nearest edge and return edge->list(direction vectors)."""
    votes: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(edges))}
    for a in arrows:
        cx, cy = a.center
        best_i = None
        best_d = 1e18
        for i, (p0, p1) in enumerate(edges):
            x0, y0 = p0
            x1, y1 = p1
            vx = x1 - x0
            vy = y1 - y0
            vv = vx * vx + vy * vy
            if vv == 0:
                continue
            t = ((cx - x0) * vx + (cy - y0) * vy) / vv
            t = max(0.0, min(1.0, t))
            px = x0 + t * vx
            py = y0 + t * vy
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None:
            votes[best_i].append(a.direction)
    return votes


def finalize_edge_directions(edges: list[tuple[tuple[int, int], tuple[int, int]]], votes: dict[int, list[tuple[float, float]]]) -> list[int]:
    """Return list of direction bits for edges: 0 keep p0->p1, 1 flip."""
    bits: list[int] = []
    for i, (p0, p1) in enumerate(edges):
        v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float32)
        v = _normalize(v)
        if i not in votes or not votes[i]:
            bits.append(0)
            continue
        s = 0.0
        for d in votes[i]:
            d = _normalize(np.array(d, dtype=np.float32))
            s += float(np.dot(v, d))
        bits.append(0 if s >= 0 else 1)
    return bits
