from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class EdgeFeature:
    """Compact, deterministic feature bundle for one edge."""

    edge_index: int
    u: tuple[float, float]
    v: tuple[float, float]
    length: float
    degree_u: int
    degree_v: int
    has_station_endpoint: bool
    is_outer_loop_candidate: bool


def _edge_length(edge: dict[str, Any], u: tuple[float, float], v: tuple[float, float]) -> float:
    geom = edge.get("geom", {})
    if geom.get("type") == "ARC" and "pts" in geom:
        pts = geom["pts"]
        if len(pts) >= 2:
            return sum(
                math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                for i in range(len(pts) - 1)
            )
    return math.hypot(v[0] - u[0], v[1] - u[1])


def _bbox_nodes(nodes: set[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    return min(xs), max(xs), min(ys), max(ys)


def _on_bbox_boundary(p: tuple[float, float], bbox: tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    minx, maxx, miny, maxy = bbox
    return (
        abs(p[0] - minx) <= eps
        or abs(p[0] - maxx) <= eps
        or abs(p[1] - miny) <= eps
        or abs(p[1] - maxy) <= eps
    )


def extract_edge_features(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
) -> list[EdgeFeature]:
    """Extract per-edge features for phase2 refinement decisions."""
    nodes = set(adj.keys())
    if not nodes:
        return []

    station_points = set(station_nodes.values())
    bbox = _bbox_nodes(nodes)
    features: list[EdgeFeature] = []

    for edge in edge_list:
        u = edge["u"]
        v = edge["v"]
        features.append(
            EdgeFeature(
                edge_index=edge["id"],
                u=u,
                v=v,
                length=_edge_length(edge, u, v),
                degree_u=len(adj.get(u, [])),
                degree_v=len(adj.get(v, [])),
                has_station_endpoint=(u in station_points or v in station_points),
                is_outer_loop_candidate=(_on_bbox_boundary(u, bbox) and _on_bbox_boundary(v, bbox)),
            )
        )

    return features
