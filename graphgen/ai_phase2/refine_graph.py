from __future__ import annotations

from typing import Any

from .features import extract_edge_features
from .reward import score_assignment_quality


SUPPORTED_MODES = {"heuristic"}


def _decide_flip_heuristic(feature, current: int) -> int:
    """Deterministic placeholder policy; only keep/flip is allowed."""
    if feature.has_station_endpoint:
        return current
    if feature.is_outer_loop_candidate and feature.degree_u == 2 and feature.degree_v == 2:
        return current
    if feature.degree_u == 1 and feature.degree_v > 1:
        return 1
    if feature.degree_v == 1 and feature.degree_u > 1:
        return 0
    return current


def refine_assignments(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
    mode: str = "heuristic",
) -> dict[int, int]:
    """Refine directed edge assignment in a phase2 pass.

    Current scope is direction refinement only (keep/flip per edge).
    Topology editing is intentionally out of scope.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported phase2 mode: {mode}")

    features = extract_edge_features(edge_list, adj, station_nodes)
    refined_assign = dict(assign)

    if mode == "heuristic":
        for feature in features:
            eid = feature.edge_index
            current = refined_assign.get(eid, 0)
            candidate = _decide_flip_heuristic(feature, current)
            if candidate != current:
                trial = dict(refined_assign)
                trial[eid] = candidate
                if score_assignment_quality(edge_list, adj, station_nodes, trial) >= score_assignment_quality(
                    edge_list, adj, station_nodes, refined_assign
                ):
                    refined_assign[eid] = candidate

    # TODO: Inject learned RL/policy model here.
    # TODO: Replace per-edge greedy evaluation with trajectory/batch policy rollout.
    return refined_assign
