"""Phase 2 graph refinement scaffold (RL-ready)."""

from .features import EdgeFeature, extract_edge_features
from .rl_env import GraphDirectionRefineEnv, build_candidate_edge_ids
from .refine_graph import refine_assignments
from .reward import (
    count_dead_ends,
    count_station_reachability_issues,
    score_assignment_quality,
)

__all__ = [
    "EdgeFeature",
    "extract_edge_features",
    "GraphDirectionRefineEnv",
    "build_candidate_edge_ids",
    "refine_assignments",
    "count_dead_ends",
    "count_station_reachability_issues",
    "score_assignment_quality",
]
