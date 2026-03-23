from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - surfaced by explicit runtime checks.
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None

from .features import EdgeFeature, extract_edge_features
from .reward import (
    count_dead_ends,
    count_station_reachability_issues,
    score_assignment_quality,
)
from src.build_graph import count_nonholonomic_branch_violations


@dataclass(frozen=True)
class GlobalMetrics:
    dead_ends: int
    station_issues: int
    nonholonomic_violations: int


def build_candidate_edge_ids(features: list[EdgeFeature]) -> list[int]:
    """Deterministic candidate edge filter for refinement actions."""
    candidate_ids: list[int] = []
    for feature in sorted(features, key=lambda f: f.edge_index):
        if feature.has_station_endpoint:
            continue
        if feature.is_outer_loop_candidate and feature.degree_u == 2 and feature.degree_v == 2:
            continue
        candidate_ids.append(feature.edge_index)
    return candidate_ids


def compute_global_metrics(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
) -> GlobalMetrics:
    return GlobalMetrics(
        dead_ends=count_dead_ends(edge_list, adj, assign),
        station_issues=count_station_reachability_issues(edge_list, station_nodes, assign),
        nonholonomic_violations=count_nonholonomic_branch_violations(edge_list, adj, assign),
    )


if gym is not None:

    class GraphDirectionRefineEnv(gym.Env):
        """Binary keep/flip RL environment for phase2 edge direction refinement."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            edge_list: list[dict[str, Any]],
            adj: dict[tuple[float, float], set[tuple[float, float]]],
            station_nodes: dict[str, tuple[float, float]],
            initial_assign: dict[int, int],
            candidate_edge_ids: list[int] | None = None,
            max_steps: int | None = None,
            step_penalty: float = 0.01,
            final_improvement_bonus: float = 0.0,
        ):
            super().__init__()
            self.edge_list = edge_list
            self.adj = adj
            self.station_nodes = station_nodes
            self.initial_assign = dict(initial_assign)
            self.features = extract_edge_features(edge_list, adj, station_nodes)
            self.feature_by_eid = {f.edge_index: f for f in self.features}

            if candidate_edge_ids is None:
                candidate_edge_ids = build_candidate_edge_ids(self.features)
            self.candidate_edge_ids = list(candidate_edge_ids)
            self.max_steps = max_steps if max_steps is not None else len(self.candidate_edge_ids)
            self.max_steps = max(self.max_steps, 1)
            self.step_penalty = float(step_penalty)
            self.final_improvement_bonus = float(final_improvement_bonus)

            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=np.full((11,), -1e6, dtype=np.float32),
                high=np.full((11,), 1e6, dtype=np.float32),
                dtype=np.float32,
            )

            self.current_assign: dict[int, int] = {}
            self.step_index = 0
            self.initial_score = 0.0
            self.current_score = 0.0

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            self.current_assign = dict(self.initial_assign)
            self.step_index = 0
            self.initial_score = score_assignment_quality(
                self.edge_list, self.adj, self.station_nodes, self.current_assign
            )
            self.current_score = self.initial_score
            return self._build_observation(), {}

        def step(self, action: int):
            if not self.candidate_edge_ids:
                return self._build_observation(), 0.0, True, False, {"score": self.current_score}

            terminated = False
            truncated = False
            edge_id = self.candidate_edge_ids[min(self.step_index, len(self.candidate_edge_ids) - 1)]
            old_score = self.current_score

            if int(action) == 1:
                self.current_assign[edge_id] = 1 - self.current_assign.get(edge_id, 0)

            new_score = score_assignment_quality(
                self.edge_list, self.adj, self.station_nodes, self.current_assign
            )
            reward = float(new_score - old_score - self.step_penalty)
            self.current_score = new_score
            self.step_index += 1

            if self.step_index >= min(len(self.candidate_edge_ids), self.max_steps):
                terminated = True
                if self.current_score > self.initial_score:
                    reward += self.final_improvement_bonus

            info = {
                "score": self.current_score,
                "initial_score": self.initial_score,
                "edge_id": edge_id,
                "step_index": self.step_index,
            }
            return self._build_observation(), reward, terminated, truncated, info

        def _build_observation(self) -> np.ndarray:
            if not self.candidate_edge_ids:
                return np.zeros((11,), dtype=np.float32)

            eid = self.candidate_edge_ids[min(self.step_index, len(self.candidate_edge_ids) - 1)]
            feat = self.feature_by_eid[eid]
            metrics = compute_global_metrics(
                self.edge_list, self.adj, self.station_nodes, self.current_assign
            )
            cur_bit = float(self.current_assign.get(eid, 0))
            remaining = max(0, min(len(self.candidate_edge_ids), self.max_steps) - self.step_index)
            denom = max(1, min(len(self.candidate_edge_ids), self.max_steps))

            obs = np.array(
                [
                    float(feat.length),
                    float(feat.degree_u),
                    float(feat.degree_v),
                    1.0 if feat.has_station_endpoint else 0.0,
                    1.0 if feat.is_outer_loop_candidate else 0.0,
                    float(metrics.dead_ends),
                    float(metrics.station_issues),
                    float(metrics.nonholonomic_violations),
                    float(self.step_index) / float(denom),
                    float(remaining) / float(denom),
                    cur_bit,
                ],
                dtype=np.float32,
            )
            return obs

        @property
        def final_assign(self) -> dict[int, int]:
            return dict(self.current_assign)

else:

    class GraphDirectionRefineEnv:  # pragma: no cover - used only when gymnasium missing.
        def __init__(self, *args, **kwargs):
            raise ImportError(f"gymnasium is required for RL env: {_GYM_IMPORT_ERROR}")
