from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass
from typing import Any

from src.build_graph import count_nonholonomic_branch_violations


Action = tuple[str, int, int | None]


@dataclass
class RepairState:
    node_degree: dict[tuple[float, float], int]
    edge_dirs: dict[int, int]
    violation_flags: dict[str, int]
    tangent_inconsistency: int


@dataclass
class GraphCase:
    name: str
    edge_list: list[dict[str, Any]]
    adj: dict[tuple[float, float], set[tuple[float, float]]]
    station_nodes: dict[str, tuple[float, float]]
    rule_assign: dict[int, int]


def _directed_edge(edge: dict[str, Any], assign: dict[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
    eid = edge["id"]
    return (edge["u"], edge["v"]) if assign.get(eid, 0) == 0 else (edge["v"], edge["u"])


def _vec_from_node(edge: dict[str, Any], node: tuple[float, float], assign: dict[int, int]) -> tuple[float, float]:
    src, dst = _directed_edge(edge, assign)
    if node == src:
        other = dst
        sign = 1.0
    elif node == dst:
        other = src
        sign = -1.0
    elif edge["u"] == node:
        other = edge["v"]
        sign = 0.0
    else:
        other = edge["u"]
        sign = 0.0
    return (other[0] - node[0], other[1] - node[1] if sign >= 0 else node[1] - other[1])


def _norm(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def count_invalid_split_merge(edge_list: list[dict[str, Any]], adj: dict[tuple[float, float], set[tuple[float, float]]], assign: dict[int, int]) -> int:
    by_node = collections.defaultdict(lambda: {"in": 0, "out": 0})
    for e in edge_list:
        src, dst = _directed_edge(e, assign)
        by_node[src]["out"] += 1
        by_node[dst]["in"] += 1

    invalid = 0
    for n in adj:
        deg = len(adj[n])
        if deg < 3:
            continue
        i = by_node[n]["in"]
        o = by_node[n]["out"]
        if not ((i == 1 and o >= 2) or (o == 1 and i >= 2)):
            invalid += 1
    return invalid


def count_tangent_inconsistency(edge_list: list[dict[str, Any]], adj: dict[tuple[float, float], set[tuple[float, float]]], assign: dict[int, int]) -> int:
    incident = collections.defaultdict(list)
    edge_by_id = {e["id"]: e for e in edge_list}
    for e in edge_list:
        incident[e["u"]].append(e["id"])
        incident[e["v"]].append(e["id"])

    inconsistency = 0
    for node, eids in incident.items():
        if len(eids) != 2:
            continue
        e1, e2 = edge_by_id[eids[0]], edge_by_id[eids[1]]
        v1 = _vec_from_node(e1, node, assign)
        v2 = _vec_from_node(e2, node, assign)
        if _norm(v1) == 0.0 or _norm(v2) == 0.0:
            continue
        cos = abs(_dot(v1, v2) / (_norm(v1) * _norm(v2)))
        if cos < 0.92:
            continue
        src1, dst1 = _directed_edge(e1, assign)
        src2, dst2 = _directed_edge(e2, assign)
        both_in = (dst1 == node and dst2 == node)
        both_out = (src1 == node and src2 == node)
        if both_in or both_out:
            inconsistency += 1
    return inconsistency


def evaluate_violations(edge_list: list[dict[str, Any]], adj: dict[tuple[float, float], set[tuple[float, float]]], station_nodes: dict[str, tuple[float, float]], assign: dict[int, int]) -> dict[str, int]:
    return {
        "nonholonomic": count_nonholonomic_branch_violations(edge_list, adj, assign),
        "invalid_split_merge": count_invalid_split_merge(edge_list, adj, assign),
        "tangent_inconsistency": count_tangent_inconsistency(edge_list, adj, assign),
        "station_unreachable": 0 if len(station_nodes) <= 1 else 0,
    }


def score_from_violations(v: dict[str, int]) -> float:
    return float(
        -(50 * v["nonholonomic"] + 40 * v["invalid_split_merge"] + 10 * v["tangent_inconsistency"] + 4 * v["station_unreachable"])
    )


class LocalDirectionRepairEnv:
    def __init__(
        self,
        edge_list: list[dict[str, Any]],
        adj: dict[tuple[float, float], set[tuple[float, float]]],
        station_nodes: dict[str, tuple[float, float]],
        initial_assign: dict[int, int],
        max_steps: int = 8,
    ):
        self.edge_list = edge_list
        self.adj = adj
        self.station_nodes = station_nodes
        self.initial_assign = dict(initial_assign)
        self.max_steps = max_steps
        self._step = 0
        self.assign = dict(initial_assign)

    def reset(self) -> RepairState:
        self._step = 0
        self.assign = dict(self.initial_assign)
        return self._state()

    def _state(self) -> RepairState:
        v = evaluate_violations(self.edge_list, self.adj, self.station_nodes, self.assign)
        return RepairState(
            node_degree={n: len(nei) for n, nei in self.adj.items()},
            edge_dirs={e["id"]: self.assign.get(e["id"], 0) for e in self.edge_list},
            violation_flags=v,
            tangent_inconsistency=v["tangent_inconsistency"],
        )

    def available_actions(self) -> list[Action]:
        return [("flip", e["id"], None) for e in self.edge_list]

    def step(self, action: Action) -> tuple[RepairState, float, bool, dict[str, Any]]:
        before = evaluate_violations(self.edge_list, self.adj, self.station_nodes, self.assign)
        if action[0] == "flip":
            eid = action[1]
            self.assign[eid] = 1 - self.assign.get(eid, 0)
        elif action[0] == "assign":
            eid = action[1]
            assert action[2] in (0, 1)
            self.assign[eid] = int(action[2])

        after = evaluate_violations(self.edge_list, self.adj, self.station_nodes, self.assign)
        reward = score_from_violations(after) - score_from_violations(before)

        self._step += 1
        done = self._step >= self.max_steps or score_from_violations(after) >= 0
        info = {
            "before": before,
            "after": after,
            "improved": sum(after.values()) < sum(before.values()),
        }
        return self._state(), float(reward), done, info


def synthetic_graph_cases() -> list[GraphCase]:
    cases: list[GraphCase] = []

    def mk(name: str, edges: list[tuple[tuple[float, float], tuple[float, float], str]]) -> GraphCase:
        edge_list = []
        adj = collections.defaultdict(set)
        for i, (u, v, kind) in enumerate(edges):
            edge_list.append({"id": i, "u": u, "v": v, "kind": kind, "geom": {"type": "LINE", "a": u, "b": v}})
            adj[u].add(v)
            adj[v].add(u)
        # intentionally imperfect starting point: all zero direction
        rule_assign = {e["id"]: 0 for e in edge_list}
        return GraphCase(name=name, edge_list=edge_list, adj=adj, station_nodes={}, rule_assign=rule_assign)

    cases.append(mk("straight_line", [((0, 0), (1, 0), "L"), ((1, 0), (2, 0), "L")]))
    cases.append(mk("simple_loop", [((0, 0), (1, 0), "L"), ((1, 0), (1, 1), "L"), ((1, 1), (0, 1), "L"), ((0, 1), (0, 0), "L")]))
    cases.append(mk("split_1_to_2", [((0, 0), (1, 0), "L"), ((1, 0), (2, 1), "L"), ((1, 0), (2, -1), "L")]))
    cases.append(mk("merge_2_to_1", [((0, 1), (1, 0), "L"), ((0, -1), (1, 0), "L"), ((1, 0), (2, 0), "L")]))
    cases.append(mk("line_arc", [((0, 0), (1, 0), "L"), ((1, 0), (2, 1), "A"), ((2, 1), (3, 1), "L")]))
    cases.append(mk("mixed_small", [((0, 0), (1, 0), "L"), ((1, 0), (2, 0), "L"), ((2, 0), (2, 1), "L"), ((2, 1), (1, 1), "L"), ((1, 1), (1, 0), "L"), ((2, 0), (3, -1), "L")]))
    return cases


def random_repair(edge_list, adj, station_nodes, assign, seed: int = 0, steps: int = 8):
    rng = random.Random(seed)
    env = LocalDirectionRepairEnv(edge_list, adj, station_nodes, assign, max_steps=steps)
    env.reset()
    for _ in range(steps):
        action = rng.choice(env.available_actions())
        _, _, done, _ = env.step(action)
        if done:
            break
    return env.assign
