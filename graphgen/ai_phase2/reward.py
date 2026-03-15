from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


def _build_directed_adj(edge_list: list[dict[str, Any]], assign: dict[int, int]) -> dict[tuple[float, float], set[tuple[float, float]]]:
    directed = defaultdict(set)
    for edge in edge_list:
        eid = edge["id"]
        if assign.get(eid, 0) == 0:
            src, dst = edge["u"], edge["v"]
        else:
            src, dst = edge["v"], edge["u"]
        directed[src].add(dst)
    return directed


def count_dead_ends(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    assign: dict[int, int],
) -> int:
    """Count nodes with no incoming or no outgoing edge."""
    directed = _build_directed_adj(edge_list, assign)
    incoming = defaultdict(int)
    outgoing = defaultdict(int)

    for src, dsts in directed.items():
        outgoing[src] += len(dsts)
        for dst in dsts:
            incoming[dst] += 1

    n_dead_ends = 0
    for node in adj.keys():
        if incoming[node] == 0 or outgoing[node] == 0:
            n_dead_ends += 1
    return n_dead_ends


def count_station_reachability_issues(
    edge_list: list[dict[str, Any]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
) -> int:
    """Count stations unreachable from at least one other station in directed traversal."""
    stations = list(station_nodes.values())
    if len(stations) <= 1:
        return 0

    directed = _build_directed_adj(edge_list, assign)

    def reachable_from(start: tuple[float, float]) -> set[tuple[float, float]]:
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in directed.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    issues = 0
    for source in stations:
        reach = reachable_from(source)
        if any(target not in reach for target in stations if target != source):
            issues += 1
    return issues


def score_assignment_quality(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
) -> float:
    """Deterministic scalar score used as an RL-ready reward baseline."""
    dead_ends = count_dead_ends(edge_list, adj, assign)
    station_issues = count_station_reachability_issues(edge_list, station_nodes, assign)
    return float(-(2 * dead_ends + 5 * station_issues))
