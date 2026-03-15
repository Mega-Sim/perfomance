"""Evaluate graph topology metrics for predicted vs ground-truth graph JSON.

Expected schema (minimal):
- nodes: [{id, x, y, type?}]
- edges: [{source, target, direction?}]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    return parser.parse_args()


def _node_key(node: dict, tolerance: float = 3.0) -> tuple[int, int, str]:
    x = int(round(float(node.get("x", 0.0)) / tolerance))
    y = int(round(float(node.get("y", 0.0)) / tolerance))
    t = str(node.get("type", "unknown"))
    return (x, y, t)


def _edge_key(edge: dict) -> tuple[int, int]:
    u = int(edge.get("source", edge.get("u", -1)))
    v = int(edge.get("target", edge.get("v", -1)))
    return (u, v)


def _direction_value(edge: dict) -> str:
    if "direction" in edge:
        return str(edge["direction"])
    bit = int(edge.get("bit", 0))
    return "reverse" if bit == 1 else "forward"


def main() -> int:
    args = parse_args()
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))

    gt_nodes = {_node_key(n) for n in gt.get("nodes", [])}
    pr_nodes = {_node_key(n) for n in pred.get("nodes", [])}

    tp_nodes = len(gt_nodes & pr_nodes)
    node_precision = tp_nodes / len(pr_nodes) if pr_nodes else 0.0
    node_recall = tp_nodes / len(gt_nodes) if gt_nodes else 0.0

    gt_edges = {_edge_key(e) for e in gt.get("edges", [])}
    pr_edges = {_edge_key(e) for e in pred.get("edges", [])}
    tp_edges = gt_edges & pr_edges
    edge_recovery_rate = len(tp_edges) / len(gt_edges) if gt_edges else 0.0

    gt_dir = {_edge_key(e): _direction_value(e) for e in gt.get("edges", [])}
    pr_dir = {_edge_key(e): _direction_value(e) for e in pred.get("edges", [])}
    direction_hits = 0
    shared = set(gt_dir.keys()) & set(pr_dir.keys())
    for edge in shared:
        if gt_dir[edge] == pr_dir[edge]:
            direction_hits += 1
    direction_accuracy = direction_hits / len(shared) if shared else 0.0

    topology_consistency = 1.0 if node_recall == 1.0 and edge_recovery_rate == 1.0 else 0.0

    result = {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "edge_recovery_rate": edge_recovery_rate,
        "direction_accuracy": direction_accuracy,
        "topology_consistency": topology_consistency,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
