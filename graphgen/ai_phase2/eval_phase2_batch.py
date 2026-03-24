from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graphgen.ai_phase2.dataset import load_dxf_paths
from graphgen.ai_phase2.refine_graph import refine_assignments
from graphgen.ai_phase2.reward import (
    count_dead_ends,
    count_station_reachability_issues,
    score_assignment_quality,
)
from src.build_graph import (
    build_graph,
    count_nonholonomic_branch_violations,
    parse_dxf,
    solve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate phase2 baseline/heuristic/PPO across multiple DXFs")
    p.add_argument("--dxf", default=None, help="Single DXF path (compatibility mode)")
    p.add_argument("--dxf_dir", default=None, help="Directory with .dxf files for batch evaluation")
    p.add_argument("--dxf_list", default=None, help="Text/JSON manifest listing DXFs")
    p.add_argument(
        "--ppo_model",
        default=None,
        help="PPO model path. Defaults to outputs/models/phase2_ppo_Drawing1.zip",
    )
    p.add_argument(
        "--model_stem",
        default=None,
        help="Optional artifact stem. Resolves to outputs/models/<stem>_<model_variant>.zip",
    )
    p.add_argument(
        "--model_variant",
        default="best",
        choices=["best", "last"],
        help="Model variant used with --model_stem (default: best)",
    )
    p.add_argument("--out_dir", default="outputs/phase2_batch_eval", help="Batch output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic PPO rollout")
    return p.parse_args()


def _metrics(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
) -> dict[str, float | int]:
    return {
        "score": score_assignment_quality(edge_list, adj, station_nodes, assign),
        "dead_ends": count_dead_ends(edge_list, adj, assign),
        "station_issues": count_station_reachability_issues(edge_list, station_nodes, assign),
        "nonholonomic_violations": count_nonholonomic_branch_violations(edge_list, adj, assign),
    }


def _write_report(summary: dict[str, Any], report_path: Path) -> None:
    report_path.write_text(
        "\n".join(
            [
                "# Phase2 Batch Evaluation Report",
                "",
                "This report summarizes a **baseline** PPO comparison over multiple layouts.",
                "PPO may be worse than heuristic depending on the current training quality.",
                "",
                f"- layouts evaluated: {summary['layout_count']}",
                f"- average phase1 score: {summary['avg_phase1_score']:.3f}",
                f"- average heuristic score: {summary['avg_heuristic_score']:.3f}",
                f"- average ppo score: {summary['avg_ppo_score']:.3f}",
                f"- PPO beat phase1 count: {summary['ppo_beat_phase1_count']}",
                f"- PPO beat heuristic count: {summary['ppo_beat_heuristic_count']}",
                "",
                "## Notes",
                "- This is still a small PPO baseline, not a final policy.",
                "- Use per-layout JSON files for detailed diagnostics.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        dxf_paths = load_dxf_paths(dxf=args.dxf, dxf_dir=args.dxf_dir, dxf_list=args.dxf_list)
    except Exception as exc:
        print(f"[ERROR] Invalid dataset input: {exc}")
        return 1

    if args.ppo_model:
        ppo_model = Path(args.ppo_model)
    elif args.model_stem:
        ppo_model = Path("outputs/models") / f"{args.model_stem}_{args.model_variant}.zip"
    else:
        ppo_model = Path("outputs/models/phase2_ppo_Drawing1.zip")
    if not ppo_model.exists():
        print(f"[ERROR] PPO model file not found: {ppo_model}")
        return 1

    out_dir = Path(args.out_dir)
    per_layout_dir = out_dir / "per_layout"
    per_layout_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for idx, dxf_path in enumerate(dxf_paths, start=1):
        print(f"[{idx}/{len(dxf_paths)}] Evaluating {dxf_path}")
        lines, arcs, texts = parse_dxf(dxf_path)
        edge_list, adj, by_ends, station_nodes = build_graph(lines, arcs, texts)
        _, _, phase1_assign = solve(edge_list, adj, by_ends, station_nodes)
        heuristic_assign = refine_assignments(
            edge_list=edge_list,
            adj=adj,
            station_nodes=station_nodes,
            assign=phase1_assign,
            mode="heuristic",
        )
        ppo_assign = refine_assignments(
            edge_list=edge_list,
            adj=adj,
            station_nodes=station_nodes,
            assign=phase1_assign,
            mode="ppo",
            model_path=ppo_model,
            seed=args.seed,
        )

        phase1 = _metrics(edge_list, adj, station_nodes, phase1_assign)
        heuristic = _metrics(edge_list, adj, station_nodes, heuristic_assign)
        ppo = _metrics(edge_list, adj, station_nodes, ppo_assign)

        row = {
            "layout": dxf_path.stem,
            "dxf": str(dxf_path),
            "phase1_score": phase1["score"],
            "phase1_dead_ends": phase1["dead_ends"],
            "phase1_station_issues": phase1["station_issues"],
            "phase1_nonholonomic_violations": phase1["nonholonomic_violations"],
            "heuristic_score": heuristic["score"],
            "heuristic_dead_ends": heuristic["dead_ends"],
            "heuristic_station_issues": heuristic["station_issues"],
            "heuristic_nonholonomic_violations": heuristic["nonholonomic_violations"],
            "ppo_score": ppo["score"],
            "ppo_dead_ends": ppo["dead_ends"],
            "ppo_station_issues": ppo["station_issues"],
            "ppo_nonholonomic_violations": ppo["nonholonomic_violations"],
            "ppo_improved_vs_phase1": bool(ppo["score"] > phase1["score"]),
            "ppo_improved_vs_heuristic": bool(ppo["score"] > heuristic["score"]),
        }
        rows.append(row)

        (per_layout_dir / f"{dxf_path.stem}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary = {
        "layout_count": len(rows),
        "avg_phase1_score": sum(float(r["phase1_score"]) for r in rows) / len(rows),
        "avg_heuristic_score": sum(float(r["heuristic_score"]) for r in rows) / len(rows),
        "avg_ppo_score": sum(float(r["ppo_score"]) for r in rows) / len(rows),
        "ppo_beat_phase1_count": sum(1 for r in rows if r["ppo_improved_vs_phase1"]),
        "ppo_beat_heuristic_count": sum(1 for r in rows if r["ppo_improved_vs_heuristic"]),
        "rows": rows,
    }

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json_path = out_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_md = out_dir / "REPORT.md"
    _write_report(summary, report_md)

    print(f"[DONE] layouts={summary['layout_count']} out_dir={out_dir}")
    print(f"       summary.csv={csv_path}")
    print(f"       summary.json={summary_json_path}")
    print(f"       report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
