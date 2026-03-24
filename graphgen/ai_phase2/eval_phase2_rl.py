from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graphgen.ai_phase2.refine_graph import refine_assignments
from graphgen.ai_phase2.reward import (
    count_dead_ends,
    count_station_reachability_issues,
    score_assignment_quality,
)
from src.build_graph import (
    build_graph,
    count_nonholonomic_branch_violations,
    dump_graph,
    parse_dxf,
    render,
    solve,
)


@dataclass(frozen=True)
class EvalMetrics:
    score: float
    dead_ends: int
    station_issues: int
    nonholonomic_violations: int


def _metrics(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
) -> EvalMetrics:
    return EvalMetrics(
        score=score_assignment_quality(edge_list, adj, station_nodes, assign),
        dead_ends=count_dead_ends(edge_list, adj, assign),
        station_issues=count_station_reachability_issues(edge_list, station_nodes, assign),
        nonholonomic_violations=count_nonholonomic_branch_violations(edge_list, adj, assign),
    )


def _render_and_dump(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
    png_out: Path,
    json_out: Path,
) -> None:
    render(edge_list, adj, station_nodes, assign, png_out, arrow_scale=6)
    dump_graph(edge_list, station_nodes, assign, json_out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate phase2 heuristic and PPO refinements")
    p.add_argument("--dxf", default="examples/Drawing1.dxf", help="Input DXF file path")
    p.add_argument(
        "--ppo_model",
        default=None,
        help="PPO model path for phase2 PPO inference",
    )
    p.add_argument(
        "--out_dir",
        default="outputs/phase2_eval",
        help="Evaluation output directory",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for PPO inference rollout")
    return p.parse_args()


def _line(label: str, m: EvalMetrics) -> str:
    return (
        f"{label:>10} | score={m.score:>7.1f} | dead_ends={m.dead_ends:>3d} | "
        f"station_issues={m.station_issues:>3d} | nonholonomic={m.nonholonomic_violations:>3d}"
    )


def main() -> int:
    args = parse_args()
    dxf_path = Path(args.dxf)
    dxf_name = dxf_path.stem
    model_path = Path(args.ppo_model) if args.ppo_model else Path("outputs/models") / f"phase2_ppo_{dxf_name}.zip"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dxf_path.exists():
        print(f"[ERROR] DXF file not found: {dxf_path}")
        return 1
    if not model_path.exists():
        print(f"[ERROR] PPO model file not found: {model_path}")
        return 1

    print(f"[1/5] Building graph from {dxf_path}")
    lines, arcs, texts = parse_dxf(dxf_path)
    edge_list, adj, by_ends, station_nodes = build_graph(lines, arcs, texts)

    print("[2/5] Solving phase1 baseline")
    _, _, phase1_assign = solve(edge_list, adj, by_ends, station_nodes)

    print("[3/5] Running heuristic phase2")
    heuristic_assign = refine_assignments(
        edge_list=edge_list,
        adj=adj,
        station_nodes=station_nodes,
        assign=phase1_assign,
        mode="heuristic",
    )

    phase1_metrics = _metrics(edge_list, adj, station_nodes, phase1_assign)
    heuristic_metrics = _metrics(edge_list, adj, station_nodes, heuristic_assign)

    phase1_png = out_dir / f"{dxf_name}_phase1.png"
    phase1_json = out_dir / f"{dxf_name}_phase1.json"
    heuristic_png = out_dir / f"{dxf_name}_heuristic.png"
    heuristic_json = out_dir / f"{dxf_name}_heuristic.json"
    ppo_png = out_dir / f"{dxf_name}_ppo.png"
    ppo_json = out_dir / f"{dxf_name}_ppo.json"

    _render_and_dump(edge_list, adj, station_nodes, phase1_assign, phase1_png, phase1_json)
    _render_and_dump(edge_list, adj, station_nodes, heuristic_assign, heuristic_png, heuristic_json)

    print("[4/5] Running PPO phase2")
    try:
        ppo_assign = refine_assignments(
            edge_list=edge_list,
            adj=adj,
            station_nodes=station_nodes,
            assign=phase1_assign,
            mode="ppo",
            model_path=model_path,
            seed=args.seed,
        )
    except Exception as exc:
        report_md = out_dir / f"{dxf_name}_report.md"
        report_md.write_text(
            "\n".join(
                [
                    "# Phase2 RL Evaluation Report",
                    "",
                    f"- source dxf: `{dxf_path}`",
                    f"- ppo model: `{model_path}`",
                    f"- seed: {args.seed}",
                    "",
                    "## Metrics (available)",
                    "",
                    "| Path | total score | dead ends | station reachability issues | nonholonomic branch violations |",
                    "|---|---:|---:|---:|---:|",
                    (
                        f"| phase1 | {phase1_metrics.score:.1f} | {phase1_metrics.dead_ends} | "
                        f"{phase1_metrics.station_issues} | {phase1_metrics.nonholonomic_violations} |"
                    ),
                    (
                        f"| phase2 heuristic | {heuristic_metrics.score:.1f} | {heuristic_metrics.dead_ends} | "
                        f"{heuristic_metrics.station_issues} | {heuristic_metrics.nonholonomic_violations} |"
                    ),
                    "",
                    "## PPO status",
                    f"- failed to run PPO evaluation: `{exc}`",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[ERROR] PPO refinement failed: {exc}")
        print(f"[INFO] Partial report written: {report_md}")
        return 1

    print("[5/5] Computing and saving report")
    ppo_metrics = _metrics(edge_list, adj, station_nodes, ppo_assign)
    _render_and_dump(edge_list, adj, station_nodes, ppo_assign, ppo_png, ppo_json)

    improved_vs_initial = ppo_metrics.score > phase1_metrics.score
    improved_vs_heuristic = ppo_metrics.score > heuristic_metrics.score

    report_md = out_dir / f"{dxf_name}_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Phase2 RL Evaluation Report",
                "",
                f"- source dxf: `{dxf_path}`",
                f"- ppo model: `{model_path}`",
                f"- seed: {args.seed}",
                "",
                "## Metrics",
                "",
                "| Path | total score | dead ends | station reachability issues | nonholonomic branch violations |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| phase1 | {phase1_metrics.score:.1f} | {phase1_metrics.dead_ends} | "
                    f"{phase1_metrics.station_issues} | {phase1_metrics.nonholonomic_violations} |"
                ),
                (
                    f"| phase2 heuristic | {heuristic_metrics.score:.1f} | {heuristic_metrics.dead_ends} | "
                    f"{heuristic_metrics.station_issues} | {heuristic_metrics.nonholonomic_violations} |"
                ),
                (
                    f"| phase2 ppo | {ppo_metrics.score:.1f} | {ppo_metrics.dead_ends} | "
                    f"{ppo_metrics.station_issues} | {ppo_metrics.nonholonomic_violations} |"
                ),
                "",
                "## Improvement checks",
                f"- PPO improved vs initial: **{improved_vs_initial}**",
                f"- PPO improved vs heuristic: **{improved_vs_heuristic}**",
                "",
                "## Artifacts",
                f"- `{phase1_png}`",
                f"- `{heuristic_png}`",
                f"- `{ppo_png}`",
                f"- `{phase1_json}`",
                f"- `{heuristic_json}`",
                f"- `{ppo_json}`",
            ]
        ),
        encoding="utf-8",
    )

    console_lines = [
        _line("phase1", phase1_metrics),
        _line("heuristic", heuristic_metrics),
        _line("ppo", ppo_metrics),
        f"ppo improved vs initial: {improved_vs_initial}",
        f"ppo improved vs heuristic: {improved_vs_heuristic}",
        f"report: {report_md}",
    ]
    print("\n".join(console_lines))

    summary_json = out_dir / f"{dxf_name}_report.json"
    summary_json.write_text(
        json.dumps(
            {
                "dxf": str(dxf_path),
                "ppo_model": str(model_path),
                "seed": args.seed,
                "phase1": phase1_metrics.__dict__,
                "heuristic": heuristic_metrics.__dict__,
                "ppo": ppo_metrics.__dict__,
                "ppo_improved_vs_initial": improved_vs_initial,
                "ppo_improved_vs_heuristic": improved_vs_heuristic,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
