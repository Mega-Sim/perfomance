"""
Phase-1 pipeline runner — DXF → Graph direct path.

기존의 이미지 경유 역추출(DXF→PNG→세그멘테이션→스켈레톤→그래프)을 제거하고
build_graph.py의 정확한 DXF 파싱 결과를 직접 사용합니다.

Closes #12: https://github.com/Mega-Sim/perfomance/issues/12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# build_graph 모듈의 핵심 함수들을 직접 import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.build_graph import (
    parse_dxf,
    build_graph,
    solve,
    render,
    render_svg,
    dump_graph,
)

OUTPUT_ROOT = Path("outputs/phase1_ai")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DXF → directed graph (direct path, no image round-trip)"
    )
    p.add_argument("--dxf", required=True, help="Input DXF file path")
    p.add_argument(
        "--out_dir",
        default=str(OUTPUT_ROOT),
        help="Output directory (default: outputs/phase1_ai)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dxf_path = Path(args.dxf)
    if not dxf_path.exists():
        print(f"[ERROR] DXF file not found: {dxf_path}")
        return 1

    # ── Step 1: DXF 파싱 & 그래프 구축 ──
    print(f"[1/4] Parsing DXF: {dxf_path}")
    lines, arcs, texts = parse_dxf(dxf_path)
    edge_list, adj, by_ends, station_nodes = build_graph(lines, arcs, texts)
    print(f"      nodes={len(adj)}, edges={len(edge_list)}, stations={len(station_nodes)}")

    # ── Step 2: 방향 결정 (outer-loop CW + propagation) ──
    print("[2/4] Solving edge directions...")
    bits, score, assign = solve(edge_list, adj, by_ends, station_nodes)

    # ── Step 3: 출력 생성 ──
    print("[3/4] Generating outputs...")

    render_png = out_dir / "directed_graph.png"
    render(edge_list, adj, station_nodes, assign, render_png, arrow_scale=6)

    graph_json = out_dir / "graph.json"
    dump_graph(edge_list, station_nodes, assign, str(graph_json))

    preview_svg = out_dir / "preview.svg"
    render_svg(edge_list, adj, station_nodes, assign, preview_svg)

    # ── Step 4: 리포트 생성 ──
    print("[4/4] Writing report...")

    graph = json.loads(graph_json.read_text(encoding="utf-8"))
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    n_stations = len(graph.get("stations", {}))

    report_path = out_dir / "REPORT.md"
    report_path.write_text(
        "\n".join([
            "# Phase1 Report — DXF Direct Path",
            "",
            f"- source dxf: `{args.dxf}`",
            f"- nodes: {n_nodes}",
            f"- edges: {n_edges} (100% directed)",
            f"- stations: {n_stations}",
            "",
            "## Produced files",
            f"- `{graph_json}` — sim_core 입력용 그래프",
            f"- `{preview_svg}` — SVG preview (DXF 기하 기반)",
            f"- `{render_png}` — matplotlib PNG",
            f"- `{report_path}` — 이 리포트",
            "",
            "## Method",
            "DXF → parse → outer-loop CW → direction propagation → export",
            "(이미지 경유 역추출 제거됨, see issue #12)",
        ]),
        encoding="utf-8",
    )

    diag_eids = [e["id"] for e in edge_list if e["kind"] == "D"]
    print(f"[OK] Done. nodes={n_nodes}, edges={n_edges}, stations={n_stations}")
    print(f"     diag_bits={bits} diag_eids={diag_eids} score={score}")
    print(f"     Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
