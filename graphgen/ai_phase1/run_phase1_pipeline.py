"""End-to-end phase-1 pipeline runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


OUTPUT_ROOT = Path("outputs/phase1_ai")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dxf", required=True)
    p.add_argument("--retrain", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    render_png = OUTPUT_ROOT / "drawing1_render.png"
    teacher_json = OUTPUT_ROOT / "drawing1_teacher.json"
    model_json = OUTPUT_ROOT / "color_model.json"
    infer_dir = OUTPUT_ROOT / "drawing1_ai"

    _run([sys.executable, "src/build_graph.py", args.dxf, str(render_png), str(teacher_json)])

    if args.retrain or not model_json.exists():
        _run(
            [
                sys.executable,
                "-m",
                "graphgen.ai_phase1.train_color_model",
                "--images_dir",
                "datasets/standard/images",
                "--out",
                str(model_json),
            ]
        )

    _run(
        [
            sys.executable,
            "-m",
            "graphgen.ai_phase1.infer_graph_from_image",
            "--image",
            str(render_png),
            "--model",
            str(model_json),
            "--out_dir",
            str(infer_dir),
        ]
    )

    graph = json.loads((infer_dir / "graph.json").read_text(encoding="utf-8"))
    counts = {"split": 0, "merge": 0, "station": 0}
    for n in graph["nodes"]:
        counts[n["type"]] = counts.get(n["type"], 0) + 1

    directed = sum(1 for e in graph["edges"] if e["dir"] is not None)
    unresolved = sum(1 for e in graph["edges"] if e["dir"] is None)

    report = OUTPUT_ROOT / "REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# Phase1 AI Report",
                "",
                f"- source dxf: `{args.dxf}`",
                f"- nodes(split): {counts.get('split', 0)}",
                f"- nodes(merge): {counts.get('merge', 0)}",
                f"- nodes(station): {counts.get('station', 0)}",
                f"- edges(total): {len(graph['edges'])}",
                f"- edges(directed): {directed}",
                f"- edges(unresolved): {unresolved}",
                "",
                "## Produced files",
                "- outputs/phase1_ai/color_model.json",
                "- outputs/phase1_ai/drawing1_render.png",
                "- outputs/phase1_ai/drawing1_teacher.json",
                "- outputs/phase1_ai/drawing1_ai/graph.json",
                "- outputs/phase1_ai/drawing1_ai/preview.svg",
                "- outputs/phase1_ai/drawing1_ai/preview.png (local, untracked)",
                "- outputs/phase1_ai/REPORT.md",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] Report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
