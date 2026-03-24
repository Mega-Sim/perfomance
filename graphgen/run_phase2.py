from __future__ import annotations

import argparse
import sys

from graphgen.ai_phase1.run_phase1_pipeline import main as run_phase1_main


def main() -> int:
    p = argparse.ArgumentParser(description="Phase2 entrypoint wrapper")
    p.add_argument("--dxf", required=True)
    p.add_argument("--out_dir", default="outputs/phase2_run")
    p.add_argument("--mode", default="rule", choices=("rule", "rl-repair"))
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    argv = [
        "run_phase1_pipeline",
        "--dxf", args.dxf,
        "--out_dir", args.out_dir,
        "--use_phase2",
        "--phase2_mode", args.mode,
        "--phase2_seed", str(args.seed),
    ]
    if args.model:
        argv.extend(["--phase2_model", args.model])

    old = sys.argv
    try:
        sys.argv = argv
        return run_phase1_main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
