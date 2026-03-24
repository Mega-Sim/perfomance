from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graphgen.ai_phase2.dataset import load_dxf_paths, split_train_val
from graphgen.ai_phase2.rl_env import GraphDirectionRefineEnv
from graphgen.ai_phase2.train_callback import ValidationEvalCallback
from src.build_graph import build_graph, parse_dxf, solve


@dataclass(frozen=True)
class GraphSample:
    dxf_path: Path
    edge_list: list[dict[str, Any]]
    adj: dict[tuple[float, float], set[tuple[float, float]]]
    station_nodes: dict[str, tuple[float, float]]
    assign: dict[int, int]


class MultiLayoutGraphEnv:
    """Sample one layout per episode, preserving single-layout behavior when N=1."""

    def __init__(self, samples: list[GraphSample], seed: int = 42):
        if not samples:
            raise ValueError("No training samples available")
        self.samples = list(samples)
        self._rng = random.Random(seed)
        self._active_env = self._make_env(self.samples[0])

        self.action_space = self._active_env.action_space
        self.observation_space = self._active_env.observation_space

    def _make_env(self, sample: GraphSample):
        return GraphDirectionRefineEnv(
            edge_list=sample.edge_list,
            adj=sample.adj,
            station_nodes=sample.station_nodes,
            initial_assign=sample.assign,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng.seed(seed)
        sample = self._rng.choice(self.samples)
        self._active_env = self._make_env(sample)
        return self._active_env.reset(seed=seed, options=options)

    def step(self, action: int):
        return self._active_env.step(action)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO baseline for phase2 direction refinement")
    p.add_argument("--dxf", default=None, help="Single input DXF file path (backward-compatible mode)")
    p.add_argument("--dxf_dir", default=None, help="Directory containing multiple .dxf files")
    p.add_argument("--dxf_list", default=None, help="Path to .txt/.json file listing DXFs")
    p.add_argument("--train_ratio", type=float, default=1.0, help="Train split ratio when using --dxf_dir/--dxf_list")
    p.add_argument("--train_list", default=None, help="Explicit train list (.txt/.json). Requires --val_list")
    p.add_argument("--val_list", default=None, help="Explicit validation list (.txt/.json). Requires --train_list")
    p.add_argument("--total_timesteps", type=int, default=5000, help="Total PPO timesteps")
    p.add_argument("--eval_freq", type=int, default=1000, help="Validation evaluation frequency in timesteps")
    p.add_argument("--best_metric", default="score", choices=["score", "dead_ends", "station_issues", "nonholonomic_violations"])
    p.add_argument("--out_stem", default=None, help="Artifact stem for outputs/models/<stem>_{last|best|train_log.*}")
    p.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True, help="Save validation-selected best checkpoint")
    p.add_argument("--save_last", action=argparse.BooleanOptionalAction, default=True, help="Save final checkpoint as <stem>_last.zip")
    p.add_argument(
        "--model_out",
        default=None,
        help="Compatibility output path for trained PPO model (.zip auto-added if omitted)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _build_sample(dxf_path: Path) -> GraphSample:
    lines, arcs, texts = parse_dxf(dxf_path)
    edge_list, adj, by_ends, station_nodes = build_graph(lines, arcs, texts)
    _, _, assign = solve(edge_list, adj, by_ends, station_nodes)
    return GraphSample(
        dxf_path=dxf_path,
        edge_list=edge_list,
        adj=adj,
        station_nodes=station_nodes,
        assign=assign,
    )


def main() -> int:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        print("[ERROR] stable-baselines3 is required for training. Install stable-baselines3 and gymnasium.")
        print(f"        import error: {exc}")
        return 1

    try:
        all_paths = load_dxf_paths(dxf=args.dxf, dxf_dir=args.dxf_dir, dxf_list=args.dxf_list)
    except Exception as exc:
        print(f"[ERROR] Invalid dataset input: {exc}")
        return 1

    try:
        train_paths, val_paths = split_train_val(
            all_paths,
            seed=args.seed,
            train_ratio=args.train_ratio,
            train_list=args.train_list,
            val_list=args.val_list,
        )
    except Exception as exc:
        print(f"[ERROR] Invalid train/validation split: {exc}")
        return 1

    print(f"[1/3] Building training graphs from {len(train_paths)} DXF(s)")
    train_samples = [_build_sample(path) for path in train_paths]
    val_samples = [_build_sample(path) for path in val_paths]

    if val_paths:
        print(f"        validation DXFs: {len(val_paths)}")

    print("[2/3] Train PPO baseline")
    env = MultiLayoutGraphEnv(samples=train_samples, seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, n_steps=64, batch_size=64)

    default_stem = train_paths[0].stem if len(train_paths) == 1 else f"multilayout_n{len(train_paths)}"
    out_stem = args.out_stem or (Path(args.model_out).with_suffix("").name if args.model_out else f"phase2_ppo_{default_stem}")
    model_dir = Path(args.model_out).parent if args.model_out else Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    last_model_path = model_dir / f"{out_stem}_last.zip"
    best_model_path = model_dir / f"{out_stem}_best.zip"
    train_log_csv = model_dir / f"{out_stem}_train_log.csv"
    train_log_json = model_dir / f"{out_stem}_train_log.json"
    train_report_md = model_dir / f"{out_stem}_train_report.md"

    callback = ValidationEvalCallback(
        val_samples=val_samples,
        eval_freq=args.eval_freq,
        best_metric=args.best_metric,
        best_model_path=best_model_path if args.save_best and val_samples else None,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    if val_samples and (args.total_timesteps % max(1, args.eval_freq) != 0):
        callback.evaluate_at_timestep(args.total_timesteps)

    if args.save_last:
        model.save(str(last_model_path))

    model_out = Path(args.model_out) if args.model_out else None
    if model_out:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_out))

    split_info = {
        "seed": args.seed,
        "eval_freq": args.eval_freq,
        "best_metric": args.best_metric,
        "total_inputs": len(all_paths),
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "train_dxfs": [str(p) for p in train_paths],
        "val_dxfs": [str(p) for p in val_paths],
        "last_model": str(last_model_path) if args.save_last else None,
        "best_model": str(best_model_path) if (args.save_best and val_samples) else None,
        "best_timestep": callback.best_timestep,
        "best_summary": callback.best_summary,
    }
    split_path = model_dir / f"{out_stem}_split.json"
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    callback.write_artifacts(
        csv_path=train_log_csv,
        json_path=train_log_json,
        report_path=train_report_md,
        split_info=split_info,
        total_timesteps=args.total_timesteps,
        last_model_path=last_model_path if args.save_last else None,
        best_model_path=best_model_path if (args.save_best and val_samples) else None,
    )

    print(f"[3/3] Saved last model: {last_model_path}" if args.save_last else "[3/3] Skipped last model save (--no-save_last)")
    if args.save_best and val_samples:
        print(f"      Saved best model: {best_model_path}")
    elif args.save_best:
        print("      Best model selection skipped (empty validation set)")
    if model_out:
        print(f"      Compatibility model: {model_out}")
    print(f"      Train log CSV: {train_log_csv}")
    print(f"      Train log JSON: {train_log_json}")
    print(f"      Train report: {train_report_md}")
    print(f"      Split metadata: {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
