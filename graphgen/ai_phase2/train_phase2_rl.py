from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build_graph import build_graph, parse_dxf, solve
from graphgen.ai_phase2.rl_env import GraphDirectionRefineEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO baseline for phase2 direction refinement")
    p.add_argument("--dxf", required=True, help="Input DXF file path")
    p.add_argument("--total_timesteps", type=int, default=5000, help="Total PPO timesteps")
    p.add_argument(
        "--model_out",
        default=None,
        help="Output path for trained PPO model (.zip auto-added if omitted)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        print("[ERROR] stable-baselines3 is required for training. Install stable-baselines3 and gymnasium.")
        print(f"        import error: {exc}")
        return 1

    dxf_path = Path(args.dxf)
    if not dxf_path.exists():
        print(f"[ERROR] DXF file not found: {dxf_path}")
        return 1
    dxf_name = dxf_path.stem

    print(f"[1/3] Build graph from {dxf_path}")
    lines, arcs, texts = parse_dxf(dxf_path)
    edge_list, adj, by_ends, station_nodes = build_graph(lines, arcs, texts)
    _, _, assign = solve(edge_list, adj, by_ends, station_nodes)

    print("[2/3] Train PPO baseline")
    env = GraphDirectionRefineEnv(
        edge_list=edge_list,
        adj=adj,
        station_nodes=station_nodes,
        initial_assign=assign,
    )
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, n_steps=64, batch_size=64)
    model.learn(total_timesteps=args.total_timesteps)

    default_model_out = Path("outputs/models") / f"phase2_ppo_{dxf_name}"
    model_out = Path(args.model_out) if args.model_out else default_model_out
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))

    print(f"[3/3] Saved model: {model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
