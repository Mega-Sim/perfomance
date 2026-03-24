from __future__ import annotations

from pathlib import Path
from typing import Any

from .rl_env import GraphDirectionRefineEnv


def run_ppo_refinement(
    edge_list: list[dict[str, Any]],
    adj: dict[tuple[float, float], set[tuple[float, float]]],
    station_nodes: dict[str, tuple[float, float]],
    assign: dict[int, int],
    model_path: str | Path,
    seed: int | None = None,
) -> dict[int, int]:
    """Run deterministic PPO policy rollout over one graph episode."""
    model_path = Path(model_path)
    if not model_path.exists():
        zipped = model_path.with_suffix(".zip")
        if zipped.exists():
            model_path = zipped
        else:
            raise FileNotFoundError(f"PPO model file not found: {model_path}")

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise ImportError("stable-baselines3 is required for PPO phase2 mode") from exc

    env = GraphDirectionRefineEnv(
        edge_list=edge_list,
        adj=adj,
        station_nodes=station_nodes,
        initial_assign=assign,
    )

    model = PPO.load(str(model_path))
    obs, _ = env.reset(seed=seed)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))

    return env.final_assign
