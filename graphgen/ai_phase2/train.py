from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

from .env import LocalDirectionRepairEnv, synthetic_graph_cases
from .policy import TabularQPolicy


def train(episodes: int, out_dir: Path, seed: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = TabularQPolicy(epsilon=0.25)
    cases = synthetic_graph_cases()
    metrics = []

    for ep in range(episodes):
        case = cases[ep % len(cases)]
        env = LocalDirectionRepairEnv(case.edge_list, case.adj, case.station_nodes, case.rule_assign, max_steps=8)
        env.reset()
        total_reward = 0.0
        for _ in range(env.max_steps):
            before = copy.deepcopy(env)
            action = policy.select_action(env)
            _, reward, done, info = env.step(action)
            after = copy.deepcopy(env)
            policy.update(before, action, reward, after)
            total_reward += reward
            if done:
                break
        metrics.append({"episode": ep, "case": case.name, "reward": total_reward, "violations": info["after"]})

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"phase2_q_{ts}.json"
    metrics_path = out_dir / f"phase2_train_metrics_{ts}.json"
    policy.save(str(model_path))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model_path, metrics_path


def main() -> int:
    p = argparse.ArgumentParser(description="Train minimal phase2 RL scaffold (tabular Q-learning)")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="outputs/models")
    args = p.parse_args()

    model, metrics = train(args.episodes, Path(args.out_dir), args.seed)
    print(f"[OK] model={model}")
    print(f"[OK] metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
