from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .env import LocalDirectionRepairEnv, evaluate_violations, random_repair, score_from_violations, synthetic_graph_cases
from .policy import GreedyPolicy, RandomPolicy, TabularQPolicy


def run_policy(case, mode: str, model_path: str | None, seed: int):
    base = dict(case.rule_assign)
    if mode == "rule-based":
        assign = base
    elif mode == "random":
        assign = random_repair(case.edge_list, case.adj, case.station_nodes, base, seed=seed, steps=8)
    else:
        env = LocalDirectionRepairEnv(case.edge_list, case.adj, case.station_nodes, base, max_steps=8)
        env.reset()
        if mode == "greedy":
            policy = GreedyPolicy()
        elif mode == "rl-repair":
            policy = TabularQPolicy.load(model_path) if model_path else GreedyPolicy()
        else:
            raise ValueError(mode)
        for _ in range(env.max_steps):
            action = policy.select_action(env)
            _, _, done, _ = env.step(action)
            if done:
                break
        assign = env.assign
    v = evaluate_violations(case.edge_list, case.adj, case.station_nodes, assign)
    return {"violations": v, "final_score": score_from_violations(v), "assign": assign}


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate phase2 rule/random/greedy/rl-repair")
    p.add_argument("--model", default=None, help="Path to trained q policy json")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in synthetic_graph_cases():
        entry = {"case": case.name}
        for mode in ("rule-based", "random", "greedy", "rl-repair"):
            entry[mode] = run_policy(case, mode, args.model, args.seed)
        rows.append(entry)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"phase2_eval_{ts}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
