from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .env import Action, LocalDirectionRepairEnv


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select_action(self, env: LocalDirectionRepairEnv) -> Action:
        return self.rng.choice(env.available_actions())


class GreedyPolicy:
    def select_action(self, env: LocalDirectionRepairEnv) -> Action:
        actions = env.available_actions()
        best_action = actions[0]
        best_reward = float("-inf")

        original = dict(env.assign)
        for action in actions:
            _, reward, _, _ = env.step(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action
            env.assign = dict(original)
        return best_action


@dataclass
class TabularQPolicy:
    alpha: float = 0.2
    gamma: float = 0.9
    epsilon: float = 0.2
    q_table: dict[str, dict[str, float]] = field(default_factory=dict)

    def _state_key(self, env: LocalDirectionRepairEnv) -> str:
        s = env._state()
        v = s.violation_flags
        return f"nh={v['nonholonomic']};sm={v['invalid_split_merge']};tan={v['tangent_inconsistency']}"

    def _action_key(self, action: Action) -> str:
        return f"{action[0]}:{action[1]}:{action[2]}"

    def select_action(self, env: LocalDirectionRepairEnv) -> Action:
        actions = env.available_actions()
        if random.random() < self.epsilon:
            return random.choice(actions)
        sk = self._state_key(env)
        qvals = self.q_table.get(sk, {})
        if not qvals:
            return actions[0]
        best = max(actions, key=lambda a: qvals.get(self._action_key(a), 0.0))
        return best

    def update(self, env_before: LocalDirectionRepairEnv, action: Action, reward: float, env_after: LocalDirectionRepairEnv):
        sk = self._state_key(env_before)
        ak = self._action_key(action)
        next_key = self._state_key(env_after)
        self.q_table.setdefault(sk, {})
        self.q_table.setdefault(next_key, {})
        old = self.q_table[sk].get(ak, 0.0)
        max_next = max(self.q_table[next_key].values(), default=0.0)
        self.q_table[sk][ak] = old + self.alpha * (reward + self.gamma * max_next - old)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon, "q_table": self.q_table}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TabularQPolicy":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(alpha=data["alpha"], gamma=data["gamma"], epsilon=data["epsilon"], q_table=data["q_table"])
