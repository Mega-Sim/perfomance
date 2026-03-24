from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphgen.ai_phase2.reward import (
    count_dead_ends,
    count_station_reachability_issues,
    score_assignment_quality,
)
from graphgen.ai_phase2.rl_env import GraphDirectionRefineEnv
from src.build_graph import count_nonholonomic_branch_violations

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - handled by runtime checks in trainer.
    BaseCallback = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class ValidationEvalRow:
    timestep: int
    avg_validation_score: float
    avg_dead_ends: float
    avg_station_issues: float
    avg_nonholonomic_violations: float
    is_best: bool


def _is_higher_better(metric: str) -> bool:
    return metric == "score"


class ValidationEvalCallback(BaseCallback):
    """Minimal periodic validation evaluator for PPO training."""

    def __init__(
        self,
        *,
        val_samples: list[Any],
        eval_freq: int,
        best_metric: str,
        best_model_path: Path | None,
    ) -> None:
        super().__init__(verbose=0)
        self.val_samples = list(val_samples)
        self.eval_freq = max(1, int(eval_freq))
        self.best_metric = best_metric
        self.best_model_path = best_model_path
        self.rows: list[ValidationEvalRow] = []
        self.best_timestep: int | None = None
        self.best_summary: dict[str, float] | None = None
        self._best_value = float("-inf") if _is_higher_better(best_metric) else float("inf")

    def _on_step(self) -> bool:
        timestep = int(self.num_timesteps)
        if not self.val_samples or timestep % self.eval_freq != 0:
            return True
        self.evaluate_at_timestep(timestep)
        return True

    def evaluate_at_timestep(self, timestep: int) -> None:
        if not self.val_samples:
            return
        summary = self._evaluate_once()
        metric_value = summary[f"avg_{self.best_metric}"]
        improved = metric_value > self._best_value if _is_higher_better(self.best_metric) else metric_value < self._best_value
        if improved:
            self._best_value = metric_value
            self.best_timestep = timestep
            self.best_summary = summary
            if self.best_model_path is not None:
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))

        self.rows.append(
            ValidationEvalRow(
                timestep=timestep,
                avg_validation_score=summary["avg_score"],
                avg_dead_ends=summary["avg_dead_ends"],
                avg_station_issues=summary["avg_station_issues"],
                avg_nonholonomic_violations=summary["avg_nonholonomic_violations"],
                is_best=bool(improved),
            )
        )

    def _evaluate_once(self) -> dict[str, float]:
        scores: list[float] = []
        dead_ends: list[float] = []
        station_issues: list[float] = []
        nonholonomic: list[float] = []

        for sample in self.val_samples:
            env = GraphDirectionRefineEnv(
                edge_list=sample.edge_list,
                adj=sample.adj,
                station_nodes=sample.station_nodes,
                initial_assign=sample.assign,
            )
            obs, _ = env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
            assign = env.final_assign
            scores.append(score_assignment_quality(sample.edge_list, sample.adj, sample.station_nodes, assign))
            dead_ends.append(float(count_dead_ends(sample.edge_list, sample.adj, assign)))
            station_issues.append(
                float(count_station_reachability_issues(sample.edge_list, sample.station_nodes, assign))
            )
            nonholonomic.append(float(count_nonholonomic_branch_violations(sample.edge_list, sample.adj, assign)))

        return {
            "avg_score": sum(scores) / len(scores),
            "avg_dead_ends": sum(dead_ends) / len(dead_ends),
            "avg_station_issues": sum(station_issues) / len(station_issues),
            "avg_nonholonomic_violations": sum(nonholonomic) / len(nonholonomic),
        }

    def write_artifacts(
        self,
        *,
        csv_path: Path,
        json_path: Path,
        report_path: Path,
        split_info: dict[str, Any],
        total_timesteps: int,
        last_model_path: Path | None,
        best_model_path: Path | None,
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            fieldnames = [
                "timestep",
                "avg_validation_score",
                "avg_dead_ends",
                "avg_station_issues",
                "avg_nonholonomic_violations",
                "is_best",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.__dict__)

        json_payload = {
            "best_metric": self.best_metric,
            "best_timestep": self.best_timestep,
            "best_summary": self.best_summary,
            "rows": [row.__dict__ for row in self.rows],
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

        no_val = split_info.get("val_count", 0) == 0
        lines = [
            "# Phase2 PPO Training Report",
            "",
            "## Run Summary",
            f"- total timesteps: {total_timesteps}",
            f"- train layouts: {split_info.get('train_count', 0)}",
            f"- validation layouts: {split_info.get('val_count', 0)}",
            f"- best metric: `{self.best_metric}`",
            f"- last model: `{last_model_path}`" if last_model_path else "- last model: `(disabled)`",
            f"- best model: `{best_model_path}`" if best_model_path else "- best model: `(disabled)`",
            "",
            "## Validation Selection",
        ]
        if no_val:
            lines.extend(
                [
                    "- Validation set is empty.",
                    "- Best-model selection was skipped; only train-only outputs are available.",
                ]
            )
        elif self.best_timestep is None:
            lines.append("- Validation set exists but no evaluation rows were recorded.")
        else:
            lines.extend(
                [
                    f"- Best checkpoint timestep: **{self.best_timestep}**",
                    (
                        "- Best validation summary: "
                        f"score={self.best_summary['avg_score']:.3f}, "
                        f"dead_ends={self.best_summary['avg_dead_ends']:.3f}, "
                        f"station_issues={self.best_summary['avg_station_issues']:.3f}, "
                        f"nonholonomic_violations={self.best_summary['avg_nonholonomic_violations']:.3f}"
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "## Logged Artifacts",
                f"- `{csv_path}`",
                f"- `{json_path}`",
            ]
        )
        report_path.write_text("\n".join(lines), encoding="utf-8")
