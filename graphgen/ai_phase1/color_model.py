"""Color prototype model used by phase-1 graph extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from graphgen.spec import load_standard_spec

CLASS_NAMES = (
    "background",
    "track_green",
    "split_red",
    "merge_purple",
    "station_blue",
    "arrow_black",
)


@dataclass
class ColorModel:
    classes: dict[str, dict[str, np.ndarray]]
    meta: dict[str, Any]

    def predict_scores(self, pix_rgb: np.ndarray) -> dict[str, float]:
        pix = np.asarray(pix_rgb, dtype=np.float32)
        scores: dict[str, float] = {}
        for cls_name, stats in self.classes.items():
            mean = stats["mean"]
            std = stats["std"]
            z = (pix - mean) / std
            scores[cls_name] = float(np.sum(z * z))
        return scores

    def predict_class(self, pix_rgb: np.ndarray) -> str:
        scores = self.predict_scores(pix_rgb)
        return min(scores, key=scores.get)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "classes": {
                cls: {
                    "mean": stats["mean"].astype(float).tolist(),
                    "std": stats["std"].astype(float).tolist(),
                }
                for cls, stats in self.classes.items()
            },
            "meta": self.meta,
        }

    def save(self, out_path: str | Path) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_json_dict(), fp, ensure_ascii=False, indent=2)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ColorModel":
        classes = {
            name: {
                "mean": np.asarray(stats["mean"], dtype=np.float32),
                "std": np.asarray(stats["std"], dtype=np.float32),
            }
            for name, stats in payload["classes"].items()
        }
        return cls(classes=classes, meta=payload.get("meta", {}))

    @classmethod
    def load(cls, model_path: str | Path) -> "ColorModel":
        with Path(model_path).open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls.from_json_dict(payload)


def load_thresholds_from_spec() -> dict[str, int]:
    spec = load_standard_spec()
    image = spec["image"]
    return {
        "bg_tol": int(image["canvas"]["background"].get("tolerance_per_channel", 10)),
        "green_min_g": int(image["track"]["color"].get("min_g", 180)),
        "green_min_gr": int(image["track"]["color"].get("min_g_minus_r", 60)),
        "green_min_gb": int(image["track"]["color"].get("min_g_minus_b", 60)),
        "red_min_r": int(image["node_marker"]["color"].get("min_r", 180)),
        "red_min_rg": int(image["node_marker"]["color"].get("min_r_minus_g", 60)),
        "red_min_rb": int(image["node_marker"]["color"].get("min_r_minus_b", 60)),
        "blue_min_b": int(image["station_marker"]["color"].get("min_b", 180)),
        "blue_min_br": int(image["station_marker"]["color"].get("min_b_minus_r", 60)),
        "blue_min_bg": int(image["station_marker"]["color"].get("min_b_minus_g", 60)),
    }


def heuristic_masks(pix_int16: np.ndarray, thresholds: dict[str, int]) -> dict[str, np.ndarray]:
    r = pix_int16[:, :, 0]
    g = pix_int16[:, :, 1]
    b = pix_int16[:, :, 2]

    bg_mask = (
        (np.abs(r - 255) <= thresholds["bg_tol"])
        & (np.abs(g - 255) <= thresholds["bg_tol"])
        & (np.abs(b - 255) <= thresholds["bg_tol"])
    )
    track_mask = (
        (g >= thresholds["green_min_g"])
        & ((g - r) >= thresholds["green_min_gr"])
        & ((g - b) >= thresholds["green_min_gb"])
    )
    split_mask = (
        (r >= thresholds["red_min_r"])
        & ((r - g) >= thresholds["red_min_rg"])
        & ((r - b) >= thresholds["red_min_rb"])
    )
    station_mask = (
        (b >= thresholds["blue_min_b"])
        & ((b - r) >= thresholds["blue_min_br"])
        & ((b - g) >= thresholds["blue_min_bg"])
    )
    merge_mask = (
        (r >= 120)
        & (b >= 120)
        & (g <= 120)
        & (np.abs(r - b) <= 80)
        & ((r - g) >= 40)
        & ((b - g) >= 40)
    )
    arrow_mask = (r <= 60) & (g <= 60) & (b <= 60)
    return {
        "background": bg_mask,
        "track_green": track_mask,
        "split_red": split_mask,
        "merge_purple": merge_mask,
        "station_blue": station_mask,
        "arrow_black": arrow_mask,
    }


def train_model(images: list[np.ndarray], thresholds: dict[str, int] | None = None) -> tuple[ColorModel, dict[str, int]]:
    thresholds = thresholds or load_thresholds_from_spec()
    samples: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}

    for arr in images:
        pix = arr.astype(np.int16)
        masks = heuristic_masks(pix, thresholds)
        for cls_name, mask in masks.items():
            if np.any(mask):
                samples[cls_name].append(arr[mask])

    classes: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for cls_name in CLASS_NAMES:
        if samples[cls_name]:
            data = np.concatenate(samples[cls_name], axis=0).astype(np.float32)
        else:
            data = np.zeros((1, 3), dtype=np.float32)
        mean = data.mean(axis=0)
        std = np.clip(data.std(axis=0), 8.0, None)
        classes[cls_name] = {"mean": mean, "std": std}
        counts[cls_name] = int(data.shape[0])

    model = ColorModel(classes=classes, meta={"trained_images": len(images)})
    return model, counts
