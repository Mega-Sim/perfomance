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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "meta": self.meta,
            "classes": {
                k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in self.classes.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "ColorModel":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = {}
        for k, v in payload["classes"].items():
            classes[k] = {kk: np.asarray(vv, dtype=np.float32) for kk, vv in v.items()}
        return ColorModel(classes=classes, meta=payload.get("meta", {}))


def _mean_cov(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)
    return mean, cov


def train_model(images: list[np.ndarray]) -> tuple[ColorModel, dict[str, int]]:
    """Train simple prototype model from images using spec thresholds."""
    spec = load_standard_spec()
    thresholds = load_thresholds_from_spec(spec)

    counts: dict[str, int] = {k: 0 for k in CLASS_NAMES}
    classes: dict[str, dict[str, np.ndarray]] = {}

    for name in CLASS_NAMES:
        px: list[np.ndarray] = []
        for img in images:
            mask = heuristic_masks(img, thresholds)[name]
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            samples = img[ys, xs].astype(np.float32)
            px.append(samples)
            counts[name] += len(samples)
        if px:
            samples = np.concatenate(px, axis=0)
            mean, cov = _mean_cov(samples)
            classes[name] = {"mean": mean, "cov": cov}
        else:
            classes[name] = {"mean": np.zeros(3, dtype=np.float32), "cov": np.eye(3, dtype=np.float32)}

    model = ColorModel(classes=classes, meta={"spec_version": spec.get("version", "unknown")})
    return model, counts


def load_thresholds_from_spec(spec: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    phase1_thresholds = spec.get("phase1", {}).get("color_thresholds", {})
    if phase1_thresholds:
        return phase1_thresholds

    image_spec = spec.get("image", {})
    track = image_spec.get("track", {}).get("color", {})
    red = image_spec.get("node_marker", {}).get("color", {})
    blue = image_spec.get("station_marker", {}).get("color", {})

    # Backward-compatible fallback derived from image-level spec.
    return {
        "background": {"lo": [230, 230, 230], "hi": [255, 255, 255]},
        "track_green": {
            "lo": [
                0,
                int(track.get("min_g", 120)),
                0,
            ],
            "hi": [
                int(max(0, int(track.get("min_g", 120)) - int(track.get("min_g_minus_r", 20)))),
                255,
                int(max(0, int(track.get("min_g", 120)) - int(track.get("min_g_minus_b", 20)))),
            ],
        },
        "split_red": {
            "lo": [
                int(red.get("min_r", 140)),
                0,
                0,
            ],
            "hi": [
                255,
                int(max(0, int(red.get("min_r", 140)) - int(red.get("min_r_minus_g", 40)))),
                int(max(0, int(red.get("min_r", 140)) - int(red.get("min_r_minus_b", 40)))),
            ],
        },
        "merge_purple": {"lo": [120, 0, 120], "hi": [255, 140, 255]},
        "station_blue": {
            "lo": [
                0,
                0,
                int(blue.get("min_b", 140)),
            ],
            "hi": [
                int(max(0, int(blue.get("min_b", 140)) - int(blue.get("min_b_minus_r", 40)))),
                int(max(0, int(blue.get("min_b", 140)) - int(blue.get("min_b_minus_g", 40)))),
                255,
            ],
        },
        "arrow_black": {"lo": [0, 0, 0], "hi": [70, 70, 70]},
    }


def heuristic_masks(img: np.ndarray, thresholds: dict[str, dict[str, list[int]]]) -> dict[str, np.ndarray]:
    """Return class masks from RGB thresholds in spec."""
    masks: dict[str, np.ndarray] = {}

    def in_range(rgb: np.ndarray, lo: list[int], hi: list[int]) -> np.ndarray:
        lo = np.asarray(lo, dtype=np.uint8)
        hi = np.asarray(hi, dtype=np.uint8)
        return np.all((rgb >= lo) & (rgb <= hi), axis=2)

    for cls in CLASS_NAMES:
        th = thresholds.get(cls)
        if not th:
            masks[cls] = np.zeros(img.shape[:2], dtype=bool)
            continue
        masks[cls] = in_range(img, th["lo"], th["hi"])

    return masks
