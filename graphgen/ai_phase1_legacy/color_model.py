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
    return spec.get("phase1", {}).get("color_thresholds", {})


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
