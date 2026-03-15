from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from graphgen.spec import load_standard_spec

CLASS_ORDER = [
    "background",
    "track",
    "split_marker",
    "merge_marker",
    "station_marker",
    "direction_marker",
]


def _mask_green_dominant(pix: np.ndarray, cfg: dict[str, int]) -> np.ndarray:
    r = pix[:, :, 0]
    g = pix[:, :, 1]
    b = pix[:, :, 2]
    return (
        (g >= int(cfg["min_g"]))
        & ((g - r) >= int(cfg["min_g_minus_r"]))
        & ((g - b) >= int(cfg["min_g_minus_b"]))
    )


def _mask_red_dominant(pix: np.ndarray, cfg: dict[str, int]) -> np.ndarray:
    r = pix[:, :, 0]
    g = pix[:, :, 1]
    b = pix[:, :, 2]
    return (
        (r >= int(cfg["min_r"]))
        & ((r - g) >= int(cfg["min_r_minus_g"]))
        & ((r - b) >= int(cfg["min_r_minus_b"]))
    )


def _mask_blue_dominant(pix: np.ndarray, cfg: dict[str, int]) -> np.ndarray:
    r = pix[:, :, 0]
    g = pix[:, :, 1]
    b = pix[:, :, 2]
    return (
        (b >= int(cfg["min_b"]))
        & ((b - r) >= int(cfg["min_b_minus_r"]))
        & ((b - g) >= int(cfg["min_b_minus_g"]))
    )


def _mask_purple_dominant(pix: np.ndarray, cfg: dict[str, int]) -> np.ndarray:
    r = pix[:, :, 0]
    g = pix[:, :, 1]
    b = pix[:, :, 2]
    return (
        (r >= int(cfg["min_r"]))
        & (b >= int(cfg["min_b"]))
        & (g <= int(cfg["max_g"]))
        & (np.abs(r - b) <= int(cfg["max_r_minus_b"]))
        & ((r - g) >= int(cfg["min_r_minus_g"]))
        & ((b - g) >= int(cfg["min_b_minus_g"]))
    )


def _mask_black_dominant(pix: np.ndarray, cfg: dict[str, int]) -> np.ndarray:
    r = pix[:, :, 0]
    g = pix[:, :, 1]
    b = pix[:, :, 2]
    return (r <= int(cfg["max_r"])) & (g <= int(cfg["max_g"])) & (b <= int(cfg["max_b"]))


def make_class_masks(rgb: np.ndarray, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    pix = rgb.astype(np.int16)
    classes = spec["image"]["classes"]

    track = _mask_green_dominant(pix, classes["track"]["color"])
    split_marker = _mask_red_dominant(pix, classes["split_marker"]["color"])
    merge_marker = _mask_purple_dominant(pix, classes["merge_marker"]["color"])
    station_marker = _mask_blue_dominant(pix, classes["station_marker"]["color"])
    direction_marker = _mask_black_dominant(pix, classes["direction_marker"]["color"])

    occupied = track | split_marker | merge_marker | station_marker | direction_marker
    background = ~occupied
    return {
        "background": background,
        "track": track,
        "split_marker": split_marker,
        "merge_marker": merge_marker,
        "station_marker": station_marker,
        "direction_marker": direction_marker,
    }


def to_index_mask(class_masks: dict[str, np.ndarray]) -> np.ndarray:
    h, w = class_masks["background"].shape
    index_mask = np.zeros((h, w), dtype=np.uint8)
    for idx, cls in enumerate(CLASS_ORDER):
        index_mask[class_masks[cls]] = idx
    return index_mask


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def split_name_for_stem(stem: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{stem}".encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"


def load_graph_label_if_exists(stem: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    graphs_dir = Path(spec["dataset"]["paths"]["graphs_dir"])
    candidate = graphs_dir / f"{stem}.json"
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def default_spec() -> dict[str, Any]:
    return load_standard_spec()
