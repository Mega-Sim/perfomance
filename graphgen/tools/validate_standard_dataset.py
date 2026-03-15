"""Fast consistency validator for the standard layout dataset."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from graphgen.spec import load_standard_spec


def _ratio(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def _warn(message: str) -> None:
    print(f"WARN: {message}")


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


def main() -> int:
    spec = load_standard_spec()
    paths = spec["dataset"]["paths"]
    naming = spec["dataset"]["file_naming"]
    image_spec = spec["image"]

    images_dir = Path(paths["images_dir"])
    stem_re = re.compile(naming["recommended_stem_regex"])

    min_w, min_h = image_spec["canvas"]["recommended_min_size_px"]
    max_w, max_h = image_spec["canvas"]["recommended_max_size_px"]

    bg_rgb = np.array(image_spec["canvas"]["background"]["rgb"], dtype=np.int16)
    bg_tol = int(image_spec["canvas"]["background"]["tolerance_per_channel"])
    bg_min_ratio = float(image_spec["canvas"]["background"]["recommended_background_ratio_min"])

    classes = image_spec["classes"]
    green_spec = classes["track"]["color"]
    split_spec = classes["split_marker"]["color"]
    merge_spec = classes["merge_marker"]["color"]
    station_spec = classes["station_marker"]["color"]
    direction_spec = classes["direction_marker"]["color"]

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        _warn(f"No PNG images found in {images_dir}. Nothing to validate yet.")
        print(json.dumps({"images_total": 0, "warnings": 1}, ensure_ascii=False))
        return 0

    warnings = 0
    green_ratios: list[float] = []
    split_ratios: list[float] = []
    merge_ratios: list[float] = []
    station_ratios: list[float] = []
    direction_ratios: list[float] = []

    for path in image_paths:
        stem = path.stem
        if not stem_re.fullmatch(stem):
            _warn(f"{path}: stem '{stem}' does not match regex {naming['recommended_stem_regex']}")
            warnings += 1

        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]
        if w < min_w or h < min_h or w > max_w or h > max_h:
            _warn(f"{path}: resolution {w}x{h} is outside recommended range {min_w}~{max_w} x {min_h}~{max_h}")
            warnings += 1

        pix = arr.astype(np.int16)
        bg_mask = np.all(np.abs(pix - bg_rgb) <= bg_tol, axis=2)
        bg_ratio = _ratio(bg_mask)
        if bg_ratio < bg_min_ratio:
            _warn(f"{path}: white background ratio {bg_ratio:.6f} < {bg_min_ratio:.2f}")
            warnings += 1

        green_mask = _mask_green_dominant(pix, green_spec)
        split_mask = _mask_red_dominant(pix, split_spec)
        merge_mask = _mask_purple_dominant(pix, merge_spec)
        station_mask = _mask_blue_dominant(pix, station_spec)
        direction_mask = _mask_black_dominant(pix, direction_spec)

        green_ratio = _ratio(green_mask)
        split_ratio = _ratio(split_mask)
        merge_ratio = _ratio(merge_mask)
        station_ratio = _ratio(station_mask)
        direction_ratio = _ratio(direction_mask)
        node_ratio = split_ratio + merge_ratio

        green_ratios.append(green_ratio)
        split_ratios.append(split_ratio)
        merge_ratios.append(merge_ratio)
        station_ratios.append(station_ratio)
        direction_ratios.append(direction_ratio)

        if green_ratio < 0.002:
            _warn(f"{path}: green pixel ratio {green_ratio:.6f} < 0.002")
            warnings += 1
        if node_ratio < 0.0002:
            _warn(f"{path}: node pixel ratio (split+merge) {node_ratio:.6f} < 0.0002")
            warnings += 1
        if direction_ratio < 0.00015:
            _warn(f"{path}: direction marker pixel ratio {direction_ratio:.6f} < 0.00015")
            warnings += 1

    summary = {
        "images_total": len(image_paths),
        "warnings": warnings,
        "green_ratio_mean": float(np.mean(green_ratios)),
        "split_ratio_mean": float(np.mean(split_ratios)),
        "merge_ratio_mean": float(np.mean(merge_ratios)),
        "station_ratio_mean": float(np.mean(station_ratios)),
        "direction_ratio_mean": float(np.mean(direction_ratios)),
        "green_ratio_min": float(np.min(green_ratios)),
        "split_ratio_min": float(np.min(split_ratios)),
        "merge_ratio_min": float(np.min(merge_ratios)),
        "station_ratio_min": float(np.min(station_ratios)),
        "direction_ratio_min": float(np.min(direction_ratios)),
    }
    print(json.dumps(summary, ensure_ascii=False))

    if all(r < 1e-6 for r in green_ratios):
        _warn("Green pixels are near-zero in all images. Dataset style may not match the standard.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
