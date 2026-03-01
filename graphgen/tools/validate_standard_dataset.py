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

    green_spec = image_spec["track"]["color"]
    red_spec = image_spec["node_marker"]["color"]
    blue_spec = image_spec["station_marker"]["color"]

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        _warn(f"No PNG images found in {images_dir}. Nothing to validate yet.")
        print(json.dumps({"images_total": 0, "warnings": 1}, ensure_ascii=False))
        return 0

    warnings = 0
    green_ratios: list[float] = []
    red_ratios: list[float] = []
    blue_ratios: list[float] = []
    purple_ratios: list[float] = []
    black_ratios: list[float] = []

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

        r = pix[:, :, 0]
        g = pix[:, :, 1]
        b = pix[:, :, 2]

        green_mask = (
            (g >= int(green_spec["min_g"]))
            & ((g - r) >= int(green_spec["min_g_minus_r"]))
            & ((g - b) >= int(green_spec["min_g_minus_b"]))
        )
        red_mask = (
            (r >= int(red_spec["min_r"]))
            & ((r - g) >= int(red_spec["min_r_minus_g"]))
            & ((r - b) >= int(red_spec["min_r_minus_b"]))
        )
        blue_mask = (
            (b >= int(blue_spec["min_b"]))
            & ((b - r) >= int(blue_spec["min_b_minus_r"]))
            & ((b - g) >= int(blue_spec["min_b_minus_g"]))
        )
        purple_mask = (
            (r >= 120)
            & (b >= 120)
            & (g <= 120)
            & (np.abs(r - b) <= 80)
            & ((r - g) >= 40)
            & ((b - g) >= 40)
        )
        black_mask = (r <= 60) & (g <= 60) & (b <= 60)

        green_ratio = _ratio(green_mask)
        red_ratio = _ratio(red_mask)
        blue_ratio = _ratio(blue_mask)
        purple_ratio = _ratio(purple_mask)
        black_ratio = _ratio(black_mask)
        node_ratio = red_ratio + purple_ratio

        green_ratios.append(green_ratio)
        red_ratios.append(red_ratio)
        blue_ratios.append(blue_ratio)
        purple_ratios.append(purple_ratio)
        black_ratios.append(black_ratio)

        if green_ratio < 0.002:
            _warn(f"{path}: green pixel ratio {green_ratio:.6f} < 0.002")
            warnings += 1
        if node_ratio < 0.0002:
            _warn(f"{path}: node pixel ratio (red+purple) {node_ratio:.6f} < 0.0002")
            warnings += 1
        if black_ratio < 0.00015:
            _warn(f"{path}: black arrow pixel ratio {black_ratio:.6f} < 0.00015")
            warnings += 1

    summary = {
        "images_total": len(image_paths),
        "warnings": warnings,
        "green_ratio_mean": float(np.mean(green_ratios)),
        "red_ratio_mean": float(np.mean(red_ratios)),
        "blue_ratio_mean": float(np.mean(blue_ratios)),
        "purple_ratio_mean": float(np.mean(purple_ratios)),
        "black_ratio_mean": float(np.mean(black_ratios)),
        "green_ratio_min": float(np.min(green_ratios)),
        "red_ratio_min": float(np.min(red_ratios)),
        "blue_ratio_min": float(np.min(blue_ratios)),
        "purple_ratio_min": float(np.min(purple_ratios)),
        "black_ratio_min": float(np.min(black_ratios)),
    }
    print(json.dumps(summary, ensure_ascii=False))

    if all(r < 1e-6 for r in green_ratios):
        _warn("Green pixels are near-zero in all images. Dataset style may not match the standard.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
