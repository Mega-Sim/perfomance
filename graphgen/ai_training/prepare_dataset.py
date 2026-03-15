"""Prepare training-ready manifest and masks for standard dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_utils import (
    CLASS_ORDER,
    default_spec,
    load_graph_label_if_exists,
    load_image,
    make_class_masks,
    split_name_for_stem,
    to_index_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="datasets/standard/spec.json")
    parser.add_argument("--out_dir", default="datasets/standard/training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = default_spec() if args.spec == "datasets/standard/spec.json" else json.loads(Path(args.spec).read_text(encoding="utf-8"))

    image_dir = Path(spec["dataset"]["paths"]["images_dir"])
    train_cfg = spec["training"]["split"]
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for image_path in sorted(image_dir.glob("*.png")):
        stem = image_path.stem
        rgb = load_image(image_path)
        class_masks = make_class_masks(rgb, spec)
        index_mask = to_index_mask(class_masks)

        mask_path = mask_dir / f"{stem}.npy"
        np.save(mask_path, index_mask)

        graph_label = load_graph_label_if_exists(stem, spec)
        split = split_name_for_stem(
            stem=stem,
            train_ratio=float(train_cfg["train_ratio"]),
            val_ratio=float(train_cfg["val_ratio"]),
            seed=int(train_cfg["seed"]),
        )

        records.append(
            {
                "stem": stem,
                "image": str(image_path),
                "mask": str(mask_path),
                "split": split,
                "has_graph_label": graph_label is not None,
                "graph": str(Path(spec["dataset"]["paths"]["graphs_dir"]) / f"{stem}.json"),
                "classes": CLASS_ORDER,
                "height": int(index_mask.shape[0]),
                "width": int(index_mask.shape[1]),
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fp:
        for item in records:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    split_counts = {"train": 0, "val": 0, "test": 0}
    for row in records:
        split_counts[row["split"]] += 1

    summary = {
        "samples": len(records),
        "split_counts": split_counts,
        "manifest": str(manifest_path),
        "mask_dir": str(mask_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
