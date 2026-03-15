"""Train/evaluate a minimal RGB prototype segmentation baseline.

This is a research scaffold, not a production inference path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_utils import CLASS_ORDER, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/standard/training/manifest.jsonl")
    parser.add_argument("--eval_split", default="val", choices=["val", "test"])
    parser.add_argument("--out", default="outputs/ai_training/segmentation_baseline_metrics.json")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def compute_prototypes(train_rows: list[dict]) -> np.ndarray:
    sums = np.zeros((len(CLASS_ORDER), 3), dtype=np.float64)
    counts = np.zeros((len(CLASS_ORDER),), dtype=np.float64)

    for row in train_rows:
        rgb = load_image(Path(row["image"]))
        mask = np.load(Path(row["mask"]))
        for cid in range(len(CLASS_ORDER)):
            cls_pixels = rgb[mask == cid]
            if cls_pixels.size == 0:
                continue
            sums[cid] += cls_pixels.mean(axis=0)
            counts[cid] += 1.0

    counts[counts == 0] = 1.0
    return (sums / counts[:, None]).astype(np.float32)


def predict_mask(rgb: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    x = rgb.reshape(-1, 3).astype(np.float32)
    d2 = ((x[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    pred = np.argmin(d2, axis=1)
    return pred.reshape(rgb.shape[:2]).astype(np.uint8)


def iou_and_f1(pred: np.ndarray, target: np.ndarray, cls_id: int) -> tuple[float, float]:
    p = pred == cls_id
    t = target == cls_id
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    denom_iou = tp + fp + fn
    iou = float(tp / denom_iou) if denom_iou else 1.0
    denom_f1 = 2 * tp + fp + fn
    f1 = float((2 * tp) / denom_f1) if denom_f1 else 1.0
    return iou, f1


def main() -> int:
    args = parse_args()
    rows = read_manifest(Path(args.manifest))
    train_rows = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == args.eval_split]

    if not train_rows or not eval_rows:
        raise RuntimeError("Not enough train/eval samples. Run prepare_dataset first.")

    prototypes = compute_prototypes(train_rows)

    class_ious = {cls: [] for cls in CLASS_ORDER}
    class_f1s = {cls: [] for cls in CLASS_ORDER}

    for row in eval_rows:
        rgb = load_image(Path(row["image"]))
        target = np.load(Path(row["mask"]))
        pred = predict_mask(rgb, prototypes)
        for cid, cls in enumerate(CLASS_ORDER):
            iou, f1 = iou_and_f1(pred, target, cid)
            class_ious[cls].append(iou)
            class_f1s[cls].append(f1)

    metrics = {
        "eval_split": args.eval_split,
        "samples": len(eval_rows),
        "macro_iou": float(np.mean([np.mean(v) for v in class_ious.values()])),
        "macro_f1": float(np.mean([np.mean(v) for v in class_f1s.values()])),
        "class_iou": {k: float(np.mean(v)) for k, v in class_ious.items()},
        "class_f1": {k: float(np.mean(v)) for k, v in class_f1s.items()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
