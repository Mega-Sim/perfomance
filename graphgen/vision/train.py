from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CLASS_ORDER, StandardVisionDataset
from .model import TinyUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train minimal vision model on datasets/standard/images")
    p.add_argument("--images_dir", default="datasets/standard/images")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--out_dir", default="outputs/vision")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = StandardVisionDataset(images_dir=args.images_dir, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(num_classes=len(CLASS_ORDER)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            steps += 1

        row = {"epoch": epoch, "loss": total_loss / max(1, steps)}
        history.append(row)
        print(json.dumps(row))

    ckpt_path = out_dir / "tiny_unet.pt"
    torch.save({"model_state": model.state_dict(), "classes": CLASS_ORDER, "image_size": args.image_size}, ckpt_path)

    metrics = {
        "images_dir": args.images_dir,
        "samples": len(ds),
        "epochs": args.epochs,
        "history": history,
        "checkpoint": str(ckpt_path),
        "labels_note": "Weak pseudo masks are generated on-the-fly from color-threshold rules.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
