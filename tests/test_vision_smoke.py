from __future__ import annotations

import torch

from graphgen.vision.dataset import CLASS_ORDER, StandardVisionDataset
from graphgen.vision.model import TinyUNet


def test_vision_dataset_and_model_smoke() -> None:
    ds = StandardVisionDataset("datasets/standard/images", image_size=128)
    assert len(ds) > 0
    sample = ds[0]

    assert sample["image"].shape == (3, 128, 128)
    assert sample["mask"].shape == (128, 128)

    model = TinyUNet(num_classes=len(CLASS_ORDER))
    with torch.no_grad():
        out = model(sample["image"].unsqueeze(0))
    assert out.shape == (1, len(CLASS_ORDER), 128, 128)
