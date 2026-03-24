from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from graphgen.ai_training.data_utils import CLASS_ORDER, default_spec, load_image, make_class_masks, to_index_mask


class StandardVisionDataset(Dataset):
    """Load images from datasets/standard/images with weak pseudo masks from color thresholds."""

    def __init__(self, images_dir: str | Path = "datasets/standard/images", image_size: int = 256):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.spec = default_spec()
        self.image_paths = sorted(self.images_dir.glob("*.png"))
        if not self.image_paths:
            raise ValueError(f"No PNG images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        rgb = load_image(image_path)
        class_masks = make_class_masks(rgb, self.spec)
        weak_mask = to_index_mask(class_masks)

        image = torch.from_numpy(np.transpose(rgb.copy(), (2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(weak_mask.astype(np.int64))

        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        return {"image": image, "mask": mask, "path": str(image_path)}


def load_image_tensor(image_path: str | Path, image_size: int = 256) -> torch.Tensor:
    rgb = load_image(Path(image_path))
    tensor = torch.from_numpy(np.transpose(rgb.copy(), (2, 0, 1))).float() / 255.0
    if tensor.shape[1] != image_size or tensor.shape[2] != image_size:
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


__all__ = ["CLASS_ORDER", "StandardVisionDataset", "load_image_tensor"]
