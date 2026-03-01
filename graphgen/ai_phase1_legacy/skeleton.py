"""Skeletonization utilities used by phase-1 legacy pipeline."""

from __future__ import annotations

import numpy as np


def zhang_suen_thinning(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning for binary image (0/1)."""
    img = img.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        # step 1
        to_remove = []
        rows, cols = img.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P = img[i, j]
                if P != 1:
                    continue
                neighbors = [
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j + 1],
                    img[i + 1, j + 1],
                    img[i + 1, j],
                    img[i + 1, j - 1],
                    img[i, j - 1],
                    img[i - 1, j - 1],
                ]
                C = sum(neighbors)
                if C < 2 or C > 6:
                    continue
                transitions = 0
                for k in range(8):
                    if neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1:
                        transitions += 1
                if transitions != 1:
                    continue
                if neighbors[0] * neighbors[2] * neighbors[4] != 0:
                    continue
                if neighbors[2] * neighbors[4] * neighbors[6] != 0:
                    continue
                to_remove.append((i, j))
        if to_remove:
            changed = True
            for i, j in to_remove:
                img[i, j] = 0

        # step 2
        to_remove = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P = img[i, j]
                if P != 1:
                    continue
                neighbors = [
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j + 1],
                    img[i + 1, j + 1],
                    img[i + 1, j],
                    img[i + 1, j - 1],
                    img[i, j - 1],
                    img[i - 1, j - 1],
                ]
                C = sum(neighbors)
                if C < 2 or C > 6:
                    continue
                transitions = 0
                for k in range(8):
                    if neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1:
                        transitions += 1
                if transitions != 1:
                    continue
                if neighbors[0] * neighbors[2] * neighbors[6] != 0:
                    continue
                if neighbors[0] * neighbors[4] * neighbors[6] != 0:
                    continue
                to_remove.append((i, j))
        if to_remove:
            changed = True
            for i, j in to_remove:
                img[i, j] = 0

    return img
