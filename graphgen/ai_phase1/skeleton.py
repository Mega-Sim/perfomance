"""Binary skeletonization utilities."""

from __future__ import annotations

import numpy as np


def _neighbors(img: np.ndarray) -> tuple[np.ndarray, ...]:
    p2 = np.roll(img, -1, axis=0)
    p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
    p4 = np.roll(img, 1, axis=1)
    p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
    p6 = np.roll(img, 1, axis=0)
    p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
    p8 = np.roll(img, -1, axis=1)
    p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
    return p2, p3, p4, p5, p6, p7, p8, p9


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    img = (binary > 0).astype(np.uint8)
    img[[0, -1], :] = 0
    img[:, [0, -1]] = 0

    changed = True
    while changed:
        changed = False
        p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(img)
        n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        s = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
        s += ((p3 == 0) & (p4 == 1)).astype(np.uint8)
        s += ((p4 == 0) & (p5 == 1)).astype(np.uint8)
        s += ((p5 == 0) & (p6 == 1)).astype(np.uint8)
        s += ((p6 == 0) & (p7 == 1)).astype(np.uint8)
        s += ((p7 == 0) & (p8 == 1)).astype(np.uint8)
        s += ((p8 == 0) & (p9 == 1)).astype(np.uint8)
        s += ((p9 == 0) & (p2 == 1)).astype(np.uint8)

        m1 = (img == 1) & (n >= 2) & (n <= 6) & (s == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
        if np.any(m1):
            img[m1] = 0
            changed = True

        p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(img)
        n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        s = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
        s += ((p3 == 0) & (p4 == 1)).astype(np.uint8)
        s += ((p4 == 0) & (p5 == 1)).astype(np.uint8)
        s += ((p5 == 0) & (p6 == 1)).astype(np.uint8)
        s += ((p6 == 0) & (p7 == 1)).astype(np.uint8)
        s += ((p7 == 0) & (p8 == 1)).astype(np.uint8)
        s += ((p8 == 0) & (p9 == 1)).astype(np.uint8)
        s += ((p9 == 0) & (p2 == 1)).astype(np.uint8)

        m2 = (img == 1) & (n >= 2) & (n <= 6) & (s == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
        if np.any(m2):
            img[m2] = 0
            changed = True

    img[[0, -1], :] = 0
    img[:, [0, -1]] = 0
    return img.astype(bool)
