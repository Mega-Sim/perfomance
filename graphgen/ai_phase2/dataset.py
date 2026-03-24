from __future__ import annotations

import json
import random
from pathlib import Path


def _normalize_dxf_path(path_value: str | Path, base_dir: Path) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _load_manifest_list(list_path: Path) -> list[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"DXF list file not found: {list_path}")

    if list_path.suffix.lower() == ".json":
        data = json.loads(list_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "dxfs" in data and isinstance(data["dxfs"], list):
                items = data["dxfs"]
            else:
                raise ValueError("JSON manifest dict must contain a list field 'dxfs'")
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("JSON manifest must be a list or an object containing 'dxfs'")
        base_dir = list_path.parent
        return [_normalize_dxf_path(item, base_dir) for item in items]

    base_dir = list_path.parent
    items: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(_normalize_dxf_path(line, base_dir))
    return items


def _collect_from_dir(dxf_dir: Path) -> list[Path]:
    if not dxf_dir.exists() or not dxf_dir.is_dir():
        raise FileNotFoundError(f"DXF directory not found: {dxf_dir}")
    return sorted(dxf_dir.glob("*.dxf"), key=lambda p: p.name.lower())


def load_dxf_paths(
    *,
    dxf: str | None = None,
    dxf_dir: str | None = None,
    dxf_list: str | None = None,
) -> list[Path]:
    selected = [v for v in (dxf, dxf_dir, dxf_list) if v]
    if len(selected) != 1:
        raise ValueError("Provide exactly one of --dxf, --dxf_dir, or --dxf_list")

    paths: list[Path]
    if dxf:
        paths = [Path(dxf).resolve()]
    elif dxf_dir:
        paths = _collect_from_dir(Path(dxf_dir).resolve())
    else:
        paths = _load_manifest_list(Path(dxf_list).resolve())

    normalized: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        normalized.append(rp)

    normalized = sorted(normalized, key=lambda p: str(p).lower())
    if not normalized:
        raise ValueError("No DXF files found from the provided input")

    missing = [str(p) for p in normalized if not p.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Some DXF files were not found:\n{preview}")

    bad_ext = [str(p) for p in normalized if p.suffix.lower() != ".dxf"]
    if bad_ext:
        preview = "\n".join(bad_ext[:5])
        raise ValueError(f"Non-DXF files found in inputs:\n{preview}")

    return normalized


def split_train_val(
    dxf_paths: list[Path],
    *,
    seed: int,
    train_ratio: float,
    train_list: str | None = None,
    val_list: str | None = None,
) -> tuple[list[Path], list[Path]]:
    if train_list or val_list:
        if not (train_list and val_list):
            raise ValueError("When using explicit split, provide both --train_list and --val_list")
        train_paths = load_dxf_paths(dxf_list=train_list)
        val_paths = load_dxf_paths(dxf_list=val_list)
        return train_paths, val_paths

    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("--train_ratio must be in the range (0, 1]")

    if len(dxf_paths) == 1 or train_ratio >= 1.0:
        return list(dxf_paths), []

    shuffled = list(dxf_paths)
    random.Random(seed).shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    train_count = max(1, min(train_count, len(shuffled) - 1))
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])
