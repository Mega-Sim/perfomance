from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TRACK_LAYER_HINTS = ("track", "rail", "line")
NODE_LAYER_HINTS = ("station", "node", "split", "merge")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a DXF as a raster image for vision inference")
    p.add_argument("--dxf", required=True, help="Input DXF path")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--size", type=int, default=256, help="Square output image size")
    p.add_argument("--margin", type=int, default=12)
    return p.parse_args()


def parse_dxf(path: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    lines: list[dict] = []
    arcs: list[dict] = []
    circles: list[dict] = []
    texts: list[dict] = []

    rows = path.read_text(errors="ignore").splitlines()
    i = 0
    in_entities = False
    cur: str | None = None
    ent: dict = {}

    def flush() -> None:
        nonlocal ent
        if not cur or not ent:
            ent = {}
            return
        if cur == "LINE":
            lines.append(ent.copy())
        elif cur == "ARC":
            arcs.append(ent.copy())
        elif cur == "CIRCLE":
            circles.append(ent.copy())
        elif cur in ("TEXT", "MTEXT"):
            texts.append(ent.copy())
        ent = {}

    while i < len(rows) - 1:
        code = rows[i].strip()
        val = rows[i + 1].strip()
        i += 2

        if code == "0" and val == "SECTION" and i + 1 < len(rows):
            if rows[i].strip() == "2" and rows[i + 1].strip() == "ENTITIES":
                in_entities = True
                continue
        if code == "0" and val == "ENDSEC":
            in_entities = False
            continue
        if not in_entities:
            continue

        if code == "0":
            flush()
            cur = val if val in ("LINE", "ARC", "CIRCLE", "TEXT", "MTEXT") else None
            continue
        if cur is None:
            continue

        if code == "8":
            ent["layer"] = val
        elif cur == "LINE":
            if code == "10":
                ent["x1"] = float(val)
            elif code == "20":
                ent["y1"] = float(val)
            elif code == "11":
                ent["x2"] = float(val)
            elif code == "21":
                ent["y2"] = float(val)
        elif cur == "ARC":
            if code == "10":
                ent["cx"] = float(val)
            elif code == "20":
                ent["cy"] = float(val)
            elif code == "40":
                ent["r"] = float(val)
            elif code == "50":
                ent["a0"] = float(val)
            elif code == "51":
                ent["a1"] = float(val)
        elif cur == "CIRCLE":
            if code == "10":
                ent["cx"] = float(val)
            elif code == "20":
                ent["cy"] = float(val)
            elif code == "40":
                ent["r"] = float(val)
        elif cur in ("TEXT", "MTEXT"):
            if code == "10":
                ent["x"] = float(val)
            elif code == "20":
                ent["y"] = float(val)
            elif code in ("1", "3"):
                ent["text"] = f"{ent.get('text', '')}{val}"
    flush()
    return lines, arcs, circles, texts


def _layer(ent: dict) -> str:
    return str(ent.get("layer", "")).strip().lower()


def _is_track_layer(name: str) -> bool:
    return (not name) or any(x in name for x in TRACK_LAYER_HINTS)


def _is_node_layer(name: str) -> bool:
    return any(x in name for x in NODE_LAYER_HINTS)


def _fit_transform(points: np.ndarray, size: int, margin: int):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scale = float((size - 2 * margin) / max(span[0], span[1]))

    def to_px(p: tuple[float, float]) -> tuple[float, float]:
        x = margin + (p[0] - mn[0]) * scale
        y = margin + (p[1] - mn[1]) * scale
        return (x, size - y)

    return to_px


def render_dxf_to_image(dxf: Path, out: Path, size: int = 256, margin: int = 12) -> Path:
    lines, arcs, circles, texts = parse_dxf(dxf)

    all_pts: list[tuple[float, float]] = []
    for ln in lines:
        if {"x1", "y1", "x2", "y2"}.issubset(ln):
            all_pts.extend([(ln["x1"], ln["y1"]), (ln["x2"], ln["y2"])])
    for arc in arcs:
        if {"cx", "cy", "r"}.issubset(arc):
            all_pts.extend([(arc["cx"] - arc["r"], arc["cy"] - arc["r"]), (arc["cx"] + arc["r"], arc["cy"] + arc["r"])])
    for txt in texts:
        if {"x", "y"}.issubset(txt):
            all_pts.append((txt["x"], txt["y"]))
    if not all_pts:
        raise ValueError(f"No renderable entities found in {dxf}")

    to_px = _fit_transform(np.asarray(all_pts, dtype=np.float64), size=size, margin=margin)

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    track_color = (0, 170, 0)
    node_color = (220, 50, 50)

    for ln in lines:
        if not {"x1", "y1", "x2", "y2"}.issubset(ln):
            continue
        lyr = _layer(ln)
        if not (_is_track_layer(lyr) or _is_node_layer(lyr)):
            continue
        color = node_color if _is_node_layer(lyr) else track_color
        draw.line([to_px((ln["x1"], ln["y1"])), to_px((ln["x2"], ln["y2"]))], fill=color, width=2)

    for arc in arcs:
        if not {"cx", "cy", "r", "a0", "a1"}.issubset(arc):
            continue
        lyr = _layer(arc)
        if not (_is_track_layer(lyr) or _is_node_layer(lyr)):
            continue
        color = node_color if _is_node_layer(lyr) else track_color
        a0 = math.radians(arc["a0"])
        a1 = math.radians(arc["a1"])
        if a1 < a0:
            a1 += 2 * math.pi
        pts = [
            to_px((arc["cx"] + arc["r"] * math.cos(t), arc["cy"] + arc["r"] * math.sin(t)))
            for t in np.linspace(a0, a1, 64)
        ]
        draw.line(pts, fill=color, width=2)

    for circ in circles:
        if not {"cx", "cy", "r"}.issubset(circ):
            continue
        if not _is_node_layer(_layer(circ)):
            continue
        x, y = to_px((circ["cx"], circ["cy"]))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=node_color)

    for txt in texts:
        if not {"x", "y"}.issubset(txt):
            continue
        if _is_node_layer(_layer(txt)) or "station" in str(txt.get("text", "")).lower():
            x, y = to_px((txt["x"], txt["y"]))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=node_color)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def main() -> int:
    args = parse_args()
    out = render_dxf_to_image(Path(args.dxf), Path(args.out), size=args.size, margin=args.margin)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
