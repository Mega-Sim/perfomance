import argparse
import json
import os
import shutil
from datetime import datetime


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="datasets/standard/images")
    ap.add_argument("--out-graphs", default="datasets/standard/graphs")
    ap.add_argument("--out-previews", default="datasets/standard/previews")
    ap.add_argument("--manifest", default="datasets/standard/manifest.jsonl")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"input dir not found: {args.in_dir}")

    ensure_dir(args.out_graphs)
    ensure_dir(args.out_previews)

    imgs = [
        os.path.join(args.in_dir, fn)
        for fn in sorted(os.listdir(args.in_dir))
        if fn.lower().endswith(".png")
    ]

    now = datetime.now().isoformat(timespec="seconds")

    with open(args.manifest, "a", encoding="utf-8") as mf:
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_json = os.path.join(args.out_graphs, f"{stem}.json")
            out_preview = os.path.join(args.out_previews, f"{stem}.png")

            if args.skip_existing and os.path.exists(out_json) and os.path.exists(out_preview):
                continue

            entry = {"ts": now, "image": img_path, "stem": stem}
            try:
                # preview는 원본 이미지 복사 (검증용)
                shutil.copyfile(img_path, out_preview)

                # placeholder directed graph JSON (다음 PR에서 실제 변환 로직으로 채움)
                data = {
                    "nodes": [],
                    "edges": [],
                    "stations": {},
                    "meta": {
                        "source_image": os.path.basename(img_path),
                        "created_at": now,
                        "note": "placeholder (converter logic will be added in next PR)",
                    },
                }
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                entry.update({"ok": True, "json": out_json, "preview": out_preview})
            except Exception as e:
                entry.update({"ok": False, "error": str(e)})

            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            mf.flush()


if __name__ == "__main__":
    main()
