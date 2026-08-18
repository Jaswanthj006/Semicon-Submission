#!/usr/bin/env python3
"""Sponsor inference entry: one reference + one search image -> print "x y".

Uses the same matcher as `python train.py localize`. Does not train or
overwrite weights. Stage1 / stage2 / stage3 stay in train.py.

  python localize.py --reference REF.png --search SEARCH.png
  python localize.py --data /path/to/split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import train

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def default_weight_dir() -> Path:
    return Path(__file__).resolve().parent / "model"


def find_pair_dirs(root: Path) -> tuple[Path, Path]:
    """Resolve reference/ and search/ under a dataset folder."""
    root = root.resolve()
    candidates = [root]
    for name in ("test", "eval", "train", "val", "optical", "dataset"):
        candidates.append(root / name)
        candidates.append(root / name / name)
    for base in candidates:
        ref_dir, sea_dir = base / "reference", base / "search"
        if ref_dir.is_dir() and sea_dir.is_dir():
            return ref_dir, sea_dir
    raise SystemExit(
        f"no reference/ and search/ folders under {root}\n"
        "expected: FOLDER/reference/*.png and FOLDER/search/*.png"
    )


def list_pairs(ref_dir: Path, sea_dir: Path) -> list[tuple[str, Path, Path]]:
    refs = sorted(
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    pairs = []
    missing = []
    for ref_path in refs:
        sea_path = sea_dir / ref_path.name
        if sea_path.is_file():
            pairs.append((ref_path.name, ref_path, sea_path))
        else:
            missing.append(ref_path.name)
    if not pairs:
        raise SystemExit(f"no matching image pairs in {ref_dir} and {sea_dir}")
    if missing:
        print(f"skipping {len(missing)} reference files with no matching search image")
    return pairs


def run_one(ref_path: Path, sea_path: Path, model, device) -> tuple[float, float]:
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(str(sea_path), cv2.IMREAD_GRAYSCALE)
    if ref is None or sea is None:
        raise SystemExit(f"could not read {ref_path} or {sea_path}")
    out = train.localize_pair(ref, sea, model, device)
    return out["x"], out["y"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", help="1000x1000 reference PNG, or a dataset folder")
    ap.add_argument("--search", help="1000x1000 search PNG")
    ap.add_argument(
        "--data",
        help="dataset folder with reference/ and search/ (runs each pair)",
    )
    ap.add_argument(
        "--out",
        default=str(default_weight_dir()),
        help="folder that contains verifier.pt (read-only)",
    )
    args = ap.parse_args()

    data_root = None
    if args.data:
        data_root = Path(args.data)
    elif args.reference and not args.search and Path(args.reference).is_dir():
        data_root = Path(args.reference)

    device = train.pick_device()
    model = train.load_verifier(Path(args.out), device)

    if data_root is not None:
        if not data_root.is_dir():
            raise SystemExit(f"not a folder: {data_root}")
        ref_dir, sea_dir = find_pair_dirs(data_root)
        pairs = list_pairs(ref_dir, sea_dir)
        for name, ref_path, sea_path in pairs:
            x, y = run_one(ref_path, sea_path, model, device)
            print(f"{name} {x:.2f} {y:.2f}")
        return

    if not args.reference or not args.search:
        raise SystemExit(
            "pass one pair:  python localize.py --reference REF.png --search SEARCH.png\n"
            "or a folder:    python localize.py --data /path/to/split"
        )

    x, y = run_one(Path(args.reference), Path(args.search), model, device)
    print(f"{x:.2f} {y:.2f}")


if __name__ == "__main__":
    main()
