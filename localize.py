#!/usr/bin/env python3
"""Sponsor inference entry: one reference + one search image -> print "x y".

Uses the same matcher as `python train.py localize`. Does not train or
overwrite weights. Stage1 / stage2 / stage3 stay in train.py.

  python localize.py --reference REF.png --search SEARCH.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import train


def default_weight_dir() -> Path:
    return Path(__file__).resolve().parent / "model"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="1000x1000 reference PNG")
    ap.add_argument("--search", required=True, help="1000x1000 search PNG")
    ap.add_argument(
        "--out",
        default=str(default_weight_dir()),
        help="folder that contains verifier.pt (read-only)",
    )
    args = ap.parse_args()

    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    if ref is None or sea is None:
        raise SystemExit("could not read reference or search image")

    device = train.pick_device()
    model = train.load_verifier(Path(args.out), device)
    out = train.localize_pair(ref, sea, model, device)
    print(f"{out['x']:.2f} {out['y']:.2f}")


if __name__ == "__main__":
    main()
