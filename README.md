# Drift-Sense — SEM pattern localization

Find a 100× / 1 nm/px **reference** patch inside a 10× / ~10 nm/px **search** image.
Both images are 1000×1000. Output is the patch **center** `(x, y)` in search pixels
(origin top-left). When several matches look equally good, pick the one closest
to the search-image center `(500, 500)`.

This is Applied Materials Track 2 (SEMICON India 2026).

## Quick start (judges)

```bash
pip install -r requirements.txt
python localize.py --reference REF.png --search SEARCH.png
```

Prints one line: `x y`

Example on the shipped eval pair (GT `844.6 285.6`):

```bash
python localize.py --reference dataset/reference/00000.png --search dataset/search/00000.png
# 844.38 285.63
```

Weights: `model/verifier.pt` (read-only). RGB optical PNGs work too — they are
loaded as grayscale.

## Repository layout

```
localize.py              inference CLI (one pair → "x y")
train.py                 stage1 proposals / stage2 train / stage3 metrics
generate_dataset.py      standalone SEM DRAM generator
model/verifier.pt        trained ranking CNN (~917 KB)
dataset/                 40 representative eval pairs (≥30 required)
  reference/  search/  manifest.csv
results/                 pass@k, PR plots, failure overlays, predictions.csv
references/CITATIONS.md  public sources for DRAM + SEM artifacts
```

Training images (`output/train` 2000, `output/val` 250) are local only. They are
not required to run `localize.py`. Recreate them with `generate_dataset.py` if
you need to retrain.

## Method (three stages)

DRAM cells repeat every few pixels at 10 nm/px. A global template match often
locks onto the **wrong period**, not a slightly wrong pixel.

Measured on val before this design:

| Rule | pass@5 |
|---|---|
| Global ZNCC max | ~0.55–0.62 |
| True window inside top-20 ZNCC peaks | ~0.85–0.90 |
| ECC / multi-map re-ranking | worse than ZNCC |

So the matcher is: **propose many look-alike windows, then learn which one is real.**

### Stage 1 — proposals (`train.py stage1`)

- Downsample the 1000×1000 reference onto a scale grid `{9.0, 9.5, 10.0, 10.5, 11.0}`
- Rotate the template `{−2, −1, 0, +1, +2}` degrees
- Zero-mean normalized cross-correlation (ZNCC) vs the search image
- Keep local maxima, NMS, **top 20** candidates

Validates **candidate recall**: is the true center inside those 20?
Eval recall@20 = **0.90**. If the true cell is not proposed, stage 2 cannot recover it.

### Stage 2 — verifier (`train.py stage2`)

A small CNN (~231k params) scores each of the 20 windows.

- Input: 2×128×128 crop (search window + matched template) plus a few scalars
- Positive = candidate within 5 px of `gt_x, gt_y`
- Hard negatives = the **other ZNCC peaks** (the aliases)
- Trained on the 2000 train pairs that had a positive

This is not a heatmap detector. Both images are different zooms; the network only
**ranks** windows that ZNCC already found.

### Stage 3 — localize (`localize.py` / `train.py stage3`)

1. Stage-1 proposals
2. Verifier softmax over the 20
3. Spec tie-break: among scores within 0.05 of the top, pick closest to `(500, 500)`
4. Parabolic sub-pixel refine on the ZNCC map

Typical runtime **~0.54 s/pair**. If `verifier.pt` is missing, falls back to ZNCC + sub-pixel.

## Synthetic data — what noise, and why

`generate_dataset.py` draws a physical DRAM layout at 1 nm/px, crops 1 µm × 1 µm
for the reference, blurs + downsamples for the search, then applies **separate**
SEM artifacts (search is always noisier). Ground truth is the crop center mapped
into search pixels, then warped with the same rotation.

Layouts are public 6F² pitch ratios (word-line ~2F, bit-line ~3F), not fab IP.
See `references/CITATIONS.md`.

Default mix (same recipe used to train):

| Bucket | Share | What it tests |
|---|---|---|
| easy | 40% | mild noise, scale=10, rot=0 — baseline accuracy |
| normal | 35% | mid noise, occasional charging |
| hard_noise | 15% | low dose, drift, charging streaks, salt-and-pepper |
| hard_geometry | 10% | crop in a periodic mat (`boundary_bias=0`) — wrong-cell trap |

**On (citeable SEM effects)**

| Artifact | Why |
|---|---|
| Gaussian beam PSF + astigmatism | finite probe size (Reimer / Goldstein) |
| Poisson shot noise (dose) | electron count statistics (Joy) |
| Additive detector noise | SEM electronics |
| Raster shear + jitter | scan drift (Jones & Nellist) |
| Charging streaks | local charging (Cazaux) |
| Salt-and-pepper | dead / hot pixels (hard_noise only) |
| Gamma | contrast / detector curve |
| Scale 9–11 | 10× FOV is not exactly 10.0 |
| Rotation ±2° | stage angle, search image only |

**Off:** speckle, barrel distortion, vignette — they turn the search into a warped
photo instead of a noisy, slightly scaled DRAM field.

## Submission eval (`dataset/`, 40 pairs)

≥30 representative cases, not a size dump. Stratified:

| IDs | Bucket | What it evaluates |
|---|---|---|
| 00000–00009 | easy | clean 10×, 0° |
| 00010–00019 | scale_rot | scale 9–11, ±2° |
| 00020–00029 | hard_noise | SEM artifacts |
| 00030–00039 | hard_geometry | repetitive DRAM |

GT locations are **not** all near center (e.g. pair 32 is at y≈62). Failures
`00025, 00027, 00029, 00032, 00035` are kept on purpose: the matcher jumped to a
look-alike mat (period alias). Overlays: `results/failures_eval/`.
Scorer table: `results/predictions.csv`.

### Results

Eval (40) · 0.54 s/pair · proposal recall 0.90

| Split | n | pass@5 | pass@2 | pass@1 | median |
|---|---|---|---|---|---|
| eval | 40 | 0.875 | 0.85 | 0.625 | 0.66 px |
| test | 250 | 0.856 | 0.844 | 0.684 | 0.64 px |

Eval by bucket: easy **1.00** · scale_rot **1.00** · hard_geometry **0.80** · hard_noise **0.70**.

## How to run

Python 3.10+, numpy, opencv-python, torch (GPU optional).

### 1. Localize one pair (required)

```bash
python localize.py --reference dataset/reference/00000.png --search dataset/search/00000.png
python localize.py --reference REF.png --search SEARCH.png --out model
```

### 2. Generate data (only if you need new pairs)

```bash
# same 40-pair eval recipe (seed 7)
python generate_dataset.py --num-samples 40 --split eval --output-dir ./dataset --seed 7

# training set used for verifier.pt (seed 42)
python generate_dataset.py --num-samples 2000 --split train --output-dir ./output --seed 42
python generate_dataset.py --num-samples 250  --split val   --output-dir ./output --seed 99
```

`--no-randomize-noise` freezes parameters (debug). Default mix is on.

### 3. Train (only if you change data or the matcher)

Needs `output/train` and `output/val` with `manifest.csv`. Overwrites `model/verifier.pt`.

```bash
python train.py stage1 --data output --split val
python train.py stage2 --data output --out model
python train.py stage3 --data output --out model --split eval
```

Do **not** retrain to run `localize.py`. Shipped weights already match this eval.

## Failure mode

When the method misses, it is almost always a **wrong DRAM period**, not a 2 px
wobble. Stage 1 never proposed the true cell, or the verifier ranked a
look-alike mat higher. Sub-pixel error on correct cells is already <1 px.
