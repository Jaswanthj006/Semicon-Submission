# Drift-Sense

> Given a 100× / 1 nm/px reference patch (1000×1000) and a 10× / ~10 nm/px search image (1000×1000), output the center `(x, y)` of the reference inside the search image. Tie-break: closest to `(500, 500)`.



---

## Our Solution — Three Stages

DRAM cells repeat every ~5–10 pixels at 10 nm/px. A global ZNCC max hits the **wrong cell** ~45% of the time — not off by 2 px, but by an entire period (100–600 px). The true window is usually among the top-20 peaks (~90% recall). So we propose many, then learn which one is real.

**Stage 1 — Propose (multi-scale ZNCC)**

The exact scale ratio varies 9×–11× and the search can be rotated ±2°. We cannot assume a fixed downscale.

- Resize the reference onto a **5-point scale grid** (9.0, 9.5, 10.0, 10.5, 11.0)
- Rotate each resized template at **5 angles** (−2°, −1°, 0°, +1°, +2°) → 25 ZNCC maps
- Extract local maxima (up to 8 per map), pool, sort by score
- Non-maximum suppression (3 px radius), keep the **top 20** candidates

Result: 20 candidate locations. The true match is among them ~90% of the time.

**Stage 2 — Rank (CNN verifier)**

The 20 candidates all look like valid matches to ZNCC. A small CNN tells the real one from the aliases.

- For each candidate: crop 128×128 from search + matched template → **2-channel 128×128 input**
- Plus **3 scalar features**: ZNCC score, scale ratio, rotation angle
- CNN (~231k params, 4 conv blocks + 2-layer head) → one score per candidate → softmax
- **Training trick:** hard negatives = the other ZNCC peaks from the same pair (the exact period aliases)

This is not a full-image detector. It only ranks windows that Stage 1 already found.

**Stage 3 — Refine (tie-break + sub-pixel)**

- Among candidates within **0.05** of the top verifier score, pick closest to **(500, 500)**
- Fit a **1D parabola** on the ZNCC map around the integer peak → sub-pixel offset
- Final: `x = xi + dx + template_width/2`, `y = yi + dy + template_width/2`

Fallback: no weights → ZNCC score alone. No peaks at all → prints `500 500`.

---

## Clone & Install

```bash
git clone https://github.com/Jaswanthj006/Semicon-Submission.git
cd Semicon-Submission
pip install -r requirements.txt
```

Python 3.10+ · Windows / Mac · GPU optional (CUDA / MPS / CPU)

Weights ship in `model/verifier.pt` — no training needed.

---

## Test on Your Dataset

**One pair:**

```bash
python localize.py --reference /path/to/REF.png --search /path/to/SEARCH.png
```

Prints one line: `x y`

**Batch (bash):**

```bash
for f in reference/*.png; do
  python localize.py --reference "$f" --search "search/$(basename "$f")"
done
```

**Batch (Windows):**

```bat
for %f in (reference\*.png) do python localize.py --reference "%f" --search "search\%~nxf"
```

**Smoke test on shipped pair:**

```bash
python localize.py --reference dataset/reference/00000.png --search dataset/search/00000.png
# Expected: ~844.38 285.63  (GT: 844.6, 285.6)
```

RGB optical PNGs work with the same command — loaded as grayscale internally.

---

## Generate Dataset & Test

One command builds reference + search + manifest with ground truth:

```bash
python generate_dataset.py --num-samples 100 --split test --output-dir ./output --seed 2026
```

Then score the model on it:

```bash
python train.py stage3 --data output --out model --split test
```

RGB optical bonus (does not overwrite `dataset/`):

```bash
python generate_dataset.py --num-samples 20 --split optical --output-dir ./dataset_optical --seed 99 --optical
```

---

## Results

| Split | n | pass@5 | pass@2 | pass@1 | median | time |
|---|---|---|---|---|---|---|
| **eval** | 40 | 0.875 | 0.85 | 0.625 | 0.66 px | 0.54 s |
| **test** | 250 | 0.856 | 0.844 | 0.684 | 0.64 px | 0.54 s |

Per bucket (eval): easy **1.00** · scale_rot **1.00** · hard_geometry **0.80** · hard_noise **0.70**

---

## Noise — What & Why

| Artifact | Why we add it | Citation |
|---|---|---|
| Poisson shot noise | Electron-dose statistics | Joy 1995 |
| Detector Gaussian | SEM electronics noise | Timischl 2015 |
| Raster shear + jitter | Scan drift | Jones & Nellist 2013 |
| Charging streaks | Local insulator charging | Cazaux 2004 |
| Salt-and-pepper | Dead/hot pixels | — |
| Scale 9–11, rot ±2° | Real FOV is not exactly 10× / 0° | Problem statement |

Speckle, barrel distortion, and vignette are deliberately **OFF** — they turn the search into a warped photo instead of a noisy, slightly scaled DRAM field.

Full details: `references/CITATIONS.md`

---

## Failure Mode & How We Addressed It

**What fails:** wrong DRAM period — the matcher jumps to a look-alike cell one repeat over (error ~100–600 px).

**Why:** at 10 nm/px, DRAM repeats every ~5–10 px. Many cells produce near-identical ZNCC scores.

**How we addressed it:**

- Stage 2 CNN trained with other ZNCC peaks as **hard negatives** (the exact aliases that cause failures)
- Spec tie-break picks the candidate nearest to (500, 500) among near-tied scores
- Reduced pass@5 failures from **~45%** (ZNCC alone) to **~12.5%** (with verifier)

Remaining misses (5 out of 40 eval): IDs `00025`, `00027`, `00029`, `00032`, `00035`. Overlays in `results/failures_eval/`.

---

## Evaluation Graph

![Precision-Recall curve](results/pr_eval.png)

Precision vs recall at different error thresholds, broken down by noise bucket.
