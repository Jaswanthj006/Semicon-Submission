# Drift-Sense

Applied Materials Track 2 — SEMICON India 2026

Find a 100×, 1 nm/px reference patch inside a 10×, ~10 nm/px search image. Both images are 1000×1000. Output is the center `(x, y)` of the match in search-image pixel coordinates, origin top-left. When more than one location is plausible, we tie-break to the point closest to the search-image center `(500, 500)`.

## Clone and install

```bash
git clone https://github.com/Jaswanthj006/Semicon-Submission.git
cd Semicon-Submission
pip install -r requirements.txt
```

Requirements: `numpy`, `opencv-python`, `torch`, `tqdm`, `matplotlib`. Python 3.10+, Windows or Mac, GPU optional (CUDA / Apple MPS / CPU all work).

If `pip` gives you CPU-only torch on a Windows machine with an NVIDIA GPU, install the CUDA build from [pytorch.org](https://pytorch.org) instead.

Weights already ship in `model/verifier.pt` — no training needed to run scoring.

## Run on your dataset

Single pair:

```bash
python localize.py --reference /path/to/REF.png --search /path/to/SEARCH.png
```

This prints one line: `x y`

By default the model looks for `verifier.pt` in `model/`. Point it elsewhere with `--out`:

```bash
python localize.py --reference REF.png --search SEARCH.png --out path/to/model
```

Batch of pairs with matching filenames in two folders:

```bash
# bash
for f in reference/*.png; do
  python localize.py --reference "$f" --search "search/$(basename "$f")"
done
```

```bat
:: Windows cmd
for %f in (reference\*.png) do python localize.py --reference "%f" --search "search\%~nxf"
```

Smoke test on a pair we ship:

```bash
python localize.py --reference dataset/reference/00000.png --search dataset/search/00000.png
```

Expected output is close to `844.38 285.63` (ground truth `844.6 285.6`). Runtime is roughly 0.5–2 s per pair on CPU.

RGB optical images run through the exact same command — `localize.py` reads inputs as grayscale internally, so no extra flag is needed.

## Method

DRAM repeats every few pixels at 10 nm/px, so a plain global ZNCC search often locks onto the wrong repeat cell rather than being slightly off on a correct one (pass@5 around 0.55–0.62 alone). The correct window is almost always inside the top-20 ZNCC peaks (~0.90 recall), so we use those peaks as candidates instead of trusting the single best one.

1. **Propose** — ZNCC over a scale grid (9.0 to 11.0) and rotation grid (−2° to +2°), then NMS down to the top 20 candidates.
2. **Rank** — a small CNN (~231k params, input is a 2×128×128 search-crop/template stack plus 3 scalar features) scores each candidate. It's trained with the other ZNCC peaks as hard negatives — it ranks proposals, it doesn't detect over the full image.
3. **Refine** — among candidates within 0.05 of the top verifier score, pick the one closest to `(500, 500)`, then apply 1D parabolic sub-pixel refinement on its ZNCC map.

If `verifier.pt` is missing, the pipeline falls back to ZNCC + sub-pixel refinement and still prints `x y`. If no ZNCC peaks are found at all, it prints `500 500` so the CLI never crashes.

## Results

Evaluated on `dataset/` (40 pairs, seed 7) and a held-out `test` split (250 pairs):

| Split | pass@5 | pass@2 | pass@1 | median error | time/pair |
|---|---|---|---|---|---|
| eval (40) | 0.875 | 0.85 | 0.625 | 0.66 px | ~0.54 s |
| test (250) | 0.856 | 0.844 | 0.684 | 0.64 px | — |

Eval by bucket: easy 1.00, scale_rot 1.00, hard_geometry 0.80, hard_noise 0.70.

Median is reported instead of mean because mean is inflated by a handful of period-jump failures. Artifacts from the eval run are in `results/metrics_eval.json`, `results/predictions.csv`, `results/pr_eval.png`, and `results/failures_eval/`.

Failure mode: when the model misses, it almost always locked onto a look-alike DRAM cell one period over, not a couple of pixels off the right one. Five eval IDs (`00025`, `00027`, `00029`, `00032`, `00035`) are kept on purpose as those period-alias misses. Overlays are in `results/failures_eval/`. On the pairs it does get right, sub-pixel error is already &lt;1 px.

## Generator and RGB inputs (optional)

`generate_dataset.py` builds synthetic pairs from physical DRAM layouts at 1 nm/px: a reference patch is cropped at 1 µm, then blurred and downsampled to build the search image, with SEM-style artifacts applied separately so the search image ends up noisier than the reference. Pitch ratios follow the public 6F² cell convention, not any fab-specific layout. Default split mix is 40% easy / 35% normal / 15% hard_noise / 10% hard_geometry, with scale 9–11 and rotation ±2° applied to the search image only. Speckle, barrel distortion, and vignette are deliberately left off. Full noise-model details are in `references/CITATIONS.md`.

`dataset/` already ships 40 grayscale eval pairs (stratified 10 per bucket above), which is what the results table uses.

```bash
python generate_dataset.py --num-samples 40 --split eval --output-dir ./dataset --seed 7
python generate_dataset.py --num-samples 20 --split optical --output-dir ./dataset_optical --seed 99 --optical
```

`--optical` produces true 3-channel BGR images into `dataset_optical/` without touching `dataset/`. `localize.py` reads them as grayscale regardless — same command, no flag needed.

Retraining is not required for scoring and only applies if you change the data or matcher:

```bash
python train.py stage1 --data output --split val
python train.py stage2 --data output --out model
python train.py stage3 --data output --out model --split eval
```

(Training images are local and gitignored, and are not needed to run inference.)

## Layout

```
localize.py
train.py
generate_dataset.py
requirements.txt
model/
  verifier.pt
dataset/
  reference/
  search/
  manifest.csv
dataset_optical/
results/
references/
  CITATIONS.md
```
