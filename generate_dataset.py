#!/usr/bin/env python3
"""Single-file Drift-Sense dataset generator.

Produces 1000x1000 grayscale Reference (100x / 1 nm/px) and Search
(10x / ~10 nm/px) PNG pairs plus manifest.csv.

Noise recipe is ON by default (same mix used to train the matcher):
  40% easy / 35% normal / 15% hard_noise / 10% hard_geometry
  scale_ratio in [9, 11], rotation_deg in [-2, 2]
  speckle / barrel / vignette OFF
  search always noisier than reference

    python generate_dataset.py --num-samples 40 --split eval --output-dir ./dataset --seed 7

Dependencies: numpy, opencv-python (no torch).
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

# =============================================================================
# Geometry
# =============================================================================
REFERENCE_SIZE_PX = 1000
SEARCH_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1

# =============================================================================
# DRAM 6F2 / FinFET presets (public pitch ratios, not fab numbers)
# =============================================================================
DRAM_1X = {
    "kind": "dram", "feature_size_nm": 32,
    "word_line_pitch_nm": 64, "word_line_width_nm": 32,
    "bit_line_pitch_nm": 96, "bit_line_width_nm": 32, "contact_diameter_nm": 32,
}
DRAM_DENSE = {
    "kind": "dram", "feature_size_nm": 24,
    "word_line_pitch_nm": 48, "word_line_width_nm": 24,
    "bit_line_pitch_nm": 72, "bit_line_width_nm": 24, "contact_diameter_nm": 24,
}
DRAM_LOOSE = {
    "kind": "dram", "feature_size_nm": 48,
    "word_line_pitch_nm": 96, "word_line_width_nm": 48,
    "bit_line_pitch_nm": 144, "bit_line_width_nm": 48, "contact_diameter_nm": 48,
}
DRAM_WIDE = {
    "kind": "dram", "feature_size_nm": 60,
    "word_line_pitch_nm": 120, "word_line_width_nm": 56,
    "bit_line_pitch_nm": 180, "bit_line_width_nm": 60, "contact_diameter_nm": 58,
}
DRAM_COMPACT = {
    "kind": "dram", "feature_size_nm": 36,
    "word_line_pitch_nm": 72, "word_line_width_nm": 30,
    "bit_line_pitch_nm": 108, "bit_line_width_nm": 34, "contact_diameter_nm": 30,
}
DRAM_LEGACY = {
    "kind": "dram", "feature_size_nm": 80,
    "word_line_pitch_nm": 160, "word_line_width_nm": 78,
    "bit_line_pitch_nm": 240, "bit_line_width_nm": 80, "contact_diameter_nm": 78,
}
FINFET_10NM = {
    "kind": "finfet", "fin_pitch_nm": 48, "fin_width_nm": 16,
    "gate_pitch_nm": 90, "gate_length_nm": 28, "contact_size_nm": 28,
}
FINFET_7NM = {
    "kind": "finfet", "fin_pitch_nm": 40, "fin_width_nm": 14,
    "gate_pitch_nm": 76, "gate_length_nm": 24, "contact_size_nm": 24,
}
FINFET_14NM = {
    "kind": "finfet", "fin_pitch_nm": 60, "fin_width_nm": 20,
    "gate_pitch_nm": 110, "gate_length_nm": 34, "contact_size_nm": 34,
}
FINFET_22NM = {
    "kind": "finfet", "fin_pitch_nm": 80, "fin_width_nm": 26,
    "gate_pitch_nm": 150, "gate_length_nm": 46, "contact_size_nm": 44,
}
FINFET_28NM = {
    "kind": "finfet", "fin_pitch_nm": 96, "fin_width_nm": 32,
    "gate_pitch_nm": 180, "gate_length_nm": 56, "contact_size_nm": 52,
}
FINFET_45NM = {
    "kind": "finfet", "fin_pitch_nm": 140, "fin_width_nm": 46,
    "gate_pitch_nm": 260, "gate_length_nm": 80, "contact_size_nm": 76,
}

PRESETS = {
    "dram_1x": DRAM_1X, "dram_dense": DRAM_DENSE, "dram_loose": DRAM_LOOSE,
    "dram_wide": DRAM_WIDE, "dram_compact": DRAM_COMPACT, "dram_legacy": DRAM_LEGACY,
    "finfet_10nm": FINFET_10NM, "finfet_7nm": FINFET_7NM, "finfet_14nm": FINFET_14NM,
    "finfet_22nm": FINFET_22NM, "finfet_28nm": FINFET_28NM, "finfet_45nm": FINFET_45NM,
}
DRAM_PRESET_NAMES = [
    "dram_1x", "dram_dense", "dram_loose", "dram_wide", "dram_compact", "dram_legacy",
]
FINFET_PRESET_NAMES = [
    "finfet_10nm", "finfet_7nm", "finfet_14nm", "finfet_22nm", "finfet_28nm", "finfet_45nm",
]


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return dict(PRESETS[name])


def presets_for_kind(kind: str):
    names = DRAM_PRESET_NAMES if kind == "dram" else FINFET_PRESET_NAMES
    return [get_preset(n) for n in names]


# =============================================================================
# Layout drawing (1 nm/px)
# =============================================================================
BACKGROUND = 40
WORD_LINE_VAL = 150
BIT_LINE_VAL = 170
CONTACT_VAL = 225
FIN_VAL = 150
GATE_VAL = 170
WIDTH_JITTER_FRACTION = 0.10


def maybe_collapse_gap(gap_nm, threshold_nm, rng, collapse_prob=0.7) -> bool:
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)


def _line_positions(size_px, pitch_nm, rng, jitter_nm) -> np.ndarray:
    positions, pos = [], rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, jitter_nm)
    return np.array(positions)


def _line_mask(size_px, positions, width_nm, collapse_threshold_nm, rng,
               linewidth_bias_nm=0.0):
    mask = np.zeros(size_px, dtype=bool)
    biased = max(width_nm + linewidth_bias_nm, 1.0)
    widths = biased * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION, size=len(positions)))
    widths = np.clip(widths, biased * 0.5, biased * 1.5)
    for i, center in enumerate(positions):
        half = widths[i] / 2.0
        lo, hi = int(round(center - half)), int(round(center + half))
        mask[max(lo, 0):min(hi, size_px)] = True
        if i + 1 < len(positions):
            nh = widths[i + 1] / 2.0
            gap = (positions[i + 1] - nh) - (center + half)
            if maybe_collapse_gap(gap, collapse_threshold_nm, rng):
                blo = int(round(center + half))
                bhi = int(round(positions[i + 1] - nh))
                mask[max(blo, 0):min(bhi, size_px)] = True
    return mask


def _round_corners(canvas, px):
    if px < 0.5:
        return canvas
    k = max(1, int(round(px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)


def generate_dram_canvas(size_px, preset, collapse_threshold_nm, rng,
                         linewidth_bias_nm=0.0, corner_rounding_px=0.0):
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    wl = _line_positions(size_px, preset["word_line_pitch_nm"], rng, 1.5)
    bl = _line_positions(size_px, preset["bit_line_pitch_nm"], rng, 1.5)
    row = _line_mask(size_px, wl, preset["word_line_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm)
    col = _line_mask(size_px, bl, preset["bit_line_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm)
    canvas[row, :] = np.maximum(canvas[row, :], WORD_LINE_VAL)
    canvas[:, col] = np.maximum(canvas[:, col], BIT_LINE_VAL)
    radius0 = max(preset["contact_diameter_nm"] + linewidth_bias_nm, 1.0) / 2.0
    for i, y in enumerate(wl):
        for j, x in enumerate(bl):
            if (i + j) % 2 == 0:
                r = max(1, int(round(radius0 * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)))))
                cv2.circle(canvas, (int(round(x)), int(round(y))), r, CONTACT_VAL, -1)
    return _round_corners(canvas, corner_rounding_px)


def generate_finfet_canvas(size_px, preset, collapse_threshold_nm, rng,
                           linewidth_bias_nm=0.0, corner_rounding_px=0.0):
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)
    fins = _line_positions(size_px, preset["fin_pitch_nm"], rng, 1.0)
    gates = _line_positions(size_px, preset["gate_pitch_nm"], rng, 1.0)
    col = _line_mask(size_px, fins, preset["fin_width_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm)
    row = _line_mask(size_px, gates, preset["gate_length_nm"],
                     collapse_threshold_nm, rng, linewidth_bias_nm)
    canvas[:, col] = np.maximum(canvas[:, col], FIN_VAL)
    canvas[row, :] = np.maximum(canvas[row, :], GATE_VAL)
    half = max(1, int(round(max(preset["contact_size_nm"] + linewidth_bias_nm, 1.0) / 2.0)))
    for i, fx in enumerate(fins):
        for j in range(len(gates) - 1):
            if (i + j) % 2 == 0:
                mid = (gates[j] + gates[j + 1]) / 2.0
                x, y = int(round(fx)), int(round(mid))
                p0 = (max(x - half, 0), max(y - half, 0))
                p1 = (min(x + half, size_px - 1), min(y + half, size_px - 1))
                cv2.rectangle(canvas, p0, p1, CONTACT_VAL, -1)
    return _round_corners(canvas, corner_rounding_px)


_GENERATORS = {"dram": generate_dram_canvas, "finfet": generate_finfet_canvas}

STRIP_BASE_VAL = 95
STRIP_LINE_VAL = 128
STRIP_LINE_PITCH_NM = 220
STRIP_LINE_WIDTH_NM = 9


def _strip_routing_texture(size_px, rng):
    canvas = np.full((size_px, size_px), STRIP_BASE_VAL, dtype=np.uint8)
    half = STRIP_LINE_WIDTH_NM / 2.0
    for positions, is_row in (
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), True),
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), False),
    ):
        for center in positions:
            lo = max(int(round(center - half)), 0)
            hi = min(int(round(center + half)), size_px)
            if is_row:
                canvas[lo:hi, :] = STRIP_LINE_VAL
            else:
                canvas[:, lo:hi] = STRIP_LINE_VAL
    return canvas


def _zone_grid(size_px, mat_size_nm, strip_width_nm):
    spans, pos, is_mat = [], 0.0, True
    while pos < size_px:
        end = min(pos + (mat_size_nm if is_mat else strip_width_nm), size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos, is_mat = end, not is_mat
    return spans


def generate_zone_canvas(size_px, kind, collapse_threshold_nm, rng,
                         mat_size_nm=2600.0, strip_width_nm=320.0,
                         linewidth_bias_nm=0.0, corner_rounding_px=0.0):
    generator = _GENERATORS[kind]
    presets = presets_for_kind(kind)
    canvas = _strip_routing_texture(size_px, rng)
    mat_rects, strip_rects = [], []
    for row_is_mat, y0, y1 in _zone_grid(size_px, mat_size_nm, strip_width_nm):
        for col_is_mat, x0, x1 in _zone_grid(size_px, mat_size_nm, strip_width_nm):
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                h, w = y1 - y0, x1 - x0
                preset = presets[int(rng.integers(0, len(presets)))]
                child = np.random.default_rng(rng.integers(0, 2**31 - 1))
                side = max(h, w)
                mat = generator(side, preset, collapse_threshold_nm, child,
                                linewidth_bias_nm=linewidth_bias_nm,
                                corner_rounding_px=corner_rounding_px)
                canvas[y0:y1, x0:x1] = mat[:h, :w]
                mat_rects.append((x0, y0, w, h))
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))
    return {"canvas": canvas, "mat_rects": mat_rects, "strip_rects": strip_rects}


# =============================================================================
# SEM imaging artifacts
# =============================================================================
def gaussian_psf_blur(img, spot_size_nm, pixel_size_nm, astigmatism_ratio=1.0):
    sx = max(spot_size_nm / pixel_size_nm, 1e-6)
    sy = max(sx * astigmatism_ratio, 1e-6)
    k = max(int(2 * round(3 * max(sx, sy)) + 1), 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sx, sigmaY=sy)


def apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    return np.clip(img.astype(np.float64) * (1.0 - strength * (r ** 2)), 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    if gamma == 1.0:
        return img
    return np.clip(np.power(np.clip(img.astype(np.float64) / 255.0, 0, 1), gamma) * 255.0,
                   0, 255).astype(np.uint8)


def apply_barrel_distortion(img, k):
    if k == 0.0:
        return img
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    factor = 1.0 + k * (nx ** 2 + ny ** 2)
    return cv2.remap(img, (nx * factor) * cx + cx, (ny * factor) * cy + cy,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_charging_streaks(img, streak_prob, intensity, rng):
    if streak_prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape[:2]
    out = img.astype(np.float64)
    n = rng.poisson(max(streak_prob * (h / 100.0), 0))
    for _ in range(n):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * rng.uniform(0.5, 1.0) * 255.0 / 10.0
    return np.clip(out, 0, 255).astype(np.uint8)


def downsample_to_size(img, out_size):
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)


def rotate_about_center(img, angle_deg, gx, gy):
    if angle_deg == 0.0:
        return img, float(gx), float(gy)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    nx = float(M[0, 0] * gx + M[0, 1] * gy + M[0, 2])
    ny = float(M[1, 0] * gx + M[1, 1] * gy + M[1, 2])
    return warped, nx, ny


def apply_raster_drift(img, shear_amplitude_px, jitter_std_px, rng):
    if shear_amplitude_px == 0 and jitter_std_px == 0:
        return img
    h, w = img.shape[:2]
    rows = np.arange(h)
    shear = shear_amplitude_px * (rows / max(h - 1, 1))
    jitter = rng.normal(0, jitter_std_px, size=h) if jitter_std_px > 0 else np.zeros(h)
    shift = (shear + jitter).astype(np.float32)
    map_x = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def add_shot_noise(img, dose, rng):
    img_f = img.astype(np.float64)
    noisy = rng.poisson(np.clip(img_f / 255.0 * dose, 0, None)).astype(np.float64)
    return np.clip(noisy / dose * 255.0, 0, 255).astype(np.uint8)


def add_detector_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, size=img.shape),
                   0, 255).astype(np.uint8)


def add_speckle_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float64) * (1.0 + rng.normal(0, sigma, size=img.shape)),
                   0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img, prob, rng):
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def image_reference(crop, pixel_size_nm, spot_size_nm, dose, rng,
                    detector_noise_sigma=2.0, drift_jitter_px=0.2,
                    astigmatism_ratio=1.0, vignette_strength=0.0, gamma=1.0,
                    barrel_distortion_k=0.0, charging_streak_prob=0.0,
                    charging_streak_intensity=0.0, speckle_sigma=0.0,
                    salt_pepper_prob=0.0):
    img = gaussian_psf_blur(crop, spot_size_nm, pixel_size_nm, astigmatism_ratio)
    img = apply_raster_drift(img, 0.0, drift_jitter_px, rng)
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    return img


def image_search(full_canvas, pixel_size_ref_nm, pixel_size_search_nm, spot_size_nm,
                 dose, rng, shear_amplitude_px=1.5, drift_jitter_px=0.5,
                 detector_noise_sigma=5.0, astigmatism_ratio=1.0, vignette_strength=0.0,
                 gamma=1.0, barrel_distortion_k=0.0, charging_streak_prob=0.0,
                 charging_streak_intensity=0.0, speckle_sigma=0.0, salt_pepper_prob=0.0,
                 search_size_px=1000):
    blurred = gaussian_psf_blur(full_canvas, spot_size_nm, pixel_size_ref_nm, astigmatism_ratio)
    down = blurred if blurred.shape[0] == search_size_px else downsample_to_size(blurred, search_size_px)
    img = apply_raster_drift(down, shear_amplitude_px, drift_jitter_px, rng)
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    return img


# =============================================================================
# One sample
# =============================================================================
@dataclass
class GenerationParams:
    beam_spot_size_nm: float = 5.0
    collapse_threshold_nm: float = 10.0
    dose_reference: float = 2000.0
    dose_search: float = 200.0
    shear_amplitude_px: float = 1.5
    drift_jitter_px: float = 0.5
    detector_noise_sigma_ref: float = 2.0
    detector_noise_sigma_search: float = 5.0
    astigmatism_ratio: float = 1.0
    vignette_strength: float = 0.0
    gamma: float = 1.0
    barrel_distortion_k: float = 0.0
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0
    speckle_sigma: float = 0.0
    salt_pepper_prob: float = 0.0
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0
    boundary_bias: float = 0.35
    linewidth_bias_nm: float = 0.0
    corner_rounding_px: float = 0.0
    scale_ratio: float = 10.0
    rotation_deg: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def fine_canvas_size_px(scale_ratio: float) -> int:
    return max(int(round(REFERENCE_SIZE_PX * float(scale_ratio))), REFERENCE_SIZE_PX)


def _pick_crop_origin(zone_result, params, rng, canvas_size):
    max_offset = canvas_size - REFERENCE_SIZE_PX
    strips = zone_result.get("strip_rects") or []
    if strips and rng.random() < params.boundary_bias:
        sx, sy, sw, sh = strips[int(rng.integers(0, len(strips)))]
        x0 = int(np.clip(sx + sw / 2.0 - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250),
                         0, max_offset))
        y0 = int(np.clip(sy + sh / 2.0 - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250),
                         0, max_offset))
        return x0, y0
    return int(rng.integers(0, max_offset + 1)), int(rng.integers(0, max_offset + 1))


def generate_sample(architecture, rng, params: GenerationParams) -> dict:
    preset = get_preset(architecture)
    zone = generate_zone_canvas(
        fine_canvas_size_px(params.scale_ratio), preset["kind"],
        params.collapse_threshold_nm, rng,
        mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
    )
    canvas = zone["canvas"]
    n = canvas.shape[0]
    x0, y0 = _pick_crop_origin(zone, params, rng, n)
    crop = canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    reference_img = image_reference(
        crop, PIXEL_SIZE_REF_NM, params.beam_spot_size_nm, params.dose_reference, rng,
        detector_noise_sigma=params.detector_noise_sigma_ref,
        drift_jitter_px=params.drift_jitter_px * 0.2,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength * 0.5,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k * 0.3,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
    )
    search_img = image_search(
        canvas, PIXEL_SIZE_REF_NM, n / float(SEARCH_SIZE_PX), params.beam_spot_size_nm,
        params.dose_search, rng,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        search_size_px=SEARCH_SIZE_PX,
    )
    scale = n / float(SEARCH_SIZE_PX)
    box = REFERENCE_SIZE_PX / scale
    cx, cy = x0 / scale + box / 2.0, y0 / scale + box / 2.0
    search_img, cx, cy = rotate_about_center(search_img, params.rotation_deg, cx, cy)
    realized = replace(params, scale_ratio=scale)
    return {
        "reference_img": reference_img, "search_img": search_img,
        "gt_x": cx, "gt_y": cy, "gt_box": (cx - box / 2.0, cy - box / 2.0, box, box),
        "architecture": architecture, "params": realized.as_dict(),
    }


# =============================================================================
# Noise curriculum (the recipe used for the submitted matcher)
# =============================================================================
BUCKET_FRACS = (("easy", 0.40), ("normal", 0.35), ("hard_noise", 0.15), ("hard_geometry", 0.10))


def bucket_counts(n):
    remaining, counts = n, []
    for name, frac in BUCKET_FRACS[:-1]:
        c = int(round(n * frac))
        counts.append((name, c))
        remaining -= c
    counts.append((BUCKET_FRACS[-1][0], remaining))
    return counts


def bucket_for_index(i, n):
    cursor = 0
    for name, count in bucket_counts(n):
        if i < cursor + count:
            return name
        cursor += count
    return BUCKET_FRACS[-1][0]


def sample_params(bucket, rng, base: GenerationParams) -> GenerationParams:
    """Per-pair SEM settings. Speckle/barrel/vignette stay off."""
    p = replace(base)
    p.speckle_sigma = 0.0
    p.barrel_distortion_k = 0.0
    p.vignette_strength = 0.0
    p.dose_reference = float(rng.uniform(1500.0, 3000.0))
    p.detector_noise_sigma_ref = float(rng.uniform(1.0, 3.0))
    p.beam_spot_size_nm = float(rng.uniform(3.0, 8.0))
    p.astigmatism_ratio = float(rng.uniform(0.88, 1.12))
    p.gamma = float(rng.uniform(0.90, 1.10))

    if bucket == "easy":
        p.dose_search = float(rng.uniform(250.0, 400.0))
        p.detector_noise_sigma_search = float(rng.uniform(4.0, 6.0))
        p.drift_jitter_px = float(rng.uniform(0.30, 0.50))
        p.shear_amplitude_px = float(rng.uniform(0.40, 1.00))
        p.charging_streak_prob = 0.0
        p.charging_streak_intensity = 0.0
        p.salt_pepper_prob = 0.0
        p.boundary_bias = 0.55
        p.gamma = float(rng.uniform(0.95, 1.05))
        p.beam_spot_size_nm = float(rng.uniform(3.0, 5.0))
    elif bucket == "normal":
        p.dose_search = float(rng.uniform(120.0, 250.0))
        p.detector_noise_sigma_search = float(rng.uniform(5.0, 8.0))
        p.drift_jitter_px = float(rng.uniform(0.40, 0.80))
        p.shear_amplitude_px = float(rng.uniform(0.80, 1.80))
        p.salt_pepper_prob = 0.0
        p.boundary_bias = 0.35
        if rng.random() < 0.28:
            p.charging_streak_prob = float(rng.uniform(1.5, 4.0))
            p.charging_streak_intensity = float(rng.uniform(1.0, 2.2))
        else:
            p.charging_streak_prob = 0.0
            p.charging_streak_intensity = 0.0
    elif bucket == "hard_noise":
        p.dose_search = float(rng.uniform(80.0, 120.0))
        p.detector_noise_sigma_search = float(rng.uniform(8.0, 10.0))
        p.drift_jitter_px = float(rng.uniform(0.60, 1.00))
        p.shear_amplitude_px = float(rng.uniform(1.20, 2.50))
        p.charging_streak_prob = float(rng.uniform(1.5, 4.0))
        p.charging_streak_intensity = float(rng.uniform(1.0, 2.2))
        p.salt_pepper_prob = float(rng.uniform(0.002, 0.010))
        p.gamma = float(rng.uniform(0.80, 1.20))
        p.boundary_bias = 0.35
    elif bucket == "hard_geometry":
        p.dose_search = float(rng.uniform(120.0, 250.0))
        p.detector_noise_sigma_search = float(rng.uniform(5.0, 8.0))
        p.drift_jitter_px = float(rng.uniform(0.50, 1.00))
        p.shear_amplitude_px = float(rng.uniform(1.50, 2.50))
        p.charging_streak_prob = 0.0
        p.charging_streak_intensity = 0.0
        p.salt_pepper_prob = 0.0
        p.boundary_bias = 0.0
    else:
        raise ValueError(f"Unknown noise bucket '{bucket}'")

    p.scale_ratio = float(rng.uniform(9.0, 11.0))
    p.rotation_deg = float(rng.uniform(-2.0, 2.0))
    return p


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--architectures", nargs="+", default=list(DRAM_PRESET_NAMES),
                   choices=list(PRESETS.keys()))
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="./dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-randomize-noise", action="store_true",
                   help="Disable the default noise mix; use fixed CLI params.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    params = GenerationParams()
    use_mix = not args.no_randomize_noise

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "architecture",
        "noise_bucket",
        "beam_spot_size_nm", "collapse_threshold_nm", "dose_reference",
        "dose_search", "detector_noise_sigma_ref", "detector_noise_sigma_search",
        "shear_amplitude_px", "drift_jitter_px",
        "astigmatism_ratio", "vignette_strength", "gamma", "barrel_distortion_k",
        "charging_streak_prob", "charging_streak_intensity",
        "speckle_sigma", "salt_pepper_prob",
        "linewidth_bias_nm", "corner_rounding_px",
        "mat_size_nm", "strip_width_nm", "boundary_bias",
        "scale_ratio", "rotation_deg", "seed",
    ]
    print(f"generating {args.num_samples} pairs  noise_mix={'ON' if use_mix else 'OFF'}  "
          f"arch={args.architectures}  seed={args.seed}")

    with open(os.path.join(split_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i in range(args.num_samples):
            architecture = args.architectures[int(rng.integers(0, len(args.architectures)))]
            if use_mix:
                bucket = bucket_for_index(i, args.num_samples)
                pair_params = sample_params(bucket, rng, params)
            else:
                bucket = "fixed"
                pair_params = params
            sample = generate_sample(architecture, rng, pair_params)
            ref_path = os.path.join(ref_dir, f"{i:05d}.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")
            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])
            gx0, gy0, gw, gh = sample["gt_box"]
            writer.writerow({
                "id": i, "reference_path": ref_path, "search_path": search_path,
                "gt_x": sample["gt_x"], "gt_y": sample["gt_y"],
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture": architecture, "noise_bucket": bucket,
                **sample["params"], "seed": args.seed,
            })
            print(
                f"[{i + 1}/{args.num_samples}] {bucket:14s} {architecture:12s} "
                f"scale={pair_params.scale_ratio:.2f} rot={pair_params.rotation_deg:+.2f} "
                f"dose_s={pair_params.dose_search:.0f} "
                f"-> gt=({sample['gt_x']:.1f}, {sample['gt_y']:.1f})"
            )
    print(f"Wrote {args.num_samples} samples to {split_dir}")


if __name__ == "__main__":
    main()
