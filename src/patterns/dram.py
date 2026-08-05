"""
DRAM-style structure generator.

Draws a folded-bitline 6F^2 cell array: horizontal word lines, vertical bit
lines, and a checkerboard of storage-node contacts (one contact per 2 cells,
matching real folded-bitline layouts rather than a naive full grid). Runs at
1 nm/px, so nanometer preset values map 1:1 to pixel offsets.

Rendering is vectorized (1D row/column masks broadcast across the canvas)
so it stays fast even at the 10000x10000 px fine-canvas size used for a
single sample.
"""

import cv2
import numpy as np

from src.structural_defects import maybe_collapse_gap

BACKGROUND = 40
WORD_LINE_VAL = 150
BIT_LINE_VAL = 170
CONTACT_VAL = 225

POSITION_JITTER_NM = 1.5


def _line_positions(size_px: int, pitch_nm: float, rng: np.random.Generator) -> np.ndarray:
    positions = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, POSITION_JITTER_NM)
    return np.array(positions)


def _line_mask(
    size_px: int,
    positions: np.ndarray,
    width_nm: float,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """1D boolean mask marking line + any bridged (collapsed) gaps."""
    mask = np.zeros(size_px, dtype=bool)
    half_w = width_nm / 2.0
    for i, center in enumerate(positions):
        lo = int(round(center - half_w))
        hi = int(round(center + half_w))
        mask[max(lo, 0):min(hi, size_px)] = True

        if i + 1 < len(positions):
            next_center = positions[i + 1]
            gap_nm = (next_center - center) - width_nm
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                bridge_lo = int(round(center + half_w))
                bridge_hi = int(round(next_center - half_w))
                mask[max(bridge_lo, 0):min(bridge_hi, size_px)] = True
    return mask


def generate_dram_canvas(
    size_px: int,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)

    word_positions = _line_positions(size_px, preset["word_line_pitch_nm"], rng)
    bit_positions = _line_positions(size_px, preset["bit_line_pitch_nm"], rng)

    row_mask = _line_mask(
        size_px, word_positions, preset["word_line_width_nm"], collapse_threshold_nm, rng
    )
    col_mask = _line_mask(
        size_px, bit_positions, preset["bit_line_width_nm"], collapse_threshold_nm, rng
    )

    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], WORD_LINE_VAL)
    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], BIT_LINE_VAL)

    radius = max(1, int(round(preset["contact_diameter_nm"] / 2.0)))
    for i, wl in enumerate(word_positions):
        for j, bl in enumerate(bit_positions):
            if (i + j) % 2 == 0:
                cv2.circle(canvas, (int(round(bl)), int(round(wl))), radius, CONTACT_VAL, -1)

    return canvas
