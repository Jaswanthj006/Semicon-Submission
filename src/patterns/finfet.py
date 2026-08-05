"""
FinFET-style structure generator.

Draws parallel vertical fins at fin_pitch/fin_width, perpendicular horizontal
gate stripes at gate_pitch/gate_length, and contact/via marks in the
diffusion regions between gates (checkerboard subset, one per 2 fin/gap
cells). Runs at 1 nm/px, so nanometer preset values map 1:1 to pixel offsets.
"""

import cv2
import numpy as np

from src.structural_defects import maybe_collapse_gap

BACKGROUND = 40
FIN_VAL = 150
GATE_VAL = 170
CONTACT_VAL = 225

POSITION_JITTER_NM = 1.0


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


def generate_finfet_canvas(
    size_px: int,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)

    fin_positions = _line_positions(size_px, preset["fin_pitch_nm"], rng)
    gate_positions = _line_positions(size_px, preset["gate_pitch_nm"], rng)

    col_mask = _line_mask(
        size_px, fin_positions, preset["fin_width_nm"], collapse_threshold_nm, rng
    )
    row_mask = _line_mask(
        size_px, gate_positions, preset["gate_length_nm"], collapse_threshold_nm, rng
    )

    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], FIN_VAL)
    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], GATE_VAL)

    half = max(1, int(round(preset["contact_size_nm"] / 2.0)))
    for i, fin_x in enumerate(fin_positions):
        for j in range(len(gate_positions) - 1):
            if (i + j) % 2 == 0:
                mid_y = (gate_positions[j] + gate_positions[j + 1]) / 2.0
                x, y = int(round(fin_x)), int(round(mid_y))
                cv2.rectangle(
                    canvas,
                    (max(x - half, 0), max(y - half, 0)),
                    (min(x + half, size_px - 1), min(y + half, size_px - 1)),
                    CONTACT_VAL,
                    -1,
                )

    return canvas
