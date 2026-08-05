"""
Orchestrates one Drift-Sense sample:

  fine canvas (1 nm/px, 10000x10000)
    -> random 1000x1000 crop = Reference Image (native res)
    -> whole-canvas beam blur + 10x downsample + search-specific noise/drift
       = Search Image (1000x1000 @ 10 nm/px)
    -> ground truth = crop location, converted to search-image pixel coords

Physical calibration is fixed by the problem statement: both images are
1000x1000 px; reference is 1 nm/px (1 um FOV), search is 10 nm/px (10 um
FOV). The 10x relationship falls directly out of that pixel-size ratio --
no separate "shrink by 10x" resize step is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from src import sem_imaging
from src.presets import get_preset
from src.patterns.dram import generate_dram_canvas
from src.patterns.finfet import generate_finfet_canvas

REFERENCE_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1
PIXEL_SIZE_SEARCH_NM = 10
SCALE_FACTOR = PIXEL_SIZE_SEARCH_NM // PIXEL_SIZE_REF_NM  # 10
FINE_CANVAS_SIZE_PX = REFERENCE_SIZE_PX * SCALE_FACTOR  # 10000


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

    def as_dict(self) -> dict:
        return asdict(self)


_GENERATORS = {
    "dram": generate_dram_canvas,
    "finfet": generate_finfet_canvas,
}


def generate_fine_canvas(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
    preset_overrides: dict | None = None,
) -> np.ndarray:
    preset = get_preset(architecture)
    if preset_overrides:
        preset.update(preset_overrides)
    generator = _GENERATORS[preset["kind"]]
    return generator(FINE_CANVAS_SIZE_PX, preset, params.collapse_threshold_nm, rng)


def generate_sample(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
    preset_overrides: dict | None = None,
) -> dict:
    fine_canvas = generate_fine_canvas(architecture, rng, params, preset_overrides)

    max_offset = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    x0 = int(rng.integers(0, max_offset + 1))
    y0 = int(rng.integers(0, max_offset + 1))
    crop = fine_canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    reference_img = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_reference,
        rng=rng,
        detector_noise_sigma=params.detector_noise_sigma_ref,
        drift_jitter_px=params.drift_jitter_px * 0.2,
    )

    search_img = sem_imaging.image_search(
        fine_canvas,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_search,
        rng=rng,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
    )

    box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR  # 100
    gt_x0 = x0 / SCALE_FACTOR
    gt_y0 = y0 / SCALE_FACTOR
    gt_cx = gt_x0 + box_w / 2.0
    gt_cy = gt_y0 + box_h / 2.0

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt_x": gt_cx,
        "gt_y": gt_cy,
        "gt_box": (gt_x0, gt_y0, box_w, box_h),
        "architecture": architecture,
        "params": params.as_dict(),
    }
