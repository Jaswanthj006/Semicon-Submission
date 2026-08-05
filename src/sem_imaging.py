"""
SEM acquisition artifacts -- applied per-image, since the reference and
search captures happen under different conditions (careful/slow vs.
fast/wide-area). This is deliberately kept separate from
structural_defects.py, which models a property of the physical device
rather than of how it was imaged.

There is a single physical beam (`beam_spot_size_nm`), applied identically
to both images as a Gaussian PSF blur *before* any downsampling. The search
image's extra softness on dense structures comes naturally from the 10x
area-average downsample on top of that shared blur -- not from a separate
"search-only blur" fudge factor.
"""

import cv2
import numpy as np


def gaussian_psf_blur(img: np.ndarray, spot_size_nm: float, pixel_size_nm: float) -> np.ndarray:
    sigma_px = max(spot_size_nm / pixel_size_nm, 1e-6)
    k = int(2 * round(3 * sigma_px) + 1)
    k = max(k, 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma_px)


def downsample_area_average(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def apply_raster_drift(
    img: np.ndarray,
    shear_amplitude_px: float,
    jitter_std_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Progressive row-to-row shear (drift accumulating over scan time) plus
    per-row jitter (vibration), mimicking real raster-scan drift artifacts.
    """
    if shear_amplitude_px == 0 and jitter_std_px == 0:
        return img
    h, w = img.shape
    rows = np.arange(h)
    shear = shear_amplitude_px * (rows / max(h - 1, 1))
    jitter = rng.normal(0, jitter_std_px, size=h) if jitter_std_px > 0 else np.zeros(h)
    row_shift = (shear + jitter).astype(np.float32)

    map_x = (np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None])
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_shot_noise(img: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise. `dose` is a proxy for electron count/dwell time --
    higher dose (slower/careful scan) means less relative noise.
    """
    img_f = img.astype(np.float64)
    counts = np.clip(img_f / 255.0 * dose, 0, None)
    noisy_counts = rng.poisson(counts).astype(np.float64)
    noisy = noisy_counts / dose * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_detector_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img
    noisy = img.astype(np.float64) + rng.normal(0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def image_reference(
    crop: np.ndarray,
    pixel_size_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    detector_noise_sigma: float = 2.0,
    drift_jitter_px: float = 0.2,
) -> np.ndarray:
    img = gaussian_psf_blur(crop, spot_size_nm, pixel_size_nm)
    img = apply_raster_drift(img, shear_amplitude_px=0.0, jitter_std_px=drift_jitter_px, rng=rng)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    return img


def image_search(
    full_canvas: np.ndarray,
    pixel_size_ref_nm: float,
    pixel_size_search_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    shear_amplitude_px: float = 1.5,
    drift_jitter_px: float = 0.5,
    detector_noise_sigma: float = 5.0,
) -> np.ndarray:
    factor = int(round(pixel_size_search_nm / pixel_size_ref_nm))
    blurred = gaussian_psf_blur(full_canvas, spot_size_nm, pixel_size_ref_nm)
    downsampled = downsample_area_average(blurred, factor)
    drifted = apply_raster_drift(downsampled, shear_amplitude_px, drift_jitter_px, rng)
    noisy = add_shot_noise(drifted, dose, rng)
    noisy = add_detector_noise(noisy, detector_noise_sigma, rng)
    return noisy
