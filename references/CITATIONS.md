# References

Public sources used to design the synthetic DRAM/SEM generator in
`generate_dataset.py`. These are not fab-proprietary layouts. Presets use
rounded public pitch *ratios* (6F² DRAM: word-line pitch ≈ 2F, bit-line pitch
≈ 3F).

## Semiconductor structures

1. Itoh, K. *VLSI Memory Chip Design*. Springer, 2001.  
   6F² DRAM cell: word-line / bit-line pitch scaling with feature size F.

2. Takai, M. et al. “A 4-F² stacked DRAM cell with poly plug.”  
   IEEE Journal of Solid-State Circuits (DRAM cell area in F²).

3. International Roadmap for Devices and Systems (IRDS) / ITRS.  
   Public FinFET fin pitch, gate pitch, and contacted poly pitch ranges
   (7–45 nm class). Used only as order-of-magnitude presets.

## SEM image formation

4. Reimer, L. *Scanning Electron Microscopy: Physics of Image Formation and
   Microanalysis*. Springer.  
   Probe size, astigmatism, and secondary-electron image formation.

5. Goldstein, J. I. et al. *Scanning Electron Microscopy and X-Ray
   Microanalysis*. Springer.  
   Interaction volume, detector noise, and magnification.

6. Joy, D. C. “Monte Carlo modeling for electron microscopy and microanalysis.”  
   Oxford University Press, 1995.  
   Electron-dose / Poisson (shot) noise.

## Scan artifacts (drift, charging, noise)

7. Jones, L. and Nellist, P. D. “Identifying and correcting scan noise and
   drift in the scanning transmission electron microscope.”  
   *Microscopy and Microanalysis*, 19(4), 2013.  
   Raster shear and frame-to-frame jitter (`shear_amplitude_px`,
   `drift_jitter_px`).

8. Cazaux, J. “Charging in scanning electron microscopy: from inside to
   outside.” *Microscopy and Microanalysis*, 10(6), 2004.  
   Local charging streaks on insulating regions.

9. Timischl, F. “The contrast-to-noise ratio for image quality analysis in
   scanning electron microscopy.” *Scanning*, 2015.  
   Detector (additive Gaussian) noise vs. Poisson shot noise.

## Geometric transforms (search vs. reference)

10. Problem statement: Applied Materials Drift-Sense (SEMICON India 2026).  
    Reference 100× / 1 nm/px, search ~10× / ~10 nm/px, both 1000×1000.
    Generator samples `scale_ratio` in [9, 11] and `rotation_deg` in [−2, 2]
    on the search image only.

11. Lewis, J. P. “Fast Normalized Cross-Correlation.” Industrial Light &
    Magic, 1995.  
    ZNCC used in stage-1 proposals (`train.py` / `localize.py`).

## What is *not* modeled (explicitly off)

Speckle, barrel distortion, and vignetting are disabled in the shipped
recipe. They are documented in Goldstein / Reimer but were left off so
the search image stays a noisy, slightly scaled/rotated version of the
same DRAM layout rather than a heavily warped optical photo.
