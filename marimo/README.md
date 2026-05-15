# slmsuite Marimo Learning Site

This directory contains the Marimo apps used as the interactive learning surface for
`slmsuite`.

The root app, `index.py`, is the guided entry point. It synthesizes the local theory
primer at `/Users/aadarwal/Downloads/slm_theory_primer.tex` with the example
notebooks, then routes readers into the concrete demos.

## Apps

| App | Route | Role |
| --- | --- | --- |
| `index.py` | `/` | Theory-guided site index and notebook menu. |
| `computational_holography.py` | `/computational` | Fourier-pair basics, GS/WGS, diffraction sampling, spots, and image targets. |
| `structured_light.py` | `/structured` | Analytic phase functions: blazes, lenses, modes, imprints, and Zernikes. |
| `experimental_holography.py` | `/experimental` | Simulated camera-SLM calibration and feedback workflows. |
| `wavefront_calibration.py` | `/wavefront` | Aberrated sources, superpixel wavefront correction, and recalibration. |
| `zernike_holography.py` | `/zernike` | Zernike indexing, modal sums, and compressed spot holography. |
| `multipoint_calibration.py` | `/multipoint` | Distributed wavefront calibration and Zernike smoothing. |
| `multiplane_holography.py` | `/multiplane` | Shared-phase multiplane image and spot objectives. |

## Learning Path

The intended order is:

1. Computational Holography
2. Structured Light
3. Experimental Holography
4. Wavefront Calibration
5. Zernike Holography
6. Multipoint Calibration
7. Multiplane Holography

The index app exposes the same order with theory notes, active equations, and
theory-to-code translations.
