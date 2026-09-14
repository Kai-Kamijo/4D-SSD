# Unsupervised Machine Learning for Automated Crystal Orientation Mapping on low-count 4D-STEM data

## Author
Kai Kamijo and Motoki Shiga

## Overview
This repository provides a pipeline to:
1) denoise low-dose 4D-STEM diffraction patterns,
2) transform diffraction images to polar coordinates and perform angular Fourier analysis,
3) segment structural regions via NMF,
4) estimate in-plane crystal orientation from Fourier phase, and
5) generate orientation maps on the scan grid.

**Input:** 4D ndarray `(Nx, Ny, H, W)`  
**Output:** region labels / component maps and orientation maps `(Nx, Ny)` (plus intermediate arrays).

## Requirements

- Python 3.10+
- NumPy, SciPy, scikit-image, torch
- MALSpy

## Installation

Install core dependencies:

```bash
pip install -r requirements.txt
```

## Acknowledgements

NMF computation in this repository is performed using MALSpy:
- Motoki Shiga, MALSpy: https://github.com/MotokiShiga/malspy