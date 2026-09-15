# Unsupervised Machine Learning for Automated Crystal Orientation Mapping on low-count 4D-STEM data

Author: Kai Kamijo, Motoki Shiga, Shusuke Kanomi, Tomohiro Miyata, Hiroshi Jinnai \
Paper: [10.21203/rs.3.rs-9374144/v1](https://www.researchsquare.com/article/rs-9374144/v1)

## Overview
This repository provides a pipeline to:
1) denoise low-dose 4D-STEM diffraction patterns
2) transform diffraction images to polar coordinates and perform angular Fourier analysis
3) segment structural regions via NMF
4) estimate in-plane crystal orientation from Fourier phase
5) generate orientation maps on the scan grid

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

## Usage

1) **(Optional) Generate synthetic data** — if you don't have real
   4D-STEM data at hand, `data/Data_Synthesizing.py` builds a synthetic
   `(100, 100, 140, 140)` low-count 4D-STEM array from the sample
   diffraction patterns in `data/`:
   ```bash
   cd data
   python Data_Synthesizing.py --output_path synthetic_data.npy
   # add --GT to save the noise-free ground truth instead
   ```

2) **Denoise** — `4DSSD/training.py` trains a self-supervised blind-spot
   denoiser and saves the denoised result. `--model` selects both the data
   layout and the matching network architecture (`2D`/`3D`/`4D`; see the
   argument table below).
   ```bash
   python 4DSSD/training.py --filename data/synthetic_data.npy --model 4D --output_path outputs
   # or run inference only, from an existing checkpoint:
   python 4DSSD/training.py --filename data/synthetic_data.npy --model 4D --checkpoint outputs/<run>/best_model_epoch*.pth
   ```

   `training.py` arguments:

   | Argument | Type / default | Description |
   |---|---|---|
   | `--filename` | str, **required** | Path to the input data: an `.npy` file (3D or 4D array; see `--model`) or, for `--model 2D`, a directory of individual image files. |
   | `--output_path` | str, default `../outputs` | Output root directory. A timestamped subdirectory is created under it for each run, holding `loss.csv`, the best checkpoint (`best_model_epoch*.pth`), and the denoised result. |
   | `--num_epochs` | int, default `100` | Maximum number of training epochs. Training may stop earlier via early stopping (patience 20 epochs, on validation loss). Ignored when `--checkpoint` is given. |
   | `--sample_size` | str, default `all` | Currently parsed but not applied — the full dataset is always used. |
   | `--model` | `{2D,3D,4D}`, default `4D` | Selects both the data layout and the matching network: `2D` — a directory of independent images, or the center frame of a 4D-STEM `.npy` array (`BlindNet2D`); `3D` — a stacked-frame volume, e.g. tilt series / focal series / video / z-stack (`BlindNet3D`); `4D` — a 4D-STEM scan grid `(Nx, Ny, H, W)` (`BlindNet4D`). |
   | `--loss` | str, default `mse` | Training loss: `mse` or `poisson`. |
   | `--anscombe` | flag, default off | Apply the Anscombe variance-stabilizing transform to the data before training/inference (useful for low-count/Poisson-noise data). |
   | `--frame_axis` | int, default `0` | For `--model 3D` only: which axis of the input `.npy` array indexes frames (the array is transposed internally so this axis becomes axis 0). |
   | `--checkpoint` | str, default none | Path to a previously saved `.pth` state dict. If given, training is skipped entirely and the checkpoint is used to run inference/reconstruction only. |

3) **Orientation mapping** — `CCA/CCA.py` provides the downstream analysis
   (high-pass filter → polar transform → angular FFT → NMF-based
   orientation estimation) applied to the denoised 4D-STEM stack. See
   `demo.ipynb` for a full worked example, from a denoised sample dataset
   through to the final orientation maps.

## Acknowledgements

NMF computation in this repository is performed using MALSpy:
- Motoki Shiga, MALSpy: https://github.com/MotokiShiga/malspy

The self-supervised blind-spot denoising networks (`4DSSD/models/blindnet.py`)
build on the blind-spot architecture and multi-frame extension described in:
- S. Laine, T. Karras, J. Lehtinen, T. Aila, "High-Quality Self-Supervised
  Deep Image Denoising," NeurIPS 2019. (`BlindNet2D`/`BlindNet4D` base
  architecture: shift-based blind-spot convolutions with 4-way rotation
  ensembling.)
- D. Y. Sheth, S. Mohan, J. Vincent, R. Manzorro, P. A. Crozier,
  M. M. Khapra, E. P. Simoncelli, C. Fernandez-Granda, "Unsupervised Deep
  Video Denoising," ICCV 2021. (`BlindNet3D`/`BlindNet4D` multi-frame
  extension: independently denoising overlapping frame groups before a
  second-stage fusion network.)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Citation
```
@article{kamijo_2026_4DSSD,
  title = {Unsupervised Machine Learning for Automated Crystal Orientation Mapping on Noisy {4D-STEM} Data},
  author = {Kamijo, Kai and Shiga, Motoki and Kanomi, Shusuke and Miyata, Tomohiro and Jinnai, Hiroshi},
  year  = {2026},
  month = apr,
  journal = {Research Square},
  doi = {10.21203/rs.3.rs-9374144/v1},
  url = {https://www.researchsquare.com/article/rs-9374144/v1},
  note = {Preprint (Version 1)}
}
```