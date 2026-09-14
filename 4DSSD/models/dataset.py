"""
PyTorch Datasets for 2D / 3D / 4D image-stack data.

Supported layouts
------------------
- 2D (DataSet2D): a directory containing one image per file (one file = one
  sample), e.g. a folder of PNG/TIFF/NPY images. This is the common layout
  for image-denoising datasets (BSD68/Set12/DIV2K-style benchmarks).
  Each sample has shape (1, H, W).

- 3D (DataSet3D): a single array (e.g. a .npy volume) holding a stack of 2D
  frames along one axis — a tilt series, focal series, video, or z-stack.
  Which axis indexes frames is not a universal convention (numpy stacks
  default to axis 0, MRC volumes are (Z, Y, X), some pipelines put the frame
  axis last), so it is configurable via `frame_axis` (default 0). The volume
  is internally transposed to (N, H, W) so the rest of the logic is
  axis-agnostic. By default only the center frame is returned, shape
  (1, H, W) — the same shape BlindNet2D expects, so that architecture can be
  reused directly for 3D data. Pass `use_only_center=False` to instead get a
  `window`-sized neighborhood of frames stacked as channels.

- 4D (DataSet4D): 4D-STEM diffraction-image stacks, shape (Nx, Ny, H, W).

Indexing convention used for 4D data:
    a_idx = index % Nx
    b_idx = index // Nx

Depending on `use_only_center`, DataSet4D returns either:
- Center-only image: shape (1, H, W)
- 3x3 neighborhood stack: shape (9, H, W), with boundary positions padded by
  the center frame

Optional random cropping (`image_size`) and the Anscombe variance-stabilizing
transform (`anscombe`) are supported across all three dataset classes.
"""

import os
import glob
import torch
import numpy as np


def anscombe_forward(y):
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 0.0)
    return 2.0 * np.sqrt(y + 3.0 / 8.0)


def _random_crop(out, size):
    """Apply the same random crop across all frames/channels of `out`."""
    H, W = out.shape[-2:]
    if size is None or size >= H or size >= W:
        return out
    h = np.random.randint(0, H - size + 1)
    w = np.random.randint(0, W - size + 1)
    return out[..., h:h + size, w:w + size]


class DataSet2D(torch.utils.data.Dataset):
    """
    A directory of individual 2D images: one file = one sample.

    Supports .npy, .tif/.tiff (via tifffile) and common raster formats
    (.png/.jpg/... via Pillow). Files are discovered recursively under
    `root_dir` and sorted for reproducible ordering.
    """

    DEFAULT_EXTENSIONS = (".npy", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, root_dir, extensions=None, image_size=None, anscombe=False):
        super().__init__()
        self.x = image_size
        self.anscombe = anscombe
        extensions = tuple(e.lower() for e in (extensions or self.DEFAULT_EXTENSIONS))

        self.files = sorted(
            f for f in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
            if os.path.isfile(f) and f.lower().endswith(extensions)
        )
        if not self.files:
            raise FileNotFoundError(
                f"No image files with extensions {extensions} found under {root_dir!r}"
            )

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _load(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            return np.load(path)
        if ext in (".tif", ".tiff"):
            import tifffile
            return tifffile.imread(path)
        from PIL import Image
        # "F" = single-channel 32-bit float; keeps raw intensities as-is.
        return np.array(Image.open(path).convert("F"))

    def __getitem__(self, index):
        img = np.asarray(self._load(self.files[index]), dtype=np.float32)

        # Collapse an incidental trailing/leading channel dim (e.g. RGB) to
        # a single plane; scientific 2D images are expected to be grayscale.
        if img.ndim == 3:
            img = img[..., 0] if img.shape[-1] in (1, 3, 4) else img[0]
        elif img.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {img.shape} from {self.files[index]}")

        out = img[None, :, :]  # (1, H, W)
        out = _random_crop(out, self.x)

        if self.anscombe:
            out = anscombe_forward(out)

        return torch.tensor(out, dtype=torch.float32)


class DataSet3D(torch.utils.data.Dataset):
    """
    A single volume of stacked 2D frames (tilt series / focal series / video
    / z-stack), read from one file.

    `frame_axis` selects which array axis indexes frames (default 0); the
    volume is transposed to (N, H, W) on load so indexing stays consistent
    regardless of how the file was saved.

    - `use_only_center=True` (default): each sample is a single frame,
      shape (1, H, W) — compatible with BlindNet2D.
    - `use_only_center=False`: each sample is a `window`-frame neighborhood
      centered on the index, shape (window, H, W). Frames past the volume
      boundary are clamped to the nearest valid frame (edge padding).
    """

    def __init__(self, filename, frame_axis=0, use_only_center=True, window=3,
                 image_size=None, anscombe=False):
        super().__init__()
        self.x = image_size
        self.anscombe = anscombe
        self.use_only_center = use_only_center
        self.window = window

        vol = self._load(filename)
        if vol.ndim != 3:
            raise ValueError(f"Expected a 3D array, got shape {vol.shape} from {filename}")
        self.img = np.moveaxis(vol, frame_axis, 0)  # -> (N, H, W)

    @staticmethod
    def _load(filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".npy":
            return np.load(filename)
        if ext in (".tif", ".tiff"):
            import tifffile
            return tifffile.imread(filename)
        raise ValueError(f"Unsupported 3D file format {ext!r} for {filename}")

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):
        if self.use_only_center:
            out = self.img[index][None, :, :]  # (1, H, W)
        else:
            n = self.img.shape[0]
            half = self.window // 2
            idxs = [min(max(index + off, 0), n - 1) for off in range(-half, self.window - half)]
            out = self.img[idxs]  # (window, H, W)

        out = _random_crop(out, self.x)

        if self.anscombe:
            out = anscombe_forward(out)

        return torch.tensor(out, dtype=torch.float32)


class DataSet4D(torch.utils.data.Dataset):
    def __init__(self, filename, image_size=None, use_only_center=False, anscombe=False):
        super().__init__()
        self.x = image_size
        self.img = np.load(filename)
        self.use_only_center = use_only_center
        self.anscombe = anscombe

    def __len__(self):
        return self.img.shape[0]*self.img.shape[1]

    def __getitem__(self, index):
        # Convert a flat index into 2D scan coordinates.
        a_idx = index % self.img.shape[0]
        b_idx = index // self.img.shape[0]
        # Center-only mode: return a single image as (1,H,W).
        if self.use_only_center:
            out = self.img[a_idx, b_idx]
            out = out[None, :, :]     # (1, H, W)

        else:
            # Neighborhood mode: gather a 3x3 spatial neighborhood around (a_idx, b_idx).
            # Index order:
            # 0 1 2
            # 3 4 5
            # 6 7 8
            positions = [
                (a_idx-1, b_idx-1),
                (a_idx-1, b_idx),
                (a_idx-1, b_idx+1),
                (a_idx, b_idx-1),
                (a_idx, b_idx),
                (a_idx, b_idx+1),
                (a_idx+1, b_idx-1),
                (a_idx+1, b_idx),
                (a_idx+1, b_idx+1),
            ]

            out = []

            # Use the center frame for out-of-bounds positions (boundary handling).
            center_frame = self.img[a_idx, b_idx]

            # Collect neighbor frames; fall back to center_frame on boundaries.
            for i, j in positions:
                if 0 <= i < self.img.shape[0] and 0 <= j < self.img.shape[1]:
                    out.append(self.img[i, j])
                else:
                    out.append(center_frame)

            out = np.array(out)
            out = _random_crop(out, self.x)

        # Anscombe transform
        if self.anscombe:
            out = anscombe_forward(out)

        return torch.tensor(out, dtype=torch.float32)

def build_dataset(path, data_type=None, frame_axis=0, use_only_center=True,
                   window=3, image_size=None, anscombe=False):
    """
    Convenience factory that picks the right Dataset class for `path`.

    `data_type` may be explicitly "2D", "3D" or "4D"; if omitted it is
    inferred: a directory -> 2D, a 3D array file -> 3D, a 4D array file -> 4D.
    """
    if data_type is None:
        if os.path.isdir(path):
            data_type = "2D"
        else:
            ndim = np.load(path, mmap_mode="r").ndim
            data_type = {2: "2D", 3: "3D", 4: "4D"}.get(ndim)
            if data_type is None:
                raise ValueError(f"Cannot infer data_type from array with ndim={ndim} for {path}")
    data_type = data_type.upper()

    if data_type == "2D":
        return DataSet2D(path, image_size=image_size, anscombe=anscombe)
    if data_type == "3D":
        return DataSet3D(path, frame_axis=frame_axis, use_only_center=use_only_center,
                          window=window, image_size=image_size, anscombe=anscombe)
    if data_type == "4D":
        return DataSet4D(path, use_only_center=use_only_center, image_size=image_size,
                          anscombe=anscombe)
    raise ValueError(f"Unknown data_type {data_type!r}")
