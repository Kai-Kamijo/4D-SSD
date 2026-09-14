"""
Rotation-Equivariant Component Analysis (CCA) for 4D-STEM diffraction data.

Downstream analysis pipeline applied to a denoised 4D-STEM stack
(Nx, Ny, H, W), corresponding to steps 2 and 4 of the pipeline described in
the top-level README:
1) `highpass_filter`  : per-diffraction-image FFT-domain Gaussian high-pass,
                        to remove low-frequency background.
2) `polar_transform`  : Cartesian (H, W) -> polar (r, theta) coordinates.
3) `fourier_transform` : 1D FFT along the angular axis, giving a (r, m)
                        representation where m is the rotational order
                        (magnitude ~ rotation-invariant, phase ~ in-plane
                        orientation). NMF on the magnitudes (via MALSpy,
                        see demo.ipynb) then yields spatial coefficients
                        `C` (Nx, Ny, n_components) and component spectra
                        `S` (R, K, n_components).
4) `orientation_map`  : robust peak detection on `S` per component, then
                        phase-based in-plane angle estimation from the
                        (r, m) Fourier phase, masked by `C`.

See demo.ipynb for an end-to-end example.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar, rotate
import scipy.ndimage as ndi

def tukey2d(h, w, alpha=0.25):
    """
    Create a 2D Tukey (tapered cosine) window by outer-product of 1D Tukey windows.
    Useful for apodization before FFT to suppress edge discontinuities / ringing.
    """
    def tukey(n, a):
        # 1D Tukey window on [0,1]
        x = np.linspace(0, 1, n)
        w = np.ones(n)

        # Taper length is a/2 on each side
        m = a / 2

        left = x < m
        right = x > 1 - m

        # Cosine taper on the left edge
        w[left] = 0.5 * (1 + np.cos(np.pi * (2 * x[left] / a - 1)))
        # Cosine taper on the right edge
        w[right] = 0.5 * (1 + np.cos(np.pi * (2 * (1 - x[right]) / a - 1)))
        return w

    # Build separable 2D window
    wy, wx = tukey(h, alpha), tukey(w, alpha)
    return np.outer(wy, wx)

def highpass_gaussian_fft(img, sigma=20, pad=0.5, apod_alpha=0.25):
    """
    High-pass filter via frequency-domain Gaussian low-pass subtraction:
      HP = 1 - exp(-r^2 / (2*sigma^2))

    Steps:
      1) Apodize with a 2D Tukey window to reduce FFT edge artifacts.
      2) Reflect-pad the image to further suppress boundary effects.
      3) FFT -> multiply by high-pass mask -> inverse FFT.
      4) Crop back to original size and clip negative values.

    Parameters
    ----------
    img : (H, W) ndarray
        Input image.
    sigma : float
        Gaussian sigma in frequency-radius units of the padded FFT grid (pixels in freq-grid).
        Larger sigma -> passes more low-frequency content (weaker high-pass).
    pad : float
        Padding ratio relative to original size (0.5 -> pad by 50% on each side).
    apod_alpha : float
        Tukey window alpha for apodization.

    Returns
    -------
    imgf : (H, W) ndarray
        High-pass filtered image (non-negative).
    """
    H, W = img.shape

    # Apodize to reduce edge discontinuities in FFT
    win = tukey2d(H, W, apod_alpha)
    imgw = img * win

    # Reflect padding to mitigate boundary artifacts
    pad_h, pad_w = int(H * pad), int(W * pad)
    imgp = np.pad(imgw, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Forward FFT (shifted so DC is at the center)
    F = np.fft.fftshift(np.fft.fft2(imgp))

    # Build radial frequency grid (in pixel units on the FFT grid)
    hh, ww = imgp.shape
    yy, xx = np.indices((hh, ww))
    cy, cx = hh // 2, ww // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Gaussian low-pass in frequency domain, then convert to high-pass
    G = np.exp(-(rr ** 2) / (2 * sigma ** 2))
    HP = 1 - G

    # Apply filter in Fourier space
    Ff = F * HP

    # Inverse FFT and crop back to original shape
    imgf = np.real(np.fft.ifft2(np.fft.ifftshift(Ff)))
    imgf = imgf[pad_h:pad_h + H, pad_w:pad_w + W]

    # Enforce non-negativity (common for intensity images)
    imgf = np.clip(imgf, 0, None)

    return imgf

def rob_z(r, k, min_T=None):
    """
    Robust threshold based on median and MAD (Median Absolute Deviation).

    Threshold:
      T = median(r) + k * (1.4826 * MAD)

    Parameters
    ----------
    r : array-like
        Input values (can include NaNs).
    k : float
        Multiplier controlling strictness of threshold.
    min_T : float or None
        Optional lower bound for the threshold.

    Returns
    -------
    T : float
        Robust threshold.
    """
    r = np.asarray(r)
    med = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - med)) + 1e-12  # avoid zero MAD
    sigma = 1.4826 * mad                         # MAD-to-sigma conversion
    T = med + k * sigma
    if min_T is not None:
        T = max(T, min_T)
    return T

def _require_4d(x, name: str):
    """Validate that the input is a 4D ndarray (Nx, Ny, H, W)."""
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"{name} must be 4D (Nx,Ny,H,W), got shape={x.shape}")
    return x

def _flatten_4d(x4):
    """Flatten (Nx,Ny,H,W) -> (N,H,W) and return (flat, (Nx,Ny))."""
    Nx, Ny, H, W = x4.shape
    return x4.reshape(Nx * Ny, H, W), (Nx, Ny)

def _unflatten_to_4d(x_flat, scan_shape):
    """Unflatten (N,...) -> (Nx,Ny,...)."""
    Nx, Ny = scan_shape
    return x_flat.reshape(Nx, Ny, *x_flat.shape[1:])


def highpass_filter(denoised_4DSTEM):
    """
    Apply high-pass filtering per diffraction image.

    Parameters
    ----------
    denoised_4DSTEM : ndarray (Nx, Ny, H, W)
        Input 4D-STEM stack.

    Returns
    -------
    out_4d : ndarray (Nx, Ny, H, W)
        High-pass filtered stack (always 4D).
    """
    x4 = _require_4d(denoised_4DSTEM, "denoised_4DSTEM")
    flat, scan_shape = _flatten_4d(x4)  # (N,H,W)

    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = highpass_gaussian_fft(flat[i])

    out4 = _unflatten_to_4d(out, scan_shape)
    if out4.ndim != 4:
        raise RuntimeError(f"highpass_filter output is not 4D: shape={out4.shape}")
    return out4


def polar_transform(highpass_data):
    """
    Rotate each image by +11 degrees and apply polar warping.

    Parameters
    ----------
    highpass_data : ndarray (Nx, Ny, H, W)
        High-pass filtered stack.

    Returns
    -------
    out_4d : ndarray (Nx, Ny, H, W)
        Polar-transformed stack (always 4D).
    """
    x4 = _require_4d(highpass_data, "highpass_data")
    flat, scan_shape = _flatten_4d(x4)  # (N,H,W)

    H, W = flat.shape[-2], flat.shape[-1]
    out = np.empty_like(flat)

    for i in range(flat.shape[0]):
        rot_image = rotate(flat[i], 11, resize=False)
        out[i] = warp_polar(
            rot_image,
            scaling="linear",
            center=None,
            output_shape=(H, W),
        )

    out4 = _unflatten_to_4d(out, scan_shape)
    if out4.ndim != 4:
        raise RuntimeError(f"polar_transform output is not 4D: shape={out4.shape}")
    return out4


def fourier_transform(polar_images):
    """
    Apply a real-valued 1D FFT along the angular axis of each polar image:
        F = np.fft.rfft(image.T, axis=1)

    `warp_polar` (used in `polar_transform`) returns each image as
    (angle, radius), i.e. `image` has shape (theta, r). Transposing to
    `image.T`, shape (r, theta), and running rFFT with axis=1 therefore
    applies the FFT along the angular axis theta for each radius r, giving
    a (r, m) representation where m is the rotational (angular-frequency)
    order.

    Parameters
    ----------
    polar_images : ndarray (Nx, Ny, H, W)
        Polar-transformed stack.

    Returns
    -------
    out_4d : ndarray (Nx, Ny, R, K) complex
        rFFT results (always 4D). R,K depend on the rFFT output shape.
    """
    x4 = _require_4d(polar_images, "polar_images")
    flat, scan_shape = _flatten_4d(x4)  # (N,H,W)

    # Compute one FFT to determine output shape and dtype
    F0 = np.fft.rfft(flat[0].T, axis=1)
    out = np.empty((flat.shape[0],) + F0.shape, dtype=F0.dtype)

    out[0] = F0
    for i in range(1, flat.shape[0]):
        out[i] = np.fft.rfft(flat[i].T, axis=1)

    out4 = _unflatten_to_4d(out, scan_shape)  # (Nx,Ny,R,K)
    if out4.ndim != 4:
        raise RuntimeError(f"fourier_transform output is not 4D: shape={out4.shape}")
    return out4


def orientation_map(
    fft_images,
    C,
    S,
    k_r=2,
    k_f=1.5,
    r_start=0,
    k_start=2,
    min_r0=20,
    manual_ranges=None,
    plot=True,
):
    """
    Parameters
    ----------
    fft_images : ndarray, shape (Nx, Ny, R, K)
    C          : ndarray, shape (Nx, Ny, n_components)
    S          : ndarray, shape (R, K, n_components)

    k_r, k_f : float
        Thresholds for automatic peak detection.

    r_start : int
        Physical start index of the sliced radial axis.

    k_start : int
        Physical start index of the sliced angular-frequency axis.
        Example: if original K was 2:8, then k_start=2.

    min_r0 : int
        Minimum radial start index in sliced coordinates for auto detection.

    manual_ranges : dict or None
        Manual specification for selected components.

        Format:
        {
            comp_index: [
                (r0, r1, col0),
                (r0, r1, col0),
                ...
            ],
            ...
        }

        where
        - r0, r1 : sliced-array radial indices (inclusive)
        - col0   : sliced-array column index

        Example:
        {
            1: [(10, 20, 2)],
            2: [(15, 30, 4), (35, 45, 2)]
        }

        If a component index exists in manual_ranges, automatic peak detection
        is skipped for that component and the manual ranges are used.

    plot : bool
        If True, diagnostic plots are shown.

    Returns
    -------
    peak_positions : list
        peak_positions[i] = [[r0_phys, r1_phys, f0_phys], ...]

    angle_maps : list
        angle_maps[i] = [masked_angle_map1, masked_angle_map2, ...]
    """
    fft_images = np.asarray(fft_images)
    C = np.asarray(C)
    S = np.asarray(S)

    if fft_images.ndim != 4:
        raise ValueError(f"fft_images must be 4D, got {fft_images.shape}")
    if C.ndim != 3:
        raise ValueError(f"C must be 3D, got {C.shape}")
    if S.ndim != 3:
        raise ValueError(f"S must be 3D, got {S.shape}")

    Nx, Ny, R, K = fft_images.shape
    Nx2, Ny2, nc1 = C.shape
    R2, K2, nc2 = S.shape

    if (Nx, Ny) != (Nx2, Ny2):
        raise ValueError(
            f"fft_images.shape[:2]={fft_images.shape[:2]} != C.shape[:2]={C.shape[:2]}"
        )
    if (R, K) != (R2, K2):
        raise ValueError(
            f"fft_images.shape[2:]={fft_images.shape[2:]} != S.shape[:2]={S.shape[:2]}"
        )
    if nc1 != nc2:
        raise ValueError(f"C.shape[2]={nc1} != S.shape[2]={nc2}")

    if manual_ranges is None:
        manual_ranges = {}

    n_components = nc1
    fft_flat = fft_images.reshape(Nx * Ny, R, K)
    comp_2d = np.argmax(C, axis=-1)

    peak_positions = []
    angle_maps = []

    if plot:
        plt.figure(figsize=(8, 4 * n_components))

    for i in range(n_components):
        maps_comp = []
        peaks_comp = []
        areas = []

        data = S[:, :, i]
        mask_2d = (comp_2d == i).astype(float)

        r_profile = np.sum(data, axis=1)

        if plot:
            plt.subplot(n_components, 2, 2 * i + 1)
            plt.gca().invert_xaxis()
            plt.plot(r_profile, np.arange(R) + r_start)

            plt.subplot(n_components, 2, 2 * i + 2)
            plt.imshow(
                data,
                origin="lower",
                aspect="auto",
                extent=[k_start, k_start + K, r_start, r_start + R - 1]
            )

        # --------------------------------------------------
        # 1) manual ranges are given for this component
        # --------------------------------------------------
        if i in manual_ranges:
            for spec in manual_ranges[i]:
                if len(spec) != 3:
                    raise ValueError(
                        f"manual_ranges[{i}] entries must be (r0, r1, col0), got {spec}"
                    )

                r0, r1, col0 = spec
                r0 = int(r0)
                r1 = int(r1)
                col0 = int(col0)

                if not (0 <= r0 <= r1 < R):
                    raise ValueError(
                        f"Invalid radial range for component {i}: {(r0, r1)} with R={R}"
                    )
                if not (0 <= col0 < K):
                    raise ValueError(
                        f"Invalid column index for component {i}: col0={col0}, K={K}"
                    )

                f0 = col0 + k_start
                band = data[r0:r1 + 1, col0]
                score = np.sum(np.abs(band))
                areas.append(score)

                if plot:
                    y0 = ((r0 + r1) / 2) + r_start
                    plt.subplot(n_components, 2, 2 * i + 2)
                    plt.scatter(f0 + 0.5, y0, c="cyan")

                alphas = []
                for ft in fft_flat:
                    Fk = ft[r0:r1 + 1, col0]
                    weights = np.abs(Fk) + 1e-8

                    if np.sum(weights) == 0:
                        alphas.append(np.nan)
                        continue

                    angles = np.angle(Fk) / f0
                    z = np.sum(weights * np.exp(1j * angles)) / np.sum(weights)
                    alpha = np.angle(z)

                    period = 2 * np.pi / f0
                    alpha = (alpha + period / 2) % period - period / 2
                    alphas.append(alpha)

                alphas = np.asarray(alphas).reshape(Nx, Ny)
                alphas = (alphas + np.pi / 2) * 180 / np.pi
                alpha_map = np.ma.array(alphas, mask=(mask_2d == 0))

                r0_phys = r0 + r_start
                r1_phys = r1 + r_start
                peaks_comp.append([r0_phys, r1_phys, f0])
                maps_comp.append(alpha_map)

        # --------------------------------------------------
        # 2) otherwise use automatic detection
        # --------------------------------------------------
        else:
            T = rob_z(r_profile, k=k_r)
            mask_r = r_profile > T
            labels, num = ndi.label(mask_r)

            if plot:
                plt.subplot(n_components, 2, 2 * i + 1)
                plt.plot(mask_r * np.max(r_profile), np.arange(R) + r_start)

            for j in range(1, num + 1):
                r_idx = np.where(labels == j)[0]
                if r_idx.size == 0:
                    continue
                if r_idx[0] < min_r0:
                    continue

                a_full = np.sum(data[r_idx[0]:r_idx[-1] + 1, :], axis=0)

                # sliced columns 0,2,4,... correspond to physical periods 2,4,6,...
                even_cols = np.arange(0, a_full.shape[0], 2)
                if even_cols.size == 0:
                    continue

                a = a_full[even_cols]
                T_angle = rob_z(a, k=k_f)

                ncheck = min(5, len(a))
                idx = np.where(a[:ncheck] > T_angle)[0]
                idx0 = 0 if idx.size == 0 else idx[0]

                col0 = even_cols[idx0]
                f0 = col0 + k_start
                score = a[idx0]
                areas.append(score)

                if plot:
                    y0 = ((r_idx[0] + r_idx[-1]) / 2) + r_start
                    plt.subplot(n_components, 2, 2 * i + 2)
                    plt.scatter(f0 + 0.5, y0, c="red")

                alphas = []
                for ft in fft_flat:
                    Fk = ft[r_idx[0]:r_idx[-1] + 1, col0]
                    weights = np.abs(Fk) + 1e-8

                    if np.sum(weights) == 0:
                        alphas.append(np.nan)
                        continue

                    angles = np.angle(Fk) / f0
                    z = np.sum(weights * np.exp(1j * angles)) / np.sum(weights)
                    alpha = np.angle(z)

                    period = 2 * np.pi / f0
                    alpha = (alpha + period / 2) % period - period / 2
                    alphas.append(alpha)

                alphas = np.asarray(alphas).reshape(Nx, Ny)
                alphas = (alphas + np.pi / 2) * 180 / np.pi
                alpha_map = np.ma.array(alphas, mask=(mask_2d == 0))

                r0_phys = r_idx[0] + r_start
                r1_phys = r_idx[-1] + r_start
                peaks_comp.append([r0_phys, r1_phys, f0])
                maps_comp.append(alpha_map)

        if len(maps_comp) == 0:
            peak_positions.append([])
            angle_maps.append([])
        else:
            order = np.argsort(areas)[::-1]
            peak_positions.append([peaks_comp[k] for k in order])
            angle_maps.append([maps_comp[k] for k in order])

    return peak_positions, angle_maps