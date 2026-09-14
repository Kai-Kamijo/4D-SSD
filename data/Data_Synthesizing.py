import numpy as np
from PIL import Image
import itertools
from skimage.transform import resize, rotate
import argparse


def load_image(filepath):
    """Load a diffraction-pattern PNG and preprocess it into a normalized 140*140 float array.

    Steps
    -----
    1) Load image and convert to grayscale.
    2) Crop a fixed ROI to match the reciprocal-space region of interest.
    3) Resize to (140, 140) with anti-aliasing.
    4) Subtract the minimum value to remove a constant background offset.

    Parameters
    ----------
    filepath : str
        Path to the PNG file.

    Returns
    -------
    sub_back : ndarray (140, 140)
        Preprocessed image (non-negative).
    """
    img = Image.open(filepath).convert("L")
    cut_array = np.array(img)[127:407,156:436]
    resized = resize(cut_array, (140, 140), anti_aliasing=True)
    sub_back = resized - np.min(resized)
    return sub_back

def ring_mask(shape, r_out, center=None):
    """Create a circular mask (disk) used to suppress the central beam region.

    Parameters
    ----------
    shape : tuple
        Image shape as (H, W).
    r_out : float
        Outer radius of the disk (in pixels).
    center : tuple or None
        Center coordinate as (cy, cx). If None, the image center is used.

    Returns
    -------
    mask : ndarray of bool, shape (H, W)
        Boolean mask where True indicates pixels inside the radius (disk region).
    """
    H, W = shape
    if center is None:
        cx = H//2
        cy = W//2

    y, x = np.ogrid[:H, :W]

    r2 = (x - cx)**2 + (y - cy)**2
    mask = (r2 <= r_out**2)

    return mask
def add_gaussian_beam(img, sigma, amplitude=1.0, center=None):
    """Add a 2D Gaussian peak (synthetic central beam) to an intensity map.

    Parameters
    ----------
    img : ndarray (H, W)
        Input image (intensity map).
    sigma : float
        Standard deviation of the Gaussian in pixels.
    amplitude : float
        Peak intensity of the Gaussian.
    center : tuple or None
        Gaussian center coordinate as (cx, cy). If None, uses the image center.

    Returns
    -------
    out : ndarray (H, W)
        New image with the Gaussian beam added.
    """
    H, W = img.shape

    if center is None:
        cx, cy = W // 2, H // 2
    else:
        cx, cy = center

    # Coordinate grids
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]

    # 2D Gaussian peak
    gauss = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    return img + gauss

def main(args):
    """Generate a synthetic 4D-STEM dataset from four reciprocal-space PNG patterns.

    This script creates a (Nx, Ny, H, W) array (here Nx=Ny=100, H=W=140) where each
    scan position is assigned a diffraction pattern according to its quadrant/region.
    Patterns are rotated/mixed and Poisson noise is applied to emulate counting statistics.

    Parameters
    ----------
    args.scale : float
        Controls the Poisson intensity (higher -> higher counts before normalization).
    args.output_path : str
        Path to save the output .npy file via np.save.
    """
    GT = args.GT
    scale = args.scale
    output_path = args.output_path

    # Load four base reciprocal-space patterns and apply a constant gain factor
    img_001 = load_image("iPS_001.png")*6
    img_111 = load_image("iPS_111.png")*6
    img_221 = load_image("iPS_221.png")*6
    img_251 = load_image("iPS_251.png")*6

    # Build a small central disk mask to suppress the direct-beam region
    mask = ring_mask(img_001.shape, 8)

    # Add a synthetic Gaussian central beam, then remove the masked disk region
    # (~mask is bitwise-not; for boolean arrays this inverts True/False)
    img_001 = add_gaussian_beam(img_001, 12, 2.5)*~mask+0.4
    img_111 = add_gaussian_beam(img_111, 12, 2.5)*~mask+0.4
    img_221 = add_gaussian_beam(img_221, 12, 2.5)*~mask+0.4
    img_251 = add_gaussian_beam(img_251, 12, 2.5)*~mask+0.4

    # Allocate synthetic 4D-STEM array: (Nx, Ny, H, W)
    arr = np.zeros((100, 100, 140, 140))

    # Populate each scan position with a region-specific pattern
    for x, y in itertools.product(range(100),range(100)):
        count = 0

        # Region 1: upper-left quadrant -> pattern 001 with Poisson noise
        if x <= 50 and y <= 50:
            arr[x,y,:,:] += img_001*scale
            count += 1

        # Region 2: upper-right quadrant -> pattern 111, rotated by an angle that depends on x
        if x > 50 and y <= 50:
            arr[x,y,:,:] += rotate(img_111*scale, angle = 4*x, mode="edge")
            count += 1

        # Region 3: bottom half -> linear mixture of patterns 221 and 251 as a function of x
        # and a fixed rotation applied to the 221 pattern
        if y > 50:
            arr[x,y,:,:] += (rotate(img_221*scale, angle = 60, mode="edge")*(1 - x/99) + (img_251*scale)*x/99)
            count += 1
        # Normalize by the number of active conditions and by scale to return to intensity-like units
        arr[x,y,:,:] = arr[x,y,:,:]/count/scale
    if GT == False:
        arr = np.random.poisson(lam=arr).astype(np.float32)
    # Save the synthetic dataset to disk
    np.save(output_path, arr)

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Data Synthesizing configuration")
    parser.add_argument('--GT', type=bool, default=False, help='making ground-truth')
    parser.add_argument('--scale', type=float, default=2, help='Poisson intensity scaling factor')
    parser.add_argument('--output_path', type=str, default='synthetic_data.npy', help='Path to save outputs')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
