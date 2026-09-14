"""
PyTorch Dataset for 4D-STEM diffraction-image stacks.

The input file is expected to be a NumPy array saved as .npy with shape:
    (Nx, Ny, H, W)

Indexing convention used throughout this repo:
    a_idx = index % Nx
    b_idx = index // Nx

Depending on `use_only_center`, the dataset returns either:
- Center-only image: shape (1, H, W)
- 3x3 neighborhood stack: shape (9, H, W), with boundary positions padded by the center frame

Optional random cropping (`image_size`) and simple geometric augmentations (`transforms`)
are supported.
"""

import torch
import numpy as np

def anscombe_forward(y):
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 0.0)
    return 2.0 * np.sqrt(y + 3.0 / 8.0)

class DataSet(torch.utils.data.Dataset): 
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
            H, W = out.shape[-2:]
            
            # Optional random crop (applied consistently across the 9-frame stack).
            if self.x is not None:
                h = np.random.randint(0, H-self.x)
                w = np.random.randint(0, W-self.x)
                out = out[:, h:h+self.x, w:w+self.x]
            
        # Anscombe transform
        if self.anscombe:
            out = anscombe_forward(out)

        return torch.tensor(out, dtype=torch.float32)
