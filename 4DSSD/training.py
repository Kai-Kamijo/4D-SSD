import os
import glob
import numpy as np
from datetime import datetime
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from models import DataSet2D, DataSet3D, DataSet4D, BlindNet2D, BlindNet3D, BlindNet4D

torch.backends.cudnn.benchmark = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# BlindNet3D's forward pass hardcodes three overlapping 3-frame groups out of
# the channel dimension ([0,1,2], [1,2,3], [2,3,4]), so it always needs
# exactly a 5-frame window.
BLINDNET3D_WINDOW = 5

def get_center(x):
    """
    x: (B, C, H, W)  C=1 (2D / 3D center-only), 5 (3D window -> BlindNet3D)
       or 9 (4D neighborhood -> BlindNet4D)
    """
    if x.shape[1] == 9:
        return x[:, 4:5, :, :]
    elif x.shape[1] == 5:
        return x[:, 2:3, :, :]
    else:
        return x

def loss_function(output, truth, loss_type="mse"):
    if(loss_type == "mse"):
        loss = F.mse_loss(output, truth)
    if(loss_type == "poisson"):
        poisson_loss_fn = nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean')
        loss = poisson_loss_fn(output, truth)
    return loss

def anscombe_inverse(z):
    z = np.asarray(z, np.float64)
    x = (z/2.0)**2 - 3.0/8.0
    return np.maximum(x, 0.0)

class EarlyStopping:
    def __init__(self, patience=20, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def build_dataset(model_name, filename, frame_axis, anscombe):
    """
    Construct the Dataset that matches what each architecture expects as
    input. The same construction is used for both training and inference/
    reconstruction, so a checkpoint trained with one call can be applied
    with the other.

    - "2D": a directory of independent images (DataSet2D), or (for backward
      compatibility) the center frame extracted from a 4D-STEM npy stack
      (DataSet4D). Samples are (1, H, W) -> BlindNet2D.
    - "3D": a single stacked volume (tilt series / focal series / video /
      z-stack); `frame_axis` selects which array axis indexes frames.
      Samples are the 5-frame window BlindNet3D expects, (5, H, W).
    - "4D": 4D-STEM scan grid; samples are the 3x3 neighborhood (9, H, W)
      BlindNet4D expects.
    """
    if model_name == "2D":
        if os.path.isdir(filename):
            return DataSet2D(filename, anscombe=anscombe)
        return DataSet4D(filename, use_only_center=True, anscombe=anscombe)
    if model_name == "3D":
        return DataSet3D(filename, frame_axis=frame_axis, use_only_center=False,
                          window=BLINDNET3D_WINDOW, anscombe=anscombe)
    if model_name == "4D":
        return DataSet4D(filename, use_only_center=False, anscombe=anscombe)
    raise ValueError(f"Unknown model_name {model_name!r}")

def build_model(model_name):
    if model_name == "2D":
        return BlindNet2D(n_channels=1, n_output=1).to(device)
    if model_name == "3D":
        return BlindNet3D(n_channels=3, n_output=1).to(device)
    if model_name == "4D":
        return BlindNet4D(n_channels=3, n_output=1).to(device)
    raise ValueError(f"Unknown model_name {model_name!r}")

def train_model(model, dataset, num_epochs, loss_type, output_path):
    """
    Run the training loop; saves loss.csv and the best checkpoint under
    `output_path`. Returns the best (lowest valid-loss) state dict, or None
    if no epoch improved on the initial best_val_loss.
    """
    n_train = int(len(dataset) * 0.8)
    n_valid = len(dataset) - n_train

    train, valid = torch.utils.data.random_split(
        dataset,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(314)
        )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    loss_csv_path = os.path.join(output_path, "loss.csv")
    with open(loss_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss", "lr"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    best_val_loss = float("inf")
    best_model = None
    early_stopper = EarlyStopping(patience=20, delta=1e-6)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch", leave=False)

        for batch in pbar:
            optimizer.zero_grad()
            x = batch.to(device)
            target = get_center(x)

            output = model(x)
            loss = loss_function(output, target, loss_type=loss_type)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3e}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                x = batch.to(device)
                target = get_center(x)
                output = model(x)
                loss = loss_function(output, target, loss_type=loss_type)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{num_epochs} | train={avg_train_loss:.3e} "
              f"| valid={avg_val_loss:.3e} | lr={current_lr:.2e}")

        with open(loss_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, current_lr])

        # -----------------------------
        # Save best model
        # -----------------------------
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()

            for f in glob.glob(os.path.join(output_path, "*.pth")):
                os.remove(f)

            save_path = os.path.join(output_path, f"best_model_epoch{epoch+1:03d}.pth")
            torch.save(best_model, save_path)

        # -----------------------------
        # Early Stopping
        # -----------------------------
        if early_stopper.step(avg_val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_model

def run_inference(model, dataset, output_path, anscombe):
    """
    Denoise every sample of `dataset` with `model` (weights already loaded,
    eval mode not required from the caller) and save the reassembled result
    under `output_path`. Dispatches on the dataset type:

    - DataSet2D: independent images -> one denoised file per input file,
      under <output_path>/denoised/.
    - DataSet3D: frame stack -> reassembled into a (N, H, W) volume,
      <output_path>/denoised.npy.
    - DataSet4D: 4D-STEM scan grid -> reassembled into (Nx, Ny, H, W),
      <output_path>/denoised.npy.
    """
    model.eval()

    if isinstance(dataset, DataSet2D):
        denoised_dir = os.path.join(output_path, "denoised")
        os.makedirs(denoised_dir, exist_ok=True)
        for k in range(len(dataset)):
            x = dataset[k].unsqueeze(0).to(device)
            with torch.no_grad():
                o = model(x).cpu().numpy()
            if anscombe:
                o = anscombe_inverse(o)
            out_name = os.path.splitext(os.path.basename(dataset.files[k]))[0] + "_denoised.npy"
            np.save(os.path.join(denoised_dir, out_name), o.astype(np.float32).squeeze(0))
        print(f"Saved {len(dataset)} denoised images to {denoised_dir}")

    elif isinstance(dataset, DataSet3D):
        denoised = np.zeros(dataset.img.shape, dtype=np.float32)
        for k in range(len(dataset)):
            x = dataset[k].unsqueeze(0).to(device)
            with torch.no_grad():
                o = model(x).cpu().numpy()
            if anscombe:
                o = anscombe_inverse(o)
            denoised[k] = o.squeeze(0).squeeze(0)
        save_path = os.path.join(output_path, "denoised.npy")
        np.save(save_path, denoised)
        print(f"Saved denoised volume to {save_path}")

    else:
        # DataSet4D: 4D-STEM scan grid -> reassemble into (Nx, Ny, H, W)
        denoised = np.zeros(dataset.img.shape, dtype=np.float32)
        length = dataset.img.shape[0]
        for k in range(len(dataset)):
            x = dataset[k].unsqueeze(0).to(device)
            with torch.no_grad():
                a_idx = k % length
                b_idx = k // length
                o = model(x).cpu().numpy()
                if anscombe:
                    o = anscombe_inverse(o)
                denoised[a_idx, b_idx] = o

        save_path = os.path.join(output_path, "denoised.npy")
        np.save(save_path, denoised.astype(np.float32))
        print(f"Saved denoised scan grid to {save_path}")

def main(args):

    filename = args.filename
    output_root = args.output_path
    num_epochs = args.num_epochs
    model_name = args.model
    loss_type = args.loss
    anscombe = args.anscombe
    frame_axis = args.frame_axis
    checkpoint = args.checkpoint

    print(
        f"file name: {filename}, output path: {output_root}, num epochs: {num_epochs}, model: {model_name}"
        )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_root, now)
    os.makedirs(output_path, exist_ok=True)

    model = build_model(model_name)

    if checkpoint:
        # Standalone inference: reuse an already-trained checkpoint instead
        # of training from scratch.
        print(f"Loading checkpoint from {checkpoint}; skipping training, running inference only.")
        best_model = torch.load(checkpoint, map_location=device)
    else:
        dataset = build_dataset(model_name, filename, frame_axis, anscombe)
        best_model = train_model(model, dataset, num_epochs, loss_type, output_path)
        if best_model is None:
            print("Training never improved on the initial loss; no checkpoint to run inference with.")
            return

    model.load_state_dict(best_model)

    inference_dataset = build_dataset(model_name, filename, frame_axis, anscombe)
    run_inference(model, inference_dataset, output_path, anscombe)

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--filename', type=str, required=True,
                         help='Path to the input data: an .npy file (3D or 4D array; see --model) '
                              'or, for --model 2D, a directory of individual image files')
    parser.add_argument('--output_path', type=str, default='../outputs', help='Path to save outputs')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sample_size', type=str, default="all", help='Sample size (currently unused)')
    parser.add_argument('--model', type=str, default="4D", choices=["2D", "3D", "4D"],
                         help='data type / architecture: 2D (image directory or 4D-STEM center frame, '
                              'BlindNet2D), 3D (stacked-frame volume, BlindNet3D), '
                              '4D (4D-STEM scan grid, BlindNet4D)')
    parser.add_argument('--loss', type=str, default="mse", help='loss function')
    # NOTE: `action='store_true'` (not `type=bool`) so the flag is off unless
    # passed -- with `type=bool`, any non-empty string (including "False")
    # parses as True, which is a common argparse footgun.
    parser.add_argument('--anscombe', action='store_true',
                         help='apply the Anscombe variance-stabilizing transform before training/inference')
    parser.add_argument('--frame_axis', type=int, default=0,
                         help='for --model 3D: which array axis of the input volume indexes frames')
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='path to a saved .pth state dict; if given, skip training and only run inference')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
