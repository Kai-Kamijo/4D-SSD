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

from models import DataSet, BlindNet2D, BlindNet4D

torch.backends.cudnn.benchmark = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_center(x):
    """
    x: (B, C, H, W)  C=1 or C=9
    """
    if x.shape[1] == 9:
        return x[:, 4:5, :, :]
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

def main(args):

    filename = args.filename
    output_path = args.output_path
    num_epochs = args.num_epochs
    sample_size = args.sample_size
    model_name = args.model
    loss_type = args.loss
    anscombe = args.anscombe

    print(
        f"file name: {filename}, output path: {output_path}, num epochs: {num_epochs}, sample_size: {sample_size}"
        )

    if model_name == "2D":
        dataset = DataSet(filename, use_only_center=True, anscombe=anscombe)
        model = BlindNet2D(n_channels=1, n_output=1).to(device)
    elif model_name == "4D":
        dataset = DataSet(filename, use_only_center=False, anscombe=anscombe)
        model = BlindNet4D(n_channels=3, n_output=1).to(device)

    if sample_size == "all":
        sample_size = int(len(dataset))
    else:
        sample_size = int(sample_size)

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

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_path, now)
    os.makedirs(output_path, exist_ok=True)
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
            x = batch.to(device).squeeze(2)  # (B, 9, H, W)
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
                x = batch.to(device).squeeze(2)
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


    if model_name == "2D":
        ds = DataSet(filename, use_only_center=True, anscombe=anscombe)
    else:
        ds = DataSet(filename, use_only_center=False, anscombe=anscombe)

    denoised = np.zeros(ds.img.shape,dtype=np.float32)
    model.load_state_dict(best_model)
    model.eval()

    length = ds.img.shape[0]
    for k in range(len(ds)):
        x = ds[k]
        x = x.squeeze(1)
        x = x.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            a_idx = k % length
            b_idx = k // length
            o  = model(x)
            o = o.cpu().numpy()
            if anscombe:
                o = anscombe_inverse(o)
            denoised[a_idx, b_idx] = o

    np.save(output_path+"/denoised.npy", denoised.astype(np.float32))

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--filename', type=str, default='../data/raw/STEM_33.npy', help='Input file name')
    parser.add_argument('--output_path', type=str, default='../outputs', help='Path to save outputs')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sample_size', type=str, default="all", help='Sample size')
    parser.add_argument('--model', type=str, default="4D", help='training model')
    parser.add_argument('--loss', type=str, default="mse", help='loss function')
    parser.add_argument('--anscombe', type=bool, default=False, help='VST spase')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
