# simple_shared_sum_model.py
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter


# ----------------------------
# Dataset
# ----------------------------
class WindAreaDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        normalize=True,
        normalize_y=True,
    ):
        """
        X: (N, L, V) float tensor
        y: (N,) float tensor
        """
        self.X = X.float()
        self.y = y.float()
        self.normalize = normalize
        self.normalize_y = normalize_y

        # Stats (computed on the *training* partition and passed in for val)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (L, V)
        y = self.y[idx]  # scalar

        if self.normalize and self.x_mean is not None and self.x_std is not None:
            # normalize per variable across time+locations using train stats
            x = (x - self.x_mean) / (self.x_std + 1e-6)

        if self.normalize_y and self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-6)

        return x, y


# ----------------------------
# Model: shared per-location MLP + sum
# ----------------------------
class SharedPerLocationSum(nn.Module):
    def __init__(self, in_dim=7, hidden=(64, 32), dropout=0.0, return_locals=False):
        super().__init__()
        # h1, h2 = hidden
        h1, h2, h3 = hidden
        self.return_locals = return_locals
        self.phi = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),  # scalar contribution per location
            nn.Softplus(),
        )

    def forward(self, x):
        """
        x: (B, L, V)
        returns:
          y_hat: (B,) predicted total
          (optionally) loc_contribs: (B, L)
        """
        B, L, V = x.shape
        z = x.view(B * L, V)  # flatten locations
        contribs = self.phi(z).view(B, L)  # (B, L)
        y_hat = contribs.sum(dim=1)  # (B,)
        if self.return_locals:
            return y_hat, contribs
        return y_hat


# ----------------------------
# Training utility
# ----------------------------
def train_model(
    data_path,
    batch_size=512,
    lr=1e-3,
    early_stopping_patience=100,
    weight_decay=1e-4,
    epochs=10,
    val_frac=0.1,
    normalize_x=True,
    normalize_y=True,
    hidden=(64, 32),
    dropout=0.0,
    seed=42,
    device=None,
    use_tensorboard=True,
    experiment_name=None,
    log_dir="runs",
    save_last=True,
    save_dir="checkpoints",
):
    torch.manual_seed(seed)

    # Load data
    blob = torch.load(data_path)
    X = blob["X"].float()  # (N, L, V)
    y = blob["y"].float()  # (N,)

    assert X.dim() == 3, f"Expected X to be (N,L,V), got {tuple(X.shape)}"
    assert y.dim() == 1 and y.shape[0] == X.shape[0], "y should be (N,) aligned with X"

    N, L, V = X.shape
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Split
    n_val = int(N * val_frac)
    n_train = N - n_val
    base_ds = torch.utils.data.TensorDataset(X, y)
    train_subset, val_subset = random_split(
        base_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    # Train normalization stats
    X_train = base_ds.tensors[0][train_subset.indices]
    y_train = base_ds.tensors[1][train_subset.indices]

    if normalize_x:
        x_mean = X_train.mean(dim=(0, 1)).view(1, V)
        x_std = (X_train.std(dim=(0, 1)) + 1e-6).view(1, V)
    else:
        x_mean = x_std = None

    if normalize_y:
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-6
    else:
        y_mean = y_std = None

    def wrap_subset(subset):
        X_sub = base_ds.tensors[0][subset.indices]
        y_sub = base_ds.tensors[1][subset.indices]
        return WindAreaDataset(
            X_sub,
            y_sub,
            x_mean,
            x_std,
            y_mean,
            y_std,
            normalize=normalize_x,
            normalize_y=normalize_y,
        )

    train_ds = wrap_subset(train_subset)
    val_ds = wrap_subset(val_subset)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Model, loss, opt
    model = SharedPerLocationSum(in_dim=V, hidden=hidden, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-4
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=1e-5
    )

    def denorm_y(t):
        return t * y_std.to(t.device) + y_mean.to(t.device) if normalize_y else t

    # >>> NEW: SummaryWriter
    writer = None
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}__{experiment_name}"
        writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

        # Optional: add a graph using one batch (ignore failures if shapes differ)
        try:
            xb0, _ = next(iter(train_loader))
            writer.add_graph(model, xb0.to(device))
        except Exception:
            pass

        # Hyperparams (logged at end with metrics too)
        writer.add_text(
            "config",
            str(
                {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "normalize_x": normalize_x,
                    "normalize_y": normalize_y,
                    "L": L,
                    "V": V,
                    "model_hidden": hidden,
                }
            ),
        )

    # Train
    best_val = float("inf")
    best_state = None
    pat_since_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            optimizer.step()
            scheduler.step(epoch - 1 + i / len(train_loader))

            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        # scheduler.step(avg_val_loss)
        last_lr = scheduler.get_last_lr()[0]

        # RMSE in original units
        with torch.no_grad():
            rmse_sum = 0.0
            n_items = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                rmse_sum += torch.sqrt(
                    ((denorm_y(preds) - denorm_y(yb)) ** 2).mean()
                ).item() * xb.size(0)
                n_items += xb.size(0)
            val_rmse = rmse_sum / max(n_items, 1)
        print(
            f"Epoch {epoch:02d} | train MSE: {avg_train_loss:.4f} | val MSE: {avg_val_loss:.4f} | val RMSE: {val_rmse:.3f} | lr: {last_lr:.1e}"
        )

        # early stopping + keep best
        if avg_val_loss < best_val - 1e-4:
            best_val = avg_val_loss
            pat_since_improve = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            pat_since_improve += 1
            if pat_since_improve >= early_stopping_patience:
                print(f"Early stop at epoch {epoch} (best val={best_val:.4f})")
                break

        if writer is not None:
            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", lr_now, epoch)
            writer.add_scalar("Loss/train_MSE", avg_train_loss, epoch)
            writer.add_scalar("Loss/val_MSE", avg_val_loss, epoch)
            writer.add_scalar("Metrics/val_RMSE_orig_units", val_rmse, epoch)
            writer.flush()

    if best_state is not None:
        model.load_state_dict(best_state)

    if writer is not None:
        hparams = {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "normalize_x": normalize_x,
            "normalize_y": normalize_y,
            "L": L,
            "V": V,
            "hidden1": hidden[0],
            "hidden2": hidden[1],
            "dropout": dropout,
        }
        final_metrics = {"hparam/val_MSE": avg_val_loss, "hparam/val_RMSE": val_rmse}
        # add_hparams writes to a child run directory
        writer.add_hparams(hparams, final_metrics)
        writer.close()

    if save_last:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{experiment_name}_last.pth")

        # Move small stats to CPU for portability
        def _cpu_or_none(t):
            return None if t is None else t.detach().cpu()

        ckpt = {
            "model_class": "SharedPerLocationSum",
            "model_kwargs": {"in_dim": V, "hidden": hidden, "dropout": dropout},
            "state_dict": model.state_dict(),
            "normalize_x": normalize_x,
            "normalize_y": normalize_y,
            "x_mean": _cpu_or_none(x_mean),
            "x_std": _cpu_or_none(x_std),
            "y_mean": _cpu_or_none(y_mean),
            "y_std": _cpu_or_none(y_std),
            "epochs": epochs,
            "final_val_MSE": avg_val_loss,
            "final_val_RMSE": val_rmse,
            "run_name": run_name,
            "data_path": data_path,
        }
        torch.save(ckpt, save_path)
        print(f"[Saved] Last model -> {save_path}")

    return model, (x_mean, x_std, y_mean, y_std)


if __name__ == "__main__":
    # Tip: pip install tensorboard
    model, stats = train_model(
        data_path="data/torch_dataset.pt",
        epochs=1000,
        val_frac=0.2,
        batch_size=512,
        lr=1e-2,  # 2e-3,
        early_stopping_patience=120,
        normalize_y=False,
        weight_decay=0,
        hidden=(64, 64, 64),
        dropout=0.05,
        use_tensorboard=True,
        experiment_name="wind_softplus_cawr_const_T",
        log_dir="runs",
        save_last=True,
        save_dir="checkpoints",
    )
