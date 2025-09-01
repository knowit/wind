# simple_shared_sum_model.py
import os
from datetime import datetime

import cf_xarray as cfxr
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter

from model import LatentSpaceSum, SharedPerLocationSum
from prepare_ensemble_data import get_all_features_for_ensemble_member


def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    batch_size,
    lr,
    warmup_epochs,
    T0_epochs,
    weight_decay,
    epochs,
    phi_width,
    phi_depth,
    latent_dims,
    rho_width,
    rho_depth,
    dropout,
    early_stopping_patience=None,
    seed=42,
    device=None,
):
    model = LatentSpaceSum(
        in_dim=X.shape[-1],
        phi_width=phi_width,
        phi_depth=phi_depth,
        latent_dims=latent_dims,
        rho_width=rho_width,
        rho_depth=rho_depth,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1, dampening=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    batches_per_epoch = (train_idx.numel() + batch_size - 1) // batch_size
    warmup_epochs = warmup_epochs
    warmup_steps = warmup_epochs * batches_per_epoch
    T0_steps = T0_epochs * batches_per_epoch

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingWarmRestarts(
                optimizer, T_0=T0_steps, T_mult=1, eta_min=1e-5
            ),
        ],
        milestones=[warmup_steps],
    )

    amp_enabled = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    # model.compile()

    # Train
    best_val = float("inf")
    best_state = None
    pat_since_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_sse = torch.tensor(0.0, device=device)
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        for i, start in enumerate(range(0, perm.numel(), batch_size)):
            idx = perm[start : start + batch_size]
            xb, yb, mb = X[idx], y[idx], mask[idx]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(xb, mb)
                loss = nn.functional.mse_loss(preds, yb)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_sse += loss.detach().float() * yb.numel()

        avg_train_loss = train_sse.item() / train_idx.numel()

        model.eval()
        val_sse = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, val_idx.numel(), batch_size):
                idx = val_idx[start : start + batch_size]
                xb, yb, mb = X[idx], y[idx], mask[idx]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(xb, mb)
                val_sse += nn.functional.mse_loss(preds, yb, reduction="sum")

        avg_val_loss = val_sse.item() / val_idx.numel()
        val_rmse = torch.sqrt(val_sse / val_idx.numel()).item()

        last_lr = scheduler.get_last_lr()[0]
        # last_lr = lr
        print(
            f"Epoch {epoch:02d} | train MSE: {avg_train_loss:.4f} | val MSE: {avg_val_loss:.4f} | val RMSE: {val_rmse:.3f} | lr: {last_lr:.1e}"
        )

        # early stopping + keep best
        patience = early_stopping_patience or 1.2 * T0_epochs
        if avg_val_loss < best_val - 1e-4:
            best_val = avg_val_loss
            pat_since_improve = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            pat_since_improve += 1
            if pat_since_improve >= patience:
                print(f"Early stop at epoch {epoch} (best val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def get_data(
    data_path, val_cutoff_date=None, bidding_area=None, features=None, device=None
):
    encoded = xr.open_dataset("data/dataset_all_zones.zarr")
    ds = cfxr.decode_compress_to_multi_index(encoded, "forecast_index")
    ds = ds.isel(forecast_index=~np.isnan(ds["y"]))  # Drop samples where y is missing
    if bidding_area is not None:
        ds = ds.sel(bidding_area=bidding_area)
    if features is not None:
        X = torch.from_numpy(
            ds["X"].sel(feature=features).values.astype(np.float32)
        ).to(device)
        mask = torch.from_numpy(
            ds["X"].sel(feature="mask").values.astype(np.float32)
        ).to(device)
    else:
        X = torch.from_numpy(ds["X"].values.astype(np.float32))
        mask = X[..., -1].to(device)
        X = X[..., :-1].to(device)

    y = torch.from_numpy(ds["y"].values.astype(np.float32)).to(device)

    N, L, V = X.shape

    # Split
    if val_cutoff_date is not None:
        val_split_date = np.datetime64(val_cutoff_date)
    else:
        val_split_date = ds.time_ref.max().item() - np.timedelta64(365, "D")

    sample_idx = torch.arange(N, device=device)  # No shuffle for time sertive eval
    train_idx = sample_idx[ds.time_ref.values < val_split_date]
    val_idx = sample_idx[ds.time_ref.values >= val_split_date]
    return X, y, mask, train_idx, val_idx


def train(
    data_path,
    features,
    bidding_area,
    batch_size,
    lr,
    warmup_epochs,
    T0_epochs,
    weight_decay,
    epochs,
    val_cutoff_date,
    normalize_x,
    phi_width,
    phi_depth,
    latent_dims,
    rho_width,
    rho_depth,
    dropout,
    early_stopping_patience=None,
    seed=42,
    device=None,
    use_tensorboard=True,
    experiment_name=None,
    log_dir="runs",
    save_last=True,
    save_dir="checkpoints",
):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    X, y, mask, train_idx, val_idx = get_data(
        data_path, val_cutoff_date, bidding_area, features, device
    )
    N, L, V = X.shape

    local_preds = X[..., -1].sum(dim=-1)
    baseline_rmse = torch.sqrt(nn.functional.mse_loss(local_preds[val_idx], y[val_idx]))
    print(X[train_idx].shape, X[val_idx].shape)

    if normalize_x:
        x_mean = X[train_idx].mean(dim=(0, 1))  # (V,)
        x_std = X[train_idx].std(dim=(0, 1)).clamp_min(1e-6)  # (V,)
        # in-place broadcasted normalization to avoid extra memory
        X.sub_(x_mean.view(1, 1, -1)).div_(x_std.view(1, 1, -1))
    else:
        x_mean = x_std = None

    model = LatentSpaceSum(
        in_dim=V,
        phi_width=phi_width,
        phi_depth=phi_depth,
        latent_dims=latent_dims,
        rho_width=rho_width,
        rho_depth=rho_depth,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1, dampening=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    train_N = train_idx.numel()
    val_N = val_idx.numel()
    batches_per_epoch = (train_N + batch_size - 1) // batch_size
    warmup_epochs = warmup_epochs
    warmup_steps = warmup_epochs * batches_per_epoch
    T0_steps = T0_epochs * batches_per_epoch

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingWarmRestarts(
                optimizer, T_0=T0_steps, T_mult=1, eta_min=1e-5
            ),
            # ConstantLR(optimizer, factor=1.0),
        ],
        milestones=[warmup_steps],
    )

    amp_enabled = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    # model.compile()

    writer = None
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}__{experiment_name}"
        writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

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
                    "L": L,
                    "V": V,
                    "phi_width": phi_width,
                    "phi_depth": phi_depth,
                    "latent_dims": latent_dims,
                    "rho_width": rho_width,
                    "rho_depth": rho_depth,
                    "T0_epochs": T0_epochs,
                    "dropout": dropout,
                }
            ),
        )

    # Train
    best_val = float("inf")
    best_state = None
    pat_since_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_sse = torch.tensor(0.0, device=device)
        perm = train_idx[torch.randperm(train_N, device=device)]
        for i, start in enumerate(range(0, perm.numel(), batch_size)):
            idx = perm[start : start + batch_size]
            xb, yb, mb = X[idx], y[idx], mask[idx]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(xb, mb)
                loss = nn.functional.mse_loss(preds, yb)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_sse += loss.detach().float() * yb.numel()

        avg_train_loss = train_sse.item() / train_N

        model.eval()
        val_sse = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, val_N, batch_size):
                idx = val_idx[start : start + batch_size]
                xb, yb, mb = X[idx], y[idx], mask[idx]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(xb, mb)
                val_sse += nn.functional.mse_loss(preds, yb, reduction="sum")

        avg_val_loss = val_sse.item() / val_N
        val_rmse = torch.sqrt(val_sse / val_N).item()

        last_lr = scheduler.get_last_lr()[0]
        # last_lr = lr
        print(
            f"Epoch {epoch:02d} | train MSE: {avg_train_loss:.4f} | val MSE: {avg_val_loss:.4f} | val RMSE: {val_rmse:.3f} | baseline: {baseline_rmse:.3f} | lr: {last_lr:.1e}"
        )

        # early stopping + keep best
        patience = early_stopping_patience or 1.2 * T0_epochs
        if avg_val_loss < best_val - 1e-4:
            best_val = avg_val_loss
            pat_since_improve = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            pat_since_improve += 1
            if pat_since_improve >= patience:
                print(f"Early stop at epoch {epoch} (best val={best_val:.4f})")
                break

        if writer is not None:
            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", lr_now, epoch)
            writer.add_scalar("Loss/train_MSE", avg_train_loss, epoch)
            writer.add_scalar("Loss/val_MSE", avg_val_loss, epoch)
            writer.add_scalar("Metrics/val_RMSE", val_rmse, epoch)
            writer.flush()

    if best_state is not None:
        model.load_state_dict(best_state)

    if writer is not None:
        writer.close()

    if save_last:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{experiment_name}_last.pth")

        def _cpu_or_none(t):
            return None if t is None else t.detach().cpu()

        ckpt = {
            "model_class": "LatentSpaceSum",
            "model_kwargs": {
                "in_dim": V,
                "phi_width": phi_width,
                "phi_depth": phi_depth,
                "latent_dims": latent_dims,
                "rho_width": rho_width,
                "rho_depth": rho_depth,
                "dropout": dropout,
            },
            "state_dict": model.state_dict(),
            "normalize_x": normalize_x,
            "x_mean": _cpu_or_none(x_mean),
            "x_std": _cpu_or_none(x_std),
            "epochs": epochs,
            "final_val_MSE": avg_val_loss,
            "final_val_RMSE": val_rmse,
            "run_name": run_name,
            "data_path": data_path,
        }
        torch.save(ckpt, save_path)
        print(f"[Saved] Last model -> {save_path}")

    return model, (x_mean, x_std)


if __name__ == "__main__":
    data_path = "data/windpower_ensemble_dataset.parquet"
    em = 0
    features = get_all_features_for_ensemble_member(em)
    for i, bidding_area in enumerate(
        ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
    ):
        model, stats = train(
            data_path="data/torch_dataset_all_zones.pt",
            features=features,
            bidding_area=bidding_area,
            epochs=500,
            val_cutoff_date="2025-01-01",
            normalize_x=True,
            batch_size=4 * 1024,
            lr=1e-3,
            warmup_epochs=5,
            T0_epochs=100,
            phi_width=32,
            phi_depth=2,
            latent_dims=16,
            rho_width=16,
            rho_depth=2,
            dropout=0.1,
            weight_decay=1e-4,
            use_tensorboard=True,
            experiment_name=f"wind_NO{i + 1}",
            log_dir="runs_all_zones",
            save_last=True,
            save_dir="checkpoints",
            device="cuda:0",
            seed=42,
        )
