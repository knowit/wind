# optuna_tune.py
import os
from datetime import datetime
from typing import Tuple

import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, Subset, TensorDataset

from model import SharedPerLocationSum  # <-- your existing model

# --------------- Data loading (once) ---------------


def load_dataset(data_path: str, device: str = "cpu"):
    blob = torch.load(data_path, map_location="cpu")
    X = blob["X"].float()  # (N, L, V)
    y = blob["y"].float()  # (N,)
    assert X.dim() == 3, f"Expected X to be (N,L,V), got {tuple(X.shape)}"
    assert y.dim() == 1 and y.shape[0] == X.shape[0], "y should be (N,) aligned with X"
    N, L, V = X.shape
    return X, y, N, L, V


def make_split_indices(N: int, val_frac: float, embargo: int = 0):
    """Chronological split with optional embargo gap (rows)."""
    n_val = int(N * val_frac)
    n_train = N - n_val
    train_end = max(0, n_train - embargo)
    train_idx = torch.arange(0, train_end)
    val_idx = torch.arange(n_train, N)
    return train_idx, val_idx


def compute_train_stats(
    X: torch.Tensor, train_idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean/std over (N,L) on train split only."""
    Xt = X[train_idx]  # (n_train, L, V)
    mean = Xt.mean(dim=(0, 1))  # (V,)
    std = Xt.std(dim=(0, 1)).clamp_min(1e-6)  # (V,)
    return mean, std


# --------------- Training for one trial ---------------


def train_one_trial(
    trial: optuna.trial.Trial,
    X: torch.Tensor,
    y: torch.Tensor,
    V: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    device: str,
    max_epochs: int,
) -> float:
    # ---- Suggest hyperparams ----
    width = trial.suggest_categorical("width", [32, 64, 128, 256])
    depth = trial.suggest_int("depth", 1, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.30)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    schedule = "CAWR"  # trial.suggest_categorical("schedule", ["constant", "CAWR"])
    # if schedule == "CAWR":
    T0_epochs = trial.suggest_categorical("T0_epochs", [20, 50, 100, 200])
    early_stopping_patience = min(110, 1.2 * T0_epochs)
    warmup_epochs = 1  # trial.suggest_categorical("warmup_epochs", [1, 3])
    batch_size = 4096

    # ---- Model / Opt / Sched ----
    model = SharedPerLocationSum(
        in_dim=V, width=width, depth=depth, dropout=dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    batches_per_epoch = (train_idx.numel() + batch_size - 1) // batch_size
    warmup_steps = max(1, warmup_epochs * batches_per_epoch)
    T0_steps = max(1, T0_epochs * batches_per_epoch)
    schedulers = [LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)]
    if schedule == "CAWR":
        schedulers.append(
            CosineAnnealingWarmRestarts(opt, T_0=T0_steps, T_mult=1, eta_min=1e-5)
        )
    scheduler = SequentialLR(
        opt,
        schedulers,
        milestones=[warmup_steps],
    )

    amp_enabled = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = float("inf")
    best_state = None
    pat = 0

    # ---- Training loop ----
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        for i, start in enumerate(range(0, perm.numel(), batch_size)):
            idx = perm[start : start + batch_size]
            xb, yb = X[idx], y[idx]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(xb)  # (B,)
                loss = nn.functional.mse_loss(preds, yb)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        # ---- Validation ----
        model.eval()
        val_sse, val_cnt = 0.0, 0
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=torch.bfloat16),
        ):
            for start in range(0, val_idx.numel(), batch_size):
                idx = val_idx[start : start + batch_size]
                xb, yb = X[idx], y[idx]
                preds = model(xb)
                val_sse += nn.functional.mse_loss(preds, yb, reduction="sum").item()
                val_cnt += yb.numel()

        val_mse = val_sse / max(1, val_cnt)

        # ---- Optuna reporting / pruning ----
        trial.report(val_mse, step=epoch)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

        print(f"Epoch {epoch:02d} | val MSE: {val_mse:.3f}")

        # ---- Early stopping (on val MSE) ----
        improved = val_mse < best_val - 1e-4
        if improved:
            best_val = val_mse
            pat = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            pat += 1
            if pat >= early_stopping_patience:
                break

    # ---- Save best checkpoint for this trial (optional) ----
    # ckpt_dir = os.path.join("checkpoints", "optuna")
    # os.makedirs(ckpt_dir, exist_ok=True)
    # if best_state is not None:
    #     torch.save(
    #         {
    #             "model_class": "SharedPerLocationSum",
    #             "model_kwargs": {"in_dim": V, "width": width, "depth": depth, "dropout": dropout},
    #             "state_dict": best_state,
    #             "x_mean": x_mean.cpu(),
    #             "x_std": x_std.cpu(),
    #             "best_val_MSE": best_val,
    #             "trial_number": trial.number,
    #         },
    #         os.path.join(ckpt_dir, f"trial_{trial.number:04d}.pth"),
    #     )

    return float(best_val)


# --------------- Main: create study & run ---------------


def main(
    data_path: str = "data/torch_dataset_all_zones.pt",
    val_frac: float = 0.2,
    embargo_rows: int = 0,
    max_epochs: int = 500,
    n_trials: int = 100,
    study_name: str = "wind_optuna",
    storage: str | None = None,  # e.g. "sqlite:///optuna.db"
    seed: int = 42,
    device: str = "cuda",
):
    # Repro
    torch.manual_seed(seed)

    blob = torch.load(data_path)
    X = blob["X"].float().to(device, non_blocking=True)  # (N, L, V)
    y = blob["y"].float().to(device, non_blocking=True)  # (N,)
    N, L, V = X.shape

    # Split
    n_val = int(N * val_frac)
    n_train = N - n_val
    # perm = torch.randperm(N, device=device)
    sample_idx = torch.arange(N, device=device)
    train_idx, val_idx = sample_idx[:n_train], sample_idx[n_train:]

    x_mean = X[train_idx].mean(dim=(0, 1))  # (V,)
    x_std = X[train_idx].std(dim=(0, 1)).clamp_min(1e-6)  # (V,)
    # in-place broadcasted normalization to avoid extra memory
    X.sub_(x_mean.view(1, 1, -1)).div_(x_std.view(1, 1, -1))

    print(
        f"Device: {device} | N={N}, L={L}, V={V} | train={train_idx.numel()} val={val_idx.numel()}"
    )

    # Study
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        # pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    def objective(trial: optuna.trial.Trial):
        # Different seed per trial for slight SGD diversity (optional)
        torch.manual_seed(seed + trial.number)
        return train_one_trial(
            trial, X, y, V, train_idx, val_idx, x_mean, x_std, device, max_epochs
        )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"Trial #{best.number}")
    print(f"Best val MSE: {best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Optional: save study summary
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("optuna_summaries", exist_ok=True)
    with open(os.path.join("optuna_summaries", f"{study_name}_{ts}.txt"), "w") as f:
        f.write(f"Best trial: {best.number}\nBest val MSE: {best.value:.6f}\nParams:\n")
        for k, v in best.params.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    # Adjust args here or wrap with argparse if you prefer.
    main(
        data_path="data/torch_dataset_all_zones.pt",
        val_frac=0.2,
        embargo_rows=0,  # set to 24 or 48 if you want a gap between train/val
        max_epochs=300,
        n_trials=100,
        study_name="wind_ts_split",
        storage="sqlite:///optuna.db",
        seed=42,
    )
