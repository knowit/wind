# simple_shared_sum_model.py
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter

from model import SharedPerLocationSum


def train_model(
    data_path,
    batch_size,
    lr,
    warmup_epochs,
    T0_epochs,
    early_stopping_patience,
    weight_decay,
    epochs,
    val_frac,
    normalize_x,
    width,
    depth,
    dropout,
    seed,
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

    # Load data
    blob = torch.load(data_path)
    X = blob["X"].float().to(device, non_blocking=True)  # (N, L, V)
    y = blob["y"].float().to(device, non_blocking=True)  # (N,)

    assert X.dim() == 3, f"Expected X to be (N,L,V), got {tuple(X.shape)}"
    assert y.dim() == 1 and y.shape[0] == X.shape[0], "y should be (N,) aligned with X"

    N, L, V = X.shape

    # Split
    n_val = int(N * val_frac)
    n_train = N - n_val
    # perm = torch.randperm(N, device=device)
    sample_idx = torch.arange(N, device=device)  # No shuffle for time sertive eval
    train_idx, val_idx = sample_idx[:n_train], sample_idx[n_train:]

    if normalize_x:
        x_mean = X[train_idx].mean(dim=(0, 1))  # (V,)
        x_std = X[train_idx].std(dim=(0, 1)).clamp_min(1e-6)  # (V,)
        # in-place broadcasted normalization to avoid extra memory
        X.sub_(x_mean.view(1, 1, -1)).div_(x_std.view(1, 1, -1))
    else:
        x_mean = x_std = None

    # Model, loss, opt
    model = SharedPerLocationSum(
        in_dim=V, width=width, depth=depth, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    batches_per_epoch = (train_idx.numel() + batch_size - 1) // batch_size
    warmup_epochs = warmup_epochs
    warmup_steps = warmup_epochs * batches_per_epoch
    T0_steps = T0_epochs * batches_per_epoch
    sched_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    sched_main = CosineAnnealingWarmRestarts(
        optimizer, T_0=T0_steps, T_mult=1, eta_min=1e-5
    )
    # scheduler = SequentialLR(
    #     optimizer, schedulers=[sched_warmup, sched_main], milestones=[warmup_steps]
    # )
    scheduler = sched_main

    model.compile()

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
                    "width": width,
                    "depth": depth,
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

        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]

        for i, start in enumerate(range(0, perm.numel(), batch_size)):
            idx = perm[start : start + batch_size]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(xb).squeeze(-1)  # forward in mixed precision

            diff = preds.float() - yb.float()
            batch_sse = (diff * diff).sum()
            loss = batch_sse / yb.numel()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            optimizer.step()
            scheduler.step()

            train_sse += batch_sse.detach()

        avg_train_loss = (train_sse / train_idx.numel()).item()  # MSE

        model.eval()
        with torch.no_grad():
            val_sse = torch.tensor(0.0, device=device)
            for start in range(0, val_idx.numel(), batch_size):
                idx = val_idx[start : start + batch_size]
                xb, yb = X[idx], y[idx]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(xb).squeeze(-1)
                val_sse += ((preds.float() - yb.float()) ** 2).sum()

        avg_val_loss = (val_sse / val_idx.numel()).item()
        val_rmse = torch.sqrt(val_sse / val_idx.numel()).item()

        last_lr = scheduler.get_last_lr()[0]
        # last_lr = lr
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
            writer.add_scalar("Metrics/val_RMSE", val_rmse, epoch)
            writer.flush()

    if best_state is not None:
        model.load_state_dict(best_state)

    if writer is not None:
        # hparams = {
        #     "lr": lr,
        #     "weight_decay": weight_decay,
        #     "batch_size": batch_size,
        #     "epochs": epochs,
        #     "normalize_x": normalize_x,
        #     "L": L,
        #     "V": V,
        #     "hidden1": hidden[0],
        #     "hidden2": hidden[1],
        #     "dropout": dropout,
        # }
        # final_metrics = {"hparam/val_MSE": avg_val_loss, "hparam/val_RMSE": val_rmse}
        # # add_hparams writes to a child run directory
        # writer.add_hparams(hparams, final_metrics)
        writer.close()

    if save_last:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{experiment_name}_last.pth")

        def _cpu_or_none(t):
            return None if t is None else t.detach().cpu()

        # ---- make sure we save CPU copies of normalization stats
        ckpt = {
            "model_class": "SharedPerLocationSum",
            "model_kwargs": {
                "in_dim": V,
                "width": width,
                "depth": depth,
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
    model, stats = train_model(
        data_path="data/torch_dataset_all_zones.pt",
        epochs=300,
        val_frac=0.2,
        normalize_x=True,
        batch_size=4 * 1024,
        lr=1e-3,
        warmup_epochs=5,
        T0_epochs=20,
        early_stopping_patience=25,
        weight_decay=1e-4,
        width=32,
        depth=2,
        # width=384,
        dropout=0.05,
        use_tensorboard=True,
        experiment_name="wind_tanh",
        log_dir="runs_all_zones",
        save_last=True,
        save_dir="checkpoints",
        device="cuda:0",
        seed=42,
    )
