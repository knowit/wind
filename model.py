import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout, expand):
        super().__init__()
        mid = width * expand
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, mid)
        self.fc2 = nn.Linear(mid, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h  # residual


class SharedPerLocationSum(nn.Module):
    def __init__(self, in_dim, width=256, depth=4, dropout=0.15):
        super().__init__()

        # stem to constant width
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.SiLU(),
        )
        # D residual MLP blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, dropout=dropout, expand=2) for _ in range(depth)]
        )

        # head -> scalar >= 0 (per location)
        self.head = nn.Linear(width, 1)
        self.softplus = nn.Softplus(beta=5.0, threshold=10.0)  # steeper, less mushy
        nn.init.constant_(
            self.head.bias, 0.05
        )  # tiny positive bias to avoid dead region

    def phi(self, x):  # x: (..., V)
        h = self.stem(x)
        h = self.blocks(h)
        out = self.head(h)
        return self.softplus(out)  # ≥ 0

    def forward(self, x):  # x: (B, L, V)
        B, L, V = x.shape
        z = x.reshape(B * L, V)
        contribs = self.phi(z).view(B, L)  # (B, L)
        y_hat = contribs.sum(dim=1)  # (B,)
        return y_hat


def load_model_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_kwargs = checkpoint.get(
        "model_kwargs",
    )
    model = SharedPerLocationSum(**model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
