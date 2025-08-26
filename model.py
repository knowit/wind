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


class LinearBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        # self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # h = self.norm(x)
        h = F.silu(self.linear(x))
        h = self.dropout(h)
        return h


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
        # self.blocks = nn.Sequential(
        #     *[LinearBlock(width, dropout=dropout) for _ in range(depth)]
        # )

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


class LatentSpaceSum(nn.Module):
    def __init__(
        self,
        in_dim,
        phi_width,
        phi_depth,
        latent_dims,
        rho_width,
        rho_depth,
        dropout=0.1,
    ):
        super().__init__()
        self.latent_dims = latent_dims
        self.phi_stem = nn.Sequential(
            nn.Linear(in_dim, phi_width),
            nn.SiLU(),
        )
        self.phi_blocks = nn.Sequential(
            *[LinearBlock(phi_width, dropout=dropout) for _ in range(phi_depth)]
        )
        self.latent_head = nn.Linear(phi_width, latent_dims)

        self.rho_stem = nn.Sequential(
            nn.Linear(latent_dims, rho_width),
            nn.SiLU(),
        )
        self.rho_blocks = nn.Sequential(
            *[LinearBlock(rho_width, dropout=dropout) for _ in range(rho_depth)]
        )
        self.head = nn.Linear(rho_width, 1)
        self.softplus = nn.Softplus(beta=5.0, threshold=10.0)

    def phi(self, x):
        h = self.phi_stem(x)
        h = self.phi_blocks(h)
        latent = self.latent_head(h)
        return latent

    def rho(self, x):
        h = self.rho_stem(x)
        h = self.rho_blocks(h)
        out = self.head(h)
        return self.softplus(out)

    def forward(self, x, mask):  # x: (B, L, V)
        B, L, V = x.shape
        z = x.reshape(B * L, V)
        latent = self.phi(z).reshape(B, L, self.latent_dims)  # (B, L, latent_dims)
        latent = latent * mask.unsqueeze(-1)
        latent_sum = latent.sum(dim=1)  # (B, latent_dims)
        y_hat = self.rho(latent_sum).squeeze()  # (B,)
        return y_hat


def load_model_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get(
        "model_kwargs",
    )
    model = LatentSpaceSum(**model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
