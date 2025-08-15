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


# import torch
# import torch.nn as nn


# class RhoAffine(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.a_raw = nn.Parameter(torch.tensor(1.0))
#         self.b_raw = nn.Parameter(torch.tensor(0.0))
#         self.sp = nn.Softplus(beta=5.0, threshold=10.0)  # steeper, less mushy

#     def forward(self, s):  # s: (B,)
#         a = self.sp(self.a_raw) + 1e-6  # slope ≥ 0
#         b = self.sp(self.b_raw)  # offset ≥ 0
#         return a * s + b  # y ≥ 0


# class SharedPerLocationSum(nn.Module):
#     def __init__(self, in_dim, hidden, dropout=0.0, return_locals=False):
#         super().__init__()
#         # h1, h2 = hidden
#         h1, h2, h3 = hidden
#         self.return_locals = return_locals
#         self.phi = nn.Sequential(
#             nn.Linear(in_dim, h1),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(h1, h2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(h2, h3),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(h3, 1),  # scalar contribution per location
#             nn.Softplus(),
#         )
#         # self.rho = RhoAffine()

#     def forward(self, x):
#         """
#         x: (B, L, V)
#         returns:
#           y_hat: (B,) predicted total
#           (optionally) loc_contribs: (B, L)
#         """
#         B, L, V = x.shape
#         z = x.view(B * L, V)  # flatten locations
#         contribs = self.phi(z).view(B, L)  # (B, L)
#         # contribs_summed = contribs.sum(dim=1)  # (B,)
#         # y_hat = self.rho(contribs_summed)
#         y_hat = contribs.sum(dim=1)  # (B,)

#         if self.return_locals:
#             return y_hat, contribs
#         return y_hat#         return y_hat
