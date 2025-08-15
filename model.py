import torch
import torch.nn as nn


class RhoAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_raw = nn.Parameter(torch.tensor(1.0))
        self.b_raw = nn.Parameter(torch.tensor(0.0))
        self.sp = nn.Softplus(beta=5.0, threshold=10.0)  # steeper, less mushy

    def forward(self, s):  # s: (B,)
        a = self.sp(self.a_raw) + 1e-6  # slope ≥ 0
        b = self.sp(self.b_raw)  # offset ≥ 0
        return a * s + b  # y ≥ 0


class SharedPerLocationSum(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.0, return_locals=False):
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
            nn.Dropout(dropout),
            nn.Linear(h3, 1),  # scalar contribution per location
            nn.Softplus(),
        )
        # self.rho = RhoAffine()

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
        # contribs_summed = contribs.sum(dim=1)  # (B,)
        # y_hat = self.rho(contribs_summed)
        y_hat = contribs.sum(dim=1)  # (B,)

        if self.return_locals:
            return y_hat, contribs
        return y_hat
