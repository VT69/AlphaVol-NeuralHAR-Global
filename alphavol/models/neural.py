"""Neural-HAR: linear HAR head (OLS-initialized, trainable) + residual network (GRN or
FT-Transformer) over standardized HAR lags and exogenous features.

Fixes vs. legacy (analysis/03_methodology_audit.md): fresh optimizer/schedule per fit (T1); stable
QLIKE without a prediction clamp (T2); seeded, deterministic training (T3); the network sees HAR
lags *and* exogenous features, as the paper describes (paper_vs_code.md C4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from alphavol.models.base import TARGET, Preprocessor, smearing, train_xy
from alphavol.models.linear import HAR


class GRNBlock(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_in, d), nn.Linear(d, d)
        self.glu = nn.Sequential(nn.Linear(d, 2 * d), nn.GLU())
        self.skip = nn.Linear(d_in, d) if d_in != d else nn.Identity()
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.fc2(nn.functional.elu(self.fc1(x))))
        return self.norm(self.glu(h) + self.skip(x))


class GRNNet(nn.Module):
    def __init__(self, d_in, d, layers, dropout):
        super().__init__()
        self.blocks = nn.Sequential(*[GRNBlock(d_in if i == 0 else d, d, dropout) for i in range(layers)])
        self.out_dim = d

    def forward(self, x):
        return self.blocks(x)


class FTTransformer(nn.Module):
    """Feature tokenizer + CLS token + Transformer encoder (Gorishniy et al. 2021)."""

    def __init__(self, d_in, d, layers, dropout, heads=2):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_in, d))
        self.b = nn.Parameter(torch.zeros(d_in, d))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads if d % heads == 0 else 1, 2 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b
        z = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        return self.norm(self.enc(z)[:, 0])


class _Net(nn.Module):
    def __init__(self, arch, d_in, hidden, layers, dropout):
        super().__init__()
        self.har = nn.Linear(3, 1)
        self.body = (GRNNet if arch == "grn" else FTTransformer)(d_in, hidden, layers, dropout)
        self.head = nn.Linear(self.body.out_dim, 1)
        nn.init.zeros_(self.head.weight)              # start exactly at the HAR prior
        nn.init.zeros_(self.head.bias)

    def forward(self, har_raw, x):
        return (self.har(har_raw) + self.head(self.body(x))).squeeze(-1)


def qlike_torch(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    u = y - f
    return (torch.exp(u) - u - 1.0).mean()


@dataclass
class NeuralHAR:
    features: list[str]
    arch: str = "grn"
    loss: str = "qlike"
    hidden: int = 32
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 60
    batch_size: int = 128
    seed: int = 0
    name: str = "NeuralHAR"
    history_: list = field(default_factory=list, init=False)

    def _tensors(self, df):
        har = torch.tensor(df[HAR].to_numpy(dtype=float), dtype=torch.float32)
        x = torch.tensor(self.prep_.transform(df), dtype=torch.float32)
        return har, x

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "NeuralHAR":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        tr = train_xy(train, self.features).dropna(subset=HAR)
        self.prep_ = Preprocessor(self.features, standardize=True).fit(tr)
        har, x = self._tensors(tr)
        y = torch.tensor(tr[TARGET].to_numpy(dtype=float), dtype=torch.float32)
        self.net_ = _Net(self.arch, x.shape[1], self.hidden, self.layers, self.dropout)
        # OLS initialization of the HAR head (+ smearing when training on QLIKE)
        X = np.c_[np.ones(len(tr)), tr[HAR].to_numpy(dtype=float)]
        beta, *_ = np.linalg.lstsq(X, tr[TARGET].to_numpy(dtype=float), rcond=None)
        b0 = beta[0] + (smearing(tr[TARGET].to_numpy() - X @ beta) if self.loss == "qlike" else 0.0)
        with torch.no_grad():
            self.net_.har.weight.copy_(torch.tensor(beta[1:], dtype=torch.float32).view(1, 3))
            self.net_.har.bias.fill_(float(b0))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)   # fresh (T1)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1), eta_min=self.lr * 0.01)
        lossf = qlike_torch if self.loss == "qlike" else nn.functional.mse_loss
        g = torch.Generator().manual_seed(self.seed)
        vt = None
        if valid is not None:
            va = train_xy(valid, self.features).dropna(subset=HAR)
            vt = self._tensors(va) + (torch.tensor(va[TARGET].to_numpy(dtype=float), dtype=torch.float32),)
        self.history_ = []
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(y), generator=g)
            for i in range(0, len(y), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = lossf(self.net_(har[idx], x[idx]), y[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            sch.step()
            if vt is not None:
                self.net_.eval()
                with torch.no_grad():
                    self.history_.append(float(qlike_torch(self.net_(vt[0], vt[1]), vt[2])))
        self.net_.eval()
        with torch.no_grad():
            fit_pred = self.net_(har, x).numpy()
        self.smear_ = smearing(tr[TARGET].to_numpy() - fit_pred) if self.loss == "mse" else 0.0
        return self

    def predict(self, context, rows) -> np.ndarray:
        df = context.loc[rows]
        har, x = self._tensors(df)
        with torch.no_grad():
            return self.net_(har, x).numpy().astype(float) + self.smear_
