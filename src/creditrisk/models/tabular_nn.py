"""FT-Transformer for tabular credit data, implemented from scratch in PyTorch.

Architecture follows Gorishniy et al., "Revisiting Deep Learning Models for
Tabular Data" (NeurIPS 2021): every numeric feature is projected to a learned
token (per-feature affine map), every categorical feature is embedded, a [CLS]
token is prepended, the token sequence passes through a standard pre-norm
Transformer encoder, and the final [CLS] representation feeds a binary head.

Trained with BCE on GPU. On tabular credit data we *expect* GBMs to remain
competitive or better — including the comparison is itself a governance point
(challenger evaluation should justify complexity, not assume it).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureTokenizer(nn.Module):
    """numeric -> per-feature affine token; categorical -> embedding token."""

    def __init__(self, n_numeric: int, cat_cardinalities: list[int], d_token: int):
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.num_bias = nn.Parameter(torch.empty(n_numeric, d_token))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        for p in [self.num_weight, self.num_bias, self.cls]:
            nn.init.uniform_(p, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias  # (B, F_num, d)
        if x_cat.shape[1]:
            cat_tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
            )
            tokens = torch.cat([tokens, cat_tokens], dim=1)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)  # (B, 1 + F, d)


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_numeric, cat_cardinalities, d_token)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.tokenizer(x_num, x_cat))
        return self.head(h[:, 0]).squeeze(-1)  # logit from [CLS]


class TabularNNClassifier:
    """sklearn-style wrapper: standardizes numerics, encodes categoricals,
    trains FTTransformer with early stopping on validation AUC."""

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 4096,
        max_epochs: int = 30,
        patience: int = 4,
        seed: int = 42,
        device: str | None = None,
    ):
        self.numeric, self.categorical = numeric, categorical
        self.hparams = dict(d_token=d_token, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history_: list[dict] = []

    # ---- encoding ------------------------------------------------------------
    def _fit_encoders(self, X: pd.DataFrame):
        self.num_mean_ = X[self.numeric].mean().values.astype(np.float32)
        self.num_std_ = (X[self.numeric].std().values + 1e-8).astype(np.float32)
        self.cat_maps_ = [
            {v: i + 1 for i, v in enumerate(sorted(X[c].astype(str).unique()))}
            for c in self.categorical
        ]  # 0 reserved for unseen

    def _encode(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x_num = (X[self.numeric].values.astype(np.float32) - self.num_mean_) / self.num_std_
        x_num = np.clip(x_num, -10, 10)
        if self.categorical:
            x_cat = np.stack(
                [
                    X[c].astype(str).map(m).fillna(0).values.astype(np.int64)
                    for c, m in zip(self.categorical, self.cat_maps_)
                ],
                axis=1,
            )
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)
        return torch.from_numpy(x_num), torch.from_numpy(x_cat)

    # ---- training --------------------------------------------------------------
    def fit(self, X_train, y_train, X_valid, y_valid):
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.seed)
        self._fit_encoders(X_train)
        cards = [len(m) + 1 for m in self.cat_maps_]
        self.model_ = FTTransformer(len(self.numeric), cards, **self.hparams).to(self.device)

        xn, xc = self._encode(X_train)
        yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
        loader = DataLoader(
            TensorDataset(xn, xc, yt), batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        # plain BCE: ~8% positives is mild imbalance, and re-weighting would
        # inflate predicted PDs — calibration matters for a credit model.
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, bad_epochs = -np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            total = 0.0
            for bn, bc, by in loader:
                bn, bc, by = bn.to(self.device), bc.to(self.device), by.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model_(bn, bc), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(by)
            sched.step()

            val_pred = self.predict_proba(X_valid)
            val_auc = roc_auc_score(y_valid, val_pred)
            self.history_.append(
                {"epoch": epoch, "train_loss": total / len(yt), "valid_auc": val_auc}
            )
            if val_auc > best_auc + 1e-5:
                best_auc, bad_epochs = val_auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model_.eval()
        xn, xc = self._encode(X)
        out = []
        for i in range(0, len(xn), 16384):
            bn = xn[i : i + 16384].to(self.device)
            bc = xc[i : i + 16384].to(self.device)
            out.append(torch.sigmoid(self.model_(bn, bc)).cpu().numpy())
        return np.concatenate(out)
