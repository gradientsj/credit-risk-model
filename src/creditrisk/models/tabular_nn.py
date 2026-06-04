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


class _FlatEncoder(nn.Module):
    """Shared front-end for flat (non-token) architectures: numeric features
    pass through; categorical features get small embeddings, concatenated."""

    def __init__(self, n_numeric: int, cat_cardinalities: list[int], d_embed: int = 8):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_embed) for card in cat_cardinalities]
        )
        self.out_dim = n_numeric + d_embed * len(cat_cardinalities)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        parts += [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=1)


class MLPResNet(nn.Module):
    """ResNet for tabular data (Gorishniy et al. 2021): pre-norm residual MLP
    blocks over a flat feature vector."""

    def __init__(self, n_numeric: int, cat_cardinalities: list[int],
                 d: int = 256, n_blocks: int = 4, dropout: float = 0.15):
        super().__init__()
        self.encoder = _FlatEncoder(n_numeric, cat_cardinalities)
        self.input_proj = nn.Linear(self.encoder.out_dim, d)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d * 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d * 2, d), nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.encoder(x_num, x_cat))
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)  # (B,) logits


class BatchEnsembleLinear(nn.Module):
    """Shared weight matrix with k rank-1 multiplicative adapters and
    per-member biases (Wen et al. 2020) — the building block of TabM."""

    def __init__(self, in_features: int, out_features: int, k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # random-sign init decorrelates ensemble members (TabM, Gorishniy et al. 2024)
        self.r = nn.Parameter(torch.empty(k, in_features).bernoulli_(0.5) * 2 - 1)
        self.s = nn.Parameter(torch.ones(k, out_features))
        self.bias = nn.Parameter(torch.zeros(k, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, k, in)
        return ((x * self.r) @ self.weight) * self.s + self.bias


class TabM(nn.Module):
    """TabM (Gorishniy et al., NeurIPS 2024): an MLP whose layers are
    BatchEnsemble-shared across k implicit members. Trains k models for ~the
    cost of one; predicts by averaging member probabilities. Returns logits of
    shape (B, k); the training wrapper averages member losses."""

    def __init__(self, n_numeric: int, cat_cardinalities: list[int],
                 d: int = 256, n_layers: int = 3, k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.encoder = _FlatEncoder(n_numeric, cat_cardinalities)
        dims = [self.encoder.out_dim] + [d] * n_layers
        self.layers = nn.ModuleList(
            [BatchEnsembleLinear(dims[i], dims[i + 1], k) for i in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = BatchEnsembleLinear(d, 1, k)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_num, x_cat).unsqueeze(1).expand(-1, self.k, -1)
        for layer in self.layers:
            h = self.dropout(torch.relu(layer(h)))
        return self.head(h).squeeze(-1)  # (B, k) logits


ARCHITECTURES = {
    "ft_transformer": FTTransformer,
    "mlp_resnet": MLPResNet,
    "tabm": TabM,
}


class TabularNNClassifier:
    """sklearn-style wrapper: standardizes numerics, encodes categoricals,
    trains the chosen architecture with early stopping on validation AUC.

    For ``arch="tabm"`` the model emits (B, k) member logits: the loss averages
    over members and prediction averages member probabilities."""

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        arch: str = "ft_transformer",
        arch_kwargs: dict | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 4096,
        max_epochs: int = 30,
        patience: int = 4,
        seed: int = 42,
        device: str | None = None,
    ):
        if arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}")
        self.numeric, self.categorical = numeric, categorical
        self.arch = arch
        self.hparams = arch_kwargs if arch_kwargs is not None else {}
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
        self.model_ = ARCHITECTURES[self.arch](
            len(self.numeric), cards, **self.hparams
        ).to(self.device)

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
                out = self.model_(bn, bc)
                if out.ndim == 2:  # TabM: (B, k) member logits — average member losses
                    by = by.unsqueeze(1).expand_as(out)
                loss = loss_fn(out, by)
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
            logits = self.model_(bn, bc)
            probs = torch.sigmoid(logits)
            if probs.ndim == 2:  # TabM: average member probabilities
                probs = probs.mean(dim=1)
            out.append(probs.cpu().numpy())
        return np.concatenate(out)
