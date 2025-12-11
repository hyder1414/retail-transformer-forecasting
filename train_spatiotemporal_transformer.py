import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import math

# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = "/content/drive/MyDrive/walmart-recruiting-store-sales-forecasting"
CSV_PATH = os.path.join(BASE_DIR, "train_merged.csv")

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, L, D)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@dataclass
class SpatioTemporalConfig:
    enc_input_length: int = 30
    dec_output_length: int = 4

    enc_feature_dim: int = 11   # overridden at runtime
    dec_feature_dim: int = 10   # overridden at runtime

    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1

    static_embed_dim: int = 16


class SpatioTemporalTransformer(nn.Module):
    """
    Encoder–decoder Transformer with static Store/Dept/Type embeddings.
    """

    def __init__(
        self,
        cfg: SpatioTemporalConfig,
        static_vocab_sizes: Dict[str, int],
        static_cols_order: List[str],
    ):
        super().__init__()
        self.cfg = cfg
        self.static_cols_order = static_cols_order

        # Static embeddings per categorical static feature
        self.static_embeddings = nn.ModuleDict()
        for col, vocab_size in static_vocab_sizes.items():
            self.static_embeddings[col] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=cfg.static_embed_dim,
            )

        total_static_dim = cfg.static_embed_dim * len(static_vocab_sizes)
        self.static_proj = nn.Linear(total_static_dim, cfg.d_model)

        # Project dynamic + static into model dimension
        self.enc_proj = nn.Linear(cfg.enc_feature_dim + cfg.d_model, cfg.d_model)
        self.dec_proj = nn.Linear(cfg.dec_feature_dim + cfg.d_model, cfg.d_model)

        self.enc_pos = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.dec_pos = PositionalEncoding(cfg.d_model, cfg.dropout)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.head = nn.Linear(cfg.d_model, 1)

    def _static_repr(self, static_idx: torch.Tensor) -> torch.Tensor:
        # static_idx: (B, num_static)
        embs = []
        for i, col in enumerate(self.static_cols_order):
            emb_layer = self.static_embeddings[col]
            emb = emb_layer(static_idx[:, i])  # (B, static_embed_dim)
            embs.append(emb)
        static_concat = torch.cat(embs, dim=-1)  # (B, total_static_dim)
        static_repr = self.static_proj(static_concat)  # (B, d_model)
        return static_repr

    def forward(
        self,
        enc_x: torch.Tensor,       # (B, L_enc, F_enc)
        dec_x: torch.Tensor,       # (B, L_dec, F_dec)
        static_idx: torch.Tensor,  # (B, num_static)
    ) -> torch.Tensor:
        B, L_enc, _ = enc_x.shape
        _, L_dec, _ = dec_x.shape
        device = enc_x.device

        static_vec = self._static_repr(static_idx)           # (B, d_model)
        static_enc = static_vec.unsqueeze(1).expand(B, L_enc, -1)
        static_dec = static_vec.unsqueeze(1).expand(B, L_dec, -1)

        enc_in = torch.cat([enc_x, static_enc], dim=-1)
        dec_in = torch.cat([dec_x, static_dec], dim=-1)

        enc_in = self.enc_proj(enc_in)
        dec_in = self.dec_proj(dec_in)

        enc_in = self.enc_pos(enc_in)
        dec_in = self.dec_pos(dec_in)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L_dec).to(device)

        out = self.transformer(enc_in, dec_in, tgt_mask=tgt_mask)  # (B, L_dec, D)
        out = self.head(out).squeeze(-1)                           # (B, L_dec)
        return out


# ============================================================
# TRAIN CONFIG & FEATURE LISTS
# ============================================================
# Encoder dynamic features (PAST) – target not included
enc_dyn_cols = [
    "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment",
    "IsHoliday",
    "DayOfWeek", "WeekOfYear",
]

# Decoder dynamic features (FUTURE covariates)
dec_dyn_cols = [
    "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment",
    "IsHoliday",
    "DayOfWeek", "WeekOfYear",
]

# Index of IsHoliday inside decoder features (for WMAE weights)
ISHOLIDAY_IDX = dec_dyn_cols.index("IsHoliday")

static_cat_cols = ["Store", "Dept", "Type"]
group_cols = ["Store", "Dept"]

# Use FULL dataset: no subsampling
MAX_WINDOWS = 0   # 0 or None → use all windows


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 20
    num_workers: int = 0
    log_sales: bool = True   # use log1p(target)


# ============================================================
# DATASET
# ============================================================
class WalmartSeq2SeqDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: SpatioTemporalConfig,
        enc_cols: List[str],
        dec_cols: List[str],
        static_cols: List[str],
        group_cols: List[str],
        log_sales: bool = False,
    ):
        self.cfg = cfg
        self.enc_cols = enc_cols
        self.dec_cols = dec_cols
        self.static_cols = static_cols
        self.group_cols = group_cols
        self.log_sales = log_sales

        # Encode static categorical variables
        self.static_encoders: Dict[str, Dict] = {}
        for col in static_cols:
            df[col] = df[col].astype("category")
            mapping = {v: i for i, v in enumerate(df[col].cat.categories)}
            self.static_encoders[col] = mapping
            df[col + "_idx"] = df[col].map(mapping).astype("int64")

        # Clean NaNs / inf in relevant columns
        dyn_cols = list(set(enc_cols + dec_cols + ["Weekly_Sales"]))
        df[dyn_cols] = df[dyn_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Sort by time within series
        df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)
        self.df = df

        self.samples = self._make_windows()

    def _make_windows(self):
        windows = []
        total_len = self.cfg.enc_input_length + self.cfg.dec_output_length
        grouped = self.df.groupby(self.group_cols, observed=False)

        for _, gdf in grouped:
            n = len(gdf)
            if n < total_len:
                continue
            for i in range(n - total_len):
                windows.append(gdf.index[i])
        return windows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        start_idx = self.samples[idx]
        pos = self.df.index.get_loc(start_idx)

        enc_start = pos
        enc_end   = pos + self.cfg.enc_input_length
        dec_start = enc_end
        dec_end   = dec_start + self.cfg.dec_output_length

        enc_df = self.df.iloc[enc_start:enc_end]
        dec_df = self.df.iloc[dec_start:dec_end]

        # Static indices from the first row
        static_idx = torch.tensor(
            [enc_df[col + "_idx"].iloc[0] for col in self.static_cols],
            dtype=torch.long,
        )

        enc_x = torch.tensor(enc_df[self.enc_cols].values, dtype=torch.float32)
        dec_x = torch.tensor(dec_df[self.dec_cols].values, dtype=torch.float32)

        # Target handling: log1p with clipping of negatives to 0
        sales = dec_df["Weekly_Sales"].values.astype("float32")
        if self.log_sales:
            sales_clipped = np.clip(sales, a_min=0.0, a_max=None)
            y = torch.tensor(np.log1p(sales_clipped), dtype=torch.float32)
        else:
            y = torch.tensor(sales, dtype=torch.float32)

        return enc_x, dec_x, static_idx, y

    @property
    def static_vocab_sizes(self):
        return {c: len(v) for c, v in self.static_encoders.items()}


# ============================================================
# TRAINING HELPERS
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total = 0.0
    for enc_x, dec_x, static_idx, y in loader:
        enc_x, dec_x = enc_x.to(device), dec_x.to(device)
        static_idx, y = static_idx.to(device), y.to(device)

        optim.zero_grad()
        pred = model(enc_x, dec_x, static_idx)

        # Clamp predictions in log-space to avoid exploding gradients
        pred = torch.clamp(pred, min=0.0, max=15.0)

        loss = criterion(pred, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        optim.step()

        total += loss.item()

    avg_loss = total / len(loader)
    print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
    return avg_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    name: str,
    log_sales: bool,
) -> Tuple[float, float]:
    """
    Evaluate in log-space for the loss, but report WMAE in original sales units.
    Returns:
        avg_loss, wmae
    """
    if len(loader) == 0:
        print(f"{name}: NO BATCHES")
        return float("nan"), float("nan")

    model.eval()
    total_loss = 0.0
    all_p, all_y, all_w = [], [], []

    with torch.no_grad():
        for enc_x, dec_x, static_idx, y in loader:
            enc_x, dec_x = enc_x.to(device), dec_x.to(device)
            static_idx, y = static_idx.to(device), y.to(device)

            pred = model(enc_x, dec_x, static_idx)
            pred = torch.clamp(pred, min=0.0, max=15.0)

            loss = criterion(pred, y)
            total_loss += loss.item()

            all_p.append(pred.cpu())
            all_y.append(y.cpu())

            # IsHoliday for WMAE weights: shape (B, L_dec)
            holiday = dec_x[..., ISHOLIDAY_IDX].cpu()
            weights = torch.where(holiday > 0.5,
                                  torch.tensor(5.0),
                                  torch.tensor(1.0))
            all_w.append(weights)

    p = torch.cat(all_p, dim=0)  # (N, L_dec)
    t = torch.cat(all_y, dim=0)  # (N, L_dec)
    w = torch.cat(all_w, dim=0)  # (N, L_dec)

    # Invert log1p transform if used
    if log_sales:
        p = torch.expm1(p)
        t = torch.expm1(t)

    abs_err = torch.abs(p - t)
    wmae = (w * abs_err).sum().item() / w.sum().item()
    avg_loss = total_loss / len(loader)

    print(f"{name}: loss={avg_loss:.4f}, WMAE={wmae:.2f}")
    return avg_loss, wmae


# ============================================================
# MAIN
# ============================================================
def main():
    cfg_train = TrainConfig()

    # Load merged data
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].map(
        {True: 1, False: 0, "TRUE": 1, "FALSE": 0}
    ).astype(int)

    # Log-scale MarkDowns, then standardize numeric covariates
    markdown_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
    for col in markdown_cols:
        if col in df.columns:
            vals = df[col].values.astype("float64")
            vals = np.clip(vals, a_min=0.0, a_max=None)
            df[col] = np.log1p(vals)

    num_cols_to_scale = [
        "Temperature", "Fuel_Price",
        "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
        "CPI","Unemployment",
        "DayOfWeek","WeekOfYear",
    ]
    for col in num_cols_to_scale:
        if col in df.columns:
            mean = df[col].mean()
            std  = df[col].std()
            if std == 0 or np.isnan(std):
                std = 1.0
            df[col] = (df[col] - mean) / std

    # Model config – wide model
    cfg = SpatioTemporalConfig(
        enc_input_length=30,
        dec_output_length=4,
        enc_feature_dim=len(enc_dyn_cols),
        dec_feature_dim=len(dec_dyn_cols),
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )

    # Dataset
    base_ds = WalmartSeq2SeqDataset(
        df, cfg,
        enc_dyn_cols, dec_dyn_cols,
        static_cat_cols, group_cols,
        log_sales=cfg_train.log_sales,
    )

    print("TOTAL windows:", len(base_ds))

    # Use full dataset (no subsampling)
    if MAX_WINDOWS and len(base_ds) > MAX_WINDOWS:
        idx = torch.randperm(len(base_ds))[:MAX_WINDOWS]
        full_ds = Subset(base_ds, idx)
        print("Subsampled windows:", len(full_ds))
    else:
        full_ds = base_ds
        print("Using all windows (no subsampling).")

    # Train/val/test split
    n = len(full_ds)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_train.batch_size,
        shuffle=True,
        num_workers=cfg_train.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_train.batch_size,
        shuffle=False,
        num_workers=cfg_train.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg_train.batch_size,
        shuffle=False,
        num_workers=cfg_train.num_workers,
    )

    static_vocab_sizes = base_ds.static_vocab_sizes
    model = SpatioTemporalTransformer(cfg, static_vocab_sizes, static_cat_cols)

    device = get_device()
    model.to(device)
    print("DEVICE:", device)

    optim = Adam(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )
    criterion = nn.L1Loss()  # L1 in log-space
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

    # Best-epoch tracking
    best_val_wmae = float("inf")
    best_epoch = -1
    best_ckpt_path = os.path.join(BASE_DIR, "spatiotemporal_transformer_best.pt")

    for epoch in range(1, cfg_train.max_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optim, device, epoch)
        _, val_wmae = evaluate(model, val_loader, criterion, device, "VAL", cfg_train.log_sales)
        scheduler.step()

        if val_wmae < best_val_wmae:
            best_val_wmae = val_wmae
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"--> New best model at epoch {epoch}: VAL WMAE={val_wmae:.2f}")

    # Load best model before final test evaluation
    print(f"\nLoading best model from epoch {best_epoch} with VAL WMAE={best_val_wmae:.2f}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    print(f"===== FINAL TEST (best epoch {best_epoch}) =====")
    test_loss, test_wmae = evaluate(model, test_loader, criterion, device, "TEST", cfg_train.log_sales)
    print(f"Final TEST: loss={test_loss:.4f}, WMAE={test_wmae:.2f}")
    print(f"Best checkpoint saved → {best_ckpt_path}")


if __name__ == "__main__":
    main()
