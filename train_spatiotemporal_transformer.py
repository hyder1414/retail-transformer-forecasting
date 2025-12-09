from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

from src.models.spatiotemporal_transformer import (
    SpatioTemporalConfig,
    SpatioTemporalTransformer,
)

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "train_merged.csv"

enc_dyn_cols = [
    "Weekly_Sales",
    "Temperature","Fuel_Price",
    "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
    "CPI","Unemployment","IsHoliday",
]

dec_dyn_cols = [
    "Temperature","Fuel_Price",
    "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
    "CPI","Unemployment","IsHoliday",
]

static_cat_cols = ["Store","Dept","Type"]
group_cols = ["Store","Dept"]

@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 10
    num_workers: int = 0
    log_sales: bool = False


# ============================================================
# DATASET
# ============================================================
class WalmartSeq2SeqDataset(Dataset):
    def __init__(self, df, cfg, enc_cols, dec_cols, static_cols, group_cols, log_sales=False):
        self.cfg = cfg
        self.enc_cols = enc_cols
        self.dec_cols = dec_cols
        self.static_cols = static_cols
        self.group_cols = group_cols
        self.log_sales = log_sales

        # Encode static categorical
        self.static_encoders = {}
        for col in static_cols:
            df[col] = df[col].astype("category")
            mapping = {v: i for i, v in enumerate(df[col].cat.categories)}
            self.static_encoders[col] = mapping
            df[col+"_idx"] = df[col].map(mapping).astype("int64")

        # Clean NaNs
        dyn_cols = list(set(enc_cols + dec_cols + ["Weekly_Sales"]))
        df[dyn_cols] = df[dyn_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)

        # Sort
        df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)
        self.df = df
        self.samples = self._make_windows()

    def _make_windows(self):
        windows = []
        step = self.cfg.enc_input_length + self.cfg.dec_output_length
        grouped = self.df.groupby(self.group_cols, observed=False)

        for _, gdf in grouped:
            n = len(gdf)
            if n < step:
                continue
            for i in range(n - step):
                windows.append(gdf.index[i])
        return windows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        pos = self.df.index.get_loc(start)

        enc_start = pos
        enc_end   = pos + self.cfg.enc_input_length
        dec_start = enc_end
        dec_end   = dec_start + self.cfg.dec_output_length

        enc_df = self.df.iloc[enc_start:enc_end]
        dec_df = self.df.iloc[dec_start:dec_end]

        static_idx = torch.tensor(
            [enc_df[col+"_idx"].iloc[0] for col in self.static_cols],
            dtype=torch.long
        )

        enc_x = torch.tensor(enc_df[self.enc_cols].values, dtype=torch.float32)
        dec_x = torch.tensor(dec_df[self.dec_cols].values, dtype=torch.float32)

        y = torch.tensor(dec_df["Weekly_Sales"].values, dtype=torch.float32)
        if self.log_sales:
            y = torch.log1p(y)

        return enc_x, dec_x, static_idx, y

    @property
    def static_vocab_sizes(self):
        return {c: len(v) for c, v in self.static_encoders.items()}


# ============================================================
# TRAINING LOOP
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optim, device, epoch):
    model.train()
    total = 0
    for enc_x, dec_x, static_idx, y in loader:
        enc_x, dec_x = enc_x.to(device), dec_x.to(device)
        static_idx, y = static_idx.to(device), y.to(device)

        optim.zero_grad()
        pred = model(enc_x, dec_x, static_idx)

        pred = torch.nan_to_num(pred)
        y = torch.nan_to_num(y)

        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        total += loss.item()
    print(f"Epoch {epoch}: train_loss={total/len(loader):.4f}")

def evaluate(model, loader, criterion, device, name, log_sales):
    if len(loader) == 0:
        print(f"{name}: NO BATCHES")
        return

    model.eval()
    total = 0
    all_p, all_y = [], []

    with torch.no_grad():
        for enc_x, dec_x, static_idx, y in loader:
            enc_x, dec_x = enc_x.to(device), dec_x.to(device)
            static_idx, y = static_idx.to(device), y.to(device)

            pred = model(enc_x, dec_x, static_idx)
            pred = torch.nan_to_num(pred)
            y = torch.nan_to_num(y)

            loss = criterion(pred, y)
            total += loss.item()

            all_p.append(pred.cpu())
            all_y.append(y.cpu())

    p = torch.cat(all_p)
    t = torch.cat(all_y)

    if log_sales:
        p = torch.expm1(p)
        t = torch.expm1(t)

    rmse = torch.sqrt(torch.mean((p - t)**2)).item()
    mae = torch.mean(torch.abs(p - t)).item()
    print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}")


# ============================================================
# MAIN
# ============================================================
def main():
    cfg_train = TrainConfig()

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].map({True:1,False:0,"TRUE":1,"FALSE":0}).astype(int)

    cfg = SpatioTemporalConfig(
        enc_input_length=30,
        dec_output_length=4,
        enc_feature_dim=len(enc_dyn_cols),
        dec_feature_dim=len(dec_dyn_cols)
    )

    full_ds = WalmartSeq2SeqDataset(df, cfg, enc_dyn_cols, dec_dyn_cols,
                                    static_cat_cols, group_cols,
                                    log_sales=cfg_train.log_sales)

    print("TOTAL windows:", len(full_ds))

    n = len(full_ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg_train.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_train.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg_train.batch_size)

    static_vocab_sizes = full_ds.static_vocab_sizes

    model = SpatioTemporalTransformer(cfg, static_vocab_sizes, static_cat_cols)
    device = get_device()
    model.to(device)

    print("DEVICE:", device)

    optim = Adam(model.parameters(), lr=cfg_train.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, cfg_train.max_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optim, device, epoch)
        evaluate(model, val_loader, criterion, device, "VAL", cfg_train.log_sales)

    print("===== FINAL TEST =====")
    evaluate(model, test_loader, criterion, device, "TEST", cfg_train.log_sales)

    torch.save(model.state_dict(), "spatiotemporal_transformer.pt")
    print("Saved checkpoint → spatiotemporal_transformer.pt")

if __name__ == "__main__":
    main()
