import os
import yaml
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from src.features.make_dataset import build_master, make_windows
from src.models.global_transformer import GlobalTransformer
from src.utils.metrics import pinball_loss, wape, save_metrics

def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)

class WindowDataModule(pl.LightningDataModule):
    """
    DataModule that takes a prebuilt long df & cardinalities so we don't
    reload/merge CSVs twice.
    """
    def __init__(self, cfg, df_long, static_cardinalities):
        super().__init__()
        self.cfg = cfg
        self.df_long = df_long
        self.static_cardinalities = static_cardinalities

    def setup(self, stage=None):
        print("Step 2/4: Building DataModule (creating sliding windows)...")
        train, val = make_windows(
            self.df_long,
            self.cfg["context_len"],
            self.cfg["horizon"],
            self.cfg["val_last_days"],
        )

        self.train_ds = TensorDataset(
            torch.tensor(train["Xc"], dtype=torch.float32),
            torch.tensor(train["Y"],  dtype=torch.float32),
            torch.tensor(train["static"], dtype=torch.long),
        )
        self.val_ds = TensorDataset(
            torch.tensor(val["Xc"], dtype=torch.float32),
            torch.tensor(val["Y"],  dtype=torch.float32),
            torch.tensor(val["static"], dtype=torch.long),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        )

class LitForecaster(pl.LightningModule):
    def __init__(self, cfg, static_cardinalities):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.quantiles = cfg["quantiles"]

        # feature dim: past_target(1) + dyn(4: dow, month, snap, price_ratio)
        d_in = 1 + 4

        self.model = GlobalTransformer(
            d_in=d_in,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            horizon=cfg["horizon"],
            quantiles=self.quantiles,
            static_cardinalities=static_cardinalities,
        )

    def forward(self, x_hist, x_static):
        return self.model(x_hist, x_static)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"])

    def training_step(self, batch, batch_idx):
        Xc, Y, S = batch
        pred = self(Xc, S)  # [B,H,Q]
        loss = pinball_loss(pred, Y, self.quantiles)
        self.log("train_pinball", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        Xc, Y, S = batch
        pred = self(Xc, S)   # [B,H,Q]
        loss = pinball_loss(pred, Y, self.quantiles)
        self.log("val_pinball", loss, prog_bar=True)

        # WAPE on median forecast
        q_index = self.quantiles.index(0.5) if 0.5 in self.quantiles else np.argmin(np.abs(np.array(self.quantiles) - 0.5))
        med = pred[:, :, q_index].detach()
        w = wape(Y.cpu().numpy().ravel(), med.cpu().numpy().ravel())
        self.log("val_wape", w, prog_bar=True)
        return {"pinball": float(loss.item()), "wape": float(w)}

def main():
    # ---- Load config
    with open("src/config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    # ---- Build long df once (faster) and get embedding cardinalities
    print("Step 1/4: Scanning raw CSVs and computing cardinalities...")
    df_long, static_cardinalities = build_master(
        cfg["raw_dir"],
        cfg["max_series"],
        cfg["min_history"],
    )

    # ---- DataModule & Model
    dm = WindowDataModule(cfg, df_long, static_cardinalities)
    model = LitForecaster(cfg, static_cardinalities)

    # ---- Trainer
    ckpt = ModelCheckpoint(monitor="val_pinball", mode="min", save_top_k=1)
    es = EarlyStopping(monitor="val_pinball", mode="min", patience=2)
    logger = CSVLogger("experiments", name="gt_m5")

    print("Step 3/4: Starting training...")
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        precision=cfg["precision"],
        logger=logger,
        callbacks=[ckpt, es],
        enable_progress_bar=True,
    )
    trainer.fit(model, dm)

    # ---- Export quick metrics for your report
    print("Step 4/4: Saving metrics...")
    metrics_path = os.path.join(cfg["metrics_dir"], "transformer_v0_metrics.json")
    os.makedirs(cfg["metrics_dir"], exist_ok=True)
    hist_path = os.path.join(logger.log_dir, "metrics.csv")
    hist = pd.read_csv(hist_path)

    best = {
        "best_val_pinball": float(hist["val_pinball"].dropna().min()) if "val_pinball" in hist else None,
        "best_val_wape": float(hist["val_wape"].dropna().min()) if "val_wape" in hist else None,
        "log_dir": logger.log_dir,
        "metrics_csv": hist_path,
    }
    save_metrics(metrics_path, best)
    print("Saved metrics to:", metrics_path)
    print("Logs at:", logger.log_dir)

if __name__ == "__main__":
    main()
