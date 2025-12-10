from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.paths import PROCESSED_DIR
from src.config.data_config import CONFIG


class WindowDataset(Dataset):
    def __init__(
        self,
        npz_path: Path | str | None = None,
        split: Literal["train", "val", "test"] = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
    ):
        # pick standardized save location
        if npz_path is None:
            npz_path = PROCESSED_DIR / f"windows_L{CONFIG.input_length}_H{CONFIG.output_length}.npz"
        self.npz_path = Path(npz_path)

        # load all arrays
        data = np.load(self.npz_path, allow_pickle=True)
        X = data["X"]  # (N, L, F)
        y = data["y"]  # (N, H)

        self.feature_cols = data["feature_cols"].tolist()

        # index of holiday flag inside feature dimension (may not exist)
        self.holiday_idx = (
            self.feature_cols.index("IsHoliday")
            if "IsHoliday" in self.feature_cols
            else None
        )

        # ----- create deterministic split -----
        N = X.shape[0]
        rng = np.random.default_rng(seed)
        indices = np.arange(N)
        rng.shuffle(indices)

        train_end = int(train_frac * N)
        val_end = int((train_frac + val_frac) * N)

        if split == "train":
            self.idx = indices[:train_end]
        elif split == "val":
            self.idx = indices[train_end:val_end]
        else:  # test
            self.idx = indices[val_end:]

        # ----- store tensors -----
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]

        x = self.X[j]   # (L, F)
        y = self.y[j]   # (H,)

        # last-step holiday flag from this window
        if self.holiday_idx is not None:
            is_holiday = float(x[-1, self.holiday_idx])
        else:
            is_holiday = 0.0

        return x, y, is_holiday


def create_dataloaders(
    batch_size: int = 256,
    num_workers: int = 0,
    npz_path: Path | str | None = None,
):
    train_ds = WindowDataset(npz_path=npz_path, split="train")
    val_ds = WindowDataset(npz_path=npz_path, split="val")
    test_ds = WindowDataset(npz_path=npz_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
