#Minimal Transformer forecaster model
from dataclasses import dataclass

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@dataclass
class TransformerConfig:
    input_length: int = 52
    output_length: int = 4
    feature_dim: int = 18

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1


class TimeSeriesTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.pos_encoder = PositionalEncoding(cfg.d_model, dropout=cfg.dropout, max_len=cfg.input_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,  # (B, L, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # use last time step embedding to predict the next H steps
        self.head = nn.Linear(cfg.d_model, cfg.output_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, F)  -> y_hat: (B, H)
        """
        x = self.input_proj(x)          # (B, L, d_model)
        x = self.pos_encoder(x)         # (B, L, d_model)
        enc = self.encoder(x)           # (B, L, d_model)
        last = enc[:, -1, :]            # (B, d_model)
        out = self.head(last)           # (B, H)
        return out
