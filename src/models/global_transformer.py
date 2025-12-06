# src/models/global_transformer.py
import math
from typing import Dict, List
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class GlobalTransformer(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float,
                 horizon: int,
                 quantiles: List[float],
                 static_cardinalities: Dict[str, int]):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        q = len(quantiles)

        # Static embeddings (repeat across time)
        self.item_emb  = nn.Embedding(static_cardinalities["item_id"],  16)
        self.dept_emb  = nn.Embedding(static_cardinalities["dept_id"],  8)
        self.cat_emb   = nn.Embedding(static_cardinalities["cat_id"],   8)
        self.store_emb = nn.Embedding(static_cardinalities["store_id"], 8)
        self.state_emb = nn.Embedding(static_cardinalities["state_id"], 6)
        static_dim = 16+8+8+8+6

        self.in_proj = nn.Linear(d_in + static_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * q)
        )

    def forward(self, x_hist: torch.Tensor, x_static: torch.Tensor):
        """
        x_hist:   [B, L, F]  (past target + dyn covariates)
        x_static: [B, 5]     (item_id, dept_id, cat_id, store_id, state_id) int64
        """
        B, L, F = x_hist.shape
        item, dept, cat, store, state = x_static.unbind(dim=1)

        s = torch.cat([
            self.item_emb(item),
            self.dept_emb(dept),
            self.cat_emb(cat),
            self.store_emb(store),
            self.state_emb(state)
        ], dim=1)  # [B, static_dim]

        s_rep = s.unsqueeze(1).expand(B, L, -1)    # repeat across time
        x = torch.cat([x_hist, s_rep], dim=2)      # [B, L, F+static_dim]

        x = self.in_proj(x)
        x = self.pos(x)
        x = self.encoder(x)                        # [B, L, d_model]
        x_last = x[:, -1, :]                       # pool last token
        out = self.head(x_last)                    # [B, H*Q]
        out = out.view(B, self.horizon, len(self.quantiles))
        return out
