# src/models/hahrt_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HAHRTConfig:
    input_dim: int
    num_stores: int
    num_depts: int
    num_store_types: int
    max_week_of_year: int
    max_year_index: int

    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    holiday_bias_strength: float = 1.5  # how strongly to bias attention toward holiday timesteps


class HolidayAwareSelfAttention(nn.Module):
    """
    Multi-head self-attention that adds a learnable bias toward timesteps
    where is_holiday == 1.

    We do standard scaled dot-product attention, but before softmax:

        scores = (QK^T / sqrt(d_head)) + bias

    where `bias` is positive on keys that correspond to holidays.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        holiday_bias_strength: float = 1.5,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.holiday_bias_strength = holiday_bias_strength

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_holiday: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        is_holiday: [B, L] (0 or 1)
        """
        B, L, _ = x.shape

        # Project to Q, K, V and reshape for multi-head: [B, h, L, d_head]
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        # scores: [B, h, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Holiday bias: we bias *keys* that correspond to holiday timesteps
        # is_holiday: [B, L] -> [B, 1, 1, L] then broadcast to [B, h, L, L]
        # so every query at any position can more easily attend to holiday keys.
        if self.holiday_bias_strength != 0.0:
            holiday_mask = is_holiday.float().unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            holiday_bias = self.holiday_bias_strength * holiday_mask
            scores = scores + holiday_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # [B, h, L, L] x [B, h, L, d_head] -> [B, h, L, d_head]
        context = torch.matmul(attn, v)

        # Merge heads: [B, h, L, d_head] -> [B, L, d_model]
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)

        out = self.out_proj(context)
        return out


class HolidayAwareTransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer, but:
      - uses HolidayAwareSelfAttention
      - supports FiLM-style modulation (gamma, beta) per-series
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 256,
        dropout: float = 0.1,
        holiday_bias_strength: float = 1.5,
    ):
        super().__init__()
        self.self_attn = HolidayAwareSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            holiday_bias_strength=holiday_bias_strength,
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        is_holiday: torch.Tensor,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, L, d_model]
        is_holiday: [B, L]
        gamma, beta: [B, d_model] (FiLM modulation), optional
        """
        # Self-attention block
        attn_out = self.self_attn(x, is_holiday)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward block
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ff)
        x = self.norm2(x)

        # FiLM adaptation (hierarchical store/dept adapters)
        if gamma is not None and beta is not None:
            # gamma, beta: [B, d_model] -> [B, 1, d_model] to broadcast over time
            gamma_b = gamma.unsqueeze(1)
            beta_b = beta.unsqueeze(1)
            x = x * (1.0 + gamma_b) + beta_b

        return x


class HAHRTModel(nn.Module):
    """
    Holiday-Aware Hierarchical Residual Transformer:
      - Input: residual + covariates sequences
      - Holiday-aware multi-head self-attention
      - Hierarchical FiLM adapters (store/dept/type/size)
      - Output: residual correction for the last timestep in the window
    """
    def __init__(self, cfg: HAHRTConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model

        # Project continuous covariates into model dimension
        self.input_proj = nn.Linear(cfg.input_dim, d_model)

        # Time embeddings
        self.week_embed = nn.Embedding(cfg.max_week_of_year + 1, d_model)
        self.year_embed = nn.Embedding(cfg.max_year_index + 1, d_model)

        # Holiday token embedding (0/1)
        self.holiday_embed = nn.Embedding(2, d_model)

        # Static / hierarchical embeddings
        # Use smaller dims then project to d_model and FiLM
        d_store = d_model // 4
        d_dept = d_model // 4
        d_type = max(d_model // 8, 4)
        d_size = max(d_model // 8, 4)

        self.store_embed = nn.Embedding(cfg.num_stores, d_store)
        self.dept_embed = nn.Embedding(cfg.num_depts, d_dept)
        self.store_type_embed = nn.Embedding(cfg.num_store_types, d_type)
        self.size_mlp = nn.Sequential(
            nn.Linear(1, d_size),
            nn.ReLU(),
            nn.Linear(d_size, d_size),
        )

        # Combine static embeddings into a single vector, then produce FiLM params
        static_total_dim = d_store + d_dept + d_type + d_size
        self.static_to_film = nn.Sequential(
            nn.Linear(static_total_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        # Stack of holiday-aware Transformer layers
        self.layers = nn.ModuleList(
            [
                HolidayAwareTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    holiday_bias_strength=cfg.holiday_bias_strength,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        self.dropout = nn.Dropout(cfg.dropout)

        # Output head: predict scalar residual correction from last timestep
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        batch keys expected (from HAHRTSequenceDataset):
          - x_cont: [B, L, input_dim]
          - week_of_year: [B, L]
          - year_idx: [B, L]
          - is_holiday: [B, L]
          - store_idx: [B]
          - dept_idx: [B]
          - store_type_idx: [B]
          - size_norm: [B]
        Returns:
          - residual_pred: [B] (predicted residual for the target timestep)
        """
        x_cont = batch["x_cont"]  # [B, L, input_dim]
        week = batch["week_of_year"].long()  # [B, L]
        year = batch["year_idx"].long()      # [B, L]
        is_holiday = batch["is_holiday"].long()  # [B, L]

        store_idx = batch["store_idx"].long()  # [B]
        dept_idx = batch["dept_idx"].long()    # [B]
        store_type_idx = batch["store_type_idx"].long()  # [B]
        size_norm = batch["size_norm"].unsqueeze(-1)  # [B, 1]

        B, L, _ = x_cont.shape

        # 1) Project continuous covariates
        x = self.input_proj(x_cont)  # [B, L, d_model]

        # 2) Add time & holiday token embeddings
        week_emb = self.week_embed(week)      # [B, L, d_model]
        year_emb = self.year_embed(year)      # [B, L, d_model]
        hol_emb = self.holiday_embed(is_holiday)  # [B, L, d_model]

        x = x + week_emb + year_emb + hol_emb
        x = self.dropout(x)

        # 3) Build static / hierarchical embeddings and FiLM params
        store_e = self.store_embed(store_idx)           # [B, d_store]
        dept_e = self.dept_embed(dept_idx)              # [B, d_dept]
        type_e = self.store_type_embed(store_type_idx)  # [B, d_type]
        size_e = self.size_mlp(size_norm)               # [B, d_size]

        static_vec = torch.cat([store_e, dept_e, type_e, size_e], dim=-1)  # [B, static_total_dim]
        film_params = self.static_to_film(static_vec)  # [B, 2 * d_model]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # each [B, d_model]

        # 4) Pass through holiday-aware Transformer layers
        for layer in self.layers:
            x = layer(x, is_holiday=is_holiday, gamma=gamma, beta=beta)

        # 5) Use the last timestep representation to predict residual correction
        last_hidden = x[:, -1, :]  # [B, d_model]
        out = self.output_head(last_hidden).squeeze(-1)  # [B]

        return out
