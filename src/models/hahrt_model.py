# src/models/hahrt_model.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class HolidayAwareEncoderLayer(nn.Module):
    """
    Wraps a standard TransformerEncoderLayer but:
    - Applies a learnable holiday gate to holiday timesteps
    - Applies FiLM-style (gamma, beta) adapters per (store, dept)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # Single learnable gain; constrained to be >= 0 via ReLU in forward
        self.holiday_gain = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        src: torch.Tensor,
        is_holiday: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, seq_len, d_model]
            is_holiday: [batch, seq_len] bool or int {0,1}
            gamma: [batch, d_model]
            beta: [batch, d_model]
            src_key_padding_mask: optional [batch, seq_len] mask
        """
        out = self.layer(src, src_key_padding_mask=src_key_padding_mask)

        if is_holiday is not None:
            # Gate holiday positions: h_t <- h_t * (1 + gain) when is_holiday=1
            gain = torch.relu(self.holiday_gain)
            gate = 1.0 + gain * is_holiday.unsqueeze(-1).to(out.dtype)
            out = out * gate

        if gamma is not None and beta is not None:
            # FiLM-style adapter: h <- h * (1 + gamma) + beta
            out = out * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        return out


class HAHRTModel(nn.Module):
    """
    Holiday-Aware Hierarchical Residual Transformer (HAHRT).

    - Input is a sequence of residual + covariate features per (store, dept)
    - Outputs a residual forecast for the next step
    """

    def __init__(
        self,
        input_dim: int,
        num_stores: int,
        num_depts: int,
        num_store_types: int,
        max_week_of_year: int,
        max_year_index: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_static: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Project continuous covariates to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Time embeddings (multi-scale)
        self.week_embed = nn.Embedding(max_week_of_year + 1, d_model)
        self.year_embed = nn.Embedding(max_year_index + 1, d_model)

        # Holiday embedding: 0 = non-holiday, 1 = holiday
        self.holiday_embed = nn.Embedding(2, d_model)

        # Static identity embeddings
        self.store_embed = nn.Embedding(num_stores, d_static)
        self.dept_embed = nn.Embedding(num_depts, d_static)
        self.store_type_embed = nn.Embedding(num_store_types, d_static)

        # Store size (normalized scalar) projected to static dim
        self.size_mlp = nn.Sequential(
            nn.Linear(1, d_static),
            nn.GELU(),
            nn.Linear(d_static, d_static),
        )

        # Combine static embeddings -> static context vector
        self.static_proj = nn.Linear(d_static, d_model)

        # Hierarchical adapter: maps static context -> gamma, beta
        self.adapter_mlp = nn.Sequential(
            nn.Linear(d_static, d_static),
            nn.GELU(),
            nn.Linear(d_static, 2 * d_model),
        )

        # Stack of holiday-aware encoder layers
        self.layers = nn.ModuleList(
            [
                HolidayAwareEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Output head: use last token representation to predict residual
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1 and "embed" not in name:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x_cont: torch.Tensor,
        week_of_year: torch.Tensor,
        year_idx: torch.Tensor,
        is_holiday: torch.Tensor,
        store_idx: torch.Tensor,
        dept_idx: torch.Tensor,
        store_type_idx: torch.Tensor,
        size_norm: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_cont: [batch, seq_len, input_dim] continuous covariates
            week_of_year: [batch, seq_len] int in [0, max_week_of_year]
            year_idx: [batch, seq_len] int in [0, max_year_index]
            is_holiday: [batch, seq_len] {0,1}
            store_idx: [batch] int
            dept_idx: [batch] int
            store_type_idx: [batch] int
            size_norm: [batch] float (already normalized)

        Returns:
            residual_pred: [batch, 1]
        """
        device = x_cont.device
        batch_size, seq_len, _ = x_cont.shape

        # Base projection of continuous covariates
        h = self.input_proj(x_cont)  # [B, L, d_model]

        # Time & holiday embeddings
        week_emb = self.week_embed(week_of_year)  # [B, L, d_model]
        year_emb = self.year_embed(year_idx)  # [B, L, d_model]
        holiday_emb = self.holiday_embed(is_holiday.long())  # [B, L, d_model]

        # Static identity embeddings
        store_e = self.store_embed(store_idx)  # [B, d_static]
        dept_e = self.dept_embed(dept_idx)  # [B, d_static]
        store_type_e = self.store_type_embed(store_type_idx)  # [B, d_static]

        size_norm = size_norm.view(batch_size, 1)
        size_e = self.size_mlp(size_norm)  # [B, d_static]

        static_context = store_e + dept_e + store_type_e + size_e  # [B, d_static]

        # Project static context into model space and broadcast
        static_ctx_proj = self.static_proj(static_context)  # [B, d_model]
        static_ctx_proj = static_ctx_proj.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine everything
        h = h + week_emb + year_emb + holiday_emb + static_ctx_proj
        h = self.dropout(h)

        # Compute FiLM adapter params from static context
        adapter_params = self.adapter_mlp(static_context)  # [B, 2*d_model]
        gamma, beta = adapter_params.chunk(2, dim=-1)  # [B, d_model] each

        # Pass through holiday-aware encoder stack
        for layer in self.layers:
            h = layer(
                h,
                is_holiday=is_holiday,
                gamma=gamma,
                beta=beta,
                src_key_padding_mask=src_key_padding_mask,
            )

        # Use the last timestep representation to predict residual
        last_h = h[:, -1, :]  # [B, d_model]
        residual_pred = self.output_head(last_h)  # [B, 1]
        return residual_pred
