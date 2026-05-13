"""
TransTab-style Feature Transformer.

Tokenisation (per the TransTab paper)
--------------------------------------
Each feature produces one token:  token_i = col_name_emb(i)  +  value_emb(x_i)

Column-name embedding
  Column name (e.g. "spend_grocery_monthly") is split on underscores into
  words ["spend", "grocery", "monthly"].  Each word is looked up in a shared
  word embedding table and the results are mean-pooled.

Value embedding – differs by feature type
  Numerical  : W_num · x_i          (nn.Linear(1 → d_model))
  Binary     : E_bin[x_i]           (nn.Embedding(2, d_model))
  Categorical: mean-pool(word_emb(value_string))
               – the categorical value string (e.g. "gold") is tokenised
               and embedded with the SAME shared word embedding table,
               enabling semantic generalisation across tables.

The shared word embedding is the only component that links column names,
categorical values and domain vocabulary – this is the core idea of TransTab.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.preprocessing import SchemaEncoding


# ──────────────────────────── TransTab tokeniser ──────────────────────────────

class TransTabTokenizer(nn.Module):
    """
    Converts (x_num, x_bin, x_cat) + SchemaEncoding → (B, F_total, d_model) token sequence.

    Parameters
    ----------
    vocab_size : int   – size of the shared word vocabulary (from TransTabVocab)
    d_model    : int   – token / model dimensionality
    dropout    : float
    """

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        # Shared word embedding: used for column names AND categorical values
        self.word_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Numerical value encoder: scalar → d_model
        self.num_val_proj = nn.Linear(1, d_model, bias=False)
        # Binary value encoder: {0, 1} → d_model
        self.bin_val_embed = nn.Embedding(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.word_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.bin_val_embed.weight, std=0.02)

    # ── column-name embedding ─────────────────────────────────────────────────

    def _col_name_emb(
        self,
        col_word_ids:  torch.Tensor,   # (F, max_words)  long
        col_word_mask: torch.Tensor,   # (F, max_words)  bool
    ) -> torch.Tensor:                 # (F, d_model)
        """Mean-pool word embeddings to obtain one vector per column name."""
        emb  = self.word_embed(col_word_ids)                   # (F, W, d)
        mask = col_word_mask.float().unsqueeze(-1)             # (F, W, 1)
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1) # (F, d)

    # ── per-type tokenisation ─────────────────────────────────────────────────

    def _tokenize_numerical(
        self,
        x_num:    torch.Tensor,   # (B, F_num)
        col_emb:  torch.Tensor,   # (F_num, d_model)
    ) -> torch.Tensor:            # (B, F_num, d_model)
        val_emb = self.num_val_proj(x_num.unsqueeze(-1))    # (B, F_num, d)
        return self.layer_norm(val_emb + col_emb.unsqueeze(0))

    def _tokenize_binary(
        self,
        x_bin:   torch.Tensor,   # (B, F_bin) long {0,1}
        col_emb: torch.Tensor,   # (F_bin, d_model)
    ) -> torch.Tensor:           # (B, F_bin, d_model)
        val_emb = self.bin_val_embed(x_bin)                  # (B, F_bin, d)
        return self.layer_norm(val_emb + col_emb.unsqueeze(0))

    def _tokenize_categorical(
        self,
        x_cat:             torch.Tensor,  # (B, F_cat) long – category codes
        col_emb:           torch.Tensor,  # (F_cat, d_model)
        cat_val_word_ids:  torch.Tensor,  # (total_vals, max_val_words) long
        cat_val_word_mask: torch.Tensor,  # (total_vals, max_val_words) bool
        cat_val_offsets:   torch.Tensor,  # (F_cat,) long
    ) -> torch.Tensor:                    # (B, F_cat, d_model)
        """
        For each (sample, feature) pair, look up the category value string's
        word IDs and mean-pool via the SHARED word embedding.
        """
        # global_val_idx[b, f] = offset[f] + x_cat[b, f]  → index into cat_val table
        global_val_idx = cat_val_offsets.unsqueeze(0) + x_cat       # (B, F_cat)
        val_word_ids   = cat_val_word_ids[global_val_idx]             # (B, F_cat, W)
        val_word_mask  = cat_val_word_mask[global_val_idx]            # (B, F_cat, W)

        # Flatten (B*F_cat) for embedding lookup, then restore shape.
        # Use reshape (not view) because advanced indexing yields non-contiguous tensors.
        B, F_cat, W = val_word_ids.shape
        flat_ids  = val_word_ids.reshape(-1, W)                       # (B*F_cat, W)
        flat_mask = val_word_mask.float().reshape(-1, W, 1)           # (B*F_cat, W, 1)
        flat_emb  = self.word_embed(flat_ids)                         # (B*F_cat, W, d)
        val_emb   = (flat_emb * flat_mask).sum(1) / flat_mask.sum(1).clamp(min=1)
        val_emb   = val_emb.reshape(B, F_cat, -1)                     # (B, F_cat, d)

        return self.layer_norm(val_emb + col_emb.unsqueeze(0))

    # ── unified forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x_num: torch.Tensor | None,   # (B, F_num)  float32  or None
        x_bin: torch.Tensor | None,   # (B, F_bin)  long {0,1}  or None
        x_cat: torch.Tensor | None,   # (B, F_cat)  long (codes)  or None
        schema_enc: SchemaEncoding,
    ) -> torch.Tensor:                # (B, F_total, d_model)
        parts: list[torch.Tensor] = []

        if x_num is not None and x_num.size(1) > 0:
            col_emb = self._col_name_emb(
                schema_enc.num_col_word_ids, schema_enc.num_col_word_mask)
            parts.append(self._tokenize_numerical(x_num, col_emb))

        if x_bin is not None and x_bin.size(1) > 0:
            col_emb = self._col_name_emb(
                schema_enc.bin_col_word_ids, schema_enc.bin_col_word_mask)
            parts.append(self._tokenize_binary(x_bin, col_emb))

        if x_cat is not None and x_cat.size(1) > 0:
            col_emb = self._col_name_emb(
                schema_enc.cat_col_word_ids, schema_enc.cat_col_word_mask)
            parts.append(self._tokenize_categorical(
                x_cat, col_emb,
                schema_enc.cat_val_word_ids,
                schema_enc.cat_val_word_mask,
                schema_enc.cat_val_offsets,
            ))

        tokens = torch.cat(parts, dim=1)   # (B, F_total, d)
        return self.dropout(tokens)


# ──────────────────────────── FeatureTransformer ──────────────────────────────

class FeatureTransformer(nn.Module):
    """
    TransTab Feature Transformer.

    Architecture:  TransTabTokenizer → [CLS] prepended → TransformerEncoder
    Output:        CLS token representation (B, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 128,
        n_heads:    int   = 4,
        n_layers:   int   = 4,
        d_ff:       int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.tokenizer = TransTabTokenizer(vocab_size, d_model, dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm    = nn.LayerNorm(d_model)

    def forward(
        self,
        x_num: torch.Tensor | None,
        x_bin: torch.Tensor | None,
        x_cat: torch.Tensor | None,
        schema_enc: SchemaEncoding,
    ) -> torch.Tensor:               # (B, d_model)
        B = _batch_size(x_num, x_bin, x_cat)
        feat_tokens = self.tokenizer(x_num, x_bin, x_cat, schema_enc)  # (B, F, d)
        cls  = self.cls_token.expand(B, -1, -1)                        # (B, 1, d)
        seq  = torch.cat([cls, feat_tokens], dim=1)                    # (B, F+1, d)
        out  = self.transformer(seq)
        return self.out_norm(out[:, 0])                                 # (B, d)

    def encode_partitions(
        self,
        x_num:       torch.Tensor | None,
        x_bin:       torch.Tensor | None,
        x_cat:       torch.Tensor | None,
        partitions:  list[list[int]],     # global feature index lists
        schema_enc:  SchemaEncoding,
    ) -> list[torch.Tensor]:              # list of (B, d_model)
        """
        Encode each vertical partition separately (used in contrastive Step 1).
        A partition is a list of global feature indices; the method extracts
        the relevant sub-tensors and sub-SchemaEncoding for each partition.
        """
        from data.preprocessing import global_partition_to_typed
        n_num = schema_enc.n_num
        n_bin = schema_enc.n_bin
        reps: list[torch.Tensor] = []
        for part_global in partitions:
            p_num, p_bin, p_cat = global_partition_to_typed(
                part_global, x_num, x_bin, x_cat, n_num, n_bin
            )
            p_schema = schema_enc.subset_global(part_global)
            reps.append(self.forward(p_num, p_bin, p_cat, p_schema))
        return reps


# ──────────────────────────── BottleneckLayer ─────────────────────────────────

class BottleneckLayer(nn.Module):
    """Compress CLS representation before the prediction head."""

    def __init__(self, d_model: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────── MultiLabelHead ──────────────────────────────────

class MultiLabelHead(nn.Module):
    """
    Dual-head output:
    * clf_logits  – binary logit per index (high / low split at 5.5)
    * reg_out     – continuous prediction in [1, 10]
    """

    def __init__(self, input_dim: int, n_labels: int = 10, dropout: float = 0.1):
        super().__init__()
        hidden = input_dim * 2
        self.shared   = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.clf_head = nn.Linear(hidden, n_labels)
        self.reg_head = nn.Linear(hidden, n_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h          = self.shared(x)
        clf_logits = self.clf_head(h)
        reg_out    = torch.sigmoid(self.reg_head(h)) * 9.0 + 1.0   # [1, 10]
        return clf_logits, reg_out


# ──────────────────────────── DomainAdaptationModel ───────────────────────────

class DomainAdaptationModel(nn.Module):
    """
    Full model: FeatureTransformer → BottleneckLayer → MultiLabelHead.

    All three steps (2, 3, 4) share this architecture.  The shared
    word embedding inside FeatureTransformer spans both source and target
    vocabularies, so NO embedding extension is needed when transferring to
    the target domain – simply pass the target SchemaEncoding to forward().
    """

    def __init__(
        self,
        vocab_size:     int,
        d_model:        int   = 128,
        n_heads:        int   = 4,
        n_layers:       int   = 4,
        d_ff:           int   = 256,
        dropout:        float = 0.1,
        bottleneck_dim: int   = 64,
        n_labels:       int   = 10,
    ):
        super().__init__()
        self.feature_transformer = FeatureTransformer(
            vocab_size, d_model, n_heads, n_layers, d_ff, dropout
        )
        self.bottleneck = BottleneckLayer(d_model, bottleneck_dim, dropout)
        self.head       = MultiLabelHead(bottleneck_dim, n_labels, dropout)

    def forward(
        self,
        x_num: torch.Tensor | None,
        x_bin: torch.Tensor | None,
        x_cat: torch.Tensor | None,
        schema_enc: SchemaEncoding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_transformer(x_num, x_bin, x_cat, schema_enc)
        b = self.bottleneck(h)
        return self.head(b)

    def encode(
        self,
        x_num: torch.Tensor | None,
        x_bin: torch.Tensor | None,
        x_cat: torch.Tensor | None,
        schema_enc: SchemaEncoding,
    ) -> torch.Tensor:
        """Return bottleneck representation (without prediction heads)."""
        h = self.feature_transformer(x_num, x_bin, x_cat, schema_enc)
        return self.bottleneck(h)


# ──────────────────────────── utilities ───────────────────────────────────────

def _batch_size(
    x_num: torch.Tensor | None,
    x_bin: torch.Tensor | None,
    x_cat: torch.Tensor | None,
) -> int:
    for t in (x_num, x_bin, x_cat):
        if t is not None:
            return t.size(0)
    raise ValueError("All feature tensors are None.")
