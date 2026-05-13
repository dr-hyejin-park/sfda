"""
TransTab-style vocabulary, feature schema, and schema encoding.

Key classes
-----------
FeatureSchema    – declares which columns are numerical / binary / categorical
                   and the string names of categorical values.
TransTabVocab    – word-level vocabulary built from column names and categorical
                   value strings (shared between both domains).
SchemaEncoding   – pre-computed word-ID tensors for every column name and every
                   categorical value; device-moveable; subset-able.

The shared vocabulary means the same word embedding matrix handles column
names from both source and target domains, enabling zero-shot generalisation
to new feature sets as described in the TransTab paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ──────────────────────────── FeatureSchema ───────────────────────────────────

@dataclass
class FeatureSchema:
    """Declares feature types and categorical value string names for one domain."""
    numerical_cols:    list[str]
    binary_cols:       list[str]
    categorical_cols:  list[str]
    # col → ordered list of string names for each category code
    # e.g. {"card_tier": ["standard", "gold", "platinum"]}
    cat_value_strings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def all_cols(self) -> list[str]:
        return self.numerical_cols + self.binary_cols + self.categorical_cols

    @property
    def n_num(self) -> int:
        return len(self.numerical_cols)

    @property
    def n_bin(self) -> int:
        return len(self.binary_cols)

    @property
    def n_cat(self) -> int:
        return len(self.categorical_cols)

    @property
    def n_total(self) -> int:
        return self.n_num + self.n_bin + self.n_cat

    def cat_cardinality(self, col: str) -> int:
        return len(self.cat_value_strings.get(col, []))


# ──────────────────────────── TransTabVocab ───────────────────────────────────

class TransTabVocab:
    """
    Word-level vocabulary built from column names and categorical value strings.

    Column names (e.g. ``spend_grocery_monthly``) are split on underscores
    and each word is assigned a unique integer ID.  The same embedding matrix
    is reused for categorical value strings (e.g. ``["bronze", "silver", ...]``),
    enabling semantic generalisation across tables.

    Special tokens
    --------------
    [PAD]  (id=0) – padding to uniform sequence length
    [UNK]  (id=1) – unknown word at inference time
    """

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self) -> None:
        self._word2id: dict[str, int] = {"[PAD]": 0, "[UNK]": 1}

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        column_names: list[str],
        cat_value_strings: dict[str, list[str]] | None = None,
    ) -> "TransTabVocab":
        """Build vocabulary from column names + categorical value strings."""
        texts: list[str] = list(column_names)
        if cat_value_strings:
            for vals in cat_value_strings.values():
                texts.extend(str(v) for v in vals)
        for word in sorted({w for t in texts for w in self._split(t)}):
            if word not in self._word2id:
                self._word2id[word] = len(self._word2id)
        return self

    def encode(self, text: str) -> list[int]:
        """Tokenize a text string to a list of word IDs."""
        return [self._word2id.get(w, self.UNK_ID) for w in self._split(text)]

    @property
    def size(self) -> int:
        return len(self._word2id)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split(text: str) -> list[str]:
        """Split on underscores/hyphens, lowercase, drop empty tokens."""
        return [w for w in text.lower().replace("-", "_").split("_") if w]


# ──────────────────────────── SchemaEncoding ──────────────────────────────────

class SchemaEncoding:
    """
    Pre-computed word-ID tensors for all features in a FeatureSchema.

    Stored as plain tensors (not nn.Buffers) so they can be moved to a device
    independently of the model and reused across batches with zero overhead.

    Attributes
    ----------
    {num,bin,cat}_col_word_ids   : (F_type, max_col_words) long – padded word IDs
    {num,bin,cat}_col_word_mask  : (F_type, max_col_words) bool  – True for real words
    cat_val_word_ids             : (total_cat_values, max_val_words) long
    cat_val_word_mask            : (total_cat_values, max_val_words) bool
    cat_val_offsets              : (F_cat,) long – offset into cat_val_word_ids per feature
    n_num, n_bin, n_cat          : int – number of features per type
    """

    def __init__(
        self,
        num_col_word_ids:  torch.Tensor,
        num_col_word_mask: torch.Tensor,
        bin_col_word_ids:  torch.Tensor,
        bin_col_word_mask: torch.Tensor,
        cat_col_word_ids:  torch.Tensor,
        cat_col_word_mask: torch.Tensor,
        cat_val_word_ids:  torch.Tensor,
        cat_val_word_mask: torch.Tensor,
        cat_val_offsets:   torch.Tensor,
        n_num: int,
        n_bin: int,
        n_cat: int,
    ) -> None:
        self.num_col_word_ids  = num_col_word_ids
        self.num_col_word_mask = num_col_word_mask
        self.bin_col_word_ids  = bin_col_word_ids
        self.bin_col_word_mask = bin_col_word_mask
        self.cat_col_word_ids  = cat_col_word_ids
        self.cat_col_word_mask = cat_col_word_mask
        self.cat_val_word_ids  = cat_val_word_ids
        self.cat_val_word_mask = cat_val_word_mask
        self.cat_val_offsets   = cat_val_offsets
        self.n_num = n_num
        self.n_bin = n_bin
        self.n_cat = n_cat

    # ── device movement ───────────────────────────────────────────────────────

    def to(self, device) -> "SchemaEncoding":
        """Return a new SchemaEncoding with all tensors on *device*."""
        enc = SchemaEncoding.__new__(SchemaEncoding)
        for attr in self._tensor_attrs():
            setattr(enc, attr, getattr(self, attr).to(device))
        enc.n_num = self.n_num
        enc.n_bin = self.n_bin
        enc.n_cat = self.n_cat
        return enc

    # ── subsetting ────────────────────────────────────────────────────────────

    def subset_global(self, global_indices: list[int]) -> "SchemaEncoding":
        """
        Return a SchemaEncoding containing only the listed global feature indices.

        Global feature ordering:
          [0 .. n_num-1]                        → numerical
          [n_num .. n_num+n_bin-1]              → binary
          [n_num+n_bin .. n_num+n_bin+n_cat-1]  → categorical
        """
        n_num, n_bin = self.n_num, self.n_bin

        num_local = sorted(i       for i in global_indices if i < n_num)
        bin_local = sorted(i - n_num        for i in global_indices if n_num <= i < n_num + n_bin)
        cat_local = sorted(i - n_num - n_bin for i in global_indices if i >= n_num + n_bin)

        def _sel(tensor: torch.Tensor, local_idx: list[int]) -> torch.Tensor:
            if not local_idx:
                return torch.zeros(0, tensor.shape[1], dtype=tensor.dtype,
                                   device=tensor.device)
            return tensor[local_idx]

        # Rebuild cat_val_offsets for the selected categorical features
        if cat_local and self.n_cat > 0:
            new_cat_val_offsets = self.cat_val_offsets[cat_local]
        else:
            new_cat_val_offsets = torch.zeros(0, dtype=torch.long,
                                               device=self.cat_val_offsets.device)

        return SchemaEncoding(
            num_col_word_ids  = _sel(self.num_col_word_ids, num_local),
            num_col_word_mask = _sel(self.num_col_word_mask, num_local),
            bin_col_word_ids  = _sel(self.bin_col_word_ids, bin_local),
            bin_col_word_mask = _sel(self.bin_col_word_mask, bin_local),
            cat_col_word_ids  = _sel(self.cat_col_word_ids, cat_local),
            cat_col_word_mask = _sel(self.cat_col_word_mask, cat_local),
            cat_val_word_ids  = self.cat_val_word_ids,   # full value table preserved
            cat_val_word_mask = self.cat_val_word_mask,
            cat_val_offsets   = new_cat_val_offsets,
            n_num = len(num_local),
            n_bin = len(bin_local),
            n_cat = len(cat_local),
        )

    def _tensor_attrs(self) -> list[str]:
        return [
            "num_col_word_ids", "num_col_word_mask",
            "bin_col_word_ids", "bin_col_word_mask",
            "cat_col_word_ids", "cat_col_word_mask",
            "cat_val_word_ids", "cat_val_word_mask",
            "cat_val_offsets",
        ]


# ──────────────────────────── factory ─────────────────────────────────────────

def _encode_names(names: list[str], vocab: TransTabVocab
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of text strings to padded word-ID tensors."""
    if not names:
        return (torch.zeros(0, 1, dtype=torch.long),
                torch.zeros(0, 1, dtype=torch.bool))
    encoded = [vocab.encode(n) for n in names]
    max_len = max(len(e) for e in encoded)
    padded  = [e + [vocab.PAD_ID] * (max_len - len(e)) for e in encoded]
    ids     = torch.tensor(padded, dtype=torch.long)
    mask    = (ids != vocab.PAD_ID)
    return ids, mask


def encode_schema(schema: FeatureSchema, vocab: TransTabVocab) -> SchemaEncoding:
    """
    Pre-compute word-ID tensors for every column name and every categorical
    value in *schema* using *vocab*.
    """
    num_ids, num_mask = _encode_names(schema.numerical_cols,   vocab)
    bin_ids, bin_mask = _encode_names(schema.binary_cols,      vocab)
    cat_ids, cat_mask = _encode_names(schema.categorical_cols, vocab)

    # Categorical value word IDs: flatten all category values across features.
    # cat_val_offsets[f] = index of the first value of feature f in the flat table.
    all_val_strings: list[str] = []
    offsets: list[int] = []
    for col in schema.categorical_cols:
        offsets.append(len(all_val_strings))
        vals = schema.cat_value_strings.get(col, [])
        all_val_strings.extend(vals)

    val_ids, val_mask = _encode_names(all_val_strings, vocab)
    cat_val_offsets = torch.tensor(offsets, dtype=torch.long) if offsets else \
                      torch.zeros(0, dtype=torch.long)

    return SchemaEncoding(
        num_col_word_ids  = num_ids,
        num_col_word_mask = num_mask,
        bin_col_word_ids  = bin_ids,
        bin_col_word_mask = bin_mask,
        cat_col_word_ids  = cat_ids,
        cat_col_word_mask = cat_mask,
        cat_val_word_ids  = val_ids,
        cat_val_word_mask = val_mask,
        cat_val_offsets   = cat_val_offsets,
        n_num = schema.n_num,
        n_bin = schema.n_bin,
        n_cat = schema.n_cat,
    )


def build_vocab(*schemas: FeatureSchema) -> TransTabVocab:
    """Build a shared vocabulary from all provided domain schemas."""
    all_cols: list[str] = []
    all_cat_vals: dict[str, list[str]] = {}
    for s in schemas:
        all_cols.extend(s.all_cols)
        all_cat_vals.update(s.cat_value_strings)
    return TransTabVocab().fit(all_cols, all_cat_vals)


# ──────────────────────────── feature partitioning ────────────────────────────

COMMON_FEATURE_PATTERNS = [
    "total_monthly_spend", "avg_transaction_amount", "n_transactions_monthly",
    "merchant_diversity_cnt", "visited_city_cnt",
    "weekend_spend_ratio", "evening_spend_ratio",
    "premium_merchant_ratio", "luxury", "travel",
    "seasonal_ratio_", "customer_tenure_months", "recency_days",
    "local_merchant_ratio", "cashback_usage_rate",
    "installment_ratio", "refund_rate", "chain_store_ratio",
]


def identify_common_features(
    src_schema: FeatureSchema,
) -> tuple[list[str], list[str]]:
    """
    Split source *numerical* columns into common and source-specific lists.
    Binary and categorical source features are always source-specific.
    Returns (common_num_cols, specific_num_cols).
    """
    common, specific = [], []
    for col in src_schema.numerical_cols:
        matched = any(pat in col for pat in COMMON_FEATURE_PATTERNS)
        (common if matched else specific).append(col)
    return common, specific


def vertical_partition(
    n_features: int,
    n_partitions: int = 4,
    random_seed: int = 0,
) -> list[list[int]]:
    """
    Randomly split global feature indices 0..n_features-1 into
    ``n_partitions`` disjoint subsets.
    """
    rng = np.random.RandomState(random_seed)
    idx = list(range(n_features))
    rng.shuffle(idx)
    return [list(part) for part in np.array_split(idx, n_partitions)]


# ──────────────────────────── batch helpers ───────────────────────────────────

def arrays_to_tensors(
    x_num: np.ndarray | None,
    x_bin: np.ndarray | None,
    x_cat: np.ndarray | None,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays (or None) to float32 / long tensors.
    Missing types become zero-column tensors so DataLoader works uniformly.
    """
    t_num = (torch.tensor(x_num, dtype=torch.float32)
             if x_num is not None else torch.zeros(n, 0, dtype=torch.float32))
    t_bin = (torch.tensor(x_bin, dtype=torch.long)
             if x_bin is not None else torch.zeros(n, 0, dtype=torch.long))
    t_cat = (torch.tensor(x_cat, dtype=torch.long)
             if x_cat is not None else torch.zeros(n, 0, dtype=torch.long))
    return t_num, t_bin, t_cat


def unpack_batch(
    t_num: torch.Tensor,
    t_bin: torch.Tensor,
    t_cat: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Convert zero-column placeholder tensors back to None."""
    return (
        t_num if t_num.size(1) > 0 else None,
        t_bin if t_bin.size(1) > 0 else None,
        t_cat if t_cat.size(1) > 0 else None,
    )


def global_partition_to_typed(
    global_indices: list[int],
    x_num: torch.Tensor | None,
    x_bin: torch.Tensor | None,
    x_cat: torch.Tensor | None,
    n_num: int,
    n_bin: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Given a list of global feature indices and the full (x_num, x_bin, x_cat)
    batch tensors, return the per-type sub-tensors for just that partition.
    """
    num_local = [i          for i in global_indices if i < n_num]
    bin_local = [i - n_num  for i in global_indices if n_num <= i < n_num + n_bin]
    cat_local = [i - n_num - n_bin for i in global_indices if i >= n_num + n_bin]

    p_num = x_num[:, num_local] if (num_local and x_num is not None) else None
    p_bin = x_bin[:, bin_local] if (bin_local and x_bin is not None) else None
    p_cat = x_cat[:, cat_local] if (cat_local and x_cat is not None) else None
    return p_num, p_bin, p_cat
