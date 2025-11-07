import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def subsequent_mask(sz: int) -> torch.Tensor:
    """Mask out subsequent positions (size: 1 x sz x sz)."""
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
    return ~mask  # True where allowed


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fast path: use PyTorch's fused SDPA (enables Flash/Memory-Efficient kernels)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = None
            if mask is not None:
                # Our convention: True means allowed. SDPA expects True=masked.
                attn_mask = ~mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
            return out, None

        # Fallback manual attention
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = attn @ v
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)  # b, h, t, d_k
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # context: if None -> self-attention
        q = self._shape(self.w_q(x))
        k_in = x if context is None else context
        k = self._shape(self.w_k(k_in))
        v = self._shape(self.w_v(k_in))

        if mask is not None:
            # Expect mask broadcastable to (b, h, t_q, t_k)
            if mask.dim() == 2:
                mask = mask[:, None, :, :]  # b,1,t,t
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]  # b,1,t,t_k

        out, _ = self.attn(q, k, v, mask)
        b, h, t, d_k = out.shape
        out = out.transpose(1, 2).contiguous().view(b, t, h * d_k)
        out = self.w_o(out)
        return self.dropout(out)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1, T, C
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), mask=tgt_mask)
        x = x + self.cross_attn(self.ln2(x), context=enc, mask=src_mask)
        x = x + self.ffn(self.ln3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)


class TransformerLanguageModel(nn.Module):
    """Encoder-only Transformer for character-level language modeling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 10000,
    ):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.tok_embedding(x)
        h = self.pos_encoding(h)
        h = self.encoder(h, src_mask)
        logits = self.lm_head(h)
        return logits


__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionwiseFFN",
    "PositionalEncoding",
    "Encoder",
    "Decoder",
    "TransformerLanguageModel",
    "subsequent_mask",
]
