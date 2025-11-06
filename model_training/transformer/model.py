import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    pe: Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Fixed: float dtype, shapes
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for batch broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] (batch_first=True)
        """
        x = x * math.sqrt(self.pe.size(2))
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BasicTransformer(nn.Module):
    def __init__(
        self,
        src_d_model: int,
        tgt_vocab: int,
        d_model: int,
        nhead: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_projection = nn.Linear(
            src_d_model, d_model
        )  # Project raw 512 to d_model if !=512
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)  # IDs -> vectors
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab)  # Hidden -> logits

    def generate_mask(self, sz) -> Tensor:
        # Causal mask for decoder self-attn
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            src: [batch, src_seq_len, 512] raw neural bins
            tgt: [batch, tgt_seq_len] phenomena IDs (teacher-forcing: input shifted, predict next)
            Masks: [batch, seq_len] bool, True for pad positions
        Returns:
            logits: [batch, tgt_seq_len, tgt_vocab]
        """
        # Project src if needed, add PE
        src_emb = self.pe(self.src_projection(src))
        # Embed tgt (shifted in training loop), add PE
        tgt_emb = self.pe(self.tgt_embedding(tgt))

        # Masks
        tgt_mask = (
            self.generate_mask(tgt.size(1)).to(tgt.device) if tgt.size(1) > 0 else None
        )

        # Full pass: encoder(src) -> memory, decoder(tgt, memory)
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.fc_out(output)

    def generate(self, src, max_new_tokens: int, sos_id: int = 0, eos_id: int = -1, pad_id: int = 1):
        self.eval()
        device = next(self.parameters()).device
        src = src.to(device)
        batch_size = src.size(0)
        tgt = torch.full(
            (batch_size, 1), sos_id, dtype=torch.long, device=device
        )  # Start with SOS
        for _ in range(max_new_tokens):
            tgt_mask = self.generate_mask(tgt.size(1)).to(device)
            logits = self(src, tgt, tgt_key_padding_mask=None, src_padding_mask=None)  # No src pad for infer
            next_token = logits[:, -1, :].argmax(-1).unsqueeze(1)  # Greedy
            tgt = torch.cat([tgt, next_token], dim=1)
            if eos_id > 0 and (next_token == eos_id).all():
                break
        return tgt  # [batch, generated_len]