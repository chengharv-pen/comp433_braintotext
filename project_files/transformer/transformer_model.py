import torch 
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, neural_dim, n_classes, n_days, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, input_dropout=0.0, patch_size=0, patch_stride=0):
        super(TransformerModel, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_days = n_days
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_dropout = input_dropout

        # Day-specific input layers (keeping from RNN)
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList([nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)])
        self.day_biases = nn.ParameterList([nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)])
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # Projection to d_model
        self.input_proj = nn.Linear(self.input_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        # Output embedding (for decoder input)
        # We need an embedding layer for the target tokens
        # n_classes is phonemes. We need to add SOS, EOS.
        # Assuming n_classes passed here is the number of phonemes (41).
        # We will use:
        # 0: PAD
        # 1..41: Phonemes
        # 42: SOS
        # 43: EOS
        self.vocab_size = n_classes + 3 # 0 to n_classes+2
        self.embedding = nn.Embedding(self.vocab_size, d_model)

        # Output projection
        self.out = nn.Linear(d_model, self.vocab_size)

        self.d_model = d_model

    def forward(self, src, day_idx, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        # src: [batch, time, features]
        # day_idx: [batch]
        # tgt: [batch, tgt_len] (indices)

        # Day-specific layer
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        src = torch.einsum("btd,bdk->btk", src, day_weights) + day_biases
        src = self.day_layer_activation(src)
        if self.input_dropout > 0:
            src = self.day_layer_dropout(src)

        # Patching
        if self.patch_size > 0:
            src = src.unsqueeze(1)
            src = src.permute(0, 3, 1, 2)
            src_unfold = src.unfold(3, self.patch_size, self.patch_stride)
            src_unfold = src_unfold.squeeze(2)
            src_unfold = src_unfold.permute(0, 2, 3, 1)
            src = src_unfold.reshape(src.size(0), src_unfold.size(1), -1)

        # Project to d_model
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Decoder input embedding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)

        # Transformer
        output = self.transformer(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_mask=tgt_mask)

        # Output logits
        logits = self.out(output)
        return logits
