import torch
from torch import nn
import math
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    """
    A simple conv block: Conv1d -> BatchNorm1d -> Activation -> (optional) Residual
    Input/Output shape for conv1d: [B, C, T]
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=None, use_residual=False):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.activation = nn.ReLU()
        self.use_residual = use_residual and (in_ch == out_ch) and (stride == 1)

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        if self.use_residual:
            out = out + x
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]

class CNNTransformer(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,

        # conv config
        conv_channels=(128, 256),
        conv_kernel_sizes=(5, 5),
        conv_strides=(2, 2),  # downsampling by 2 each block
        conv_residual=(False, True),

        # transformer config
        n_layers=3,
        n_heads=8,
        dim_feedforward=2048,
        trans_dropout=0.1,
        input_dropout=0.0,
        activation="gelu",
        max_len=5000,
    ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of features, number of units for linear layer
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes
        dim_feedforward (int)  - dimensionality of hidden units in each transformer layer
        trans_dropout (float)  - percentage of units to dropout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers
        activation (str)       - the activation function used for a transformer layer
        '''
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh
        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList([nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)])
        self.day_biases = nn.ParameterList([nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)])
        self.day_layer_dropout = nn.Dropout(input_dropout)

        # 1d conv front-end
        conv_blocks = []
        in_ch = neural_dim
        for i, (out_ch, k, s, res) in enumerate(zip(conv_channels, conv_kernel_sizes, conv_strides, conv_residual)):
            conv_blocks.append(ConvBlock1D(in_ch, out_ch, kernel_size=k, stride=s, padding=(k-1)//2, use_residual=res))
            in_ch = out_ch
        self.conv_frontend = nn.Sequential(*conv_blocks)
        self.conv_output_channels = in_ch  # final channel size

        # Project conv features to transformer d_model
        self.input_proj = nn.Linear(self.conv_output_channels, n_units)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(n_units, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_units,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            batch_first=True,
            activation=activation,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # LayerNorm before head
        self.final_ln = nn.LayerNorm(n_units)
        self.out = nn.Linear(n_units, n_classes)

        # Explicit initialization call
        self._init_weights()

    def forward(self, x, day_idx, states=None, return_state=False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexes corresponding to the day of each example in the batch x.
        '''
        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the output of the day specific layer
        if self.day_layer_dropout.p > 0:
            x = self.day_layer_dropout(x)

        # Pass through convs: conv1d expects [B, C, T]
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.conv_frontend(x)  # [B, C_out, T_down]
        x = x.permute(0, 2, 1)  # transpose to [B, T_down, C_out]

        # Project to transformer dim
        x = self.input_proj(x)  # [B, T_down, n_units]
        x = F.gelu(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.encoder(x) # [B, T', n_units]

        # Final normalization
        x = self.final_ln(x)

        # CTC logits
        logits = self.out(x)  # [B, T', n_classes]

        # compatibility with evaluation script
        if return_state:
            return logits, None  # no recurrent state for CNN
        else:
            return logits


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)