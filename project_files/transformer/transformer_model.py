import torch
from torch import nn
import math

class CustomTransformer(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        n_layers = 6,
        n_heads = 8,
        dim_feedforward = 2048,
        trans_dropout = 0.0,
        input_dropout = 0.0,
        patch_size = 0,
        patch_stride = 0,
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
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input
        '''
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes

        self.trans_dropout = trans_dropout
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if patch_size > 0:
            self.input_size *= patch_size

        # NEW: Project into transformer dimension
        self.input_proj = nn.Linear(self.input_size, n_units)

        # NEW: Positional Encoding
        self.pos_encoding = PositionalEncoding(n_units)

        # NEW: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_units,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.out = nn.Linear(self.n_units, self.n_classes)

        # Explicit initialization call
        self._init_weights()

    def forward(self, x, day_idx):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexes corresponding to the day of each example in the batch x.
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0:
            x = x.unsqueeze(1)  # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)  # [batches, feature_dim, 1, timesteps]

            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size,
                                self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]

            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)  # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Project to d_model
        x = self.input_proj(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.encoder(x)

        # Logits for CTC
        logits = self.out(x) # [batch, time, n_classes]
        return logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        return x + self.pe[:, : x.size(1)]