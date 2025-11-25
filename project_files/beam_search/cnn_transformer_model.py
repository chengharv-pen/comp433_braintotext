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


class CNNTransformerForGeneration(nn.Module):
    """
    CNN-Transformer model with custom beam search implementation.
    """
    
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
        enc_layers=3,
        dec_layers=2,
        n_heads=8,
        dim_feedforward=2048,
        trans_dropout=0.1,
        input_dropout=0.0,
        activation="gelu",
        max_len=10000,
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # INTRODUCING TRANSFORMER DECODER LAYERS
        # WE DO NOT CHANGE n_classes, we just add 2 internal slots
        # Slot 41 = SOS
        # Slot 42 = EOS
        self.internal_vocab_size = n_classes + 2
        
        # Special token IDs
        self.pad_token_id = 0  # PAD/BLANK
        self.bos_token_id = 41  # SOS
        self.eos_token_id = 42  # EOS

        # Embedding for the target sequence (class indices)
        self.tgt_embedding = nn.Embedding(self.internal_vocab_size, n_units)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_units,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            batch_first=True,
            activation=activation
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # LayerNorm before head
        self.final_ln = nn.LayerNorm(n_units)
        self.out = nn.Linear(n_units, self.internal_vocab_size) # output size will also be 43

        # Explicit initialization call
        self._init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x, day_idx):
        """
        Encode neural data to memory representation.
        
        Args:
            x: [B, T, neural_dim] - input neural features
            day_idx: [B] - day indices for day-specific layers
            
        Returns:
            memory: [B, T_down, n_units] - encoded representation
        """
        # Apply day-specific layer
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

        # Encode
        memory = self.encoder(x)  # [B, T_down, n_units]
        
        return memory

    def forward(self, x=None, day_idx=None, states=None, return_state=False, tgt=None, 
                encoder_outputs=None, decoder_input_ids=None, **kwargs):
        '''
        Forward pass supporting both training and HuggingFace generation.
        
        For training:
            x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
            day_idx  (tensor)  - tensor which is a list of day indexes corresponding to the day of each example in the batch x.
            tgt      (tensor)  - target sequence for teacher forcing
            
        For HuggingFace generation:
            encoder_outputs - tuple containing (memory,) from encode()
            decoder_input_ids - input token ids for the decoder
        '''
        # HuggingFace generation path
        if encoder_outputs is not None:
            # Extract memory from encoder_outputs
            if isinstance(encoder_outputs, tuple):
                memory = encoder_outputs[0]
            else:
                memory = encoder_outputs
            
            # decoder_input_ids is the target sequence
            tgt = decoder_input_ids
            
            # Embed and decode
            tgt_emb = self.tgt_embedding(tgt)  # [B, Seq_Len, n_units]
            tgt_emb = self.pos_encoding(tgt_emb)

            # Generate Causal Mask
            seq_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)

            # Transformer Decoder
            x = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )

            # Final Projection
            x = self.final_ln(x)
            logits = self.out(x)  # [B, Seq_Len, vocab_size]
            
            return logits
        
        # Standard training path
        if x is None:
            raise ValueError("Either 'x' or 'encoder_outputs' must be provided.")
            
        # --------------------------
        # Part A: Encode Neural Data
        # --------------------------
        memory = self.encode(x, day_idx)

        # --------------------------
        # Part B: Decode to Sequence
        # --------------------------

        if tgt is None:
            raise ValueError("In Encoder-Decoder training, 'tgt' (target indices) must be provided.")

        # 1. Prepare Target (Embed + PE)
        tgt_emb = self.tgt_embedding(tgt)  # [B, Seq_Len, n_units]
        tgt_emb = self.pos_encoding(tgt_emb)

        # 2. Generate Causal Mask
        # This prevents the decoder from peeking at future tokens
        seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # 3. Transformer Decoder
        # output takes info from 'tgt' and cross-attends to 'memory'
        x = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [B, Seq_Len, n_units]

        # 4. Final Projection
        x = self.final_ln(x)
        logits = self.out(x)  # [B, Seq_Len, n_classes]

        # compatibility with evaluation script
        if return_state:
            return logits, None  # no recurrent state for CNN
        else:
            return logits

    def prepare_inputs_for_generation(self, decoder_input_ids, encoder_outputs=None, **kwargs):
        """
        Prepare inputs for HuggingFace's generate() method.
        """
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
        }
    
    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def beam_search_decode(self, memory, num_beams=5, max_length=100, length_penalty=1.0):
        """
        Perform beam search decoding.
        
        Custom implementation that maintains num_beams hypotheses per batch element,
        expanding and pruning at each step.
        
        Args:
            memory: [B, T, D] - encoder output
            num_beams: number of beams for beam search
            max_length: maximum sequence length
            length_penalty: length penalty for beam search (>1 favors longer sequences)
            
        Returns:
            generated: [B, max_seq_len] - best generated token ids for each batch element
        """
        batch_size = memory.size(0)
        device = memory.device
        vocab_size = self.internal_vocab_size
        
        # Expand memory for beam search: [B, T, D] -> [B * num_beams, T, D]
        memory_expanded = memory.unsqueeze(1).repeat(1, num_beams, 1, 1)
        memory_expanded = memory_expanded.view(batch_size * num_beams, memory.size(1), memory.size(2))
        
        # Initialize sequences with SOS token: [B * num_beams, 1]
        sequences = torch.full(
            (batch_size * num_beams, 1), 
            self.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Initialize scores: [B * num_beams]
        # First beam of each batch has score 0, others have -inf so they don't get selected initially
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = float('-inf')
        beam_scores = beam_scores.view(-1)  # [B * num_beams]
        
        # Track which beams are done (hit EOS)
        done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        
        for step in range(max_length - 1):
            # Get current sequence length
            cur_len = sequences.size(1)
            
            # Generate causal mask
            tgt_mask = self.generate_square_subsequent_mask(cur_len).to(device)
            
            # Embed and decode
            tgt_emb = self.tgt_embedding(sequences)
            tgt_emb = self.pos_encoding(tgt_emb)
            
            out = self.decoder(tgt=tgt_emb, memory=memory_expanded, tgt_mask=tgt_mask)
            out = self.final_ln(out)
            logits = self.out(out)  # [B * num_beams, cur_len, vocab_size]
            
            # Get logits for next token
            next_token_logits = logits[:, -1, :]  # [B * num_beams, vocab_size]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # [B * num_beams, vocab_size]
            
            # Add current beam scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)  # [B * num_beams, vocab_size]
            
            # For finished beams, only allow PAD token
            next_token_scores[done] = float('-inf')
            next_token_scores[done, self.pad_token_id] = beam_scores[done]
            
            # Reshape for selecting top-k across all beams for each batch element
            # [B, num_beams * vocab_size]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Select top 2 * num_beams to account for EOS
            next_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Compute beam indices and token indices
            next_beam_indices = next_tokens // vocab_size  # Which beam
            next_token_indices = next_tokens % vocab_size  # Which token
            
            # Build next sequences
            # Gather the sequences from the selected beams
            batch_beam_indices = (
                torch.arange(batch_size, device=device).unsqueeze(-1) * num_beams + next_beam_indices
            )  # [B, 2 * num_beams]
            
            # Select top num_beams for each batch
            next_sequences = []
            next_beam_scores = []
            next_done = []
            
            for b in range(batch_size):
                beam_idx = 0
                for k in range(2 * num_beams):
                    if beam_idx >= num_beams:
                        break
                    
                    global_beam_idx = batch_beam_indices[b, k].item()
                    token_id = next_token_indices[b, k].item()
                    score = next_scores[b, k].item()
                    
                    # Skip if this results in a duplicate sequence
                    new_seq = torch.cat([
                        sequences[global_beam_idx],
                        torch.tensor([token_id], device=device, dtype=torch.long)
                    ], dim=0)
                    
                    next_sequences.append(new_seq)
                    
                    # Apply length penalty to score
                    if token_id == self.eos_token_id or done[global_beam_idx]:
                        # Apply length penalty for finished sequence
                        seq_len = new_seq.size(0)
                        length_factor = ((5 + seq_len) / 6) ** length_penalty
                        next_beam_scores.append(score / length_factor)
                        next_done.append(True)
                    else:
                        next_beam_scores.append(score)
                        next_done.append(False)
                    
                    beam_idx += 1
            
            # Stack into tensors
            # Pad sequences to same length
            max_len = max(seq.size(0) for seq in next_sequences)
            padded_sequences = torch.full(
                (batch_size * num_beams, max_len), 
                self.pad_token_id, 
                dtype=torch.long, 
                device=device
            )
            for i, seq in enumerate(next_sequences):
                padded_sequences[i, :seq.size(0)] = seq
            
            sequences = padded_sequences
            beam_scores = torch.tensor(next_beam_scores, device=device, dtype=torch.float)
            done = torch.tensor(next_done, device=device, dtype=torch.bool)
            
            # Check if all beams are done
            if done.view(batch_size, num_beams).all(dim=1).all():
                break
        
        # Select best sequence for each batch element
        beam_scores = beam_scores.view(batch_size, num_beams)
        best_beam_indices = beam_scores.argmax(dim=1)  # [B]
        
        # Gather best sequences
        sequences = sequences.view(batch_size, num_beams, -1)
        best_sequences = torch.stack([
            sequences[b, best_beam_indices[b]] 
            for b in range(batch_size)
        ])
        
        return best_sequences

    def greedy_decode(self, memory, max_length=100):
        """
        Perform greedy decoding (for comparison with beam search).
        
        Args:
            memory: [B, T, D] - encoder output
            max_length: maximum sequence length
            
        Returns:
            generated: [B, max_seq_len] - generated token ids
        """
        batch_size = memory.size(0)
        device = memory.device
        
        generated = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
        
        for _ in range(max_length - 1):
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(device)
            tgt_emb = self.tgt_embedding(generated)
            tgt_emb = self.pos_encoding(tgt_emb)
            
            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            out = self.final_ln(out)
            step_logits = self.out(out)  # [B, T, Vocab]
            
            next_token = torch.argmax(step_logits[:, -1, :], dim=-1).unsqueeze(1)
            generated = torch.cat((generated, next_token), dim=1)
            
            if (next_token == self.eos_token_id).all():
                break
        
        return generated

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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)


# Alias for backward compatibility
CNNTransformer = CNNTransformerForGeneration