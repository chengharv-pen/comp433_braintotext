# Transformer Model for Brain-to-Text

This directory contains the transformer-based model for decoding neural signals to phonemes.

## Files:
- `model.py`: BasicTransformer model with positional encoding
- `dataset.py`: SimpleBrainToTextDataset for loading HDF5 data
- `trainer.py`: TransformerTrainer with mixed precision and optimizations
- `train_transformer.py`: Script to train for 1 hour
- `run_experiments.py`: Script to run hyperparameter experiments
- `plot_losses.py`: Script to plot training curves

## Usage:
```bash
cd transformer
python train_transformer.py  # Train for 1 hour
python run_experiments.py    # Run hyperparameter sweep
python plot_losses.py        # Plot results
```

## Features:
- Mixed precision training (BF16)
- torch.compile for speed
- Data augmentation toggle
- Fixed-time training for fair comparisons
- Automatic loss logging and plotting