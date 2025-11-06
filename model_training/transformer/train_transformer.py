#!/usr/bin/env python3

"""
Script to train the transformer model on brain-to-text data with optimizations for RTX 5080.
"""

from transformer.trainer import TransformerTrainer
import json

def main():
    config = {
        'dataset_dir': '../../data/hdf5_data_final',  # Adjust path as needed
        'sessions': [
            't15.2023.08.11',
            't15.2023.08.13',
            # Add more sessions as needed, or use subset for fast experiments
        ],
        'test_percentage': 0.1,
        'seed': 42,
        'batch_size': 64,  # Increased for GPU utilization
        'num_workers': 4,  # For faster data loading
        'use_augmentation': False,  # Toggle: Set to True to enable data augmentation
        'augmentation_params': {
            'white_noise_std': 1.0,
            'constant_offset_std': 0.2
        },
        'src_d_model': 512,  # Input neural features
        'tgt_vocab': 43,  # 0:pad, 1-40:phonemes, 41:SOS, 42:EOS
        'd_model': 256,
        'nhead': 8,
        'n_encoder_layers': 3,
        'n_decoder_layers': 3,
        'd_ff': 512,
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'use_amp': True,  # Mixed precision for speed
        'sos_id': 41,
        'eos_id': 42,
        'log_every': 20  # Log every 20 batches
    }

    print("Initializing Transformer Trainer...")
    trainer = TransformerTrainer(config)

    print("Starting training for 1 hour...")
    train_losses, val_losses, val_pers = trainer.train(max_time_seconds=3600, log_losses=True)  # 1 hour

    # Save results
    results = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_pers': val_pers
    }
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f)

    print("Results saved to experiment_results.json")

if __name__ == "__main__":
    main()