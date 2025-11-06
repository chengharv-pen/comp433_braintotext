#!/usr/bin/env python3

"""
Run hyperparameter experiments for fixed time and compare results.
"""

import os
import json
from transformer.trainer import TransformerTrainer
import subprocess

def run_experiment(config, exp_name, max_time=3600):
    print(f"Running experiment: {exp_name}")
    config['log_every'] = 50  # Less frequent logging for experiments

    trainer = TransformerTrainer(config)
    train_losses, val_losses, val_pers = trainer.train(max_time_seconds=max_time, log_losses=False)

    results = {
        'exp_name': exp_name,
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_pers': val_pers,
        'final_per': val_pers[-1] if val_pers else None
    }

    with open(f'results_{exp_name}.json', 'w') as f:
        json.dump(results, f)

    return results

def main():
    base_config = {
        'dataset_dir': '../../data/hdf5_data_final',
        'sessions': ['t15.2023.08.11', 't15.2023.08.13'],  # Subset for fast experiments
        'test_percentage': 0.1,
        'seed': 42,
        'num_workers': 4,
        'use_augmentation': False,
        'augmentation_params': {'white_noise_std': 1.0, 'constant_offset_std': 0.2},
        'src_d_model': 512,
        'tgt_vocab': 43,
        'nhead': 8,
        'n_encoder_layers': 3,
        'n_decoder_layers': 3,
        'd_ff': 512,
        'dropout': 0.1,
        'weight_decay': 1e-4,
        'use_amp': True,
        'sos_id': 41,
        'eos_id': 42,
        'log_every': 20
    }

    experiments = [
        {'batch_size': 32, 'lr': 1e-4, 'd_model': 256, 'exp_name': 'baseline'},
        {'batch_size': 64, 'lr': 1e-4, 'd_model': 256, 'exp_name': 'larger_batch'},
        {'batch_size': 64, 'lr': 5e-4, 'd_model': 256, 'exp_name': 'higher_lr'},
        {'batch_size': 64, 'lr': 1e-4, 'd_model': 512, 'exp_name': 'larger_model'},
        {'batch_size': 64, 'lr': 1e-4, 'd_model': 256, 'use_augmentation': True, 'exp_name': 'with_aug'},
    ]

    results = []
    for exp in experiments:
        config = base_config.copy()
        config.update(exp)
        result = run_experiment(config, exp['exp_name'], max_time=1800)  # 30 min per experiment
        results.append(result)

    # Save summary
    summary = {
        'experiments': [{'name': r['exp_name'], 'final_per': r['final_per'], 'config': r['config']} for r in results]
    }
    with open('experiment_summary.json', 'w') as f:
        json.dump(summary, f)

    print("All experiments complete. Run 'python plot_losses.py' on individual results or create comparison plots.")

if __name__ == "__main__":
    main()