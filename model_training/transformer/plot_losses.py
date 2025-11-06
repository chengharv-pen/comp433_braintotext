#!/usr/bin/env python3

"""
Plot training losses from experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)

    train_losses = data['train_losses']
    val_losses = data['val_losses']
    val_pers = data['val_pers']

    # Smooth train losses for plotting (since many batches)
    window_size = 100
    if len(train_losses) > window_size:
        smoothed_train = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
    else:
        smoothed_train = train_losses

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax1.plot(smoothed_train, label='Train Loss (smoothed)')
    ax1.plot(np.linspace(0, len(smoothed_train), len(val_losses)), val_losses, label='Val Loss')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot PER
    ax2.plot(val_pers, label='Validation PER')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('PER')
    ax2.set_title('Validation Phoneme Error Rate')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_losses('experiment_results.json')