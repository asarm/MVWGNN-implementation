"""
Visualization utilities for MVWGNN
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    """
    Plot training history curves.

    Args:
        history: Dict with training history
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Training and Validation Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation and Test MAE
    axes[1].plot(history['val_mae'], label='Val MAE', color='blue')
    axes[1].plot(history['test_mae'], label='Test MAE', color='red', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Validation vs Test MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Validation and Test RMSE
    axes[2].plot(history['val_rmse'], label='Val RMSE', color='orange')
    axes[2].plot(history['test_rmse'], label='Test RMSE', color='red', linestyle='--')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Validation vs Test RMSE')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()
