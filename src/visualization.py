"""
This file contains a function to plot the training and validation losses from a model checkpoint.

Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
        swathi.thiru078@gmail.com
"""

import matplotlib.pyplot as plt
import torch

def plot_losses(model_checkpoint_path):
    """
    Plots the training and validation losses from a model checkpoint.

    This function loads a PyTorch model checkpoint, retrieves the training and validation loss histories,
    and visualizes them as line plots over the training epochs.

    Parameters:
    - model_checkpoint_path (str): Path to the PyTorch checkpoint file (.pth or .pt) that contains
      loss histories under the keys 'training_losses' and 'validation_losses'.

    Returns:
    - None: The function displays the plot but does not return any value.

    Example:
    >>> # Assume 'model_checkpoint.pth' contains loss histories
    >>> plot_losses('model_checkpoint.pth')

    # Output: A plot with training and validation loss curves, labeled by epoch.

    """
    # Load the checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

    # Retrieve losses
    training_losses = checkpoint.get('training_losses', [])
    validation_losses = checkpoint.get('validation_losses', [])
    #training_losses = [0.8, 0.6, 0.4, 0.2]
    #validation_losses = [0.9, 0.7, 0.5, 0.3]

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # update the path to the model.pth file after training/fine-tuning the model
    plot_losses("weights/new/structVisualization/with_modified_val_loss_4/model.pth")


if __name__ == "__main__":
    main()