import numpy as np
from hyperparameter_search import *
from utils import *
from train import *
from test import test
import matplotlib.pyplot as plt

def hyperparameter_search(X_train, y_train, X_val, y_val, input_size, hidden_sizes, learning_rates, 
                         reg_strengths, activations=['relu'], batch_size=128, num_epochs=5):
    
    results = []
    best_val_acc = 0
    best_params = {}
    
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_strengths:
                for activation in activations:
                    print(f"Training with hidden_size={hidden_size}, lr={lr}, reg={reg}, activation={activation}")
                    
                    # Create model
                    model = ThreeLayerNet(input_size, hidden_size, hidden_size, 10, activation=activation)
                    
                    # Create optimizer
                    optimizer = SGDOptimizer(learning_rate=lr)
                    
                    # Train model
                    _, _, val_accs = train(model, X_train, y_train, X_val, y_val, optimizer,
                                          batch_size=batch_size, num_epochs=num_epochs, reg_strength=reg)
                    
                    # Record the best validation accuracy
                    val_acc = val_accs[-1]
                    
                    # Save results
                    results.append({
                        'hidden_size': hidden_size,
                        'learning_rate': lr,
                        'reg_strength': reg,
                        'activation': activation,
                        'val_acc': val_acc
                    })
                    
                    # Update best parameters if necessary
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {
                            'hidden_size': hidden_size,
                            'learning_rate': lr,
                            'reg_strength': reg,
                            'activation': activation
                        }
    
    # Print best parameters
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best parameters: {best_params}")
    
    return results, best_params

def visualize_training(train_losses, val_losses, val_accs):
    """
    Visualize training and validation losses, and validation accuracy.
    """
    
    # Create figure with three subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training and validation loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot validation accuracy
    ax2.plot(epochs, val_accs, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def visualize_weights(model, save_path='weight_visualization.png'):
    """
    Visualize the weights of the first layer as images.
    """
    
    # Get the first layer weights
    W1 = model.params['W1']
    
    # Assuming input is 3072 (32x32x3) for CIFAR-10
    W1_reshaped = W1.reshape(-1, 32, 32, 3)
    
    # Create a grid of weight visualizations
    n_filters = min(25, W1_reshaped.shape[0])  # Show at most 25 filters
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    
    plt.figure(figsize=(12, 12))
    for i in range(n_filters):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Normalize for better visualization
        filter_img = W1_reshaped[i]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        
        plt.imshow(filter_img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()