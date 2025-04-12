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

def visualize_training(train_losses, val_losses, val_accs, title=None):
    """
    Visualize training and validation losses, and validation accuracy.
    """
    
    # Create figure with three subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training and validation loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title(f'Loss Curves {title}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot validation accuracy
    ax2.plot(epochs, val_accs, 'g-', label='Validation Accuracy')
    ax2.set_title(f'Validation Accuracy {title}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def learning_rate_search(args):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    input_size = X_train.shape[1]  # 3072 for CIFAR-10 (32x32x3)

    # Check hyperparameters
    assert len(args.hidden_sizes.split(',')) == 1
    assert len(args.reg_strengths.split(',')) == 1
    
    # Define hyperparameters to search
    hidden_sizes = [int(item) for item in args.hidden_sizes.split(',')]
    learning_rates = [float(item) for item in args.learning_rates.split(',')]
    reg_strengths = [float(item) for item in args.reg_strengths.split(',')]
    
    for learning_rate in learning_rates:
        model = ThreeLayerNet(
            input_size, 
            hidden_sizes[0], 
            hidden_sizes[0], 
            10,  # 10 classes for CIFAR-10
            activation='relu'
        )
        
        optimizer = SGDOptimizer(learning_rate=learning_rate)
        
        train_losses, val_losses, val_accs = train(
            model, X_train, y_train, X_val, y_val, optimizer,
            reg_strength=reg_strengths[0],
            num_epochs=20,  # Train for more epochs
        )
        
        # Visualize training progress
        visualize_training(train_losses, val_losses, val_accs, title=f'of learning rate {learning_rate}')

def reg_strength_search(args):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    input_size = X_train.shape[1]  # 3072 for CIFAR-10 (32x32x3)
    
    # Check hyperparameters
    assert len(args.hidden_sizes.split(',')) == 1
    assert len(args.learning_rates.split(',')) == 1

    # Define hyperparameters to search
    hidden_sizes = [int(item) for item in args.hidden_sizes.split(',')]
    learning_rates = [float(item) for item in args.learning_rates.split(',')]
    reg_strengths = [float(item) for item in args.reg_strengths.split(',')]

    for reg_strength in reg_strengths:
        model = ThreeLayerNet(
            input_size, 
            hidden_sizes[0], 
            hidden_sizes[0], 
            10,  # 10 classes for CIFAR-10
            activation='relu'
        )
        
        optimizer = SGDOptimizer(learning_rate=learning_rates[0])
        
        train_losses, val_losses, val_accs = train(
            model, X_train, y_train, X_val, y_val, optimizer,
            reg_strength=reg_strength,
            num_epochs=20,  # Train for more epochs
        )
        
        # Visualize training progress
        visualize_training(train_losses, val_losses, val_accs, title=f'of reg_strength {reg_strength}')

def hidden_size_search(args):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    input_size = X_train.shape[1]  # 3072 for CIFAR-10 (32x32x3)
    
    # Check hyperparameters
    assert len(args.reg_strengths.split(',')) == 1
    assert len(args.learning_rates.split(',')) == 1

    # Define hyperparameters to search
    hidden_sizes = [int(item) for item in args.hidden_sizes.split(',')]
    learning_rates = [float(item) for item in args.learning_rates.split(',')]
    reg_strengths = [float(item) for item in args.reg_strengths.split(',')]
    
    for hidden_size in hidden_sizes:
        model = ThreeLayerNet(
            input_size, 
            hidden_size, 
            hidden_size, 
            10,  # 10 classes for CIFAR-10
            activation='relu'
        )
        
        optimizer = SGDOptimizer(learning_rate=learning_rates[0])
        
        train_losses, val_losses, val_accs = train(
            model, X_train, y_train, X_val, y_val, optimizer,
            reg_strength=reg_strengths[0],
            num_epochs=20,  # Train for more epochs
        )
        
        # Visualize training progress
        visualize_training(train_losses, val_losses, val_accs, title=f'of hidden size {hidden_size}')

def hyper_parameter_search(args):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    input_size = X_train.shape[1]  # 3072 for CIFAR-10 (32x32x3)
    
    # Define hyperparameters to search
    hidden_sizes = [int(item) for item in args.hidden_sizes.split(',')]
    learning_rates = [float(item) for item in args.learning_rates.split(',')]
    reg_strengths = [float(item) for item in args.reg_strengths.split(',')]
    
    # Perform hyperparameter search
    print("Performing hyperparameter search...")
    results, best_params = hyperparameter_search(
        X_train, y_train, X_val, y_val, input_size,
        hidden_sizes, learning_rates, reg_strengths
    )
    
    # Train the best model for more epochs
    print("Training the best model...")
    print(results)
    model = ThreeLayerNet(
        input_size, 
        best_params['hidden_size'], 
        best_params['hidden_size'], 
        10,  # 10 classes for CIFAR-10
        activation=best_params['activation']
    )
    
    optimizer = SGDOptimizer(learning_rate=best_params['learning_rate'])
    
    train_losses, val_losses, val_accs = train(
        model, X_train, y_train, X_val, y_val, optimizer,
        reg_strength=best_params['reg_strength'],
        num_epochs=20,  # Train for more epochs
        save_path='best_model.npz'
    )
    
    # Visualize training progress
    visualize_training(train_losses, val_losses, val_accs)
    
    # Load best model and test
    best_model = ThreeLayerNet(input_size, best_params['hidden_size'], best_params['hidden_size'], 10, activation=best_params['activation'])
    test_acc, _ = test(best_model, X_test, y_test, model_path='best_model.npz')
    
    # Visualize weights
    visualize_weights(best_model)
    
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Best parameters: {best_params}")

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