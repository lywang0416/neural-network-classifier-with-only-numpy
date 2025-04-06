from hyperparameter_search import *
from utils import *
from train import *
from test import test

def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    input_size = X_train.shape[1]  # 3072 for CIFAR-10 (32x32x3)
    
    # Define hyperparameters to search
    hidden_sizes = [128]
    learning_rates = [1e-3]
    reg_strengths = [1e-2]
    
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
    best_model = ThreeLayerNet(input_size, best_params['hidden_size'], 10, activation=best_params['activation'])
    test_acc, _ = test(best_model, X_test, y_test, model_path='best_model.npz')
    
    # Visualize weights
    visualize_weights(best_model)
    
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Best parameters: {best_params}")

if __name__ == "__main__":
    main()