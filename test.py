import numpy as np

def test(model, X_test, y_test, model_path=None):
    # Load model if a path is provided
    if model_path:
        model.load(model_path)
    
    # Forward pass
    scores = model.forward(X_test)
    
    # Compute predictions
    preds = np.argmax(scores, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(preds == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy, preds