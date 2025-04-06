import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, activation='relu'):
        # Initialize weights and biases
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size_1) * 0.01
        self.params['b1'] = np.zeros(hidden_size_1)
        self.params['W2'] = np.random.randn(hidden_size_1, hidden_size_2) * 0.01
        self.params['b2'] = np.zeros(hidden_size_2)
        self.params['W3'] = np.random.randn(hidden_size_2, output_size) * 0.01
        self.params['b3'] = np.zeros(output_size)
        
        # Store activations and gradients
        self.cache = {}
        self.gradients = {}
        self.activation_type = activation
        
    def forward(self, X):
        # First layer
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.activate(Z1)
        
        # Second layer
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.activate(Z2)
        
        # Output layer
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        scores = Z3  # No activation for the output layer (will use softmax with loss)
        
        # Cache for backpropagation
        self.cache['X'] = X
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        self.cache['Z2'] = Z2
        self.cache['A2'] = A2
        self.cache['Z3'] = Z3
        
        return scores
    
    def activate(self, x):
        if self.activation_type == 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")
    
    def activate_derivative(self, x):
        if self.activation_type == 'relu':
            return (x > 0).astype(float)
        elif self.activation_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation_type == 'tanh':
            return 1 - np.power(np.tanh(x), 2)
        else:
            raise ValueError("Unsupported activation function")
    
    def backward(self, dscores):
        batch_size = dscores.shape[0]
        
        # Gradients for output layer
        self.gradients['W3'] = self.cache['A2'].T.dot(dscores)
        self.gradients['b3'] = np.sum(dscores, axis=0)
        
        # Gradients for hidden layer 2
        dA2 = dscores.dot(self.params['W3'].T)
        dZ2 = dA2 * self.activate_derivative(self.cache['Z2'])
        self.gradients['W2'] = self.cache['A1'].T.dot(dZ2)
        self.gradients['b2'] = np.sum(dZ2, axis=0)
        
        # Gradients for hidden layer 1
        dA1 = dZ2.dot(self.params['W2'].T)
        dZ1 = dA1 * self.activate_derivative(self.cache['Z1'])
        self.gradients['W1'] = self.cache['X'].T.dot(dZ1)
        self.gradients['b1'] = np.sum(dZ1, axis=0)
        
        # Normalize gradients by batch size
        for key in self.gradients:
            self.gradients[key] /= batch_size
    
    def save(self, filepath):
        np.savez(filepath, **self.params)
    
    def load(self, filepath):
        data = np.load(filepath)
        for key in self.params:
            self.params[key] = data[key]

class SGDOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, model):
        # Initialize velocity if not already done
        if not self.velocity:
            for key in model.params:
                self.velocity[key] = np.zeros_like(model.params[key])
        
        # Update parameters using momentum
        for key in model.params:
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * model.gradients[key]
            model.params[key] += self.velocity[key]
    
    def decay_learning_rate(self, decay_rate):
        self.learning_rate *= decay_rate

def softmax(x):
    # For numerical stability, subtract the max value
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(scores, y):
    batch_size = scores.shape[0]
    
    # Compute softmax probabilities
    probs = softmax(scores)
    
    # Compute cross-entropy loss
    correct_logprobs = -np.log(probs[np.arange(batch_size), y])
    loss = np.sum(correct_logprobs) / batch_size
    
    # Compute gradients
    dscores = probs.copy()
    dscores[np.arange(batch_size), y] -= 1
    
    return loss, dscores

def l2_regularization(model, reg_strength):
    # Compute L2 regularization loss
    reg_loss = 0.5 * reg_strength * (
        np.sum(model.params['W1'] * model.params['W1']) + 
        np.sum(model.params['W2'] * model.params['W2']) + 
        np.sum(model.params['W3'] * model.params['W3'])
    )
    
    # Add regularization gradients
    model.gradients['W1'] += reg_strength * model.params['W1']
    model.gradients['W2'] += reg_strength * model.params['W2']
    model.gradients['W3'] += reg_strength * model.params['W3']
    
    return reg_loss

def load_cifar10():
    """
    Load CIFAR-10 dataset and preprocess it.
    """
    try:
        import pickle
        import os
        import urllib.request
        import tarfile
        
        # Download CIFAR-10 if not already present
        if not os.path.exists('cifar-10-batches-py'):
            print("Downloading CIFAR-10 dataset...")
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            urllib.request.urlretrieve(url, "cifar-10-python.tar.gz")
            with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
                tar.extractall()
            print("Download complete.")
        
        # Load training data
        X_train = []
        y_train = []
        for batch in range(1, 6):
            with open(f'cifar-10-batches-py/data_batch_{batch}', 'rb') as fo:
                batch_data = pickle.load(fo, encoding='bytes')
            X_train.append(batch_data[b'data'])
            y_train.extend(batch_data[b'labels'])
        
        X_train = np.vstack(X_train).astype(np.float32)
        y_train = np.array(y_train)
        
        # Load test data
        with open('cifar-10-batches-py/test_batch', 'rb') as fo:
            test_data = pickle.load(fo, encoding='bytes')
        X_test = test_data[b'data'].astype(np.float32)
        y_test = np.array(test_data[b'labels'])
        
        # Reshape and normalize the data
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)
        
        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Normalize the data to [0, 1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Split training data into training and validation sets
        split = int(0.9 * X_train.shape[0])
        X_val = X_train[split:]
        y_val = y_train[split:]
        X_train = X_train[:split]
        y_train = y_train[:split]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        return None