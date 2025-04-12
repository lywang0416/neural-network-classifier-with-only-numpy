from utils import *

def train(model, X_train, y_train, X_val, y_val, optimizer, 
          batch_size=128, num_epochs=10, reg_strength=0.001,
          lr_decay=0.95, lr_decay_every=1, save_path=None):
    
    # Keep track of the best validation accuracy and corresponding model
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    
    for epoch in range(num_epochs):
        # Shuffle the training data
        shuffle_idx = np.random.permutation(num_train)
        X_train_shuffled = X_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]
        
        # Decay learning rate at the end of epoch if specified
        if epoch > 0 and epoch % lr_decay_every == 0:
            optimizer.decay_learning_rate(lr_decay)
            print(f"Learning rate decayed to {optimizer.learning_rate}")
        
        epoch_train_loss = 0.0
        
        # Mini-batch training
        for i in range(iterations_per_epoch):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_train)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            scores = model.forward(X_batch)
            
            # Compute loss and gradients
            loss, dscores = cross_entropy_loss(scores, y_batch)
            model.backward(dscores)
            
            # Add L2 regularization
            reg_loss = l2_regularization(model, reg_strength)
            loss += reg_loss
            
            # Update parameters
            optimizer.update(model)
            
            epoch_train_loss += loss
            
            # Print progress
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{iterations_per_epoch}, Loss: {loss:.4f}")
        
        # Compute average training loss for the epoch
        avg_train_loss = epoch_train_loss / iterations_per_epoch
        train_losses.append(avg_train_loss)
        
        # Validate model
        val_scores = model.forward(X_val)
        val_loss, _ = cross_entropy_loss(val_scores, y_val)
        val_loss += 0.5 * reg_strength * (
            np.sum(model.params['W1'] * model.params['W1']) + 
            np.sum(model.params['W2'] * model.params['W2']) + 
            np.sum(model.params['W3'] * model.params['W3'])
        )
        val_losses.append(val_loss)
        
        # Compute validation accuracy
        val_preds = np.argmax(val_scores, axis=1)
        val_acc = np.mean(val_preds == y_val)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(save_path)
            print(f"New best model saved as {save_path} and with validation accuracy: {best_val_acc:.4f}")
    
    return train_losses, val_losses, val_accs