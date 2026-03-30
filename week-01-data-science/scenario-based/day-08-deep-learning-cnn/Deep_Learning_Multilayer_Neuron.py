"""
Deep Learning - Multilayer Neural Network
Medium Level Project

A multilayer neural network with backpropagation for classification
"""

import numpy as np
import matplotlib.pyplot as plt

class MultilayerNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000):
        """
        Initialize Multilayer Neural Network
        
        Parameters:
        - layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        - learning_rate: Learning rate for gradient descent
        - epochs: Number of training iterations
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.losses = []
        
        # Initialize weights and biases
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases for all layers"""
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            bias = np.zeros((1, self.layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        
        Returns:
        - activations: List of activations for each layer
        """
        activations = [X]
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            # Apply activation function
            if i < len(self.weights) - 1:
                # Hidden layers use sigmoid
                a = self.sigmoid(z)
            else:
                # Output layer uses sigmoid
                a = self.sigmoid(z)
            
            activations.append(a)
        
        return activations
    
    def backward_propagation(self, X, y, activations):
        """
        Backward propagation to compute gradients
        
        Parameters:
        - X: Input data
        - y: True labels
        - activations: Activations from forward propagation
        """
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer error
        output_error = activations[-1] - y
        deltas[-1] = output_error * self.sigmoid_derivative(activations[-1])
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[i+1], self.weights[i+1].T)
            deltas[i] = error * self.sigmoid_derivative(activations[i+1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """
        Train the neural network
        
        Parameters:
        - X: Training features
        - y: Training labels
        """
        for epoch in range(self.epochs):
            # Forward propagation
            activations = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(y, activations[-1])
            self.losses.append(loss)
            
            # Backward propagation
            self.backward_propagation(X, y, activations)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        activations = self.forward_propagation(X)
        return (activations[-1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        activations = self.forward_propagation(X)
        return activations[-1]
    
    def plot_loss(self):
        """Plot training loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Multilayer Neural Network - Training Loss')
        plt.grid(True)
        plt.savefig('neural_network_loss.png', dpi=300, bbox_inches='tight')
        plt.show()


# Example: XOR Problem
def example_xor():
    """Example solving XOR problem (non-linearly separable)"""
    print("=" * 50)
    print("Multilayer Neural Network - XOR Problem")
    print("=" * 50)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create neural network: 2 inputs -> 4 hidden -> 1 output
    nn = MultilayerNeuralNetwork(
        layer_sizes=[2, 4, 1],
        learning_rate=0.5,
        epochs=5000
    )
    
    # Train the network
    nn.fit(X, y)
    
    # Make predictions
    print("\nPredictions:")
    predictions = nn.predict(X)
    probabilities = nn.predict_proba(X)
    
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Predicted: {predictions[i][0]}, "
              f"Probability: {probabilities[i][0]:.4f}, Actual: {y[i][0]}")
    
    # Plot loss
    nn.plot_loss()
    
    return nn


# Example: Binary Classification
def example_binary_classification():
    """Example with synthetic binary classification data"""
    print("\n" + "=" * 50)
    print("Multilayer Neural Network - Binary Classification")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    
    # Class 0
    X_class0 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
    y_class0 = np.zeros((n_samples//2, 1))
    
    # Class 1
    X_class1 = np.random.randn(n_samples//2, 2) + np.array([-2, -2])
    y_class1 = np.ones((n_samples//2, 1))
    
    # Combine data
    X = np.vstack([X_class0, X_class1])
    y = np.vstack([y_class0, y_class1])
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Create neural network: 2 inputs -> 8 hidden -> 4 hidden -> 1 output
    nn = MultilayerNeuralNetwork(
        layer_sizes=[2, 8, 4, 1],
        learning_rate=0.1,
        epochs=2000
    )
    
    # Train the network
    nn.fit(X, y)
    
    # Calculate accuracy
    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y) * 100
    print(f"\nTraining Accuracy: {accuracy:.2f}%")
    
    return nn


if __name__ == "__main__":
    # Run examples
    nn_xor = example_xor()
    nn_binary = example_binary_classification()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
