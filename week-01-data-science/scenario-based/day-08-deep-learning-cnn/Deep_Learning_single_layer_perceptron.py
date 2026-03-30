"""
Deep Learning - Single Layer Perceptron
Medium Level Project

A single layer perceptron for binary classification
"""

import numpy as np
import matplotlib.pyplot as plt

class SingleLayerPerceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Initialize the Single Layer Perceptron
        
        Parameters:
        - learning_rate: Learning rate for weight updates
        - epochs: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []
    
    def activation_function(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        """
        Train the perceptron
        
        Parameters:
        - X: Training features (n_samples, n_features)
        - y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_errors = 0
            
            for idx, x_i in enumerate(X):
                # Calculate linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Apply activation function
                y_predicted = self.activation_function(linear_output)
                
                # Calculate error
                error = y[idx] - y_predicted
                
                # Update weights and bias
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
                
                # Track errors
                epoch_errors += int(error != 0)
            
            self.errors.append(epoch_errors)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Errors = {epoch_errors}")
    
    def predict(self, X):
        """Make predictions"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)
    
    def plot_errors(self):
        """Plot training errors over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.errors)), self.errors, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Errors')
        plt.title('Single Layer Perceptron - Training Errors')
        plt.grid(True)
        plt.savefig('perceptron_errors.png', dpi=300, bbox_inches='tight')
        plt.show()


# Example: AND Gate
def example_and_gate():
    """Example using AND gate logic"""
    print("=" * 50)
    print("Single Layer Perceptron - AND Gate Example")
    print("=" * 50)
    
    # AND gate dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    # Create and train perceptron
    perceptron = SingleLayerPerceptron(learning_rate=0.1, epochs=50)
    perceptron.fit(X, y)
    
    # Make predictions
    print("\nPredictions:")
    for i in range(len(X)):
        prediction = perceptron.predict(X[i:i+1])[0]
        print(f"Input: {X[i]} -> Predicted: {prediction}, Actual: {y[i]}")
    
    # Plot errors
    perceptron.plot_errors()
    
    return perceptron


# Example: OR Gate
def example_or_gate():
    """Example using OR gate logic"""
    print("\n" + "=" * 50)
    print("Single Layer Perceptron - OR Gate Example")
    print("=" * 50)
    
    # OR gate dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    # Create and train perceptron
    perceptron = SingleLayerPerceptron(learning_rate=0.1, epochs=50)
    perceptron.fit(X, y)
    
    # Make predictions
    print("\nPredictions:")
    for i in range(len(X)):
        prediction = perceptron.predict(X[i:i+1])[0]
        print(f"Input: {X[i]} -> Predicted: {prediction}, Actual: {y[i]}")
    
    return perceptron


if __name__ == "__main__":
    # Run examples
    perceptron_and = example_and_gate()
    perceptron_or = example_or_gate()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
