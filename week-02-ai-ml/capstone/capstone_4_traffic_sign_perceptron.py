# Capstone Project 4: Smart Traffic Sign Recognition System
# Stage 1: Perceptron-Based Neural Network
# 
# SCENARIO:
# A city is building a Smart Traffic Monitoring System to improve road safety.
# Every intersection will have cameras that detect traffic signs automatically.
# The system must recognize: Stop sign, Speed limit sign, Pedestrian crossing
# 
# CHALLENGES:
# 1. System must understand basic classification logic (learning from features)
# 2. Must classify images of traffic signs captured by cameras
# 
# STAGE 1 GOAL:
# Before building a full image model, implement a Perceptron-based neural 
# network to understand how neural networks make decisions.
# 
# The perceptron will predict whether a traffic sign means "STOP" or "NOT STOP"
# using extracted features from the sign.
# 
# DATASET: DatasetCapstoneProject4.csv
# Features: Red_Color_Intensity, Circular_Shape, Text_Present, Edge_Count
# Target: Stop_Sign (0 or 1)
# 
# REAL-WORLD USE:
# - Autonomous vehicles: Real-time sign detection
# - Traffic monitoring cameras: Automated recognition
# - Driver assistance systems: Alert drivers about signs

import pandas as pd
import numpy as np

# Step 1: Load the traffic sign dataset
# This contains extracted features from traffic sign images
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'DatasetCapstoneProject4.csv')
df = pd.DataFrame(pd.read_csv(dataset_path))

print(df.head())

# Step 2: Separate features (X) and target variable (y)
# X = sign features (color, shape, text, edges)
# y = Stop_Sign (1 if STOP sign, 0 if NOT STOP)
X = df.drop(['Sign_ID', 'Stop_Sign'], axis=1)
print(X.head())

y = df['Stop_Sign']
print(y.head())

# Step 3: Create a Perceptron class (simple neural network)
# A perceptron is the most basic form of a neural network
# It learns by adjusting weights based on prediction errors
class Perceptron:
    def __init__(self, n_features):
        # Initialize weights to zero for each feature
        self.weights = np.zeros(n_features)
        # Initialize bias to zero
        self.bias = 0.0
    
    def sigmoid(self, x):
        # Sigmoid activation function: converts any value to 0-1 range
        # This gives us probability-like outputs
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        # Forward pass: calculate output
        # Linear combination: sum of (weights * features) + bias
        linear = np.dot(x, self.weights) + self.bias
        # Apply sigmoid to get probability
        return self.sigmoid(linear)

# Step 4: Initialize the perceptron model
# X.shape[1] gives us the number of features (4 in this case)
model = Perceptron(X.shape[1])
learning_rate = 0.1  # How fast the model learns

# Step 5: Training loop - teach the perceptron to recognize STOP signs
# We train for 100 epochs (complete passes through the data)
for epoch in range(100):
    # Go through each traffic sign in the dataset
    for i in range(len(X)):
        # Get features for this sign
        xi = X.iloc[i].values.astype(float)
        # Get actual label (STOP or NOT STOP)
        target = float(y.iloc[i])
        
        # Make a prediction
        output = model.forward(xi)
        prediction = 1 if output >= 0.5 else 0
        
        # Calculate error (how wrong we were)
        error = target - prediction
        
        # Update weights and bias based on error
        # This is how the perceptron learns!
        model.weights += learning_rate * error * xi
        model.bias += learning_rate * error

# Print progress every 10 epochs
if (epoch + 1) % 10 == 0:
    print(f"Epoch {epoch + 1} completed")

# Step 6: Test with a new traffic sign
# Features: [Red_Color_Intensity=0.9, Circular_Shape=0.85, Text_Present=1, Edge_Count=0]
# This looks like a STOP sign (high red, has text)
test = np.array([0.9, 0.85, 1, 0], dtype=float)

# Step 7: Make prediction
result = model.forward(test)

# Step 8: Display results
if result >= 0.5:
    print('Traffic sign is STOP')
else:
    print("Traffic sign is NOT STOP")

print("\nPrediction Probability:", result)
print("Predicted Class:", 1 if result >= 0.5 else 0)

# HOW IT WORKS:
# 1. Perceptron looks at sign features (color, shape, text, edges)
# 2. Multiplies each feature by a learned weight
# 3. Adds all weighted features together with a bias
# 4. Applies sigmoid function to get probability (0 to 1)
# 5. If probability > 0.5, predicts STOP sign
# 
# NEXT STEPS (Stage 2):
# - Implement CNN for actual image recognition
# - Add more sign categories (Speed Limit, Pedestrian)
# - Handle real camera images with noise and varying conditions
# - Deploy on edge devices for real-time processing
# 
# IMPACT:
# - Improved road safety through automated detection
# - Support for autonomous vehicle navigation
# - Reduced human error in sign recognition
