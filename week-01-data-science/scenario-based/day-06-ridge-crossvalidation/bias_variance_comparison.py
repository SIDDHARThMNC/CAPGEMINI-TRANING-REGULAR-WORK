"""
SCENARIO: Comprehensive Bias-Variance Tradeoff Comparison

This file demonstrates the bias-variance tradeoff across multiple scenarios to help understand
when models are too simple (high bias) or too complex (high variance).

CONCEPTS COVERED:
1. Underfitting (High Bias) - Model too simple
2. Overfitting (High Variance) - Model too complex
3. Balanced Model - Just right complexity

REAL-WORLD APPLICATIONS:
- Student exam prediction
- Athlete performance prediction
- Any prediction task where model complexity matters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(123)

def create_dataset(n_samples=30, noise_level=2):
    X = np.linspace(0, 8, n_samples).reshape(-1, 1)
    y = (12 * np.sin(X * 0.8) + 5).ravel() + np.random.normal(scale=noise_level, size=n_samples)
    return X, y

def train_and_evaluate_models(X_train, y_train, X_test):
    models = {
        'Linear (deg=1)': make_pipeline(PolynomialFeatures(1), LinearRegression()),
        'Balanced (deg=3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        'Complex (deg=10)': make_pipeline(PolynomialFeatures(10), LinearRegression())
    }
    
    predictions = {}
    scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        predictions[name] = y_pred_test
        scores[name] = {
            'train_r2': r2_score(y_train, y_pred_train),
            'train_mse': mean_squared_error(y_train, y_pred_train)
        }
    
    return predictions, scores

X_train, y_train = create_dataset(n_samples=30, noise_level=2)
X_test = np.linspace(0, 8, 100).reshape(-1, 1)

predictions, scores = train_and_evaluate_models(X_train, y_train, X_test)

fig = plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.scatter(X_train, y_train, color="gray", s=50, label="Training Data", zorder=3)
plt.plot(X_test, predictions['Linear (deg=1)'], color="red", linewidth=2, label="Linear Model")
plt.xlabel("Input Feature", fontweight='bold')
plt.ylabel("Target Value", fontweight='bold')
plt.title("High Bias (Underfitting)\nToo Simple", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.scatter(X_train, y_train, color="gray", s=50, label="Training Data", zorder=3)
plt.plot(X_test, predictions['Balanced (deg=3)'], color="green", linewidth=2, label="Balanced Model")
plt.xlabel("Input Feature", fontweight='bold')
plt.ylabel("Target Value", fontweight='bold')
plt.title("Balanced Model\nJust Right", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.scatter(X_train, y_train, color="gray", s=50, label="Training Data", zorder=3)
plt.plot(X_test, predictions['Complex (deg=10)'], color="blue", linewidth=2, label="Complex Model")
plt.xlabel("Input Feature", fontweight='bold')
plt.ylabel("Target Value", fontweight='bold')
plt.title("High Variance (Overfitting)\nToo Complex", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

model_names = list(scores.keys())
train_r2_values = [scores[name]['train_r2'] for name in model_names]
train_mse_values = [scores[name]['train_mse'] for name in model_names]

plt.subplot(2, 3, 4)
colors = ['red', 'green', 'blue']
bars = plt.bar(model_names, train_r2_values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('R² Score', fontweight='bold')
plt.title('Training R² Score Comparison', fontsize=12, fontweight='bold')
plt.ylim([0, 1])
plt.xticks(rotation=15, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, train_r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', fontweight='bold')

plt.subplot(2, 3, 5)
bars = plt.bar(model_names, train_mse_values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Mean Squared Error', fontweight='bold')
plt.title('Training MSE Comparison', fontsize=12, fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, train_mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.2f}', ha='center', fontweight='bold')

plt.subplot(2, 3, 6)
complexity_levels = ['Low\n(Linear)', 'Medium\n(Poly 3)', 'High\n(Poly 10)']
bias_levels = [8, 3, 1]
variance_levels = [2, 3, 9]

x_pos = np.arange(len(complexity_levels))
width = 0.35

bars1 = plt.bar(x_pos - width/2, bias_levels, width, label='Bias', color='red', edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x_pos + width/2, variance_levels, width, label='Variance', color='blue', edgecolor='black', linewidth=1.5)

plt.xlabel('Model Complexity', fontweight='bold')
plt.ylabel('Error Level', fontweight='bold')
plt.title('Bias-Variance Tradeoff', fontsize=12, fontweight='bold')
plt.xticks(x_pos, complexity_levels)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Bias-Variance Tradeoff: Comprehensive Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('Day_6_Mar01_Ridge_CrossValidation/bias_variance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*80)
print("BIAS-VARIANCE TRADEOFF: COMPREHENSIVE ANALYSIS")
print("="*80)

print("\nMODEL PERFORMANCE SUMMARY:")
print("-" * 80)
for name, score in scores.items():
    print(f"\n{name}:")
    print(f"  Training R²: {score['train_r2']:.4f}")
    print(f"  Training MSE: {score['train_mse']:.4f}")

print("\n" + "="*80)
print("KEY CONCEPTS EXPLAINED")
print("="*80)

print("\n1. HIGH BIAS (Underfitting)")
print("-" * 40)
print("CHARACTERISTICS:")
print("  - Model is too simple")
print("  - Makes strong assumptions about data")
print("  - Poor performance on both training and test data")
print("  - Consistent errors (systematic mistakes)")
print()
print("EXAMPLE: Linear model for non-linear data")
print("  - Assumes straight-line relationship")
print("  - Misses curves and patterns")
print("  - Predictions consistently off")
print()
print("REAL-WORLD ANALOGY:")
print("  - Using only temperature to predict ice cream sales")
print("  - Ignoring day of week, holidays, weather, location")

print("\n2. HIGH VARIANCE (Overfitting)")
print("-" * 40)
print("CHARACTERISTICS:")
print("  - Model is too complex")
print("  - Memorizes training data including noise")
print("  - Great performance on training data")
print("  - Poor performance on new/test data")
print("  - Predictions change dramatically with small data changes")
print()
print("EXAMPLE: Polynomial degree 10 model")
print("  - Follows every bump in training data")
print("  - Captures noise as if it's a pattern")
print("  - Fails on new data")
print()
print("REAL-WORLD ANALOGY:")
print("  - Memorizing exam answers without understanding concepts")
print("  - Works for exact same questions, fails on variations")

print("\n3. BALANCED MODEL (Just Right)")
print("-" * 40)
print("CHARACTERISTICS:")
print("  - Appropriate complexity for the problem")
print("  - Captures true patterns, ignores noise")
print("  - Good performance on both training and test data")
print("  - Generalizes well to new data")
print()
print("EXAMPLE: Polynomial degree 3 model")
print("  - Flexible enough for curves")
print("  - Simple enough to avoid memorizing noise")
print("  - Reliable predictions")
print()
print("REAL-WORLD ANALOGY:")
print("  - Understanding concepts and applying to new problems")
print("  - Works on both practice and real exams")

print("\n" + "="*80)
print("THE BIAS-VARIANCE TRADEOFF")
print("="*80)
print()
print("As model complexity INCREASES:")
print("  ↓ Bias decreases (model fits training data better)")
print("  ↑ Variance increases (model becomes sensitive to noise)")
print()
print("As model complexity DECREASES:")
print("  ↑ Bias increases (model too simple, misses patterns)")
print("  ↓ Variance decreases (model more stable, less sensitive)")
print()
print("GOAL: Find the sweet spot where Total Error is minimized")
print("      Total Error = Bias² + Variance + Irreducible Error")

print("\n" + "="*80)
print("PRACTICAL RECOMMENDATIONS")
print("="*80)
print()
print("If your model has HIGH BIAS (underfitting):")
print("  - Increase model complexity")
print("  - Add more features")
print("  - Use more flexible algorithms")
print("  - Reduce regularization")
print()
print("If your model has HIGH VARIANCE (overfitting):")
print("  - Decrease model complexity")
print("  - Get more training data")
print("  - Use regularization (Ridge, Lasso)")
print("  - Remove irrelevant features")
print("  - Use cross-validation")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
