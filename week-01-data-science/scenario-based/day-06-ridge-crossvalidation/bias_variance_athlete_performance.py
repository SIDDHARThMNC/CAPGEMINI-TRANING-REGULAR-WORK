"""
SCENARIO: Predicting Athlete Performance - Bias vs Variance Tradeoff

A sports academy wants to build a model to predict athlete sprint times (in seconds) based on training
hours. They collect data from 30 athletes, but the sprint times are noisy because of other factors 
(like diet, fatigue, or weather).

They try two different models:
- Linear Model (straight line) → very simple, assumes sprint times improve perfectly with more
  training hours.
- Polynomial Model (degree 10 curve) → very complex, tries to follow every bump in the data.

QUESTIONS:
- Part A: If the linear model consistently predicts sprint times that are too fast or too slow compared
  to actual results, what does this show about bias?
- Part B: If the polynomial model fits the training data almost perfectly but gives very different
  predictions when tested on new athletes, what does this show about variance?
- Part C: Which model is likely to generalize better to new athletes, and why?
- Part D (Applied): How would you explain the difference between "high bias" and "high variance"
  to a coach who doesn't know machine learning?
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)
X = np.linspace(0, 10, 30).reshape(-1, 1)
y = (15 - 0.8 * X + 0.3 * np.sin(2 * X)).ravel() + np.random.normal(scale=0.5, size=30)

linear_model = make_pipeline(PolynomialFeatures(1), LinearRegression())
poly_model = make_pipeline(PolynomialFeatures(10), LinearRegression())
balanced_model = make_pipeline(PolynomialFeatures(3), LinearRegression())

linear_model.fit(X, y)
poly_model.fit(X, y)
balanced_model.fit(X, y)

X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_linear = linear_model.predict(X_test)
y_poly = poly_model.predict(X_test)
y_balanced = balanced_model.predict(X_test)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_linear, color="red", label="Linear Model")
plt.xlabel("Training Hours")
plt.ylabel("Sprint Time (seconds)")
plt.title("High Bias (Underfitting)")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_balanced, color="green", label="Poly deg=3")
plt.xlabel("Training Hours")
plt.ylabel("Sprint Time (seconds)")
plt.title("Balanced Model")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_poly, color="blue", label="Poly deg=10")
plt.xlabel("Training Hours")
plt.ylabel("Sprint Time (seconds)")
plt.title("High Variance (Overfitting)")
plt.legend()

plt.suptitle("Bias vs Variance: Athlete Sprint Performance", fontsize=14)
plt.tight_layout()
plt.savefig('Day_6_Mar01_Ridge_CrossValidation/bias_variance_athlete_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*80)
print("BIAS VS VARIANCE TRADEOFF - ATHLETE SPRINT PERFORMANCE")
print("="*80)

print("\nANSWERS TO QUESTIONS:")
print("\nPart A: High Bias (Linear Model)")
print("-" * 40)
print("If the linear model consistently predicts sprint times that are too fast or too slow")
print("compared to actual results, this shows HIGH BIAS. The model is too simple and assumes")
print("a perfectly linear relationship between training hours and sprint times. It fails to")
print("capture the true complexity of athletic performance (like diminishing returns from")
print("training, fatigue effects, or optimal training zones), leading to systematic prediction")
print("errors (underfitting).")

print("\nPart B: High Variance (Polynomial Model)")
print("-" * 40)
print("If the polynomial model fits the training data almost perfectly but gives very different")
print("predictions when tested on new athletes, this shows HIGH VARIANCE. The model is too")
print("complex and memorizes the training data, including random noise from factors like weather,")
print("diet, or daily fatigue. It's overly sensitive to small fluctuations and doesn't capture")
print("the general pattern, so it fails to predict new athletes' performance (overfitting).")

print("\nPart C: Which Model Generalizes Better?")
print("-" * 40)
print("The BALANCED MODEL (Polynomial degree 3) is likely to generalize better to new athletes.")
print("Why? It achieves the right balance:")
print("  - Flexible enough to capture non-linear patterns (training improvements plateau)")
print("  - Simple enough to ignore random noise (daily variations in performance)")
print("  - Captures the general trend without memorizing individual quirks")
print("This model will make reliable predictions for new athletes.")

print("\nPart D: Explaining to a Coach (Non-Technical)")
print("-" * 40)
print("HIGH BIAS is like a coach who uses the same training plan for every athlete.")
print("  - Too rigid, doesn't adapt to individual needs")
print("  - Consistent mistakes because the approach is too simple")
print("  - Example: 'Everyone should run 10 miles daily' (ignores sprinters vs marathoners)")
print()
print("HIGH VARIANCE is like a coach who completely changes the training plan based on")
print("one bad practice session.")
print("  - Too reactive, doesn't see the bigger picture")
print("  - Inconsistent, doesn't work when conditions change")
print("  - Example: Athlete had a bad day due to poor sleep, so coach assumes they need")
print("    completely different training (overreacting to noise)")
print()
print("BALANCED APPROACH is like a coach who adapts training based on proven principles")
print("and general patterns.")
print("  - Flexible enough to adjust for athlete type (sprinter vs distance)")
print("  - Stable enough to stick with proven methods")
print("  - Example: Adjusts intensity based on athlete's progress over weeks, not daily mood")
print()
print("KEY INSIGHT FOR COACHES:")
print("  - High Bias = Ignoring important differences (too simple)")
print("  - High Variance = Overreacting to random noise (too complex)")
print("  - Good Model = Responds to real patterns, ignores random fluctuations")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
