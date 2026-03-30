"""
SCENARIO: Predicting Student Exam Scores - Bias vs Variance Tradeoff

A school wants to build a model to predict student exam scores based on study hours. They collect data
from 30 students, but the scores are noisy because of other factors (like sleep, stress, or health).

They try two different models:
- Linear Model (straight line) → very simple, assumes scores increase perfectly with study hours.
- Polynomial Model (degree 10 curve) → very complex, tries to follow every bump in the data.

QUESTIONS:
- Part A: If the linear model consistently predicts too low or too high compared to actual scores,
  what does this show about bias?
- Part B: If the polynomial model fits the training data almost perfectly but gives very different 
  predictions when tested on new students, what does this show about variance?
- Part C: Which model is likely to generalize better to new students, and why?
- Part D (Applied): How would you explain the difference between "high bias" and "high variance" to
  a teacher who doesn't know machine learning?
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)
X = np.linspace(0, 6, 30).reshape(-1, 1)
y = (10 * np.sin(X)).ravel() + np.random.normal(scale=3, size=30)

linear_model = make_pipeline(PolynomialFeatures(1), LinearRegression())
poly_model = make_pipeline(PolynomialFeatures(10), LinearRegression())
balanced_model = make_pipeline(PolynomialFeatures(3), LinearRegression())

linear_model.fit(X, y)
poly_model.fit(X, y)
balanced_model.fit(X, y)

X_test = np.linspace(0, 6, 100).reshape(-1, 1)
y_linear = linear_model.predict(X_test)
y_poly = poly_model.predict(X_test)
y_balanced = balanced_model.predict(X_test)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_linear, color="red", label="Linear Model")
plt.title("High Bias (Underfitting)")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_balanced, color="green", label="Poly deg=3")
plt.title("Balanced Model")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_poly, color="blue", label="Poly deg=10")
plt.title("High Variance (Overfitting)")
plt.legend()

plt.suptitle("Bias vs Variance: Student Exam Scores", fontsize=14)
plt.tight_layout()
plt.savefig('Day_6_Mar01_Ridge_CrossValidation/bias_variance_student_exam.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*80)
print("BIAS VS VARIANCE TRADEOFF - STUDENT EXAM SCORES")
print("="*80)

print("\nANSWERS TO QUESTIONS:")
print("\nPart A: High Bias (Linear Model)")
print("-" * 40)
print("If the linear model consistently predicts too low or too high compared to actual scores,")
print("this shows HIGH BIAS. The model is too simple to capture the true relationship between")
print("study hours and exam scores. It makes strong assumptions (linear relationship) that don't")
print("match reality, leading to systematic errors (underfitting).")

print("\nPart B: High Variance (Polynomial Model)")
print("-" * 40)
print("If the polynomial model fits the training data almost perfectly but gives very different")
print("predictions when tested on new students, this shows HIGH VARIANCE. The model is too complex")
print("and memorizes the training data, including noise. It's overly sensitive to small changes")
print("in the data and doesn't generalize well to new students (overfitting).")

print("\nPart C: Which Model Generalizes Better?")
print("-" * 40)
print("The BALANCED MODEL (Polynomial degree 3) is likely to generalize better to new students.")
print("Why? It strikes a balance between:")
print("  - Being flexible enough to capture the true pattern (not too simple)")
print("  - Not being so complex that it memorizes noise (not too complicated)")
print("This is the essence of the bias-variance tradeoff.")

print("\nPart D: Explaining to a Teacher (Non-Technical)")
print("-" * 40)
print("HIGH BIAS is like a teacher who uses only one teaching method for all students.")
print("  - Too rigid, misses individual differences")
print("  - Consistent mistakes because the approach is too simple")
print("  - Example: 'All students learn best by reading' (ignores visual/hands-on learners)")
print()
print("HIGH VARIANCE is like a teacher who creates a completely custom lesson for each student")
print("based on one day's observation.")
print("  - Too personalized, doesn't work when student behavior changes")
print("  - Inconsistent, doesn't generalize to new situations")
print("  - Example: Student was tired one day, so teacher assumes they're always tired")
print()
print("BALANCED APPROACH is like a teacher who adapts methods based on general learning styles")
print("but doesn't over-customize.")
print("  - Flexible enough to handle differences")
print("  - Stable enough to work consistently")
print("  - Example: Uses mix of reading, visuals, and practice based on proven patterns")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
