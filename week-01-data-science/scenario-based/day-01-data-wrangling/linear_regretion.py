import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys

print("=" * 70, flush=True)
print("LINEAR REGRESSION - HOUSE PRICE PREDICTION", flush=True)
print("=" * 70, flush=True)

data = {
    "Area_sqft": [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500],
    "Price_lakhs": [30, 40, 50, 60, 68, 75, 85, 95, 105, 120]
}

df = pd.DataFrame(data)
print("\nDataset:", flush=True)
print(df, flush=True)

X = df[['Area_sqft']]
y = df['Price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 70, flush=True)
print("MODEL PARAMETERS", flush=True)
print("=" * 70, flush=True)
print(f"Coefficient (Slope): {model.coef_[0]:.4f}", flush=True)
print(f"Intercept: {model.intercept_:.4f}", flush=True)

y_pred = model.predict(X_test)

print("\n" + "=" * 70, flush=True)
print("PREDICTIONS ON TEST DATA", flush=True)
print("=" * 70, flush=True)
for i, (area, actual, predicted) in enumerate(zip(X_test['Area_sqft'], y_test, y_pred)):
    print(f"Test {i+1}: Area = {area} sqft, Actual = {actual} lakhs, Predicted = {predicted:.2f} lakhs", flush=True)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70, flush=True)
print("MODEL EVALUATION", flush=True)
print("=" * 70, flush=True)
print(f"Mean Absolute Error (MAE): {mae:.2f} lakhs", flush=True)
print(f"R² Score: {r2:.4f}", flush=True)

print("\n" + "=" * 70, flush=True)
print("PREDICT NEW HOUSE PRICES", flush=True)
print("=" * 70, flush=True)
new_areas = [[1500], [2300], [3000]]
new_predictions = model.predict(new_areas)
for area, price in zip(new_areas, new_predictions):
    print(f"Area = {area[0]} sqft → Predicted Price = {price:.2f} lakhs", flush=True)

print("\nGenerating visualization...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(X, y, color='blue', label='Actual Data', s=100, alpha=0.7)
axes[0].plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
axes[0].scatter(X_test, y_test, color='green', s=150, marker='s', label='Test Data', edgecolors='black', linewidths=2)
new_areas_array = np.array(new_areas)
axes[0].scatter(new_areas_array, new_predictions, color='orange', s=200, marker='*', 
                label='New Predictions', edgecolors='black', linewidths=2)
axes[0].set_xlabel('Area (sqft)', fontsize=12, weight='bold')
axes[0].set_ylabel('Price (lakhs)', fontsize=12, weight='bold')
axes[0].set_title('House Price Prediction - Linear Regression', fontsize=14, weight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, color='purple', s=100, alpha=0.7, edgecolors='black')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price (lakhs)', fontsize=12, weight='bold')
axes[1].set_ylabel('Residuals', fontsize=12, weight='bold')
axes[1].set_title('Residual Plot - Model Accuracy', fontsize=14, weight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('house_price_prediction_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'house_price_prediction_visualization.png'", flush=True)

print("\n" + "=" * 70, flush=True)
print("EXECUTION COMPLETE!", flush=True)
print("=" * 70, flush=True)
