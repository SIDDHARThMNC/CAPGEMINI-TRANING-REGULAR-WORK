import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Starting Linear Regression Analysis...")
print("=" * 70)

data = {
    "Area_sqft": [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500],
    "Price_lakhs": [30, 40, 50, 60, 68, 75, 85, 95, 105, 120]
}

df = pd.DataFrame(data)
print("\nDataset:")
print(df.to_string())

X = df[['Area_sqft']]
y = df['Price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)
print("Coefficient (Slope): {:.4f}".format(model.coef_[0]))
print("Intercept: {:.4f}".format(model.intercept_))

y_pred = model.predict(X_test)

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA")
print("=" * 70)
test_data = list(zip(X_test['Area_sqft'], y_test, y_pred))
for i, (area, actual, predicted) in enumerate(test_data):
    print("Test {}: Area = {} sqft, Actual = {} lakhs, Predicted = {:.2f} lakhs".format(
        i+1, area, actual, predicted))

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print("Mean Absolute Error (MAE): {:.2f} lakhs".format(mae))
print("R-squared Score: {:.4f}".format(r2))

print("\n" + "=" * 70)
print("PREDICT NEW HOUSE PRICES")
print("=" * 70)
new_areas = [[1500], [2300], [3000]]
new_predictions = model.predict(new_areas)
for area, price in zip(new_areas, new_predictions):
    print("Area = {} sqft -> Predicted Price = {:.2f} lakhs".format(area[0], price))

print("\nCreating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(X, y, color='blue', label='Actual Data', s=100, alpha=0.7)
axes[0].plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
axes[0].scatter(X_test, y_test, color='green', s=150, marker='s', label='Test Data', 
                edgecolors='black', linewidths=2)
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
output_file = 'linear_regression_output.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization saved as: {}".format(output_file))
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
