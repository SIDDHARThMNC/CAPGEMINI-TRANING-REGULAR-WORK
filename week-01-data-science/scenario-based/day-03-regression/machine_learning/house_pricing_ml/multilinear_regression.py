import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

data = {
    "Area_sqft": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 900, 1600, 1400, 2100],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 2, 3, 3, 4],
    "Age_years": [15, 10, 8, 5, 4, 3, 2, 1, 12, 6, 7, 3],
    "Distance_km": [12, 10, 8, 6, 5, 4, 3, 2, 11, 7, 9, 4],
    "Price_lakhs": [40, 50, 62, 75, 90, 105, 120, 140, 45, 80, 70, 110]
}

df = pd.DataFrame(data)

print("=" * 70)
print("MULTIPLE LINEAR REGRESSION - HOUSE PRICE PREDICTION")
print("=" * 70)
print("\nScenario: Predicting House Price Based on Multiple Factors")
print("\nA real estate company wants to predict the price of a house based on:")
print("  • Area (square feet)")
print("  • Number of bedrooms")
print("  • Age of the house (years)")
print("  • Distance from city center (km)")
print("\nFormula: Price = b0 + b1(Area) + b2(Bedrooms) + b3(Age) + b4(Distance)")

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df)

df.to_csv('machine_learning/house_pricing_ml/house_dataset.csv', index=False)
print("\n✓ Dataset saved as 'house_dataset.csv'")

X = df[['Area_sqft', 'Bedrooms', 'Age_years', 'Distance_km']]
y = df['Price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)
print(f"Intercept (b0): {model.intercept_:.4f}")
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

print("\n" + "=" * 70)
print("EQUATION")
print("=" * 70)
equation = f"Price = {model.intercept_:.2f}"
for feature, coef in zip(X.columns, model.coef_):
    sign = "+" if coef >= 0 else ""
    equation += f" {sign} {coef:.2f}({feature})"
print(equation)

y_pred = model.predict(X_test)

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA")
print("=" * 70)
test_results = X_test.copy()
test_results['Actual_Price'] = y_test.values
test_results['Predicted_Price'] = y_pred
test_results['Error'] = test_results['Actual_Price'] - test_results['Predicted_Price']
print(test_results.to_string(index=False))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print(f"Mean Absolute Error (MAE): {mae:.2f} lakhs")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} lakhs")
print(f"R² Score: {r2:.4f}")

print("\n" + "=" * 70)
print("PREDICT NEW HOUSE PRICES")
print("=" * 70)
new_houses = pd.DataFrame({
    'Area_sqft': [1300, 1900, 2400],
    'Bedrooms': [3, 4, 5],
    'Age_years': [6, 3, 2],
    'Distance_km': [8, 5, 3]
})
new_predictions = model.predict(new_houses)
print("\nNew House Predictions:")
for i, (idx, row) in enumerate(new_houses.iterrows()):
    print(f"\nHouse {i+1}:")
    print(f"  Area: {row['Area_sqft']} sqft, Bedrooms: {row['Bedrooms']}, Age: {row['Age_years']} years, Distance: {row['Distance_km']} km")
    print(f"  Predicted Price: {new_predictions[i]:.2f} lakhs")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(df['Area_sqft'], df['Price_lakhs'], color='blue', s=100, alpha=0.6, edgecolors='black')
axes[0, 0].set_xlabel('Area (sqft)', fontsize=11, weight='bold')
axes[0, 0].set_ylabel('Price (lakhs)', fontsize=11, weight='bold')
axes[0, 0].set_title('Area vs Price', fontsize=12, weight='bold')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(df['Bedrooms'], df['Price_lakhs'], color='green', s=100, alpha=0.6, edgecolors='black')
axes[0, 1].set_xlabel('Number of Bedrooms', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Price (lakhs)', fontsize=11, weight='bold')
axes[0, 1].set_title('Bedrooms vs Price', fontsize=12, weight='bold')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].scatter(df['Age_years'], df['Price_lakhs'], color='red', s=100, alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('Age (years)', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Price (lakhs)', fontsize=11, weight='bold')
axes[1, 0].set_title('Age vs Price', fontsize=12, weight='bold')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(y_test, y_pred, color='purple', s=100, alpha=0.6, edgecolors='black')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1, 1].set_xlabel('Actual Price (lakhs)', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Predicted Price (lakhs)', fontsize=11, weight='bold')
axes[1, 1].set_title('Actual vs Predicted', fontsize=12, weight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('machine_learning/house_pricing_ml/house_price_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'house_price_analysis.png'")

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)
print(feature_importance.to_string(index=False))

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
