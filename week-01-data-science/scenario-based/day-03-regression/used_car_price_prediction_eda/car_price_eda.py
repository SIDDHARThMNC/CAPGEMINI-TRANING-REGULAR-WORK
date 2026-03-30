import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

print("=" * 70)
print("MULTIPLE LINEAR REGRESSION WITH EDA - USED CAR PRICE PREDICTION")
print("=" * 70)

print("\nScenario: Predicting Used Car Prices")
print("\nA used car dealership wants to predict the selling price based on:")
print("  ⚙️  Engine Size (Liters)")
print("  🛣️  Mileage (thousand km driven)")
print("  📅 Car Age (years)")
print("  💨 Horsepower")

data = {
    "Engine_Size": [1.2, 1.5, 1.8, 2.0, 2.2, 1.3, 1.6, 2.4, 2.0, 1.4, 1.7, 2.5, 1.8, 2.2, 1.5],
    "Mileage": [90, 70, 60, 50, 40, 85, 65, 30, 45, 80, 55, 25, 50, 35, 75],
    "Age": [8, 6, 5, 4, 3, 7, 6, 2, 4, 7, 5, 1, 3, 2, 6],
    "Horsepower": [80, 95, 110, 130, 150, 85, 100, 180, 140, 90, 115, 200, 125, 160, 105],
    "Price": [3.5, 5, 6, 8, 10, 4, 5.5, 14, 9, 4.5, 6.5, 16, 8.5, 12, 5.2]
}

df = pd.DataFrame(data)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

df.to_csv('used_car_price_prediction_eda/car_price_dataset.csv', index=False)
print("\n✓ Dataset saved as 'car_price_dataset.csv'")

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

print("\n1. DATASET INFO:")
print("   Total Records: {}".format(len(df)))
print("   Features: {}".format(list(df.columns[:-1])))
print("   Target: Price (lakhs)")

print("\n2. STATISTICAL SUMMARY:")
print(df.describe().to_string())

print("\n3. CORRELATION ANALYSIS:")
correlation = df.corr()
print(correlation['Price'].sort_values(ascending=False).to_string())

print("\n" + "=" * 70)
print("BUILDING MULTIPLE LINEAR REGRESSION MODEL")
print("=" * 70)

X = df[['Engine_Size', 'Mileage', 'Age', 'Horsepower']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nMODEL PARAMETERS:")
print("Intercept (b0): {:.4f}".format(model.intercept_))
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print("  {}: {:.4f}".format(feature, coef))

print("\nEQUATION:")
equation = "Price = {:.2f}".format(model.intercept_)
for feature, coef in zip(X.columns, model.coef_):
    sign = "+" if coef >= 0 else ""
    equation += " {} {:.2f}({})".format(sign, coef, feature)
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
print("Mean Absolute Error (MAE): {:.2f} lakhs".format(mae))
print("Root Mean Squared Error (RMSE): {:.2f} lakhs".format(rmse))
print("R-squared Score: {:.4f}".format(r2))

print("\n" + "=" * 70)
print("PREDICT NEW CAR PRICES")
print("=" * 70)
new_cars = pd.DataFrame({
    'Engine_Size': [1.6, 2.0, 1.8],
    'Mileage': [60, 40, 50],
    'Age': [5, 3, 4],
    'Horsepower': [105, 140, 120]
})
new_predictions = model.predict(new_cars)
print("\nNew Car Predictions:")
for i, (idx, row) in enumerate(new_cars.iterrows()):
    print("\nCar {}:".format(i+1))
    print("  Engine: {}L, Mileage: {}k km, Age: {} years, HP: {}".format(
        row['Engine_Size'], row['Mileage'], row['Age'], row['Horsepower']))
    print("  Predicted Price: {:.2f} lakhs".format(new_predictions[i]))

print("\nCreating EDA visualizations...")

fig = plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
plt.scatter(df['Engine_Size'], df['Price'], color='blue', s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Engine Size (L)', fontsize=10, weight='bold')
plt.ylabel('Price (lakhs)', fontsize=10, weight='bold')
plt.title('Engine Size vs Price', fontsize=11, weight='bold')
plt.grid(alpha=0.3)

plt.subplot(3, 3, 2)
plt.scatter(df['Mileage'], df['Price'], color='green', s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Mileage (thousand km)', fontsize=10, weight='bold')
plt.ylabel('Price (lakhs)', fontsize=10, weight='bold')
plt.title('Mileage vs Price', fontsize=11, weight='bold')
plt.grid(alpha=0.3)

plt.subplot(3, 3, 3)
plt.scatter(df['Age'], df['Price'], color='red', s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Age (years)', fontsize=10, weight='bold')
plt.ylabel('Price (lakhs)', fontsize=10, weight='bold')
plt.title('Age vs Price', fontsize=11, weight='bold')
plt.grid(alpha=0.3)

plt.subplot(3, 3, 4)
plt.scatter(df['Horsepower'], df['Price'], color='orange', s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Horsepower', fontsize=10, weight='bold')
plt.ylabel('Price (lakhs)', fontsize=10, weight='bold')
plt.title('Horsepower vs Price', fontsize=11, weight='bold')
plt.grid(alpha=0.3)

plt.subplot(3, 3, 5)
mask = np.triu(np.ones_like(correlation, dtype=bool))
cmap = plt.cm.coolwarm
im = plt.imshow(correlation, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.colorbar(im)
plt.title('Correlation Heatmap', fontsize=11, weight='bold')

plt.subplot(3, 3, 6)
plt.scatter(y_test, y_pred, color='purple', s=100, alpha=0.6, edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price (lakhs)', fontsize=10, weight='bold')
plt.ylabel('Predicted Price (lakhs)', fontsize=10, weight='bold')
plt.title('Actual vs Predicted', fontsize=11, weight='bold')
plt.grid(alpha=0.3)

plt.subplot(3, 3, 7)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='brown', s=100, alpha=0.6, edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price (lakhs)', fontsize=10, weight='bold')
plt.ylabel('Residuals', fontsize=10, weight='bold')
plt.title('Residual Plot', fontsize=11, weight='bold')
plt.grid(alpha=0.3)

plt.subplot(3, 3, 8)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(model.coef_)
}).sort_values('Coefficient', ascending=True)
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='teal', alpha=0.7)
plt.xlabel('Absolute Coefficient', fontsize=10, weight='bold')
plt.title('Feature Importance', fontsize=11, weight='bold')
plt.grid(alpha=0.3, axis='x')

plt.subplot(3, 3, 9)
plt.hist(df['Price'], bins=8, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Price (lakhs)', fontsize=10, weight='bold')
plt.ylabel('Frequency', fontsize=10, weight='bold')
plt.title('Price Distribution', fontsize=11, weight='bold')
plt.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('used_car_price_prediction_eda/car_price_eda_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ EDA Visualization saved as 'car_price_eda_analysis.png'")

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)
print(feature_importance_df.to_string(index=False))

print("\n" + "=" * 70)
print("KEY INSIGHTS FROM EDA")
print("=" * 70)
print("1. Horsepower has the strongest positive correlation with price")
print("2. Mileage and Age have negative correlation (higher values = lower price)")
print("3. Engine Size shows positive correlation with price")
print("4. Model achieves R² score of {:.4f} indicating good fit".format(r2))

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
