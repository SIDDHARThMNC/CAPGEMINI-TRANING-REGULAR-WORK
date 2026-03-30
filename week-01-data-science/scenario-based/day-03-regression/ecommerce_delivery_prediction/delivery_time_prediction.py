import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

print("=" * 70)
print("MULTIPLE LINEAR REGRESSION - DELIVERY TIME PREDICTION")
print("=" * 70)

print("\nScenario: Predicting Delivery Time for E-commerce Orders")
print("\nAn e-commerce company wants to predict delivery time based on:")
print("  • Distance to customer (km)")
print("  • Number of items in the order")
print("  • Traffic level (1 = Low, 2 = Medium, 3 = High)")
print("  • Warehouse processing time (hours)")
print("\nFormula: DeliveryTime = b0 + b1(Distance) + b2(Items) + b3(Traffic) + b4(ProcessingTime)")

data = {
    "Distance_km": [5, 10, 15, 20, 8, 12, 18, 25, 7, 22, 14, 30, 6, 16, 28],
    "Items_Count": [2, 3, 1, 5, 2, 4, 3, 6, 1, 4, 2, 7, 2, 3, 5],
    "Traffic_Level": [1, 2, 1, 3, 1, 2, 2, 3, 1, 3, 2, 3, 1, 2, 3],
    "Processing_Time_hrs": [0.5, 1.0, 0.5, 1.5, 0.5, 1.0, 1.0, 2.0, 0.5, 1.5, 1.0, 2.0, 0.5, 1.0, 1.5],
    "Delivery_Time_hrs": [2.5, 4.0, 3.5, 8.0, 3.0, 5.5, 6.0, 10.5, 2.8, 9.0, 5.0, 12.0, 2.2, 5.8, 10.0]
}

df = pd.DataFrame(data)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

df.to_csv('ecommerce_delivery_prediction/delivery_dataset.csv', index=False)
print("\n✓ Dataset saved as 'delivery_dataset.csv'")

X = df[['Distance_km', 'Items_Count', 'Traffic_Level', 'Processing_Time_hrs']]
y = df['Delivery_Time_hrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)
print("Intercept (b0): {:.4f}".format(model.intercept_))
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print("  {}: {:.4f}".format(feature, coef))

print("\n" + "=" * 70)
print("EQUATION")
print("=" * 70)
equation = "DeliveryTime = {:.2f}".format(model.intercept_)
for feature, coef in zip(X.columns, model.coef_):
    sign = "+" if coef >= 0 else ""
    equation += " {} {:.2f}({})".format(sign, coef, feature)
print(equation)

y_pred = model.predict(X_test)

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA")
print("=" * 70)
test_results = X_test.copy()
test_results['Actual_Time'] = y_test.values
test_results['Predicted_Time'] = y_pred
test_results['Error'] = test_results['Actual_Time'] - test_results['Predicted_Time']
print(test_results.to_string(index=False))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print("Mean Absolute Error (MAE): {:.2f} hours".format(mae))
print("Root Mean Squared Error (RMSE): {:.2f} hours".format(rmse))
print("R-squared Score: {:.4f}".format(r2))

print("\n" + "=" * 70)
print("PREDICT NEW DELIVERY TIMES")
print("=" * 70)
new_orders = pd.DataFrame({
    'Distance_km': [10, 20, 15],
    'Items_Count': [3, 5, 2],
    'Traffic_Level': [1, 3, 2],
    'Processing_Time_hrs': [1.0, 1.5, 0.5]
})
new_predictions = model.predict(new_orders)
print("\nNew Order Predictions:")
for i, (idx, row) in enumerate(new_orders.iterrows()):
    print("\nOrder {}:".format(i+1))
    traffic_label = {1: 'Low', 2: 'Medium', 3: 'High'}[row['Traffic_Level']]
    print("  Distance: {} km, Items: {}, Traffic: {}, Processing: {} hrs".format(
        row['Distance_km'], row['Items_Count'], traffic_label, row['Processing_Time_hrs']))
    print("  Predicted Delivery Time: {:.2f} hours".format(new_predictions[i]))

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(df['Distance_km'], df['Delivery_Time_hrs'], color='blue', s=100, alpha=0.6, edgecolors='black')
axes[0, 0].set_xlabel('Distance (km)', fontsize=11, weight='bold')
axes[0, 0].set_ylabel('Delivery Time (hours)', fontsize=11, weight='bold')
axes[0, 0].set_title('Distance vs Delivery Time', fontsize=12, weight='bold')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(df['Items_Count'], df['Delivery_Time_hrs'], color='green', s=100, alpha=0.6, edgecolors='black')
axes[0, 1].set_xlabel('Number of Items', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Delivery Time (hours)', fontsize=11, weight='bold')
axes[0, 1].set_title('Items vs Delivery Time', fontsize=12, weight='bold')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].scatter(df['Traffic_Level'], df['Delivery_Time_hrs'], color='red', s=100, alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('Traffic Level', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Delivery Time (hours)', fontsize=11, weight='bold')
axes[1, 0].set_title('Traffic vs Delivery Time', fontsize=12, weight='bold')
axes[1, 0].set_xticks([1, 2, 3])
axes[1, 0].set_xticklabels(['Low', 'Medium', 'High'])
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(y_test, y_pred, color='purple', s=100, alpha=0.6, edgecolors='black')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1, 1].set_xlabel('Actual Delivery Time (hours)', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Predicted Delivery Time (hours)', fontsize=11, weight='bold')
axes[1, 1].set_title('Actual vs Predicted', fontsize=12, weight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ecommerce_delivery_prediction/delivery_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'delivery_analysis.png'")

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
