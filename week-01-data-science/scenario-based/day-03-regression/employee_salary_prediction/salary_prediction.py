import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

print("=" * 70)
print("MULTIPLE LINEAR REGRESSION - EMPLOYEE SALARY PREDICTION")
print("=" * 70)

print("\nScenario: Predicting Employee Salary Based on Multiple Factors")
print("\nA company wants to predict employee salary based on:")
print("  • Years of Experience")
print("  • Education Level (1 = Bachelor, 2 = Master, 3 = PhD)")
print("  • Number of Skills Known")
print("  • Performance Rating (1 to 5)")
print("\nFormula: Salary = b0 + b1(Experience) + b2(Education) + b3(Skills) + b4(Performance)")

data = {
    "Years_Experience": [2, 3, 5, 7, 4, 6, 8, 10, 1, 9, 3, 5, 7, 2, 6],
    "Education_Level": [1, 1, 2, 2, 1, 2, 3, 3, 1, 3, 1, 2, 2, 1, 3],
    "Skills_Known": [3, 4, 5, 6, 4, 5, 7, 8, 2, 7, 3, 5, 6, 3, 6],
    "Performance_Rating": [3, 4, 4, 5, 3, 4, 5, 5, 3, 5, 3, 4, 4, 3, 5],
    "Salary_Lakhs": [4.5, 5.2, 7.0, 9.5, 6.0, 8.0, 12.0, 15.0, 3.8, 13.5, 5.0, 7.5, 9.0, 4.8, 11.0]
}

df = pd.DataFrame(data)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

df.to_csv('employee_salary_prediction/employee_salary_dataset.csv', index=False)
print("\n✓ Dataset saved as 'employee_salary_dataset.csv'")

X = df[['Years_Experience', 'Education_Level', 'Skills_Known', 'Performance_Rating']]
y = df['Salary_Lakhs']

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
equation = "Salary = {:.2f}".format(model.intercept_)
for feature, coef in zip(X.columns, model.coef_):
    sign = "+" if coef >= 0 else ""
    equation += " {} {:.2f}({})".format(sign, coef, feature)
print(equation)

y_pred = model.predict(X_test)

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA")
print("=" * 70)
test_results = X_test.copy()
test_results['Actual_Salary'] = y_test.values
test_results['Predicted_Salary'] = y_pred
test_results['Error'] = test_results['Actual_Salary'] - test_results['Predicted_Salary']
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
print("PREDICT NEW EMPLOYEE SALARIES")
print("=" * 70)
new_employees = pd.DataFrame({
    'Years_Experience': [4, 8, 6],
    'Education_Level': [1, 3, 2],
    'Skills_Known': [4, 7, 5],
    'Performance_Rating': [4, 5, 4]
})
new_predictions = model.predict(new_employees)
print("\nNew Employee Predictions:")
for i, (idx, row) in enumerate(new_employees.iterrows()):
    print("\nEmployee {}:".format(i+1))
    print("  Experience: {} years, Education: Level {}, Skills: {}, Performance: {}".format(
        row['Years_Experience'], row['Education_Level'], row['Skills_Known'], row['Performance_Rating']))
    print("  Predicted Salary: {:.2f} lakhs".format(new_predictions[i]))

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(df['Years_Experience'], df['Salary_Lakhs'], color='blue', s=100, alpha=0.6, edgecolors='black')
axes[0, 0].set_xlabel('Years of Experience', fontsize=11, weight='bold')
axes[0, 0].set_ylabel('Salary (lakhs)', fontsize=11, weight='bold')
axes[0, 0].set_title('Experience vs Salary', fontsize=12, weight='bold')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(df['Education_Level'], df['Salary_Lakhs'], color='green', s=100, alpha=0.6, edgecolors='black')
axes[0, 1].set_xlabel('Education Level', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Salary (lakhs)', fontsize=11, weight='bold')
axes[0, 1].set_title('Education vs Salary', fontsize=12, weight='bold')
axes[0, 1].set_xticks([1, 2, 3])
axes[0, 1].set_xticklabels(['Bachelor', 'Master', 'PhD'])
axes[0, 1].grid(alpha=0.3)

axes[1, 0].scatter(df['Skills_Known'], df['Salary_Lakhs'], color='red', s=100, alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('Number of Skills', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Salary (lakhs)', fontsize=11, weight='bold')
axes[1, 0].set_title('Skills vs Salary', fontsize=12, weight='bold')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(y_test, y_pred, color='purple', s=100, alpha=0.6, edgecolors='black')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1, 1].set_xlabel('Actual Salary (lakhs)', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Predicted Salary (lakhs)', fontsize=11, weight='bold')
axes[1, 1].set_title('Actual vs Predicted', fontsize=12, weight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('employee_salary_prediction/salary_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'salary_analysis.png'")

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
