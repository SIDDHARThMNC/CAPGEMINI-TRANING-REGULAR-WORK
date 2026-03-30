"""
SCENARIO: Predicting Patient Recovery Time

A hospital research team wants to build a model to predict patient recovery time (in days) after
surgery based on several factors.

FEATURES:
- Age of the patient
- Number of hours of post-surgery physiotherapy per week
- Pre-existing health conditions (numeric severity score)
- Length of hospital stay (days)
- Average sleep hours during recovery

DATASET:
- 1,000 patients

OBJECTIVES:
1. Use Linear Regression for recovery time prediction
2. Apply 5-Fold Cross-Validation
3. Evaluate using R² as the performance metric
4. Analyze model stability across folds
5. Identify key factors affecting recovery time

BUSINESS IMPACT:
- Better resource planning for hospital beds
- Personalized recovery plans
- Early identification of patients needing extended care
- Improved patient communication about expected recovery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("PATIENT RECOVERY TIME PREDICTION - LINEAR REGRESSION WITH CROSS-VALIDATION")
print("="*80)

n_patients = 1000

data = {
    'Patient_ID': range(1, n_patients + 1),
    'Age': np.random.uniform(18, 85, n_patients),
    'Physiotherapy_Hours_Per_Week': np.random.uniform(0, 20, n_patients),
    'Health_Condition_Severity': np.random.uniform(0, 10, n_patients),
    'Hospital_Stay_Days': np.random.uniform(1, 15, n_patients),
    'Average_Sleep_Hours': np.random.uniform(4, 10, n_patients)
}

df = pd.DataFrame(data)

df['Recovery_Time_Days'] = (
    df['Age'] * 0.3 +
    df['Physiotherapy_Hours_Per_Week'] * (-1.5) +
    df['Health_Condition_Severity'] * 2.5 +
    df['Hospital_Stay_Days'] * 1.8 +
    df['Average_Sleep_Hours'] * (-0.8) +
    np.random.normal(0, 5, n_patients) +
    20
).clip(5, 120)

print("\nDataset Created:")
print(f"Total Patients: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Statistics:")
print(df.describe())

print("\nRecovery Time Distribution:")
print(f"Mean Recovery Time: {df['Recovery_Time_Days'].mean():.2f} days")
print(f"Median Recovery Time: {df['Recovery_Time_Days'].median():.2f} days")
print(f"Min Recovery Time: {df['Recovery_Time_Days'].min():.2f} days")
print(f"Max Recovery Time: {df['Recovery_Time_Days'].max():.2f} days")

df.to_csv('Day_6_Mar01_Ridge_CrossValidation/patient_recovery_data.csv', index=False)
print("\nDataset saved to: patient_recovery_data.csv")

X = df[['Age', 'Physiotherapy_Hours_Per_Week', 'Health_Condition_Severity', 
        'Hospital_Stay_Days', 'Average_Sleep_Hours']]
y = df['Recovery_Time_Days']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*80)
print("LINEAR REGRESSION WITH 5-FOLD CROSS-VALIDATION")
print("="*80)

lr_model = LinearRegression()

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(lr_model, X_scaled, y, cv=kfold, scoring='r2')

print("\n5-Fold Cross-Validation Results:")
print(f"R² Scores for each fold: {cv_scores}")
print(f"Mean R² Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
print(f"Min R² Score: {cv_scores.min():.4f}")
print(f"Max R² Score: {cv_scores.max():.4f}")

scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

cv_results = cross_validate(
    lr_model, 
    X_scaled, 
    y, 
    cv=kfold, 
    scoring=scoring_metrics,
    return_train_score=True
)

print("\n" + "="*80)
print("DETAILED PERFORMANCE METRICS")
print("="*80)

print("\nR² Scores:")
print(f"  Training R² (mean): {cv_results['train_r2'].mean():.4f} ± {cv_results['train_r2'].std():.4f}")
print(f"  Validation R² (mean): {cv_results['test_r2'].mean():.4f} ± {cv_results['test_r2'].std():.4f}")

print("\nMean Squared Error (MSE):")
print(f"  Training MSE (mean): {-cv_results['train_neg_mean_squared_error'].mean():.4f} ± {cv_results['train_neg_mean_squared_error'].std():.4f}")
print(f"  Validation MSE (mean): {-cv_results['test_neg_mean_squared_error'].mean():.4f} ± {cv_results['test_neg_mean_squared_error'].std():.4f}")

print("\nMean Absolute Error (MAE):")
print(f"  Training MAE (mean): {-cv_results['train_neg_mean_absolute_error'].mean():.4f} days")
print(f"  Validation MAE (mean): {-cv_results['test_neg_mean_absolute_error'].mean():.4f} days")

overfitting_check = cv_results['train_r2'].mean() - cv_results['test_r2'].mean()
print(f"\nOverfitting Check:")
print(f"  Difference (Train R² - Val R²): {overfitting_check:.4f}")
if overfitting_check < 0.05:
    print("  Status: Excellent generalization")
elif overfitting_check < 0.1:
    print("  Status: Good generalization")
else:
    print("  Status: Possible overfitting detected")

lr_model.fit(X_scaled, y)
y_pred = lr_model.predict(X_scaled)

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients (Impact on Recovery Time):")
for idx, row in feature_importance.iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']}: {row['Coefficient']:.4f} ({direction} recovery time)")

fig = plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
plt.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black', linewidth=2)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold Number', fontweight='bold', fontsize=11)
plt.ylabel('R² Score', fontweight='bold', fontsize=11)
plt.title('5-Fold Cross-Validation: R² Scores', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 1])

plt.subplot(2, 3, 2)
metrics = ['R² (Train)', 'R² (Val)', 'MSE (Train)', 'MSE (Val)']
values = [
    cv_results['train_r2'].mean(),
    cv_results['test_r2'].mean(),
    -cv_results['train_neg_mean_squared_error'].mean() / 100,
    -cv_results['test_neg_mean_squared_error'].mean() / 100
]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = plt.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Score', fontweight='bold', fontsize=11)
plt.title('Performance Metrics Comparison\n(MSE scaled by 0.01)', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.subplot(2, 3, 3)
colors_feat = ['red' if x > 0 else 'green' for x in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], 
         color=colors_feat, edgecolor='black', linewidth=1.5)
plt.xlabel('Coefficient Value', fontweight='bold', fontsize=11)
plt.title('Feature Impact on Recovery Time\n(Red=Increases, Green=Decreases)', 
          fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 3, 4)
plt.scatter(y, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Recovery Time (days)', fontweight='bold', fontsize=11)
plt.ylabel('Predicted Recovery Time (days)', fontweight='bold', fontsize=11)
plt.title('Actual vs Predicted Recovery Time', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Recovery Time (days)', fontweight='bold', fontsize=11)
plt.ylabel('Residuals (days)', fontweight='bold', fontsize=11)
plt.title('Residual Plot', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.hist(df['Recovery_Time_Days'], bins=30, color='skyblue', edgecolor='black', linewidth=1.5)
plt.axvline(df['Recovery_Time_Days'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {df["Recovery_Time_Days"].mean():.1f} days')
plt.axvline(df['Recovery_Time_Days'].median(), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {df["Recovery_Time_Days"].median():.1f} days')
plt.xlabel('Recovery Time (days)', fontweight='bold', fontsize=11)
plt.ylabel('Number of Patients', fontweight='bold', fontsize=11)
plt.title('Distribution of Recovery Times', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Patient Recovery Time Prediction - Linear Regression with 5-Fold CV', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_6_Mar01_Ridge_CrossValidation/patient_recovery_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: patient_recovery_analysis.png")

plt.show()

results_df = pd.DataFrame({
    'Metric': ['Mean R² (CV)', 'Std R² (CV)', 'Mean MSE (CV)', 'Mean MAE (CV)', 
               'Train R²', 'Val R²', 'Overfitting Gap'],
    'Value': [
        cv_scores.mean(),
        cv_scores.std(),
        -cv_results['test_neg_mean_squared_error'].mean(),
        -cv_results['test_neg_mean_absolute_error'].mean(),
        cv_results['train_r2'].mean(),
        cv_results['test_r2'].mean(),
        overfitting_check
    ]
})

results_df.to_csv('Day_6_Mar01_Ridge_CrossValidation/patient_recovery_cv_results.csv', index=False)
print("Results saved to: patient_recovery_cv_results.csv")

print("\n" + "="*80)
print("KEY INSIGHTS & CLINICAL IMPLICATIONS")
print("="*80)

print("\n1. Model Performance:")
print(f"   The model explains {cv_scores.mean()*100:.2f}% of variance in recovery time")
print(f"   Average prediction error: ±{-cv_results['test_neg_mean_absolute_error'].mean():.2f} days")

print("\n2. Model Stability:")
print(f"   R² varies from {cv_scores.min():.4f} to {cv_scores.max():.4f} across folds")
if cv_scores.std() < 0.05:
    print("   Very stable model - consistent predictions across different patient groups")
elif cv_scores.std() < 0.1:
    print("   Stable model - reliable predictions")
else:
    print("   Some variability - model performance depends on patient characteristics")

print("\n3. Top Factors Affecting Recovery:")
for i, row in feature_importance.head(3).iterrows():
    impact = "prolongs" if row['Coefficient'] > 0 else "shortens"
    print(f"   {i+1}. {row['Feature']} - {impact} recovery time")

print("\n4. Clinical Recommendations:")
top_feature = feature_importance.iloc[0]
if 'Physiotherapy' in top_feature['Feature'] and top_feature['Coefficient'] < 0:
    print("   Increase physiotherapy hours to reduce recovery time")
if 'Health_Condition_Severity' in feature_importance['Feature'].values:
    severity_coef = feature_importance[feature_importance['Feature'] == 'Health_Condition_Severity']['Coefficient'].values[0]
    if severity_coef > 0:
        print("   Patients with pre-existing conditions need extended care planning")
if 'Sleep' in str(feature_importance['Feature'].values):
    sleep_coef = feature_importance[feature_importance['Feature'] == 'Average_Sleep_Hours']['Coefficient'].values[0]
    if sleep_coef < 0:
        print("   Ensure patients get adequate sleep during recovery")

print("\n5. Resource Planning:")
print(f"   Expected recovery time range: {df['Recovery_Time_Days'].quantile(0.25):.1f} - {df['Recovery_Time_Days'].quantile(0.75):.1f} days (50% of patients)")
print(f"   Plan for extended care (>90 days): {(df['Recovery_Time_Days'] > 90).sum()} patients ({(df['Recovery_Time_Days'] > 90).sum()/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
