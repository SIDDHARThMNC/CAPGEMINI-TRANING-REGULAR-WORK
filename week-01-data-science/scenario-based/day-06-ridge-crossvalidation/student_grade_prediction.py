"""
SCENARIO: University Student Grade Prediction with Cross-Validation

A university wants to build a predictive model to estimate student grades based on multiple factors.

FEATURES:
- Study hours per week
- Attendance percentage
- Previous exam score
- Average sleep hours

DATASET:
- 200 students

OBJECTIVES:
1. Use Ridge Regression for grade prediction
2. Apply K-Fold Cross-Validation (5 folds, shuffled) to check model stability
3. Multi-metric evaluation using R² and MSE
4. Compare training vs validation scores
5. Stratified K-Fold CV for pass/fail classification using Logistic Regression

EVALUATION STRATEGIES:
- Basic K-Fold CV: Assess R² score stability across folds
- Multi-metric: Compare R² and MSE on training and validation sets
- Stratified K-Fold: For classification task (pass/fail prediction)

BUSINESS IMPACT:
- Early identification of struggling students
- Resource allocation for tutoring
- Personalized academic support
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("STUDENT GRADE PREDICTION WITH RIDGE REGRESSION & CROSS-VALIDATION")
print("="*80)

n_students = 200

data = {
    'Student_ID': range(1, n_students + 1),
    'Study_Hours_Per_Week': np.random.uniform(2, 40, n_students),
    'Attendance_Percentage': np.random.uniform(40, 100, n_students),
    'Previous_Exam_Score': np.random.uniform(30, 100, n_students),
    'Average_Sleep_Hours': np.random.uniform(3, 10, n_students)
}

df = pd.DataFrame(data)

df['Final_Grade'] = (
    df['Study_Hours_Per_Week'] * 1.5 +
    df['Attendance_Percentage'] * 0.3 +
    df['Previous_Exam_Score'] * 0.35 +
    df['Average_Sleep_Hours'] * 1.5 +
    np.random.normal(0, 10, n_students)
).clip(25, 100)

df['Pass_Fail'] = (df['Final_Grade'] >= 60).astype(int)

print("\nDataset Created:")
print(f"Total Students: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Statistics:")
print(df.describe())

df.to_csv('Day_6_Mar01_Ridge_CrossValidation/student_data.csv', index=False)

X = df[['Study_Hours_Per_Week', 'Attendance_Percentage', 'Previous_Exam_Score', 'Average_Sleep_Hours']]
y_regression = df['Final_Grade']
y_classification = df['Pass_Fail']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*80)
print("PART 1: RIDGE REGRESSION WITH K-FOLD CROSS-VALIDATION")
print("="*80)

ridge_model = Ridge(alpha=1.0)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(ridge_model, X_scaled, y_regression, cv=kfold, scoring='r2')

print("\nK-Fold Cross-Validation Results (5 folds):")
print(f"R² Scores for each fold: {cv_scores}")
print(f"Mean R² Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
print(f"Min R² Score: {cv_scores.min():.4f}")
print(f"Max R² Score: {cv_scores.max():.4f}")

print("\n" + "="*80)
print("PART 2: MULTI-METRIC EVALUATION (R² AND MSE)")
print("="*80)

scoring_metrics = ['r2', 'neg_mean_squared_error']

cv_results = cross_validate(
    ridge_model, 
    X_scaled, 
    y_regression, 
    cv=kfold, 
    scoring=scoring_metrics,
    return_train_score=True
)

print("\nMulti-Metric Cross-Validation Results:")
print(f"\nR² Scores:")
print(f"  Training R² (mean): {cv_results['train_r2'].mean():.4f} ± {cv_results['train_r2'].std():.4f}")
print(f"  Validation R² (mean): {cv_results['test_r2'].mean():.4f} ± {cv_results['test_r2'].std():.4f}")

print(f"\nMean Squared Error:")
print(f"  Training MSE (mean): {-cv_results['train_neg_mean_squared_error'].mean():.4f} ± {cv_results['train_neg_mean_squared_error'].std():.4f}")
print(f"  Validation MSE (mean): {-cv_results['test_neg_mean_squared_error'].mean():.4f} ± {cv_results['test_neg_mean_squared_error'].std():.4f}")

overfitting_check = cv_results['train_r2'].mean() - cv_results['test_r2'].mean()
print(f"\nOverfitting Check:")
print(f"  Difference (Train R² - Val R²): {overfitting_check:.4f}")
if overfitting_check < 0.05:
    print("  Status: Good generalization (minimal overfitting)")
elif overfitting_check < 0.1:
    print("  Status: Acceptable (slight overfitting)")
else:
    print("  Status: Possible overfitting detected")

print("\n" + "="*80)
print("PART 3: STRATIFIED K-FOLD FOR PASS/FAIL CLASSIFICATION")
print("="*80)

logistic_model = LogisticRegression(random_state=42, max_iter=1000)

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stratified_scores = cross_val_score(
    logistic_model, 
    X_scaled, 
    y_classification, 
    cv=stratified_kfold, 
    scoring='accuracy'
)

print("\nStratified K-Fold Cross-Validation Results:")
print(f"Accuracy Scores for each fold: {stratified_scores}")
print(f"Mean Accuracy: {stratified_scores.mean():.4f}")
print(f"Standard Deviation: {stratified_scores.std():.4f}")

print("\nClass Distribution Check:")
for fold_idx, (train_idx, val_idx) in enumerate(stratified_kfold.split(X_scaled, y_classification), 1):
    train_dist = y_classification.iloc[train_idx].value_counts(normalize=True)
    val_dist = y_classification.iloc[val_idx].value_counts(normalize=True)
    print(f"\nFold {fold_idx}:")
    print(f"  Training - Pass: {train_dist.get(1, 0):.2%}, Fail: {train_dist.get(0, 0):.2%}")
    print(f"  Validation - Pass: {val_dist.get(1, 0):.2%}, Fail: {val_dist.get(0, 0):.2%}")

ridge_model.fit(X_scaled, y_regression)
y_pred_regression = ridge_model.predict(X_scaled)

logistic_model.fit(X_scaled, y_classification)
y_pred_classification = logistic_model.predict(X_scaled)

fig = plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
plt.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black', linewidth=2)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold Number', fontweight='bold')
plt.ylabel('R² Score', fontweight='bold')
plt.title('K-Fold CV: R² Scores Across Folds', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 2)
metrics = ['R² (Train)', 'R² (Val)', 'MSE (Train)', 'MSE (Val)']
values = [
    cv_results['train_r2'].mean(),
    cv_results['test_r2'].mean(),
    -cv_results['train_neg_mean_squared_error'].mean() / 10,
    -cv_results['test_neg_mean_squared_error'].mean() / 10
]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = plt.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Score', fontweight='bold')
plt.title('Multi-Metric Comparison\n(MSE scaled by 0.1)', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', fontweight='bold')

plt.subplot(2, 3, 3)
plt.bar(range(1, 6), stratified_scores, color='green', edgecolor='black', linewidth=2)
plt.axhline(y=stratified_scores.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {stratified_scores.mean():.4f}')
plt.xlabel('Fold Number', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Stratified K-Fold: Classification Accuracy', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 4)
plt.scatter(y_regression, y_pred_regression, alpha=0.6, edgecolors='black')
plt.plot([y_regression.min(), y_regression.max()], 
         [y_regression.min(), y_regression.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Grade', fontweight='bold')
plt.ylabel('Predicted Grade', fontweight='bold')
plt.title('Ridge Regression: Actual vs Predicted', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
cm = confusion_matrix(y_classification, y_pred_classification)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Classification Confusion Matrix', fontsize=13, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

plt.subplot(2, 3, 6)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': ridge_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

colors_feat = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], 
         color=colors_feat, edgecolor='black', linewidth=1.5)
plt.xlabel('Ridge Coefficient', fontweight='bold')
plt.title('Feature Importance (Ridge Regression)', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

plt.suptitle('Student Grade Prediction - Ridge Regression with Cross-Validation', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_6_Mar01_Ridge_CrossValidation/grade_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")

plt.show()

results_df = pd.DataFrame({
    'Metric': ['Mean R² (CV)', 'Std R² (CV)', 'Mean MSE (CV)', 'Classification Accuracy'],
    'Value': [
        cv_scores.mean(),
        cv_scores.std(),
        -cv_results['test_neg_mean_squared_error'].mean(),
        stratified_scores.mean()
    ]
})

results_df.to_csv('Day_6_Mar01_Ridge_CrossValidation/cv_results.csv', index=False)

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n1. Model Stability:")
print(f"   R² varies from {cv_scores.min():.4f} to {cv_scores.max():.4f} across folds")
print(f"   Low standard deviation ({cv_scores.std():.4f}) indicates stable model")

print("\n2. Generalization:")
if overfitting_check < 0.05:
    print("   Model generalizes well with minimal overfitting")
else:
    print("   Some overfitting detected - consider regularization adjustment")

print("\n3. Classification Performance:")
print(f"   Pass/Fail prediction accuracy: {stratified_scores.mean():.2%}")
print("   Stratified K-Fold ensures balanced class distribution")

print("\n4. Most Important Features:")
print(f"   Top predictor: {feature_importance.iloc[0]['Feature']}")
print(f"   Coefficient: {feature_importance.iloc[0]['Coefficient']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
