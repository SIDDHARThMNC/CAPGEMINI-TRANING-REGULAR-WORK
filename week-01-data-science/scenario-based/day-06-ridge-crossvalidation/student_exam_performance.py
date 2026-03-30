"""
SCENARIO: Predicting Student Exam Performance

A university research team wants to build a model to predict student exam scores (out of 100)
based on several factors.

FEATURES:
- Number of study hours per week
- Attendance percentage in lectures
- Prior GPA (Grade Point Average)
- Participation in group projects (numeric engagement score)
- Average sleep hours during exam preparation

DATASET:
- 800 students across different departments

OBJECTIVES:
1. Use Linear Regression for exam score prediction
2. Apply 5-Fold Cross-Validation
3. Evaluate using R² as the performance metric
4. Analyze model stability and performance
5. Identify key factors affecting exam performance

BUSINESS IMPACT:
- Early identification of at-risk students
- Personalized study recommendations
- Resource allocation for tutoring programs
- Improved academic support strategies
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
print("STUDENT EXAM PERFORMANCE PREDICTION - LINEAR REGRESSION WITH 5-FOLD CV")
print("="*80)

n_students = 800

data = {
    'Student_ID': range(1, n_students + 1),
    'Study_Hours_Per_Week': np.random.uniform(2, 40, n_students),
    'Attendance_Percentage': np.random.uniform(40, 100, n_students),
    'Prior_GPA': np.random.uniform(1.5, 4.0, n_students),
    'Group_Project_Engagement': np.random.uniform(0, 10, n_students),
    'Average_Sleep_Hours': np.random.uniform(4, 10, n_students)
}

df = pd.DataFrame(data)

df['Exam_Score'] = (
    df['Study_Hours_Per_Week'] * 1.2 +
    df['Attendance_Percentage'] * 0.35 +
    df['Prior_GPA'] * 8.5 +
    df['Group_Project_Engagement'] * 1.8 +
    df['Average_Sleep_Hours'] * 0.9 +
    np.random.normal(0, 5, n_students)
).clip(0, 100)

print("\nDataset Created:")
print(f"Total Students: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Statistics:")
print(df.describe())

print("\nExam Score Distribution:")
print(f"Mean Exam Score: {df['Exam_Score'].mean():.2f}")
print(f"Median Exam Score: {df['Exam_Score'].median():.2f}")
print(f"Min Exam Score: {df['Exam_Score'].min():.2f}")
print(f"Max Exam Score: {df['Exam_Score'].max():.2f}")
print(f"Standard Deviation: {df['Exam_Score'].std():.2f}")

grade_distribution = pd.cut(df['Exam_Score'], bins=[0, 60, 70, 80, 90, 100], 
                             labels=['F (<60)', 'D (60-70)', 'C (70-80)', 'B (80-90)', 'A (90-100)'])
print("\nGrade Distribution:")
print(grade_distribution.value_counts().sort_index())

df.to_csv('Day_6_Mar01_Ridge_CrossValidation/student_exam_data.csv', index=False)
print("\nDataset saved to: student_exam_data.csv")

X = df[['Study_Hours_Per_Week', 'Attendance_Percentage', 'Prior_GPA', 
        'Group_Project_Engagement', 'Average_Sleep_Hours']]
y = df['Exam_Score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*80)
print("LINEAR REGRESSION WITH 5-FOLD CROSS-VALIDATION")
print("="*80)

lr_model = LinearRegression()

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(lr_model, X_scaled, y, cv=kfold, scoring='r2')

print("\n5-Fold Cross-Validation Results:")
print(f"R² Scores for each fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"\nMean R² Score: {cv_scores.mean():.4f}")
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
train_mse = -cv_results['train_neg_mean_squared_error'].mean()
val_mse = -cv_results['test_neg_mean_squared_error'].mean()
print(f"  Training MSE (mean): {train_mse:.4f}")
print(f"  Validation MSE (mean): {val_mse:.4f}")
print(f"  RMSE (Validation): {np.sqrt(val_mse):.4f} points")

print("\nMean Absolute Error (MAE):")
train_mae = -cv_results['train_neg_mean_absolute_error'].mean()
val_mae = -cv_results['test_neg_mean_absolute_error'].mean()
print(f"  Training MAE (mean): {train_mae:.4f} points")
print(f"  Validation MAE (mean): {val_mae:.4f} points")

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

print("\nFeature Coefficients (Impact on Exam Score):")
for idx, row in feature_importance.iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']}: {row['Coefficient']:.4f} ({direction} exam score)")

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
metrics = ['R² (Train)', 'R² (Val)', 'MAE (Train)', 'MAE (Val)']
values = [
    cv_results['train_r2'].mean(),
    cv_results['test_r2'].mean(),
    train_mae / 100,
    val_mae / 100
]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = plt.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Score', fontweight='bold', fontsize=11)
plt.title('Performance Metrics Comparison\n(MAE scaled by 0.01)', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.subplot(2, 3, 3)
colors_feat = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], 
         color=colors_feat, edgecolor='black', linewidth=1.5)
plt.xlabel('Coefficient Value', fontweight='bold', fontsize=11)
plt.title('Feature Impact on Exam Score\n(Green=Positive, Red=Negative)', 
          fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 3, 4)
plt.scatter(y, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Exam Score', fontweight='bold', fontsize=11)
plt.ylabel('Predicted Exam Score', fontweight='bold', fontsize=11)
plt.title('Actual vs Predicted Exam Scores', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Exam Score', fontweight='bold', fontsize=11)
plt.ylabel('Residuals', fontweight='bold', fontsize=11)
plt.title('Residual Plot', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.hist(df['Exam_Score'], bins=30, color='skyblue', edgecolor='black', linewidth=1.5)
plt.axvline(df['Exam_Score'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {df["Exam_Score"].mean():.1f}')
plt.axvline(df['Exam_Score'].median(), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {df["Exam_Score"].median():.1f}')
plt.xlabel('Exam Score', fontweight='bold', fontsize=11)
plt.ylabel('Number of Students', fontweight='bold', fontsize=11)
plt.title('Distribution of Exam Scores', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Student Exam Performance Prediction - Linear Regression with 5-Fold CV', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_6_Mar01_Ridge_CrossValidation/student_exam_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: student_exam_analysis.png")

plt.show()

results_df = pd.DataFrame({
    'Metric': ['Mean R² (CV)', 'Std R² (CV)', 'Mean RMSE (CV)', 'Mean MAE (CV)', 
               'Train R²', 'Val R²', 'Overfitting Gap'],
    'Value': [
        cv_scores.mean(),
        cv_scores.std(),
        np.sqrt(val_mse),
        val_mae,
        cv_results['train_r2'].mean(),
        cv_results['test_r2'].mean(),
        overfitting_check
    ]
})

results_df.to_csv('Day_6_Mar01_Ridge_CrossValidation/student_exam_cv_results.csv', index=False)
print("Results saved to: student_exam_cv_results.csv")

print("\n" + "="*80)
print("KEY INSIGHTS & ACADEMIC RECOMMENDATIONS")
print("="*80)

print("\n1. Model Performance:")
print(f"   The model explains {cv_scores.mean()*100:.2f}% of variance in exam scores")
print(f"   Average prediction error: ±{val_mae:.2f} points")
print(f"   Root Mean Squared Error: {np.sqrt(val_mse):.2f} points")

print("\n2. Model Stability:")
print(f"   R² varies from {cv_scores.min():.4f} to {cv_scores.max():.4f} across folds")
if cv_scores.std() < 0.05:
    print("   Very stable model - consistent predictions across different student groups")
elif cv_scores.std() < 0.1:
    print("   Stable model - reliable predictions")
else:
    print("   Some variability - model performance depends on student characteristics")

print("\n3. Top Factors Affecting Exam Performance:")
for i, (idx, row) in enumerate(feature_importance.head(3).iterrows(), 1):
    impact = "improves" if row['Coefficient'] > 0 else "reduces"
    print(f"   {i}. {row['Feature']} - {impact} exam score")

print("\n4. Academic Recommendations:")
top_feature = feature_importance.iloc[0]
if 'Prior_GPA' in top_feature['Feature']:
    print("   Prior academic performance is the strongest predictor")
    print("   Focus on building strong foundational knowledge")
if 'Study_Hours' in feature_importance['Feature'].values:
    study_coef = feature_importance[feature_importance['Feature'] == 'Study_Hours_Per_Week']['Coefficient'].values[0]
    if study_coef > 0:
        print("   Increase study hours for better exam performance")
if 'Attendance' in str(feature_importance['Feature'].values):
    attendance_coef = feature_importance[feature_importance['Feature'] == 'Attendance_Percentage']['Coefficient'].values[0]
    if attendance_coef > 0:
        print("   Regular class attendance significantly impacts exam scores")
if 'Group_Project' in str(feature_importance['Feature'].values):
    project_coef = feature_importance[feature_importance['Feature'] == 'Group_Project_Engagement']['Coefficient'].values[0]
    if project_coef > 0:
        print("   Active participation in group projects enhances learning")

print("\n5. Student Performance Categories:")
print(f"   High Performers (>80): {(df['Exam_Score'] > 80).sum()} students ({(df['Exam_Score'] > 80).sum()/len(df)*100:.1f}%)")
print(f"   Average Performers (60-80): {((df['Exam_Score'] >= 60) & (df['Exam_Score'] <= 80)).sum()} students ({((df['Exam_Score'] >= 60) & (df['Exam_Score'] <= 80)).sum()/len(df)*100:.1f}%)")
print(f"   At-Risk Students (<60): {(df['Exam_Score'] < 60).sum()} students ({(df['Exam_Score'] < 60).sum()/len(df)*100:.1f}%)")

print("\n6. Intervention Strategies:")
if (df['Exam_Score'] < 60).sum() > 0:
    print(f"   {(df['Exam_Score'] < 60).sum()} students need immediate academic support")
    print("   Recommend tutoring programs and study skills workshops")
if val_mae < 5:
    print("   Model is accurate enough for early warning systems")
else:
    print("   Consider additional features for improved prediction accuracy")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
