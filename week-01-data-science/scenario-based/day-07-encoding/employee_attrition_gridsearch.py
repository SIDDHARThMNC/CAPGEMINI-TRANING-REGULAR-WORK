"""
SCENARIO: Employee Attrition Prediction using Pipeline and GridSearchCV

A company wants to predict whether an employee will leave the company (Attrition) or stay, 
so HR can take preventive action.

FEATURES:
- age: Age of the employee (numeric)
- years_experience: Total years of work experience (numeric)
- department: Department (Sales, IT, HR) (categorical)
- education_level: Bachelor, Master, PhD (categorical)
- attrition: Target variable (1 = Employee leaves, 0 = Employee stays)

PIPELINE APPROACH:
The data science team builds a Pipeline that:
1. Scales numeric features using StandardScaler
2. Encodes categorical features using OneHotEncoder
3. Uses Logistic Regression for classification

HYPERPARAMETER TUNING:
Uses GridSearchCV to find the best hyperparameters:
- C: [0.1, 1, 10]
- solver: ['liblinear', 'lbfgs']
- Evaluation: 5-fold cross-validation using accuracy

QUESTIONS:
Part A: Why is predicting attrition important for HR departments?
Part B: How does GridSearchCV help find optimal model parameters?
Part C: What is the difference between 'liblinear' and 'lbfgs' solvers?
Part D: How can HR use these predictions to reduce employee turnover?
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("EMPLOYEE ATTRITION PREDICTION - PIPELINE + GRIDSEARCHCV")
print("="*70)

np.random.seed(42)

n_employees = 250

data = {
    'age': np.random.randint(22, 60, n_employees),
    'years_experience': np.random.randint(0, 35, n_employees),
    'department': np.random.choice(['Sales', 'IT', 'HR'], n_employees, p=[0.4, 0.35, 0.25]),
    'education_level': np.random.choice(['Bachelor', 'Master', 'PhD'], n_employees, p=[0.5, 0.35, 0.15])
}

df = pd.DataFrame(data)

attrition_prob = (
    (df['age'] < 30) * 0.25 +
    (df['years_experience'] < 3) * 0.3 +
    (df['department'] == 'Sales') * 0.2 +
    (df['education_level'] == 'Bachelor') * 0.15 +
    0.05
)

df['attrition'] = (np.random.random(n_employees) < attrition_prob).astype(int)

print("\nDataset Overview:")
print(df.head(10))

print("\nDataset Statistics:")
print(df.describe())

print("\nAttrition Distribution:")
print(df['attrition'].value_counts())
print(f"Attrition Rate: {df['attrition'].mean():.2%}")

print("\nDepartment Distribution:")
print(df['department'].value_counts())

print("\nEducation Level Distribution:")
print(df['education_level'].value_counts())

print("\nAttrition by Department:")
print(df.groupby('department')['attrition'].mean().sort_values(ascending=False))

print("\nAttrition by Education Level:")
print(df.groupby('education_level')['attrition'].mean().sort_values(ascending=False))

X = df[['age', 'years_experience', 'department', 'education_level']]
y = df['attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "="*70)
print("BUILDING ML PIPELINE")
print("="*70)

numeric_features = ['age', 'years_experience']
categorical_features = ['department', 'education_level']

print("\nNumeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

print("\nPipeline Structure:")
print(pipeline)

print("\n" + "="*70)
print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*70)

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}

print("\nParameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)
print(f"\nTotal combinations to test: {total_combinations}")
print("Cross-Validation Folds: 5")
print(f"Total model fits: {total_combinations * 5}")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("\nTraining GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\n" + "="*70)
print("GRIDSEARCHCV RESULTS")
print("="*70)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

results_df = pd.DataFrame(grid_search.cv_results_)
results_summary = results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']].sort_values('rank_test_score')

print("\nAll Parameter Combinations (Ranked):")
print(results_summary.to_string(index=False))

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stays', 'Leaves']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives (Correctly predicted Stays): {cm[0,0]}")
print(f"False Positives (Incorrectly predicted Leaves): {cm[0,1]}")
print(f"False Negatives (Incorrectly predicted Stays): {cm[1,0]}")
print(f"True Positives (Correctly predicted Leaves): {cm[1,1]}")

baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

baseline_pipeline.fit(X_train, y_train)
baseline_pred = baseline_pipeline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print(f"\nBaseline Model Accuracy (default C=1.0): {baseline_accuracy:.4f}")
print(f"Tuned Model Accuracy: {accuracy:.4f}")
print(f"Improvement: {(accuracy - baseline_accuracy):.4f} ({((accuracy - baseline_accuracy)/baseline_accuracy)*100:.2f}%)")

print("\n" + "="*70)
print("PREDICTING ATTRITION FOR NEW EMPLOYEES")
print("="*70)

new_employees = pd.DataFrame({
    'age': [28, 45, 32, 50, 26],
    'years_experience': [2, 18, 8, 25, 1],
    'department': ['Sales', 'IT', 'HR', 'IT', 'Sales'],
    'education_level': ['Bachelor', 'Master', 'PhD', 'Master', 'Bachelor']
})

print("\nNew Employee Data:")
print(new_employees)

predictions = best_model.predict(new_employees)
probabilities = best_model.predict_proba(new_employees)

print("\nAttrition Predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    result = "Leaves" if pred == 1 else "Stays"
    confidence = prob[pred] * 100
    attrition_risk = prob[1] * 100
    risk_level = "HIGH" if attrition_risk > 60 else "MEDIUM" if attrition_risk > 40 else "LOW"
    print(f"Employee {i+1}: {result} (Confidence: {confidence:.2f}%, Attrition Risk: {attrition_risk:.2f}% - {risk_level})")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

fig = plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
attrition_counts = df['attrition'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.bar(['Stays', 'Leaves'], attrition_counts.values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Count', fontweight='bold')
plt.title('Attrition Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(attrition_counts.values):
    plt.text(i, v + 3, str(v), ha='center', fontweight='bold')

plt.subplot(2, 3, 2)
c_values = [0.1, 1, 10]
c_scores_liblinear = []
c_scores_lbfgs = []
for c in c_values:
    liblinear_score = results_df[(results_df['param_classifier__C'] == c) & 
                                  (results_df['param_classifier__solver'] == 'liblinear')]['mean_test_score'].values[0]
    lbfgs_score = results_df[(results_df['param_classifier__C'] == c) & 
                              (results_df['param_classifier__solver'] == 'lbfgs')]['mean_test_score'].values[0]
    c_scores_liblinear.append(liblinear_score)
    c_scores_lbfgs.append(lbfgs_score)

x_pos = np.arange(len(c_values))
width = 0.35
plt.bar(x_pos - width/2, c_scores_liblinear, width, label='liblinear', color='steelblue', edgecolor='black')
plt.bar(x_pos + width/2, c_scores_lbfgs, width, label='lbfgs', color='coral', edgecolor='black')
plt.xlabel('C Parameter', fontweight='bold')
plt.ylabel('Mean CV Accuracy', fontweight='bold')
plt.title('GridSearchCV: Solver Comparison', fontsize=12, fontweight='bold')
plt.xticks(x_pos, c_values)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stays', 'Leaves'], yticklabels=['Stays', 'Leaves'])
plt.title('Confusion Matrix (Tuned Model)', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

plt.subplot(2, 3, 4)
plt.plot(fpr, tpr, color='darkorange', linewidth=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curve', fontsize=12, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
dept_attrition = df.groupby(['department', 'attrition']).size().unstack(fill_value=0)
dept_attrition.plot(kind='bar', color=colors, edgecolor='black', linewidth=2, ax=plt.gca())
plt.xlabel('Department', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Attrition by Department', fontsize=12, fontweight='bold')
plt.legend(['Stays', 'Leaves'])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 6)
models = ['Baseline\n(C=1.0)', 'Tuned\n(GridSearch)']
accuracies = [baseline_accuracy, accuracy]
colors_model = ['#3498db', '#2ecc71']
bars = plt.bar(models, accuracies, color=colors_model, edgecolor='black', linewidth=2)
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Model Comparison', fontsize=12, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.3f}', ha='center', fontweight='bold')

plt.suptitle('Employee Attrition Prediction - Pipeline + GridSearchCV', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_7_Mar02_Encoding/employee_attrition_gridsearch_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")

plt.show()

results_df_export = pd.DataFrame({
    'Employee': [f'Employee {i+1}' for i in range(len(new_employees))],
    'Age': new_employees['age'].values,
    'Experience': new_employees['years_experience'].values,
    'Department': new_employees['department'].values,
    'Education': new_employees['education_level'].values,
    'Prediction': ['Leaves' if p == 1 else 'Stays' for p in predictions],
    'Attrition_Risk': [prob[1] * 100 for prob in probabilities],
    'Risk_Level': ['HIGH' if prob[1] > 0.6 else 'MEDIUM' if prob[1] > 0.4 else 'LOW' for prob in probabilities]
})

results_df_export.to_csv('Day_7_Mar02_Encoding/employee_attrition_gridsearch_predictions.csv', index=False)

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why is predicting attrition important for HR?")
print("  - Cost reduction: Hiring and training new employees is expensive")
print("  - Proactive retention: Identify at-risk employees before they leave")
print("  - Workforce planning: Anticipate staffing needs and gaps")
print("  - Employee satisfaction: Address issues causing dissatisfaction")
print("  - Competitive advantage: Retain top talent and institutional knowledge")
print("  - ROI: Predictive models help allocate retention budget effectively")

print("\nPart B: How GridSearchCV helps find optimal parameters:")
print("  - Automated search: Tests all parameter combinations systematically")
print("  - Cross-validation: Evaluates each combination on 5 different data splits")
print("  - Prevents overfitting: Uses validation data not seen during training")
print("  - Objective selection: Chooses parameters with best average performance")
print(f"  - Result: Found C={grid_search.best_params_['classifier__C']}, solver={grid_search.best_params_['classifier__solver']}")

print("\nPart C: Difference between 'liblinear' and 'lbfgs' solvers:")
print("  liblinear:")
print("    - Uses coordinate descent algorithm")
print("    - Good for small datasets")
print("    - Supports L1 and L2 regularization")
print("    - Faster for high-dimensional sparse data")
print("  lbfgs:")
print("    - Uses Limited-memory BFGS algorithm")
print("    - Better for larger datasets")
print("    - Only supports L2 regularization")
print("    - More memory efficient for dense data")
print(f"  - Best solver for this dataset: {grid_search.best_params_['classifier__solver']}")

print("\nPart D: How HR can use predictions to reduce turnover:")
print("  - High-risk employees: Offer retention bonuses, promotions, or flexible work")
print("  - Department-specific: Address issues in high-attrition departments (Sales)")
print("  - Experience-based: Mentorship programs for employees with <3 years experience")
print("  - Age-based: Career development opportunities for younger employees")
print("  - Regular monitoring: Track attrition risk quarterly and intervene early")
print("  - Exit interviews: Validate model predictions with actual reasons for leaving")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"\n1. Best CV Score: {grid_search.best_score_:.2%}")
print(f"2. Test Accuracy: {accuracy:.2%}")
print(f"3. ROC-AUC Score: {roc_auc:.4f}")
print(f"4. Attrition Rate: {df['attrition'].mean():.2%}")
print(f"5. Highest Risk Department: {df.groupby('department')['attrition'].mean().idxmax()}")
print(f"6. Total combinations tested: {total_combinations}")

print("\n" + "="*70)
print("HR ACTION PLAN")
print("="*70)

high_risk_employees = results_df_export[results_df_export['Risk_Level'] == 'HIGH']
print(f"\n1. Immediate attention needed for {len(high_risk_employees)} high-risk employees")
print("2. Implement retention programs in Sales department")
print("3. Create mentorship for employees with <3 years experience")
print("4. Conduct satisfaction surveys for employees under 30")
print("5. Review compensation packages for Bachelor degree holders")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
