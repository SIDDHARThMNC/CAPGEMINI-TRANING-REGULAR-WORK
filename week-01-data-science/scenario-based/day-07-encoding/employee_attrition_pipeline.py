"""
SCENARIO: Employee Attrition Prediction

A company wants to predict whether an employee is likely to stay or leave based on 
their personal and workplace information.

FEATURES:
- age: Age of the employee (numeric)
- years_experience: Number of years of work experience (numeric)
- department: Department (HR, IT, Sales) (categorical)
- education: Education level (Graduate, Postgraduate) (categorical)
- attrition: Target variable (1 = Leaves, 0 = Stays)

PIPELINE APPROACH:
The data science team builds a machine learning pipeline that:
1. Standardizes numeric features (age, years_experience)
2. Converts categorical features (department, education) into numerical format using One-Hot Encoding
3. Combines preprocessing and model training into a single pipeline
4. Uses Logistic Regression to predict employee attrition

QUESTIONS:
Part A: Why is predicting employee attrition important for companies?
Part B: How does the pipeline ensure consistent preprocessing for new employees?
Part C: What role does standardization play in this prediction task?
Part D: How can the company use these predictions to reduce attrition?
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("EMPLOYEE ATTRITION PREDICTION - ML PIPELINE")
print("="*70)

np.random.seed(42)

n_employees = 150

data = {
    'age': np.random.randint(22, 60, n_employees),
    'years_experience': np.random.randint(0, 35, n_employees),
    'department': np.random.choice(['HR', 'IT', 'Sales'], n_employees),
    'education': np.random.choice(['Graduate', 'Postgraduate'], n_employees)
}

df = pd.DataFrame(data)

attrition_prob = (
    (df['age'] < 30) * 0.3 +
    (df['years_experience'] < 3) * 0.25 +
    (df['department'] == 'Sales') * 0.2 +
    (df['education'] == 'Graduate') * 0.15
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

print("\nEducation Distribution:")
print(df['education'].value_counts())

X = df[['age', 'years_experience', 'department', 'education']]
y = df['attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "="*70)
print("BUILDING ML PIPELINE")
print("="*70)

numeric_features = ['age', 'years_experience']
categorical_features = ['department', 'education']

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
print("TRAINING THE MODEL")
print("="*70)

pipeline.fit(X_train, y_train)

print("\nModel trained successfully!")

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print(f"\nAccuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stays', 'Leaves']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n" + "="*70)
print("PREDICTING ATTRITION FOR NEW EMPLOYEES")
print("="*70)

new_employees = pd.DataFrame({
    'age': [28, 45, 32, 50],
    'years_experience': [2, 15, 5, 20],
    'department': ['Sales', 'IT', 'HR', 'IT'],
    'education': ['Graduate', 'Postgraduate', 'Graduate', 'Postgraduate']
})

print("\nNew Employee Data:")
print(new_employees)

predictions = pipeline.predict(new_employees)
probabilities = pipeline.predict_proba(new_employees)

print("\nAttrition Predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    result = "Leaves" if pred == 1 else "Stays"
    confidence = prob[pred] * 100
    risk = prob[1] * 100
    print(f"Employee {i+1}: {result} (Confidence: {confidence:.2f}%, Attrition Risk: {risk:.2f}%)")

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
    plt.text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.subplot(2, 3, 2)
dept_attrition = df.groupby(['department', 'attrition']).size().unstack(fill_value=0)
dept_attrition.plot(kind='bar', color=colors, edgecolor='black', linewidth=2, ax=plt.gca())
plt.xlabel('Department', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Attrition by Department', fontsize=12, fontweight='bold')
plt.legend(['Stays', 'Leaves'])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stays', 'Leaves'], yticklabels=['Stays', 'Leaves'])
plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
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
plt.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black', linewidth=2)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Cross-Validation Scores', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 6)
edu_attrition = df.groupby(['education', 'attrition']).size().unstack(fill_value=0)
edu_attrition.plot(kind='bar', color=colors, edgecolor='black', linewidth=2, ax=plt.gca())
plt.xlabel('Education Level', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Attrition by Education', fontsize=12, fontweight='bold')
plt.legend(['Stays', 'Leaves'])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Employee Attrition Prediction - ML Pipeline Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_7_Mar02_Encoding/employee_attrition_pipeline_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")

plt.show()

results_df = pd.DataFrame({
    'Employee': [f'Employee {i+1}' for i in range(len(new_employees))],
    'Age': new_employees['age'].values,
    'Experience': new_employees['years_experience'].values,
    'Department': new_employees['department'].values,
    'Education': new_employees['education'].values,
    'Prediction': ['Leaves' if p == 1 else 'Stays' for p in predictions],
    'Attrition_Risk': [prob[1] * 100 for prob in probabilities]
})

results_df.to_csv('Day_7_Mar02_Encoding/employee_attrition_predictions.csv', index=False)

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why is predicting attrition important?")
print("  - Cost savings: Hiring and training new employees is expensive")
print("  - Retention strategies: Identify at-risk employees early")
print("  - Workforce planning: Anticipate staffing needs")
print("  - Employee satisfaction: Address issues before employees leave")
print("  - Competitive advantage: Retain top talent and institutional knowledge")

print("\nPart B: How pipeline ensures consistent preprocessing:")
print("  - Same transformations applied to training and new data")
print("  - Scaler uses training data statistics (mean, std)")
print("  - One-Hot Encoder uses same categories from training")
print("  - No manual intervention needed for new predictions")
print("  - Prevents errors from inconsistent preprocessing")

print("\nPart C: Role of standardization:")
print("  - Age range: 22-60 years")
print("  - Experience range: 0-35 years")
print("  - Different scales affect Logistic Regression coefficients")
print("  - Standardization ensures equal contribution from both features")
print("  - Improves model convergence and interpretability")

print("\nPart D: Using predictions to reduce attrition:")
print("  - High-risk employees: Offer retention bonuses or promotions")
print("  - Department-specific: Address issues in high-attrition departments")
print("  - Career development: Provide training for employees with low experience")
print("  - Work-life balance: Improve conditions for younger employees")
print("  - Regular monitoring: Track attrition risk over time")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"\n1. Model Accuracy: {accuracy:.2%}")
print(f"2. ROC-AUC Score: {roc_auc:.4f}")
print(f"3. Attrition Rate: {df['attrition'].mean():.2%}")
print(f"4. Cross-Validation Mean: {cv_scores.mean():.2%}")
print(f"5. High-Risk Departments: {df.groupby('department')['attrition'].mean().idxmax()}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n1. Focus retention efforts on employees with <3 years experience")
print("2. Implement mentorship programs in high-attrition departments")
print("3. Offer career development opportunities for younger employees")
print("4. Conduct regular satisfaction surveys to identify issues early")
print("5. Use model predictions to prioritize retention interventions")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
