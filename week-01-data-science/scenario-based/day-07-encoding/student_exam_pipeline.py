"""
SCENARIO: Student Exam Pass Prediction

A university wants to predict whether a student will pass or fail an exam based on 
their personal and academic information.

FEATURES:
- age: Age of the student (numeric)
- hours_study: Number of hours the student studies per day (numeric)
- gender: Male or Female (categorical)
- school: School type (Government or Private) (categorical)
- pass_exam: Target variable (1 = Pass, 0 = Fail)

PIPELINE APPROACH:
The data science team builds a machine learning pipeline that:
1. Standardizes numeric features (age, hours_study) so they are on the same scale
2. Converts categorical features (gender, school) into numerical format using One-Hot Encoding
3. Combines preprocessing and model training into a single pipeline
4. Uses Logistic Regression to predict whether a student will pass

QUESTIONS:
Part A: Why do we need to standardize numeric features before training?
Part B: What is the advantage of using a pipeline instead of manual preprocessing?
Part C: How does One-Hot Encoding help the model understand categorical features?
Part D: What happens if we don't standardize features in Logistic Regression?
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("STUDENT EXAM PASS PREDICTION - ML PIPELINE")
print("="*70)

np.random.seed(42)

n_students = 100

data = {
    'age': np.random.randint(17, 23, n_students),
    'hours_study': np.random.randint(1, 8, n_students),
    'gender': np.random.choice(['Male', 'Female'], n_students),
    'school': np.random.choice(['Government', 'Private'], n_students)
}

df = pd.DataFrame(data)

df['pass_exam'] = ((df['hours_study'] >= 4) | 
                   ((df['hours_study'] >= 3) & (df['school'] == 'Private'))).astype(int)

noise = np.random.choice([0, 1], size=n_students, p=[0.85, 0.15])
df['pass_exam'] = (df['pass_exam'] + noise) % 2

print("\nDataset Overview:")
print(df.head(10))

print("\nDataset Statistics:")
print(df.describe())

print("\nClass Distribution:")
print(df['pass_exam'].value_counts())
print(f"Pass Rate: {df['pass_exam'].mean():.2%}")

X = df[['age', 'hours_study', 'gender', 'school']]
y = df['pass_exam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*70)
print("BUILDING ML PIPELINE")
print("="*70)

numeric_features = ['age', 'hours_study']
categorical_features = ['gender', 'school']

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
print(f"\nAccuracy: {accuracy:.4f} ({accuracy:.2%})")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n" + "="*70)
print("MAKING PREDICTIONS FOR NEW STUDENTS")
print("="*70)

new_students = pd.DataFrame({
    'age': [19, 20, 18, 21],
    'hours_study': [3, 6, 1, 5],
    'gender': ['Female', 'Male', 'Female', 'Male'],
    'school': ['Government', 'Private', 'Government', 'Private']
})

print("\nNew Student Data:")
print(new_students)

predictions = pipeline.predict(new_students)
probabilities = pipeline.predict_proba(new_students)

print("\nPredictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    result = "Pass" if pred == 1 else "Fail"
    confidence = prob[pred] * 100
    print(f"Student {i+1}: {result} (Confidence: {confidence:.2f}%)")

fig = plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
pass_counts = df['pass_exam'].value_counts()
colors = ['#e74c3c', '#2ecc71']
plt.bar(['Fail', 'Pass'], pass_counts.values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Count', fontweight='bold')
plt.title('Pass/Fail Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(pass_counts.values):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.subplot(2, 3, 2)
avg_hours = df.groupby('pass_exam')['hours_study'].mean()
plt.bar(['Fail', 'Pass'], avg_hours.values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Average Study Hours', fontweight='bold')
plt.title('Study Hours by Outcome', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(avg_hours.values):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

plt.subplot(2, 3, 4)
plt.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black', linewidth=2)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Cross-Validation Scores', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 5)
gender_pass = df.groupby(['gender', 'pass_exam']).size().unstack(fill_value=0)
gender_pass.plot(kind='bar', color=colors, edgecolor='black', linewidth=2, ax=plt.gca())
plt.xlabel('Gender', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Pass/Fail by Gender', fontsize=12, fontweight='bold')
plt.legend(['Fail', 'Pass'])
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 6)
school_pass = df.groupby(['school', 'pass_exam']).size().unstack(fill_value=0)
school_pass.plot(kind='bar', color=colors, edgecolor='black', linewidth=2, ax=plt.gca())
plt.xlabel('School Type', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Pass/Fail by School Type', fontsize=12, fontweight='bold')
plt.legend(['Fail', 'Pass'])
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Student Exam Pass Prediction - ML Pipeline Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_7_Mar02_Encoding/student_exam_pipeline_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")

plt.show()

results_df = pd.DataFrame({
    'Student': [f'Student {i+1}' for i in range(len(new_students))],
    'Age': new_students['age'].values,
    'Study_Hours': new_students['hours_study'].values,
    'Gender': new_students['gender'].values,
    'School': new_students['school'].values,
    'Prediction': ['Pass' if p == 1 else 'Fail' for p in predictions],
    'Confidence': [prob[pred] * 100 for pred, prob in zip(predictions, probabilities)]
})

results_df.to_csv('Day_7_Mar02_Encoding/student_predictions.csv', index=False)

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why standardize numeric features?")
print("  - Different scales: age (17-22) vs hours_study (1-7)")
print("  - Logistic Regression uses gradient descent - sensitive to feature scales")
print("  - Standardization: mean=0, std=1 for all features")
print("  - Prevents features with larger values from dominating the model")
print("  - Improves convergence speed and model performance")

print("\nPart B: Advantages of using a pipeline:")
print("  - Prevents data leakage: preprocessing fitted only on training data")
print("  - Cleaner code: single fit() and predict() calls")
print("  - Reproducibility: same preprocessing applied to new data")
print("  - Easy deployment: save entire pipeline as single object")
print("  - Cross-validation: preprocessing included in each fold")

print("\nPart C: How One-Hot Encoding helps:")
print("  - Converts categories to binary features")
print("  - Gender: Male/Female → [0] or [1] (with drop='first')")
print("  - School: Government/Private → [0] or [1]")
print("  - No ordinal relationship assumed (Male ≠ 0, Female ≠ 1)")
print("  - Model learns separate coefficients for each category")

print("\nPart D: Without standardization:")
print("  - Features with larger ranges dominate gradient updates")
print("  - Slower convergence or failure to converge")
print("  - Coefficients become difficult to interpret")
print("  - Model may perform poorly on test data")
print("  - Example: age (17-22) has smaller impact than hours_study (1-7)")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"\n1. Model Accuracy: {accuracy:.2%}")
print(f"2. Cross-Validation Mean: {cv_scores.mean():.2%}")
print(f"3. Pass Rate in Dataset: {df['pass_exam'].mean():.2%}")
print(f"4. Average Study Hours (Pass): {df[df['pass_exam']==1]['hours_study'].mean():.2f}")
print(f"5. Average Study Hours (Fail): {df[df['pass_exam']==0]['hours_study'].mean():.2f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
