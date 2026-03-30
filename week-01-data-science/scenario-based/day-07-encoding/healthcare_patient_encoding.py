"""
SCENARIO: Healthcare Patient Records Encoding

You are working as a data scientist in a hospital. The hospital wants to build a machine 
learning model to predict patient recovery time based on demographic and treatment details.

BUSINESS CONTEXT:
- Treatment Type: Different medical procedures (Surgery, Therapy, Medication)
- Hospital Wing: Location where patient is admitted (East, West, North, South)
- Recovery Days: Numeric values representing recovery time

PROBLEM:
The dataset contains categorical variables (Treatment Type and Hospital Wing) that must be 
converted into numeric form before modeling.

CHALLENGE:
Machine learning models cannot directly interpret text categories, so you need to encode 
categorical features into numbers.

ENCODING TECHNIQUES:
1. Label Encoding - Assigns numeric codes to categories
2. One-Hot Encoding - Creates binary columns for each category
3. Comparison of both methods for healthcare data

QUESTIONS:
Part A: Why is proper encoding critical in healthcare ML models?
Part B: Which encoding method is better for Treatment Type and Hospital Wing?
Part C: How does encoding affect prediction accuracy for recovery time?
Part D: What are the risks of using wrong encoding in medical predictions?
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("HEALTHCARE PATIENT RECORDS ENCODING")
print("="*70)

np.random.seed(42)

data = pd.DataFrame({
    'Patient_ID': [f'P{str(i).zfill(3)}' for i in range(1, 21)],
    'Treatment_Type': ['Surgery', 'Therapy', 'Medication', 'Surgery', 'Therapy',
                       'Medication', 'Surgery', 'Therapy', 'Medication', 'Surgery',
                       'Therapy', 'Medication', 'Surgery', 'Therapy', 'Medication',
                       'Surgery', 'Therapy', 'Medication', 'Surgery', 'Therapy'],
    'Hospital_Wing': ['East', 'West', 'North', 'South', 'East', 'West', 'North', 'South',
                      'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South',
                      'East', 'West', 'North', 'South'],
    'Age': [45, 32, 58, 41, 29, 67, 38, 52, 44, 36, 61, 48, 33, 55, 42, 39, 50, 35, 47, 31]
})

recovery_base = {
    'Surgery': 15,
    'Therapy': 8,
    'Medication': 5
}

data['Recovery_Days'] = data.apply(
    lambda row: recovery_base[row['Treatment_Type']] + 
                np.random.randint(-3, 4) + 
                (row['Age'] - 40) * 0.1,
    axis=1
).round().astype(int)

print("\nOriginal Patient Dataset:")
print(data.head(10))

print("\nDataset Summary:")
print(f"Total Patients: {len(data)}")
print(f"Treatment Types: {data['Treatment_Type'].unique()}")
print(f"Hospital Wings: {data['Hospital_Wing'].unique()}")
print(f"Average Recovery Days: {data['Recovery_Days'].mean():.2f}")

print("\n" + "="*70)
print("METHOD 1: LABEL ENCODING")
print("="*70)

label_encoder_treatment = LabelEncoder()
label_encoder_wing = LabelEncoder()

data['Treatment_Label'] = label_encoder_treatment.fit_transform(data['Treatment_Type'])
data['Wing_Label'] = label_encoder_wing.fit_transform(data['Hospital_Wing'])

print("\nLabel Encoded Data:")
print(data[['Patient_ID', 'Treatment_Type', 'Treatment_Label', 
            'Hospital_Wing', 'Wing_Label', 'Recovery_Days']].head(10))

print("\nLabel Encoding Mappings:")
print("\nTreatment Type Mapping:")
for i, label in enumerate(label_encoder_treatment.classes_):
    print(f"  {label} → {i}")

print("\nHospital Wing Mapping:")
for i, label in enumerate(label_encoder_wing.classes_):
    print(f"  {label} → {i}")

print("\n" + "="*70)
print("METHOD 2: ONE-HOT ENCODING")
print("="*70)

onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')

treatment_wing_encoded = onehot_encoder.fit_transform(data[['Treatment_Type', 'Hospital_Wing']])

onehot_columns = onehot_encoder.get_feature_names_out(['Treatment_Type', 'Hospital_Wing'])
onehot_df = pd.DataFrame(treatment_wing_encoded, columns=onehot_columns)

print("\nOne-Hot Encoded Features:")
print(onehot_df.head(10))

print("\nOne-Hot Encoding Mappings:")
print(f"Original Features: Treatment_Type, Hospital_Wing")
print(f"Encoded Features: {list(onehot_columns)}")
print(f"Note: drop='first' prevents multicollinearity")

data_with_onehot = pd.concat([data[['Patient_ID', 'Age', 'Recovery_Days']], onehot_df], axis=1)

print("\nDataset Ready for ML Model:")
print(data_with_onehot.head(10))

print("\n" + "="*70)
print("MODEL COMPARISON: LABEL VS ONE-HOT ENCODING")
print("="*70)

X_label = data[['Treatment_Label', 'Wing_Label', 'Age']]
X_onehot = pd.concat([data[['Age']], onehot_df], axis=1)
y = data['Recovery_Days']

X_label_train, X_label_test, y_train, y_test = train_test_split(
    X_label, y, test_size=0.3, random_state=42
)

X_onehot_train, X_onehot_test, _, _ = train_test_split(
    X_onehot, y, test_size=0.3, random_state=42
)

lr_label = LinearRegression()
lr_label.fit(X_label_train, y_train)
y_pred_label = lr_label.predict(X_label_test)

lr_onehot = LinearRegression()
lr_onehot.fit(X_onehot_train, y_train)
y_pred_onehot = lr_onehot.predict(X_onehot_test)

dt_label = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_label.fit(X_label_train, y_train)
y_pred_dt_label = dt_label.predict(X_label_test)

dt_onehot = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_onehot.fit(X_onehot_train, y_train)
y_pred_dt_onehot = dt_onehot.predict(X_onehot_test)

print("\nLinear Regression Performance:")
print(f"  Label Encoding - R²: {r2_score(y_test, y_pred_label):.4f}, MSE: {mean_squared_error(y_test, y_pred_label):.4f}")
print(f"  One-Hot Encoding - R²: {r2_score(y_test, y_pred_onehot):.4f}, MSE: {mean_squared_error(y_test, y_pred_onehot):.4f}")

print("\nDecision Tree Performance:")
print(f"  Label Encoding - R²: {r2_score(y_test, y_pred_dt_label):.4f}, MSE: {mean_squared_error(y_test, y_pred_dt_label):.4f}")
print(f"  One-Hot Encoding - R²: {r2_score(y_test, y_pred_dt_onehot):.4f}, MSE: {mean_squared_error(y_test, y_pred_dt_onehot):.4f}")

fig = plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
treatment_counts = data['Treatment_Type'].value_counts()
colors_treatment = ['#e74c3c', '#3498db', '#2ecc71']
plt.bar(treatment_counts.index, treatment_counts.values, color=colors_treatment, 
        edgecolor='black', linewidth=2)
plt.xlabel('Treatment Type', fontweight='bold')
plt.ylabel('Patient Count', fontweight='bold')
plt.title('Treatment Type Distribution', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 2)
wing_counts = data['Hospital_Wing'].value_counts()
colors_wing = ['#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
plt.bar(wing_counts.index, wing_counts.values, color=colors_wing, 
        edgecolor='black', linewidth=2)
plt.xlabel('Hospital Wing', fontweight='bold')
plt.ylabel('Patient Count', fontweight='bold')
plt.title('Hospital Wing Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 3)
avg_recovery = data.groupby('Treatment_Type')['Recovery_Days'].mean().sort_values(ascending=False)
plt.barh(avg_recovery.index, avg_recovery.values, color=colors_treatment, 
         edgecolor='black', linewidth=2)
plt.xlabel('Average Recovery Days', fontweight='bold')
plt.ylabel('Treatment Type', fontweight='bold')
plt.title('Average Recovery by Treatment', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(avg_recovery.values):
    plt.text(v + 0.3, i, f'{v:.1f}', va='center', fontweight='bold')

plt.subplot(2, 3, 4)
models = ['Linear Reg\n(Label)', 'Linear Reg\n(One-Hot)', 'Decision Tree\n(Label)', 'Decision Tree\n(One-Hot)']
r2_scores = [
    r2_score(y_test, y_pred_label),
    r2_score(y_test, y_pred_onehot),
    r2_score(y_test, y_pred_dt_label),
    r2_score(y_test, y_pred_dt_onehot)
]
colors_models = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']
bars = plt.bar(models, r2_scores, color=colors_models, edgecolor='black', linewidth=2)
plt.ylabel('R² Score', fontweight='bold')
plt.title('Model Performance Comparison', fontsize=12, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{score:.3f}', ha='center', fontweight='bold')

plt.subplot(2, 3, 5)
plt.scatter(y_test, y_pred_onehot, alpha=0.6, edgecolors='black', s=100, label='One-Hot')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Recovery Days', fontweight='bold')
plt.ylabel('Predicted Recovery Days', fontweight='bold')
plt.title('Linear Regression: Actual vs Predicted\n(One-Hot Encoding)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
encoding_comparison = pd.DataFrame({
    'Encoding': ['Label\nEncoding', 'One-Hot\nEncoding'],
    'Features': [2, len(onehot_columns)],
    'Best For': ['Tree Models', 'Linear Models']
})
x_pos = np.arange(len(encoding_comparison))
bars = plt.bar(x_pos, encoding_comparison['Features'], 
               color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
plt.xticks(x_pos, encoding_comparison['Encoding'])
plt.ylabel('Number of Features', fontweight='bold')
plt.title('Feature Count Comparison', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, (bar, feat, best) in enumerate(zip(bars, encoding_comparison['Features'], encoding_comparison['Best For'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{feat}\n({best})', ha='center', fontweight='bold', fontsize=9)

plt.suptitle('Healthcare Patient Records - Encoding Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_7_Mar02_Encoding/healthcare_encoding_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")

plt.show()

data.to_csv('Day_7_Mar02_Encoding/healthcare_label_encoded.csv', index=False)
data_with_onehot.to_csv('Day_7_Mar02_Encoding/healthcare_onehot_encoded.csv', index=False)

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why is proper encoding critical in healthcare ML models?")
print("  - Patient safety: Wrong predictions can lead to incorrect treatment plans")
print("  - Resource allocation: Accurate recovery time helps hospital planning")
print("  - Cost optimization: Better predictions reduce unnecessary hospital stays")
print("  - Treatment effectiveness: Models help identify best treatment for conditions")

print("\nPart B: Which encoding is better for Treatment Type and Hospital Wing?")
print("  Treatment Type:")
print("    - One-Hot Encoding recommended for linear models")
print("    - No natural order between Surgery, Therapy, Medication")
print("    - Prevents false relationships (Surgery ≠ 2 × Medication)")
print("  Hospital Wing:")
print("    - One-Hot Encoding for linear models")
print("    - Label Encoding acceptable for tree-based models")
print("    - Wings are independent locations with no hierarchy")

print("\nPart C: How does encoding affect prediction accuracy?")
print(f"  Linear Regression:")
print(f"    - Label Encoding R²: {r2_score(y_test, y_pred_label):.4f}")
print(f"    - One-Hot Encoding R²: {r2_score(y_test, y_pred_onehot):.4f}")
print(f"    - One-Hot performs better for linear models")
print(f"  Decision Tree:")
print(f"    - Both encodings work well (tree-based models handle both)")

print("\nPart D: Risks of using wrong encoding in medical predictions?")
print("  - Incorrect recovery time estimates → patient dissatisfaction")
print("  - Resource misallocation → bed shortages or wastage")
print("  - Treatment bias → model may favor certain treatments incorrectly")
print("  - Legal implications → wrong predictions can lead to liability issues")
print("  - Loss of trust → patients and doctors lose confidence in AI systems")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"\n1. Total Patients Analyzed: {len(data)}")
print(f"2. Treatment with Longest Recovery: {avg_recovery.index[0]} ({avg_recovery.values[0]:.1f} days)")
print(f"3. Treatment with Shortest Recovery: {avg_recovery.index[-1]} ({avg_recovery.values[-1]:.1f} days)")
print(f"4. Best Model: Linear Regression with One-Hot Encoding (R²: {r2_score(y_test, y_pred_onehot):.4f})")
print(f"5. Feature Count: Label (2) vs One-Hot ({len(onehot_columns)})")

print("\n" + "="*70)
print("RECOMMENDATIONS FOR HOSPITAL")
print("="*70)

print("\n1. Use One-Hot Encoding for linear models predicting recovery time")
print("2. Consider tree-based models if interpretability is less critical")
print("3. Collect more patient data to improve model accuracy")
print("4. Include additional features: diagnosis, severity, comorbidities")
print("5. Regularly validate model predictions against actual outcomes")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
