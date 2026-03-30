"""
SCENARIO: Diabetes Risk Prediction

A healthcare provider wants to predict diabetes risk based on patient BMI data.

FEATURES:
- BMI (Body Mass Index): Weight-to-height ratio

TARGET:
- Diabetic: 1 (Has diabetes)
- Non-Diabetic: 0 (No diabetes)

OBJECTIVE:
- Build Logistic Regression model to classify diabetes risk
- Evaluate model performance with accuracy, precision, recall
- Identify BMI threshold for high-risk patients

BUSINESS IMPACT:
- Early intervention for at-risk patients
- Preventive healthcare programs
- Resource allocation for diabetes management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 70)
print("LOGISTIC REGRESSION - DIABETES RISK PREDICTION")
print("=" * 70)

print("\nScenario: Predicting Diabetes Risk 🏥")
print("\nContext:")
print("A hospital wants to predict whether patients are at risk of developing")
print("diabetes based on their BMI (Body Mass Index).")
print("\nTarget Variable:")
print("  • 1 = Diabetes")
print("  • 0 = No Diabetes")

df = pd.read_csv('diabetes_risk_prediction/BMI_dataset - Sheet1.csv')

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Total Patients: {}".format(len(df)))
print("Diabetes Cases: {}".format(df['Diabetes'].sum()))
print("No Diabetes: {}".format(len(df) - df['Diabetes'].sum()))
print("Diabetes Rate: {:.1f}%".format(df['Diabetes'].mean() * 100))

print("\n" + "=" * 70)
print("BMI CATEGORIES (WHO Classification)")
print("=" * 70)
print("Underweight: BMI < 18.5")
print("Normal weight: BMI 18.5 - 24.9")
print("Overweight: BMI 25.0 - 29.9")
print("Obese: BMI ≥ 30.0")

X = df[['BMI']]
y = df['Diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)
print("Training Set: {} samples".format(len(X_train)))
print("Test Set: {} samples".format(len(X_test)))

model = LogisticRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)
print("Coefficient (β1): {:.4f}".format(model.coef_[0][0]))
print("Intercept (β0): {:.4f}".format(model.intercept_[0]))

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION EQUATION")
print("=" * 70)
print("log(p/(1-p)) = {:.4f} + {:.4f} * BMI".format(model.intercept_[0], model.coef_[0][0]))
print("\nWhere p = Probability of having diabetes")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA")
print("=" * 70)
test_results = X_test.copy()
test_results['Actual'] = y_test.values
test_results['Predicted'] = y_pred
test_results['Probability'] = y_pred_proba
print(test_results.to_string(index=False))

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
print("                  Predicted")
print("              No Diabetes  Diabetes")
print("Actual No Diabetes    {}         {}".format(conf_matrix[0][0], conf_matrix[0][1]))
print("       Diabetes       {}         {}".format(conf_matrix[1][0], conf_matrix[1][1]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

bmi_range = np.linspace(15, 45, 100).reshape(-1, 1)
probabilities = model.predict_proba(bmi_range)[:, 1]

threshold_bmi = None
for bmi, prob in zip(bmi_range, probabilities):
    if prob >= 0.5:
        threshold_bmi = bmi[0]
        break

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
if threshold_bmi:
    print("The probability of diabetes crosses 50% at BMI: {:.1f}".format(threshold_bmi))
    if threshold_bmi < 25:
        category = "Normal weight"
    elif threshold_bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    print("This falls in the '{}' category".format(category))
else:
    print("The probability never crosses 50% in the given BMI range")

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS FOR NEW PATIENTS")
print("=" * 70)
new_bmis = pd.DataFrame({'BMI': [20, 25, 30, 35, 40]})
new_predictions = model.predict(new_bmis)
new_probabilities = model.predict_proba(new_bmis)[:, 1]

for bmi, pred, prob in zip(new_bmis['BMI'], new_predictions, new_probabilities):
    diabetes_status = "Diabetes" if pred == 1 else "No Diabetes"
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    print("BMI {:.1f} ({}): {} (Probability: {:.2f}%)".format(bmi, category, diabetes_status, prob * 100))

print("\nCreating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(df[df['Diabetes']==0]['BMI'], df[df['Diabetes']==0]['Diabetes'], 
                color='green', s=100, label='No Diabetes', alpha=0.7, marker='o')
axes[0].scatter(df[df['Diabetes']==1]['BMI'], df[df['Diabetes']==1]['Diabetes'], 
                color='red', s=100, label='Diabetes', alpha=0.7, marker='x')
axes[0].plot(bmi_range, probabilities, color='blue', linewidth=3, label='Logistic Curve')
axes[0].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='50% Threshold')
if threshold_bmi:
    axes[0].axvline(x=threshold_bmi, color='purple', linestyle='--', linewidth=2, 
                    label='BMI at 50% ({:.1f})'.format(threshold_bmi))

axes[0].axvspan(18.5, 25, alpha=0.1, color='green', label='Normal Weight')
axes[0].axvspan(25, 30, alpha=0.1, color='yellow', label='Overweight')
axes[0].axvspan(30, 45, alpha=0.1, color='red', label='Obese')

axes[0].set_xlabel('BMI (Body Mass Index)', fontsize=12, weight='bold')
axes[0].set_ylabel('Probability of Diabetes', fontsize=12, weight='bold')
axes[0].set_title('Logistic Regression: Diabetes Risk vs BMI', fontsize=14, weight='bold')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.3)
axes[0].set_ylim(-0.1, 1.1)

cm_display = conf_matrix
im = axes[1].imshow(cm_display, cmap='Greens', aspect='auto')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['No Diabetes', 'Diabetes'])
axes[1].set_yticklabels(['No Diabetes', 'Diabetes'])
axes[1].set_xlabel('Predicted', fontsize=12, weight='bold')
axes[1].set_ylabel('Actual', fontsize=12, weight='bold')
axes[1].set_title('Confusion Matrix', fontsize=14, weight='bold')

for i in range(2):
    for j in range(2):
        text = axes[1].text(j, i, cm_display[i, j], ha="center", va="center", 
                           color="white" if cm_display[i, j] > cm_display.max()/2 else "black",
                           fontsize=20, weight='bold')

plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('diabetes_risk_prediction/diabetes_risk_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'diabetes_risk_analysis.png'")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("1. The logistic curve shows diabetes risk increases with BMI")
print("2. Patients with BMI < 25 (Normal weight) have lower diabetes risk")
print("3. Patients with BMI ≥ 30 (Obese) have significantly higher risk")
print("4. The model can help identify high-risk patients for early intervention")
print("5. Accuracy of {:.2f}% indicates good predictive performance".format(accuracy * 100))

print("\n" + "=" * 70)
print("CLINICAL RECOMMENDATIONS")
print("=" * 70)
print("• BMI < 25: Low risk - Maintain healthy lifestyle")
print("• BMI 25-30: Moderate risk - Weight management recommended")
print("• BMI ≥ 30: High risk - Medical intervention and lifestyle changes needed")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
