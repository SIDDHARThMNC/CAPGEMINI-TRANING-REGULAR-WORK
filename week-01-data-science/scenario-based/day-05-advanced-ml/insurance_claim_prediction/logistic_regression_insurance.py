import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 70)
print("LOGISTIC REGRESSION - INSURANCE CLAIM PREDICTION")
print("=" * 70)

print("\nScenario: Predicting Insurance Claims 🚗")
print("\nContext:")
print("A car insurance company wants to predict whether a driver is likely")
print("to file a claim in the next year based on their age.")
print("\nTarget Variable:")
print("  • 1 = Claim Filed")
print("  • 0 = No Claim")

data = {
    "Age": [18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65],
    "Claim_Filed": [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

df.to_csv('insurance_claim_prediction/insurance_claim_dataset.csv', index=False)
print("\n✓ Dataset saved as 'insurance_claim_dataset.csv'")

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Total Drivers: {}".format(len(df)))
print("Claims Filed: {}".format(df['Claim_Filed'].sum()))
print("No Claims: {}".format(len(df) - df['Claim_Filed'].sum()))
print("Claim Rate: {:.1f}%".format(df['Claim_Filed'].mean() * 100))

X = df[['Age']]
y = df['Claim_Filed']

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
print("log(p/(1-p)) = {:.4f} + {:.4f} * Age".format(model.intercept_[0], model.coef_[0][0]))
print("\nWhere p = Probability of filing a claim")

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
print("                Predicted")
print("              No Claim  Claim")
print("Actual No Claim    {}      {}".format(conf_matrix[0][0], conf_matrix[0][1]))
print("       Claim       {}      {}".format(conf_matrix[1][0], conf_matrix[1][1]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Claim', 'Claim']))

age_range = np.linspace(18, 65, 100).reshape(-1, 1)
probabilities = model.predict_proba(age_range)[:, 1]

threshold_age = None
for age, prob in zip(age_range, probabilities):
    if prob >= 0.5:
        threshold_age = age[0]
        break

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
if threshold_age:
    print("The probability of filing a claim crosses 50% at age: {:.1f} years".format(threshold_age))
else:
    print("The probability never crosses 50% in the given age range")

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS FOR NEW DRIVERS")
print("=" * 70)
new_ages = pd.DataFrame({'Age': [21, 30, 40, 50, 60]})
new_predictions = model.predict(new_ages)
new_probabilities = model.predict_proba(new_ages)[:, 1]

for age, pred, prob in zip(new_ages['Age'], new_predictions, new_probabilities):
    claim_status = "Claim" if pred == 1 else "No Claim"
    print("Age {} years: {} (Probability: {:.2f}%)".format(age, claim_status, prob * 100))

print("\nCreating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(df[df['Claim_Filed']==0]['Age'], df[df['Claim_Filed']==0]['Claim_Filed'], 
                color='green', s=100, label='No Claim', alpha=0.7, marker='o')
axes[0].scatter(df[df['Claim_Filed']==1]['Age'], df[df['Claim_Filed']==1]['Claim_Filed'], 
                color='red', s=100, label='Claim Filed', alpha=0.7, marker='x')
axes[0].plot(age_range, probabilities, color='blue', linewidth=3, label='Logistic Curve')
axes[0].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='50% Threshold')
if threshold_age:
    axes[0].axvline(x=threshold_age, color='purple', linestyle='--', linewidth=2, 
                    label='Age at 50% ({:.1f})'.format(threshold_age))
axes[0].set_xlabel('Age (years)', fontsize=12, weight='bold')
axes[0].set_ylabel('Probability of Filing Claim', fontsize=12, weight='bold')
axes[0].set_title('Logistic Regression: Claim Probability vs Age', fontsize=14, weight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(-0.1, 1.1)

cm_display = conf_matrix
im = axes[1].imshow(cm_display, cmap='Blues', aspect='auto')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['No Claim', 'Claim'])
axes[1].set_yticklabels(['No Claim', 'Claim'])
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
plt.savefig('insurance_claim_prediction/logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'logistic_regression_analysis.png'")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("1. The logistic curve shows how claim probability decreases with age")
print("2. Younger drivers (< 30 years) have higher claim probability")
print("3. Older drivers (> 40 years) have lower claim probability")
print("4. The model can help insurance companies adjust premiums based on age")
print("5. Accuracy of {:.2f}% indicates good predictive performance".format(accuracy * 100))

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
