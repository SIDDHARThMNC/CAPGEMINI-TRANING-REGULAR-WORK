import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 70)
print("DECISION TREE - UNIVERSITY ADMISSIONS PREDICTION")
print("=" * 70)

print("\nScenario: University Admissions Rulebook 🎓")
print("\nContext:")
print("You're an admissions officer at a university. Every day, students")
print("apply for admission, and you need to decide whether to accept or")
print("reject them. Instead of guessing, you build a rulebook")
print("(that's your Decision Tree).")

print("\n" + "=" * 70)
print("THE DATA")
print("=" * 70)
print("Each applicant has:")
print("  • High School GPA - How well they performed academically")
print("  • Entrance Exam Score - Their standardized test performance")
print("  • Extracurriculars - 1 = active in clubs/sports, 0 = not active")
print("\nTarget Variable:")
print("  • 1 = Accepted")
print("  • 0 = Rejected")

df = pd.read_csv('university.py/University Dataset - Sheet1.csv')

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string(index=False))

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Total Applications: {}".format(len(df)))
print("Accepted: {}".format(df['Admission_Label'].sum()))
print("Rejected: {}".format(len(df) - df['Admission_Label'].sum()))
print("Acceptance Rate: {:.1f}%".format(df['Admission_Label'].mean() * 100))

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)
print(df.describe().to_string())

X = df[['HighSchool_GPA', 'Exam_Score', 'Extracurriculars']].values
y = df['Admission_Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)
print("Training Set: {} applications".format(len(X_train)))
print("Test Set: {} applications".format(len(X_test)))

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

print("\n" + "=" * 70)
print("DECISION TREE MODEL TRAINED")
print("=" * 70)
print("Max Depth: 3")
print("Number of Features: 3")

y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA")
print("=" * 70)
test_df = pd.DataFrame(X_test, columns=['GPA', 'Exam_Score', 'Extracurriculars'])
test_df['Actual'] = y_test
test_df['Predicted'] = y_pred
test_df['Status'] = test_df['Predicted'].apply(lambda x: 'Accepted' if x == 1 else 'Rejected')
print(test_df.to_string(index=False))

conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
print("              Predicted")
print("           Rejected  Accepted")
print("Actual Rejected  {}         {}".format(conf_matrix[0][0], conf_matrix[0][1]))
print("       Accepted  {}         {}".format(conf_matrix[1][0], conf_matrix[1][1]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Accepted']))

feature_importance = dt_model.feature_importances_
features = ['HighSchool_GPA', 'Exam_Score', 'Extracurriculars']

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
for feature, importance in zip(features, feature_importance):
    print("{}: {:.2f}%".format(feature, importance * 100))

print("\n" + "=" * 70)
print("PREDICT FOR NEW APPLICANTS")
print("=" * 70)
new_applicants = [
    [3.5, 1180, 1],
    [2.8, 950, 0],
    [3.9, 1420, 1]
]

for i, applicant in enumerate(new_applicants, 1):
    prediction = dt_model.predict([applicant])
    prediction_proba = dt_model.predict_proba([applicant])
    status = "Accepted" if prediction[0] == 1 else "Rejected"
    print("\nApplicant {}: GPA={}, Exam={}, Extracurriculars={}".format(
        i, applicant[0], applicant[1], applicant[2]))
    print("Decision: {}".format(status))
    print("Acceptance Probability: {:.2f}%".format(prediction_proba[0][1] * 100))

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

plot_tree(dt_model, ax=axes[0, 0], feature_names=features, 
          class_names=['Rejected', 'Accepted'], filled=True, 
          rounded=True, fontsize=9)
axes[0, 0].set_title('Decision Tree Structure', fontsize=13, weight='bold', pad=10)

axes[0, 1].barh(features, feature_importance, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                edgecolor='black', linewidth=1.5)
axes[0, 1].set_xlabel('Importance', fontsize=11, weight='bold')
axes[0, 1].set_title('Feature Importance', fontsize=13, weight='bold')
axes[0, 1].grid(alpha=0.3, axis='x')

cm_display = conf_matrix
im = axes[1, 0].imshow(cm_display, cmap='RdYlGn', aspect='auto')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Rejected', 'Accepted'])
axes[1, 0].set_yticklabels(['Rejected', 'Accepted'])
axes[1, 0].set_xlabel('Predicted', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Actual', fontsize=11, weight='bold')
axes[1, 0].set_title('Confusion Matrix', fontsize=13, weight='bold')

for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, cm_display[i, j], ha="center", va="center",
                              color="white" if cm_display[i, j] > cm_display.max()/2 else "black",
                              fontsize=20, weight='bold')
plt.colorbar(im, ax=axes[1, 0])

accepted = df[df['Admission_Label']==1]
rejected = df[df['Admission_Label']==0]

axes[1, 1].scatter(rejected['HighSchool_GPA'], rejected['Exam_Score'], 
                   color='red', s=120, label='Rejected', alpha=0.7, 
                   marker='x', linewidths=2)
axes[1, 1].scatter(accepted['HighSchool_GPA'], accepted['Exam_Score'], 
                   color='green', s=120, label='Accepted', alpha=0.7, 
                   marker='o', edgecolors='black')
axes[1, 1].set_xlabel('High School GPA', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Exam Score', fontsize=11, weight='bold')
axes[1, 1].set_title('GPA vs Exam Score', fontsize=13, weight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('university.py/admission_decision_tree_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'admission_decision_tree_analysis.png'")

print("\n" + "=" * 70)
print("HOW DECISION TREE WORKS")
print("=" * 70)
print("1. The tree asks questions about features (GPA, Exam Score, etc.)")
print("2. Based on answers, it follows branches to make decisions")
print("3. Each split divides data into more homogeneous groups")
print("4. Leaf nodes contain final decisions (Accept/Reject)")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("1. GPA is the most important factor ({:.1f}%)".format(feature_importance[0] * 100))
print("2. Higher GPA and exam scores increase acceptance chances")
print("3. Extracurricular activities also play a role in decisions")
print("4. Decision trees create interpretable rules for admissions officers")

print("\n" + "=" * 70)
print("ADMISSION GUIDELINES")
print("=" * 70)
print("Based on the model:")
print("  • GPA > 3.3 with good exam score → High acceptance chance")
print("  • GPA < 3.0 → Likely rejection")
print("  • Extracurriculars can boost borderline applications")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
