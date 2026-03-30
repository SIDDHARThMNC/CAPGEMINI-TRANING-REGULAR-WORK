"""
SCENARIO: Email Spam Detection

An email service provider wants to automatically filter spam messages to protect users.

FEATURES:
- Email content and characteristics
- Word frequency patterns
- Sender information

TARGET:
- Spam: 1 (Unwanted email)
- Ham: 0 (Legitimate email)

OBJECTIVE:
- Build Logistic Regression classifier to detect spam
- Minimize false positives (legitimate emails marked as spam)
- Maximize spam detection rate

BUSINESS IMPACT:
- Improved user experience
- Reduced inbox clutter
- Protection from phishing and malicious emails
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 70)
print("LOGISTIC REGRESSION - EMAIL SPAM DETECTION")
print("=" * 70)

print("\nScenario: Detecting Spam Emails 📧")
print("\nContext:")
print("An email service provider wants to build a model to detect whether")
print("an incoming email is spam or not spam using three features:")
print("  • Word Count - Total number of words in the email")
print("  • Has Link - Whether the email contains a hyperlink (1=yes, 0=no)")
print("  • Caps Ratio - Proportion of words written in ALL CAPS")
print("\nTarget Variable:")
print("  • 1 = Spam")
print("  • 0 = Not Spam")

X = np.array([[50, 1, 0.8],   # SPAM
              [200, 0, 0.1],  # Not spam
              [30, 1, 0.9],   # SPAM
              [180, 0, 0.05], # Not spam
              [10, 1, 0.95],  # SPAM
              [220, 0, 0.08]]) # Not spam

y = np.array([1, 0, 1, 0, 1, 0])  # 1=Spam, 0=Not spam

df = pd.DataFrame(X, columns=['Word_Count', 'Has_Link', 'Caps_Ratio'])
df['Spam'] = y

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

df.to_csv('email_spam_detection/spam_dataset.csv', index=False)
print("\n✓ Dataset saved as 'spam_dataset.csv'")

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Total Emails: {}".format(len(df)))
print("Spam Emails: {}".format(df['Spam'].sum()))
print("Not Spam: {}".format(len(df) - df['Spam'].sum()))
print("Spam Rate: {:.1f}%".format(df['Spam'].mean() * 100))

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)
print(df.describe().to_string())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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
print("Intercept (β0): {:.4f}".format(model.intercept_[0]))
print("\nCoefficients:")
feature_names = ['Word_Count', 'Has_Link', 'Caps_Ratio']
for feature, coef in zip(feature_names, model.coef_[0]):
    print("  {}: {:.4f}".format(feature, coef))

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION EQUATION")
print("=" * 70)
equation = "log(p/(1-p)) = {:.4f}".format(model.intercept_[0])
for feature, coef in zip(feature_names, model.coef_[0]):
    sign = "+" if coef >= 0 else ""
    equation += " {} {:.4f}*{}".format(sign, coef, feature)
print(equation)
print("\nWhere p = Probability of being spam")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA")
print("=" * 70)
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['Actual'] = y_test
test_df['Predicted'] = y_pred
test_df['Spam_Probability'] = y_pred_proba
print(test_df.to_string(index=False))

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
print("              Predicted")
print("           Not Spam  Spam")
print("Actual Not Spam  {}      {}".format(conf_matrix[0][0], conf_matrix[0][1]))
print("       Spam      {}      {}".format(conf_matrix[1][0], conf_matrix[1][1]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

print("\n" + "=" * 70)
print("INTERPRETING PROBABILITIES")
print("=" * 70)
print("If an email has a 70% probability of being spam:")
print("  • The model is 70% confident it's spam")
print("  • There's a 30% chance it's legitimate")
print("  • Typically, emails with >50% probability are classified as spam")
print("  • Higher probability = Higher confidence in spam classification")

print("\n" + "=" * 70)
print("TEST ON NEW EMAILS")
print("=" * 70)
new_emails = np.array([
    [100, 0, 0.10],  # Normal email
    [40, 1, 0.85],   # Suspicious
    [200, 0, 0.05],  # Normal email
    [15, 1, 0.95]    # Very suspicious
])

new_predictions = model.predict(new_emails)
new_probabilities = model.predict_proba(new_emails)[:, 1]

print("\nNew Email Predictions:")
for i, (email, pred, prob) in enumerate(zip(new_emails, new_predictions, new_probabilities)):
    spam_status = "SPAM" if pred == 1 else "NOT SPAM"
    link_status = "Yes" if email[1] == 1 else "No"
    print("\nEmail {}:".format(i+1))
    print("  Words: {}, Has Link: {}, Caps Ratio: {:.2f}".format(
        int(email[0]), link_status, email[2]))
    print("  Prediction: {} (Probability: {:.2f}%)".format(spam_status, prob * 100))

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(df[df['Spam']==0]['Word_Count'], df[df['Spam']==0]['Caps_Ratio'], 
                   color='green', s=150, label='Not Spam', alpha=0.7, marker='o', edgecolors='black', linewidths=2)
axes[0, 0].scatter(df[df['Spam']==1]['Word_Count'], df[df['Spam']==1]['Caps_Ratio'], 
                   color='red', s=150, label='Spam', alpha=0.7, marker='x', linewidths=3)
axes[0, 0].set_xlabel('Word Count', fontsize=11, weight='bold')
axes[0, 0].set_ylabel('Caps Ratio', fontsize=11, weight='bold')
axes[0, 0].set_title('Word Count vs Caps Ratio', fontsize=12, weight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

has_link_spam = df[df['Has_Link']==1]['Spam'].value_counts()
no_link_spam = df[df['Has_Link']==0]['Spam'].value_counts()
x = ['No Link', 'Has Link']
spam_counts = [no_link_spam.get(1, 0), has_link_spam.get(1, 0)]
not_spam_counts = [no_link_spam.get(0, 0), has_link_spam.get(0, 0)]
x_pos = np.arange(len(x))
axes[0, 1].bar(x_pos - 0.2, not_spam_counts, 0.4, label='Not Spam', color='green', alpha=0.7, edgecolor='black')
axes[0, 1].bar(x_pos + 0.2, spam_counts, 0.4, label='Spam', color='red', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Link Presence', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Count', fontsize=11, weight='bold')
axes[0, 1].set_title('Link Presence vs Spam', fontsize=12, weight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(x)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3, axis='y')

cm_display = conf_matrix
im = axes[1, 0].imshow(cm_display, cmap='RdYlGn_r', aspect='auto')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Not Spam', 'Spam'])
axes[1, 0].set_yticklabels(['Not Spam', 'Spam'])
axes[1, 0].set_xlabel('Predicted', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Actual', fontsize=11, weight='bold')
axes[1, 0].set_title('Confusion Matrix', fontsize=12, weight='bold')

for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, cm_display[i, j], ha="center", va="center", 
                              color="white" if cm_display[i, j] > cm_display.max()/2 else "black",
                              fontsize=20, weight='bold')
plt.colorbar(im, ax=axes[1, 0])

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': np.abs(model.coef_[0])
}).sort_values('Coefficient', ascending=True)
axes[1, 1].barh(feature_importance['Feature'], feature_importance['Coefficient'], 
                color='teal', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Absolute Coefficient', fontsize=11, weight='bold')
axes[1, 1].set_title('Feature Importance', fontsize=12, weight='bold')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('email_spam_detection/spam_detection_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'spam_detection_analysis.png'")

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)
feature_imp = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)
print(feature_imp.to_string(index=False))

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("1. Emails with high Caps Ratio (>0.8) are strong spam indicators")
print("2. Presence of links combined with high caps ratio = likely spam")
print("3. Low word count (<50) with links and high caps = very suspicious")
print("4. Model accuracy of {:.2f}% shows good performance".format(accuracy * 100))
print("5. Combining multiple features improves spam detection")

print("\n" + "=" * 70)
print("PRACTICAL RECOMMENDATIONS")
print("=" * 70)
print("• Flag emails with >50% spam probability for review")
print("• Emails with >80% probability can be auto-filtered")
print("• Monitor false positives to avoid blocking legitimate emails")
print("• Regularly update model with new spam patterns")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
