import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

print("=" * 70)
print("EMAIL SPAM FILTER - MODEL EVALUATION")
print("=" * 70)

print("\nScenario: Email Spam Filter 📧")
print("\nContext:")
print("A company has built a machine learning model to detect spam emails.")
print("They want to evaluate the model using multiple metrics:")
print("  • Accuracy - Overall correctness")
print("  • Precision - How many predicted spam are actually spam")
print("  • Recall - How many actual spam were caught")
print("  • F1 Score - Balance between precision and recall")
print("  • ROC-AUC - Model's ability to distinguish classes")

y_true = [1,0,1,1,0,1,0,0,1,0]
y_pred = [1,0,1,0,0,1,1,0,1,0]
y_prob = [0.9,0.1,0.8,0.4,0.2,0.85,0.6,0.15,0.7,0.3]

print("\n" + "=" * 70)
print("DATA")
print("=" * 70)
print("True Labels (y_true): 1 = Spam, 0 = Not Spam")
print("Predictions (y_pred): What the model guessed")
print("Probabilities (y_prob): Model's confidence (0-1)")

print("\n{:<10} {:<15} {:<15} {:<15}".format("Email", "Actual", "Predicted", "Probability"))
print("-" * 70)
for i in range(len(y_true)):
    actual = "Spam" if y_true[i] == 1 else "Not Spam"
    predicted = "Spam" if y_pred[i] == 1 else "Not Spam"
    print("{:<10} {:<15} {:<15} {:<15.2f}".format(i+1, actual, predicted, y_prob[i]))

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_true, y_pred)
print("              Predicted")
print("           Not Spam  Spam")
print("Actual Not Spam  {}        {}".format(cm[0][0], cm[0][1]))
print("       Spam      {}        {}".format(cm[1][0], cm[1][1]))

tn, fp, fn, tp = cm.ravel()
print("\nBreakdown:")
print("  True Negatives (TN): {} - Correctly identified Not Spam".format(tn))
print("  False Positives (FP): {} - Wrongly marked as Spam".format(fp))
print("  False Negatives (FN): {} - Missed Spam emails".format(fn))
print("  True Positives (TP): {} - Correctly identified Spam".format(tp))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

print("\n" + "=" * 70)
print("MODEL EVALUATION METRICS")
print("=" * 70)

print("\n1. ACCURACY: {:.2f}%".format(accuracy * 100))
print("   Definition: (TP + TN) / Total")
print("   Calculation: ({} + {}) / {} = {:.2f}%".format(tp, tn, len(y_true), accuracy * 100))
print("   Meaning: {}% of all predictions were correct".format(int(accuracy * 100)))

print("\n2. PRECISION: {:.2f}%".format(precision * 100))
print("   Definition: TP / (TP + FP)")
print("   Calculation: {} / ({} + {}) = {:.2f}%".format(tp, tp, fp, precision * 100))
print("   Meaning: When model says 'Spam', it's correct {:.0f}% of the time".format(precision * 100))

print("\n3. RECALL (Sensitivity): {:.2f}%".format(recall * 100))
print("   Definition: TP / (TP + FN)")
print("   Calculation: {} / ({} + {}) = {:.2f}%".format(tp, tp, fn, recall * 100))
print("   Meaning: Model catches {:.0f}% of all actual spam emails".format(recall * 100))

print("\n4. F1 SCORE: {:.2f}%".format(f1 * 100))
print("   Definition: 2 * (Precision * Recall) / (Precision + Recall)")
print("   Calculation: 2 * ({:.2f} * {:.2f}) / ({:.2f} + {:.2f}) = {:.2f}%".format(
    precision, recall, precision, recall, f1 * 100))
print("   Meaning: Harmonic mean of precision and recall")

print("\n5. ROC-AUC SCORE: {:.2f}".format(roc_auc))
print("   Definition: Area Under the ROC Curve")
print("   Range: 0.5 (random) to 1.0 (perfect)")
print("   Meaning: Model's ability to distinguish spam from not spam")

print("\n" + "=" * 70)
print("METRIC INTERPRETATION")
print("=" * 70)
print("High Precision ({:.0f}%):".format(precision * 100))
print("  ✓ Few false alarms - users won't miss important emails")
print("  ✗ But {} spam emails were missed (False Negatives)".format(fn))

print("\nHigh Recall ({:.0f}%):".format(recall * 100))
print("  ✓ Catches most spam - inbox stays clean")
print("  ✗ But {} legitimate emails marked as spam (False Positives)".format(fp))

print("\nF1 Score ({:.0f}%):".format(f1 * 100))
print("  • Balances precision and recall")
print("  • Good for imbalanced datasets")

print("\nROC-AUC ({:.2f}):".format(roc_auc))
if roc_auc >= 0.9:
    print("  • Excellent discrimination ability")
elif roc_auc >= 0.8:
    print("  • Good discrimination ability")
elif roc_auc >= 0.7:
    print("  • Acceptable discrimination ability")
else:
    print("  • Poor discrimination ability")

print("\n" + "=" * 70)
print("BUSINESS IMPACT")
print("=" * 70)
print("Cost of False Positives (FP = {}):".format(fp))
print("  • Important emails go to spam folder")
print("  • User frustration and missed opportunities")

print("\nCost of False Negatives (FN = {}):".format(fn))
print("  • Spam reaches inbox")
print("  • Cluttered inbox, potential security risks")

print("\nModel Trade-off:")
if precision > recall:
    print("  • Model is conservative - prefers to let spam through")
    print("  • Better for avoiding false alarms")
else:
    print("  • Model is aggressive - prefers to catch all spam")
    print("  • Better for keeping inbox clean")

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

im = axes[0, 0].imshow(cm, cmap='RdYlGn', aspect='auto')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(['Not Spam', 'Spam'])
axes[0, 0].set_yticklabels(['Not Spam', 'Spam'])
axes[0, 0].set_xlabel('Predicted', fontsize=11, weight='bold')
axes[0, 0].set_ylabel('Actual', fontsize=11, weight='bold')
axes[0, 0].set_title('Confusion Matrix', fontsize=13, weight='bold')

for i in range(2):
    for j in range(2):
        text = axes[0, 0].text(j, i, cm[i, j], ha="center", va="center",
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=24, weight='bold')
plt.colorbar(im, ax=axes[0, 0])

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

bars = axes[0, 1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Score (%)', fontsize=11, weight='bold')
axes[0, 1].set_title('Model Performance Metrics', fontsize=13, weight='bold')
axes[0, 1].set_ylim(0, 100)
axes[0, 1].grid(alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    '{:.1f}%'.format(value), ha='center', va='bottom', 
                    fontsize=10, weight='bold')

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
axes[1, 0].plot(fpr, tpr, color='#e74c3c', linewidth=3, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
axes[1, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Classifier')
axes[1, 0].set_xlabel('False Positive Rate', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('True Positive Rate', fontsize=11, weight='bold')
axes[1, 0].set_title('ROC Curve', fontsize=13, weight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

x_pos = np.arange(len(y_true))
axes[1, 1].bar(x_pos, y_prob, color=['red' if y == 1 else 'green' for y in y_true], 
               alpha=0.7, edgecolor='black')
axes[1, 1].axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1, 1].set_xlabel('Email Number', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Spam Probability', fontsize=11, weight='bold')
axes[1, 1].set_title('Model Confidence per Email', fontsize=13, weight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels([str(i+1) for i in range(len(y_true))])
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('email_spam_evaluation/spam_evaluation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'spam_evaluation_analysis.png'")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("To improve the model:")
print("  1. Collect more training data")
print("  2. Add more features (sender, subject keywords, etc.)")
print("  3. Adjust classification threshold based on business needs")
print("  4. Use ensemble methods for better accuracy")
print("  5. Regularly retrain with new spam patterns")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
