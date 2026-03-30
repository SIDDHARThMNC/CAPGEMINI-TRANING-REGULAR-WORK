"""
SCENARIO: Predicting Titanic Survival

Researchers are studying the Titanic disaster and want to build models that predict 
whether a passenger would survive or not based on their information.

Features used:
- Passenger class (pclass)
- Gender (sex)
- Age (age)
- Number of siblings/spouses aboard (sibsp)
- Number of parents/children aboard (parch)
- Ticket fare (fare)

Label:
- 1 = Survived
- 0 = Died

The researchers train three different models:
- Logistic Regression
- K-Nearest Neighbors (KNN) with k=5
- Decision Tree with max depth = 4

They then evaluate each model using a classification report (precision, recall, F1-score, accuracy).

QUESTIONS:
- Which model performs best at predicting survival, and why?
- How does Logistic Regression differ from Decision Tree in terms of interpretability?
- Why is scaling applied before training Logistic Regression and KNN, but not strictly needed for Decision Trees?
- Looking at the classification report, what do precision and recall mean in the context of survival predictions?
  * Precision: Of those predicted to survive, how many actually survived?
  * Recall: Of all who truly survived, how many were correctly predicted?
- If you were a historian, which model would you trust more to explain survival patterns, and why?
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TITANIC SURVIVAL PREDICTION - MODEL COMPARISON")
print("="*80)

df = sns.load_dataset('titanic')

print("\nOriginal Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
target = 'survived'

df_clean = df[features + [target]].copy()
df_clean = df_clean.dropna()

print(f"\nCleaned Dataset Shape: {df_clean.shape}")

df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})

print("\nFeature Statistics:")
print(df_clean.describe())

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining Set: {X_train.shape[0]} samples")
print(f"Test Set: {X_test.shape[0]} samples")
print(f"Survival Rate in Training: {y_train.mean():.2%}")
print(f"Survival Rate in Test: {y_test.mean():.2%}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"\nAccuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=['Died', 'Survived']))

print("\n" + "="*80)
print("MODEL 2: K-NEAREST NEIGHBORS (k=5)")
print("="*80)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f"\nAccuracy: {knn_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, knn_pred, target_names=['Died', 'Survived']))

print("\n" + "="*80)
print("MODEL 3: DECISION TREE (max_depth=4)")
print("="*80)

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"\nAccuracy: {dt_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, dt_pred, target_names=['Died', 'Survived']))

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN (k=5)', 'Decision Tree (depth=4)'],
    'Accuracy': [lr_accuracy, knn_accuracy, dt_accuracy],
    'Scaling Required': ['Yes', 'Yes', 'No']
})

print("\n", results.to_string(index=False))

best_model = results.loc[results['Accuracy'].idxmax(), 'Model']
best_accuracy = results['Accuracy'].max()
print(f"\nBest Model: {best_model} with {best_accuracy:.4f} accuracy")

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

print("\nLogistic Regression Coefficients:")
lr_coef = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(lr_coef.to_string(index=False))

print("\nDecision Tree Feature Importances:")
dt_importance = pd.DataFrame({
    'Feature': features,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(dt_importance.to_string(index=False))

fig = plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Died', 'Survived'], 
            yticklabels=['Died', 'Survived'], cbar_kws={'label': 'Count'})
plt.title('Logistic Regression\nConfusion Matrix', fontsize=13, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

plt.subplot(2, 3, 2)
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', xticklabels=['Died', 'Survived'], 
            yticklabels=['Died', 'Survived'], cbar_kws={'label': 'Count'})
plt.title('KNN (k=5)\nConfusion Matrix', fontsize=13, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

plt.subplot(2, 3, 3)
cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', xticklabels=['Died', 'Survived'], 
            yticklabels=['Died', 'Survived'], cbar_kws={'label': 'Count'})
plt.title('Decision Tree (depth=4)\nConfusion Matrix', fontsize=13, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

plt.subplot(2, 3, 4)
accuracies = [lr_accuracy, knn_accuracy, dt_accuracy]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = plt.bar(['Logistic\nRegression', 'KNN\n(k=5)', 'Decision Tree\n(depth=4)'], 
               accuracies, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.subplot(2, 3, 5)
lr_coef_sorted = lr_coef.sort_values('Coefficient')
colors_coef = ['red' if x < 0 else 'green' for x in lr_coef_sorted['Coefficient']]
plt.barh(lr_coef_sorted['Feature'], lr_coef_sorted['Coefficient'], color=colors_coef, 
         edgecolor='black', linewidth=1.5)
plt.xlabel('Coefficient Value', fontweight='bold')
plt.title('Logistic Regression\nFeature Coefficients', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 3, 6)
dt_importance_sorted = dt_importance.sort_values('Importance')
plt.barh(dt_importance_sorted['Feature'], dt_importance_sorted['Importance'], 
         color='#e74c3c', edgecolor='black', linewidth=1.5)
plt.xlabel('Importance Score', fontweight='bold')
plt.title('Decision Tree\nFeature Importances', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.suptitle('Titanic Survival Prediction - Model Comparison Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Capstone_Projects_Feb28_2026/titanic_survival_prediction_capstone 1/titanic_model_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'titanic_model_comparison.png'")

plt.show()

results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN (k=5)', 'Decision Tree (depth=4)'],
    'Accuracy': [lr_accuracy, knn_accuracy, dt_accuracy],
    'Predictions_Died': [sum(lr_pred == 0), sum(knn_pred == 0), sum(dt_pred == 0)],
    'Predictions_Survived': [sum(lr_pred == 1), sum(knn_pred == 1), sum(dt_pred == 1)]
})

results_df.to_csv('titanic_survival_prediction_capstone 1/model_comparison_results.csv', index=False)
print("Results saved to 'model_comparison_results.csv'")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n1. Best Performing Model:")
print(f"   {best_model} achieved the highest accuracy of {best_accuracy:.4f}")

print("\n2. Interpretability:")
print("   - Logistic Regression: Linear relationships, coefficient-based")
print("   - Decision Tree: Rule-based, easy to visualize decision paths")
print("   - KNN: Instance-based, less interpretable")

print("\n3. Scaling Requirement:")
print("   - Logistic Regression & KNN: Require feature scaling")
print("   - Decision Tree: Scale-invariant, no scaling needed")

print("\n4. Most Important Features:")
print(f"   - Logistic Regression: {lr_coef.iloc[0]['Feature']} (coef: {lr_coef.iloc[0]['Coefficient']:.4f})")
print(f"   - Decision Tree: {dt_importance.iloc[0]['Feature']} (importance: {dt_importance.iloc[0]['Importance']:.4f})")

print("\n5. Precision vs Recall:")
print("   - Precision: Of predicted survivors, how many actually survived")
print("   - Recall: Of actual survivors, how many were correctly predicted")

print("\n6. Historical Analysis:")
print("   - Decision Tree: Best for explaining survival patterns")
print("   - Reason: Clear rules (e.g., 'if female and 1st class, then survived')")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
