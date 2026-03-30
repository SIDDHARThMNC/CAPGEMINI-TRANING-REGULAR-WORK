"""
SCENARIO: Customer Purchase Prediction using Pipeline and GridSearchCV

A retail company wants to predict whether a customer will purchase a product (Yes/No) 
based on their profile.

FEATURES:
- age: Age of the customer (numeric)
- income: Monthly income (numeric)
- gender: Male/Female (categorical)
- city: City of residence (categorical)
- purchased: Target variable (1 = Purchased, 0 = Not Purchased)

PIPELINE APPROACH:
The data science team creates a Pipeline that performs:
1. Scaling of numeric features using StandardScaler
2. Encoding of categorical features using OneHotEncoder
3. Classification using LogisticRegression

HYPERPARAMETER TUNING:
Uses GridSearchCV to find the best hyperparameters:
- C: Regularization strength (inverse)
- solver: Optimization algorithm
- max_iter: Maximum iterations for convergence

QUESTIONS:
Part A: What is GridSearchCV and why is it important for model optimization?
Part B: How does the pipeline prevent data leakage during cross-validation?
Part C: What is the role of the 'C' parameter in Logistic Regression?
Part D: How do we interpret the best parameters found by GridSearchCV?
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
print("CUSTOMER PURCHASE PREDICTION - PIPELINE + GRIDSEARCHCV")
print("="*70)

np.random.seed(42)

n_customers = 200

data = {
    'age': np.random.randint(18, 65, n_customers),
    'income': np.random.randint(20000, 100000, n_customers),
    'gender': np.random.choice(['Male', 'Female'], n_customers),
    'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], n_customers)
}

df = pd.DataFrame(data)

purchase_prob = (
    (df['age'] > 30) * 0.2 +
    (df['income'] > 50000) * 0.3 +
    (df['city'] == 'Mumbai') * 0.15 +
    (df['city'] == 'Bangalore') * 0.15 +
    0.1
)

df['purchased'] = (np.random.random(n_customers) < purchase_prob).astype(int)

print("\nDataset Overview:")
print(df.head(10))

print("\nDataset Statistics:")
print(df.describe())

print("\nPurchase Distribution:")
print(df['purchased'].value_counts())
print(f"Purchase Rate: {df['purchased'].mean():.2%}")

print("\nCity Distribution:")
print(df['city'].value_counts())

print("\nGender Distribution:")
print(df['gender'].value_counts())

X = df[['age', 'income', 'gender', 'city']]
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "="*70)
print("BUILDING ML PIPELINE")
print("="*70)

numeric_features = ['age', 'income']
categorical_features = ['gender', 'city']

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
    ('classifier', LogisticRegression(random_state=42))
])

print("\nPipeline Structure:")
print(pipeline)

print("\n" + "="*70)
print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*70)

param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__max_iter': [100, 200, 500]
}

print("\nParameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)
print(f"\nTotal combinations to test: {total_combinations}")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nTraining GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\n" + "="*70)
print("GRIDSEARCHCV RESULTS")
print("="*70)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

results_df = pd.DataFrame(grid_search.cv_results_)
top_5 = results_df.nsmallest(5, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]

print("\nTop 5 Parameter Combinations:")
print(top_5.to_string(index=False))

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
print(classification_report(y_test, y_pred, target_names=['Not Purchased', 'Purchased']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

baseline_pipeline.fit(X_train, y_train)
baseline_pred = baseline_pipeline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print(f"\nBaseline Model Accuracy (default params): {baseline_accuracy:.4f}")
print(f"Tuned Model Accuracy: {accuracy:.4f}")
print(f"Improvement: {(accuracy - baseline_accuracy):.4f} ({((accuracy - baseline_accuracy)/baseline_accuracy)*100:.2f}%)")

print("\n" + "="*70)
print("MAKING PREDICTIONS FOR NEW CUSTOMERS")
print("="*70)

new_customers = pd.DataFrame({
    'age': [25, 45, 35, 50],
    'income': [30000, 80000, 55000, 95000],
    'gender': ['Female', 'Male', 'Female', 'Male'],
    'city': ['Mumbai', 'Bangalore', 'Delhi', 'Chennai']
})

print("\nNew Customer Data:")
print(new_customers)

predictions = best_model.predict(new_customers)
probabilities = best_model.predict_proba(new_customers)

print("\nPurchase Predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    result = "Purchased" if pred == 1 else "Not Purchased"
    confidence = prob[pred] * 100
    purchase_prob = prob[1] * 100
    print(f"Customer {i+1}: {result} (Confidence: {confidence:.2f}%, Purchase Prob: {purchase_prob:.2f}%)")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

fig = plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
purchase_counts = df['purchased'].value_counts()
colors = ['#e74c3c', '#2ecc71']
plt.bar(['Not Purchased', 'Purchased'], purchase_counts.values, color=colors, 
        edgecolor='black', linewidth=2)
plt.ylabel('Count', fontweight='bold')
plt.title('Purchase Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(purchase_counts.values):
    plt.text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.subplot(2, 3, 2)
c_values = [0.01, 0.1, 1, 10, 100]
c_scores = []
for c in c_values:
    scores = results_df[results_df['param_classifier__C'] == c]['mean_test_score']
    c_scores.append(scores.mean())

plt.plot(c_values, c_scores, marker='o', linewidth=2, markersize=8, color='steelblue')
plt.xscale('log')
plt.xlabel('C (Regularization)', fontweight='bold')
plt.ylabel('Mean CV Accuracy', fontweight='bold')
plt.title('GridSearchCV: C Parameter Impact', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Purchased', 'Purchased'], 
            yticklabels=['Not Purchased', 'Purchased'])
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
city_purchase = df.groupby(['city', 'purchased']).size().unstack(fill_value=0)
city_purchase.plot(kind='bar', color=colors, edgecolor='black', linewidth=2, ax=plt.gca())
plt.xlabel('City', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Purchase by City', fontsize=12, fontweight='bold')
plt.legend(['Not Purchased', 'Purchased'])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 6)
models = ['Baseline\n(Default)', 'Tuned\n(GridSearch)']
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

plt.suptitle('Customer Purchase Prediction - Pipeline + GridSearchCV', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_7_Mar02_Encoding/customer_purchase_gridsearch_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")

plt.show()

results_df = pd.DataFrame({
    'Customer': [f'Customer {i+1}' for i in range(len(new_customers))],
    'Age': new_customers['age'].values,
    'Income': new_customers['income'].values,
    'Gender': new_customers['gender'].values,
    'City': new_customers['city'].values,
    'Prediction': ['Purchased' if p == 1 else 'Not Purchased' for p in predictions],
    'Purchase_Probability': [prob[1] * 100 for prob in probabilities]
})

results_df.to_csv('Day_7_Mar02_Encoding/customer_purchase_predictions.csv', index=False)

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: What is GridSearchCV?")
print("  - Automated hyperparameter tuning technique")
print("  - Tests all combinations of parameters in param_grid")
print("  - Uses cross-validation to evaluate each combination")
print("  - Selects best parameters based on scoring metric")
print("  - Prevents overfitting by validating on multiple folds")
print("  - Importance: Finds optimal model configuration automatically")

print("\nPart B: How pipeline prevents data leakage:")
print("  - Preprocessing fitted only on training data in each CV fold")
print("  - Test fold never sees training data during preprocessing")
print("  - StandardScaler uses only training fold statistics")
print("  - OneHotEncoder learns categories only from training fold")
print("  - Ensures realistic performance estimates")

print("\nPart C: Role of 'C' parameter:")
print("  - C = Inverse of regularization strength")
print("  - Small C (e.g., 0.01): Strong regularization, simpler model")
print("  - Large C (e.g., 100): Weak regularization, complex model")
print(f"  - Best C found: {grid_search.best_params_['classifier__C']}")
print("  - Balances bias-variance tradeoff")

print("\nPart D: Interpreting best parameters:")
print(f"  - Best C: {grid_search.best_params_['classifier__C']}")
print(f"  - Best solver: {grid_search.best_params_['classifier__solver']}")
print(f"  - Best max_iter: {grid_search.best_params_['classifier__max_iter']}")
print("  - These parameters gave highest cross-validation accuracy")
print("  - Use these for production deployment")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"\n1. Best CV Score: {grid_search.best_score_:.2%}")
print(f"2. Test Accuracy: {accuracy:.2%}")
print(f"3. ROC-AUC Score: {roc_auc:.4f}")
print(f"4. Improvement over baseline: {((accuracy - baseline_accuracy)/baseline_accuracy)*100:.2f}%")
print(f"5. Total combinations tested: {total_combinations}")
print(f"6. Purchase Rate: {df['purchased'].mean():.2%}")

print("\n" + "="*70)
print("BUSINESS RECOMMENDATIONS")
print("="*70)

print("\n1. Target customers with age > 30 and income > 50,000")
print("2. Focus marketing campaigns in Mumbai and Bangalore")
print("3. Use model predictions to personalize product recommendations")
print("4. Retrain model quarterly with new customer data")
print("5. Monitor model performance and retune hyperparameters if needed")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
