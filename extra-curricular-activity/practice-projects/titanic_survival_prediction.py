# Date: 28/2/2026
# Scenario Question: Predicting Titanic Survival
# Researchers are studying the Titanic disaster and want to build models that predict whether a
#  passenger would survive or not survive based on their information.
# - Features used:
# - Passenger class (pclass)
# - Gender (sex)
# - Age (age)
# - Number of siblings/spouses aboard (sibsp)
# - Number of parents/children aboard (parch)
# - Ticket fare (fare)
# - Label:
# - 1 = Survived
# - 0 = Died
# The researchers train three different models:
# - Logistic Regression
# - K-Nearest Neighbors (KNN) with k=5
# - Decision Tree with max depth = 4
# They then evaluate each model using a classification report (precision, recall, F1-score, accuracy).

# ❓ Questions for Learners
# - Which model performs best at predicting survival, and why?
# - How does Logistic Regression differ from Decision Tree in terms of interpretability?
# - Why is scaling applied before training Logistic Regression and KNN, but not strictly needed
#  for Decision Trees?
# - Looking at the classification report, what do precision and recall mean in the context of survival
#  predictions?
# - Precision → Of those predicted to survive, how many actually survived?
# - Recall → Of all who truly survived, how many were correctly predicted?
# - If you were a historian, which model would you trust more to explain survival patterns, and why?

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns

# Load dataset
df = sns.load_dataset('titanic')
print(df.head())

# Logistic Regression
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
y = df['survived']

# Encode gender
X['sex'] = X['sex'].map({'male': 0, 'female': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

lr_pred = model.predict(X_test_scaled)
print('\nLogistic Regression')
print("Accuracy:", accuracy_score(y_test, lr_pred))

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)
print("\nKNN (k=5)")
print("Accuracy:", accuracy_score(y_test, knn_pred))

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)
print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, tree_pred))

# Question 1. Which model performs well?
# Decision Tree may get slightly better recall for survival
# Logistic Regression gives balanced precision + recall
# KNN performs well only if features are scaled properly

# - How does Logistic Regression differ from Decision Tree in terms of interpretability?
# Ans. Logistic Regression is interpretable through feature coefficients and odds ratios, providing a statistically
# grounded explanation of how each variable influences survival probability.
# Decision Trees are interpretable through explicit rule-based splits, making them easier to visualize and 
# understand for non-technical audiences.
# Logistic Regression provides quantitative insight, while Decision Trees provide structural decision logic.

# Why is scaling applied before training Logistic Regression and KNN, but not strictly needed
#  for Decision Trees?
# Ans. Scaling bring all the data in similar range which makes the prediction and classification easy and fast
