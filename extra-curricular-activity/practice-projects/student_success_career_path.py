# Description
# Capstone Project: Student Success & Career Path Prediction

# Scenario
# The university wants to analyze student performance data to:
# Predict exam scores (Regression).
# Classify students into "At Risk" vs. "On Track" categories (Classification).
# Cluster students into groups with similar study habits (Clustering).
# Recommend interventions (extra tutoring, workshops, counseling).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, silhouette_score

# Load dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Student_Success_Career_Path.csv")
df = pd.read_csv(csv_path)
print(df.head())

# Prepare data
df_model = df.copy()

# Remove ID
df_model = df_model.drop("Student_ID", axis=1)

# Encode Gender
df_model["Gender"] = df_model["Gender"].map({"Male": 0, "Female": 1})

# ============================================
# Regression - Predict Final Exam Score
# ============================================
X_reg = df_model.drop(["Final_Exam_Score", "Pass_Fail"], axis=1)
y_reg = df_model["Final_Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

pred = reg_model.predict(X_test_scaled)

print("\nRegression Results")
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# ============================================
# Classification - Predict Pass/Fail
# ============================================
df_model["Pass_Fail"] = df_model["Pass_Fail"].map({"Fail": 0, "Pass": 1})

X_clf = df_model.drop("Pass_Fail", axis=1)
y_clf = df_model["Pass_Fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_scaled, y_train)

pred = clf_model.predict(X_test_scaled)

print("\nClassification Results")
print(classification_report(y_test, pred))

# ============================================
# Clustering - Group students by study habits
# ============================================
features = df_model[
    ["Hours_Studied",
     "Attendance (%)",
     "Assignments_Submitted",
     "Participation_Score"]
]

X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df_model["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nSilhouette Score:", silhouette_score(X_scaled, df_model["Cluster"]))

print("\nCluster Means:")
print(df_model.groupby("Cluster").mean())
