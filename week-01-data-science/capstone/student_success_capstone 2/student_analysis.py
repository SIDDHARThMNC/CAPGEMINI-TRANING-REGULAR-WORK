"""
CAPSTONE PROJECT: Student Success & Career Path Prediction

SCENARIO:
The university wants to analyze student performance data to:
- Predict exam scores (Regression)
- Classify students into "At Risk" vs "On Track" categories (Classification)
- Cluster students into groups with similar study habits (Clustering)
- Recommend interventions (extra tutoring, workshops, counseling)

DATASET FEATURES:
- Student_ID: Unique identifier
- Hours_Studied: Weekly study hours
- Attendance (%): Class attendance percentage
- Assignments_Submitted: Number of assignments completed
- Previous_Sem_GPA: GPA from previous semester
- Participation_Score: Class participation rating
- Final_Exam_Score: Final exam score (target for regression)
- Pass_Fail: Pass or Fail status (target for classification)
- Career_Readiness_Score: Career preparation score
- Age: Student age
- Gender: Male or Female

OBJECTIVES:
1. REGRESSION: Predict final exam scores based on study habits and performance
2. CLASSIFICATION: Identify students at risk of failing
3. CLUSTERING: Group students by similar characteristics for targeted interventions
4. RECOMMENDATIONS: Provide specific intervention strategies for each student group

QUESTIONS TO EXPLORE:
- What factors most strongly predict exam success?
- Can we identify at-risk students early enough to help them?
- What are the distinct student profiles in our dataset?
- What interventions would be most effective for each group?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("STUDENT SUCCESS & CAREER PATH PREDICTION - CAPSTONE PROJECT")
print("="*90)

df = pd.read_csv('student_success_capstone 2/student_dataset.csv')

print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Statistics:")
print(df.describe())

df['Gender_Encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Pass_Fail_Encoded'] = df['Pass_Fail'].map({'Fail': 0, 'Pass': 1})

print("\n" + "="*90)
print("PART 1: REGRESSION - PREDICTING FINAL EXAM SCORES")
print("="*90)

features_reg = ['Hours_Studied', 'Attendance (%)', 'Assignments_Submitted', 
                'Previous_Sem_GPA', 'Participation_Score', 'Gender_Encoded']
X_reg = df[features_reg]
y_reg = df['Final_Exam_Score']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
lr_pred = lr_model.predict(X_test_reg)
lr_mse = mean_squared_error(y_test_reg, lr_pred)
lr_r2 = r2_score(y_test_reg, lr_pred)

print(f"\nLinear Regression Results:")
print(f"  MSE: {lr_mse:.4f}")
print(f"  RMSE: {np.sqrt(lr_mse):.4f}")
print(f"  R² Score: {lr_r2:.4f}")

feature_importance = pd.DataFrame({
    'Feature': features_reg,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Importance (Coefficients):")
print(feature_importance.to_string(index=False))

print("\n" + "="*90)
print("PART 2: CLASSIFICATION - PASS/FAIL PREDICTION")
print("="*90)

features_clf = ['Hours_Studied', 'Attendance (%)', 'Assignments_Submitted', 
                'Previous_Sem_GPA', 'Participation_Score', 'Gender_Encoded']
X_clf = df[features_clf]
y_clf = df['Pass_Fail_Encoded']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_clf_scaled, y_train_clf)
log_pred = log_model.predict(X_test_clf_scaled)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test_clf, log_pred, target_names=['Fail', 'Pass']))

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_clf, y_train_clf)
dt_pred = dt_model.predict(X_test_clf)

print("\nDecision Tree Classification Report:")
print(classification_report(y_test_clf, dt_pred, target_names=['Fail', 'Pass']))

print("\n" + "="*90)
print("PART 3: CLUSTERING - STUDENT STUDY HABIT GROUPS")
print("="*90)

cluster_features = ['Hours_Studied', 'Attendance (%)', 'Participation_Score']
X_cluster = df[cluster_features]

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

inertias = []
silhouette_scores = []
K_range = range(2, 6)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, labels))

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_cluster_scaled)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster Characteristics:")
cluster_summary = df.groupby('Cluster')[cluster_features + ['Final_Exam_Score', 'Career_Readiness_Score']].mean()
print(cluster_summary)

cluster_names = {}
for i in range(optimal_k):
    avg_score = cluster_summary.loc[i, 'Final_Exam_Score']
    if avg_score >= 75:
        cluster_names[i] = "High Achievers"
    elif avg_score >= 60:
        cluster_names[i] = "Steady Performers"
    else:
        cluster_names[i] = "Needs Support"

df['Cluster_Name'] = df['Cluster'].map(cluster_names)

print("\n" + "="*90)
print("STUDENT SEGMENTS & INTERVENTIONS")
print("="*90)

for cluster_id, name in cluster_names.items():
    cluster_df = df[df['Cluster'] == cluster_id]
    fail_count = (cluster_df['Pass_Fail'] == 'Fail').sum()
    
    print(f"\n{name} (Cluster {cluster_id}):")
    print(f"  Students: {len(cluster_df)}")
    print(f"  Avg Exam Score: {cluster_df['Final_Exam_Score'].mean():.2f}")
    print(f"  Avg Career Readiness: {cluster_df['Career_Readiness_Score'].mean():.2f}")
    print(f"  Failed Students: {fail_count}")
    
    if "High Achievers" in name:
        print("  Interventions:")
        print("     - Leadership opportunities & peer tutoring roles")
        print("     - Advanced placement programs")
        print("     - Career mentorship & networking events")
    elif "Steady" in name:
        print("  Interventions:")
        print("     - Optional study groups & workshops")
        print("     - Career guidance sessions")
        print("     - Skill development programs")
    else:
        print("  Interventions:")
        print("     - Mandatory tutoring sessions")
        print("     - Study skills workshop")
        print("     - Weekly progress monitoring")

df.to_csv('student_success_capstone 2/student_results.csv', index=False)

fig = plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.scatter(y_test_reg, lr_pred, alpha=0.7, edgecolors='black', s=100)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 
         'r--', linewidth=2)
plt.xlabel('Actual Exam Score', fontweight='bold')
plt.ylabel('Predicted Exam Score', fontweight='bold')
plt.title(f'Exam Score Prediction\nR² = {lr_r2:.4f}', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
top_features = feature_importance.head(6)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors, edgecolor='black')
plt.xlabel('Coefficient', fontweight='bold')
plt.title('Feature Importance\n(Exam Score)', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 3, 3)
cm_log = confusion_matrix(y_test_clf, log_pred)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Pass/Fail Classification\n(Logistic Regression)', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')

plt.subplot(2, 3, 4)
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('Inertia', fontweight='bold')
plt.title('Elbow Method', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

plt.subplot(2, 3, 5)
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('Silhouette Score', fontweight='bold')
plt.title('Cluster Quality', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

plt.subplot(2, 3, 6)
cluster_counts = df['Cluster_Name'].value_counts()
colors_pie = plt.cm.Set3(range(len(cluster_counts)))
plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', 
        colors=colors_pie, startangle=90, textprops={'fontweight': 'bold'})
plt.title('Student Distribution\nby Segment', fontsize=12, fontweight='bold')

plt.suptitle('Student Success & Career Path Prediction - Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('student_success_capstone 2/analysis_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")
plt.show()

print("\n" + "="*90)
print("CAPSTONE PROJECT COMPLETE")
print("="*90)
