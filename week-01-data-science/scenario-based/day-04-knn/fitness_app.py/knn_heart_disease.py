import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 70)
print("K-NEAREST NEIGHBORS (KNN) - HEART DISEASE RISK PREDICTION")
print("=" * 70)

print("\nScenario: Fitness App Heart Disease Risk Assessment 🩺")
print("\nContext:")
print("A fitness app wants to predict whether a person is at risk of heart")
print("disease based on three lifestyle indicators:")
print("  • Exercise Level - Hours of physical activity per week")
print("  • Diet Quality - Rating from 1-5 (higher = healthier)")
print("  • Stress Level - Rating from 1-5 (higher = more stress)")
print("\nTarget Variable:")
print("  • 1 = At Risk")
print("  • 0 = Not at Risk")

df = pd.read_csv('fitness_app.py/Fitness_app_dataset - Sheet1.csv')

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Total Users: {}".format(len(df)))
print("At Risk: {}".format(df['AtRisk'].sum()))
print("Not at Risk: {}".format(len(df) - df['AtRisk'].sum()))
print("Risk Rate: {:.1f}%".format(df['AtRisk'].mean() * 100))

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)
print(df.describe().to_string())

X = df[['Exercise', 'Diet', 'Stress']].values
y = df['AtRisk'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)
print("Training Set: {} samples".format(len(X_train)))
print("Test Set: {} samples".format(len(X_test)))

print("\n" + "=" * 70)
print("WHY SCALING MATTERS IN KNN")
print("=" * 70)
print("Before Scaling:")
print("  Exercise: range {}-{}".format(df['Exercise'].min(), df['Exercise'].max()))
print("  Diet: range {}-{}".format(df['Diet'].min(), df['Diet'].max()))
print("  Stress: range {}-{}".format(df['Stress'].min(), df['Stress'].max()))
print("\nWithout scaling, features with larger ranges would dominate distance")
print("calculations. Scaling ensures all features contribute equally.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nAfter Scaling:")
print("  All features have mean ≈ 0 and standard deviation ≈ 1")
print("  This ensures fair contribution from all features")

k_values = [1, 3, 5]
accuracies = []
models = {}

print("\n" + "=" * 70)
print("TRAINING MODELS WITH DIFFERENT K VALUES")
print("=" * 70)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    models[k] = knn
    print("\nK = {}: Accuracy = {:.2f}%".format(k, accuracy * 100))

best_k = k_values[np.argmax(accuracies)]
best_model = models[best_k]

print("\n" + "=" * 70)
print("BEST MODEL SELECTION")
print("=" * 70)
print("Best K value: {}".format(best_k))
print("Best Accuracy: {:.2f}%".format(max(accuracies) * 100))

y_pred_best = best_model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred_best)

print("\n" + "=" * 70)
print("PREDICTIONS ON TEST DATA (K={})".format(best_k))
print("=" * 70)
test_df = pd.DataFrame(X_test, columns=['Exercise', 'Diet', 'Stress'])
test_df['Actual'] = y_test
test_df['Predicted'] = y_pred_best
print(test_df.to_string(index=False))

print("\n" + "=" * 70)
print("MODEL EVALUATION (K={})".format(best_k))
print("=" * 70)
print("Accuracy: {:.2f}%".format(max(accuracies) * 100))
print("\nConfusion Matrix:")
print("              Predicted")
print("           Not at Risk  At Risk")
print("Actual Not at Risk  {}         {}".format(conf_matrix[0][0], conf_matrix[0][1]))
print("       At Risk      {}         {}".format(conf_matrix[1][0], conf_matrix[1][1]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Not at Risk', 'At Risk']))

print("\n" + "=" * 70)
print("PREDICT FOR NEW USER")
print("=" * 70)
new_user = np.array([[4, 3, 4]])
new_user_scaled = scaler.transform(new_user)
prediction = best_model.predict(new_user_scaled)
prediction_proba = best_model.predict_proba(new_user_scaled)

print("New User Profile: [Exercise=4 hrs/week, Diet=3, Stress=4]")
print("\nPrediction: {}".format("At Risk" if prediction[0] == 1 else "Not at Risk"))
print("Probability of Being at Risk: {:.2f}%".format(prediction_proba[0][1] * 100))
print("Probability of Not Being at Risk: {:.2f}%".format(prediction_proba[0][0] * 100))

neighbors = best_model.kneighbors(new_user_scaled, return_distance=True)
print("\nNearest {} Neighbors:".format(best_k))
for i, (dist, idx) in enumerate(zip(neighbors[0][0], neighbors[1][0])):
    neighbor_data = X_train[idx]
    neighbor_label = y_train[idx]
    print("  Neighbor {}: Exercise={}, Diet={}, Stress={} → {} (Distance: {:.2f})".format(
        i+1, int(neighbor_data[0]), int(neighbor_data[1]), int(neighbor_data[2]),
        "At Risk" if neighbor_label == 1 else "Not at Risk", dist))

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(k_values, [acc * 100 for acc in accuracies], marker='o', linewidth=2, 
                markersize=10, color='blue')
axes[0, 0].set_xlabel('K Value', fontsize=11, weight='bold')
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11, weight='bold')
axes[0, 0].set_title('Model Accuracy vs K Value', fontsize=12, weight='bold')
axes[0, 0].set_xticks(k_values)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label='Best K={}'.format(best_k))
axes[0, 0].legend(fontsize=10)

axes[0, 1].scatter(df[df['AtRisk']==0]['Exercise'], 
                   df[df['AtRisk']==0]['Stress'], 
                   color='green', s=100, label='Not at Risk', alpha=0.7, marker='o', edgecolors='black')
axes[0, 1].scatter(df[df['AtRisk']==1]['Exercise'], 
                   df[df['AtRisk']==1]['Stress'], 
                   color='red', s=100, label='At Risk', alpha=0.7, marker='x', linewidths=2)
axes[0, 1].scatter(new_user[0][0], new_user[0][2], color='blue', s=200, marker='*', 
                   label='New User', edgecolors='black', linewidths=2)
axes[0, 1].set_xlabel('Exercise Hours/Week', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Stress Level', fontsize=11, weight='bold')
axes[0, 1].set_title('Exercise vs Stress Level', fontsize=12, weight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

cm_display = conf_matrix
im = axes[1, 0].imshow(cm_display, cmap='RdYlGn', aspect='auto')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Not at Risk', 'At Risk'])
axes[1, 0].set_yticklabels(['Not at Risk', 'At Risk'])
axes[1, 0].set_xlabel('Predicted', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Actual', fontsize=11, weight='bold')
axes[1, 0].set_title('Confusion Matrix (K={})'.format(best_k), fontsize=12, weight='bold')

for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, cm_display[i, j], ha="center", va="center", 
                              color="white" if cm_display[i, j] > cm_display.max()/2 else "black",
                              fontsize=20, weight='bold')
plt.colorbar(im, ax=axes[1, 0])

categories = ['Exercise', 'Diet', 'Stress']
at_risk_avg = [df[df['AtRisk']==1]['Exercise'].mean(),
               df[df['AtRisk']==1]['Diet'].mean(),
               df[df['AtRisk']==1]['Stress'].mean()]
not_risk_avg = [df[df['AtRisk']==0]['Exercise'].mean(),
                df[df['AtRisk']==0]['Diet'].mean(),
                df[df['AtRisk']==0]['Stress'].mean()]

x_pos = np.arange(len(categories))
axes[1, 1].bar(x_pos - 0.2, not_risk_avg, 0.4, label='Not at Risk', color='green', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x_pos + 0.2, at_risk_avg, 0.4, label='At Risk', color='red', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Lifestyle Factor', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Average Value', fontsize=11, weight='bold')
axes[1, 1].set_title('Average Lifestyle Factors by Risk Status', fontsize=12, weight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fitness_app.py/heart_disease_knn_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'heart_disease_knn_analysis.png'")

print("\n" + "=" * 70)
print("HOW CHANGING K AFFECTS PREDICTIONS")
print("=" * 70)
print("K=1: Uses only closest neighbor - sensitive to noise")
print("K=3: Majority vote from 3 neighbors - balanced approach")
print("K=5: Majority vote from 5 neighbors - smoother boundaries")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("1. People at risk tend to have:")
print("   - Lower exercise hours")
print("   - Lower diet quality")
print("   - Higher stress levels")
print("2. Best K={} provides {:.2f}% accuracy".format(best_k, max(accuracies) * 100))
print("3. Feature scaling is crucial for fair distance calculations")
print("4. KNN is effective for health risk assessment")

print("\n" + "=" * 70)
print("HEALTH RECOMMENDATIONS")
print("=" * 70)
print("To reduce heart disease risk:")
print("  • Increase exercise to 5+ hours per week")
print("  • Improve diet quality (aim for 4-5 rating)")
print("  • Manage stress levels (aim for 1-2 rating)")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
