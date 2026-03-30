import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 70)
print("K-NEAREST NEIGHBORS (KNN) - MOVIE RECOMMENDATION")
print("=" * 70)

print("\nScenario: Movie Recommendation System 🎬")
print("\nContext:")
print("A streaming platform wants to recommend movies to users based on")
print("their preferences. Each movie is rated on three aspects:")
print("  • Action Rating - How action-packed it is (1-5)")
print("  • Comedy Rating - How funny it is (1-5)")
print("  • Drama Rating - How emotional it is (1-5)")
print("\nTarget Variable:")
print("  • 1 = Will Like")
print("  • 0 = Won't Like")

X = np.array([[5,2,3],[4,1,4],[1,5,2],[2,4,1],[5,1,5],[3,5,1],[1,4,3],[5,3,4],[2,1,4],[3,4,2]])
y = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

df = pd.DataFrame(X, columns=['Action_Rating', 'Comedy_Rating', 'Drama_Rating'])
df['Will_Like'] = y

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print(df.to_string())

df.to_csv('movie_recommendation_knn/movie_dataset.csv', index=False)
print("\n✓ Dataset saved as 'movie_dataset.csv'")

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Total Movies: {}".format(len(df)))
print("Will Like: {}".format(df['Will_Like'].sum()))
print("Won't Like: {}".format(len(df) - df['Will_Like'].sum()))
print("Like Rate: {:.1f}%".format(df['Will_Like'].mean() * 100))

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)
print(df.describe().to_string())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)
print("Training Set: {} samples".format(len(X_train)))
print("Test Set: {} samples".format(len(X_test)))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)
print("Features have been standardized (mean=0, std=1)")
print("This is important for KNN as it uses distance calculations")

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
test_df = pd.DataFrame(X_test, columns=['Action_Rating', 'Comedy_Rating', 'Drama_Rating'])
test_df['Actual'] = y_test
test_df['Predicted'] = y_pred_best
print(test_df.to_string(index=False))

print("\n" + "=" * 70)
print("MODEL EVALUATION (K={})".format(best_k))
print("=" * 70)
print("Accuracy: {:.2f}%".format(max(accuracies) * 100))
print("\nConfusion Matrix:")
print("              Predicted")
print("           Won't Like  Will Like")
print("Actual Won't Like  {}         {}".format(conf_matrix[0][0], conf_matrix[0][1]))
print("       Will Like   {}         {}".format(conf_matrix[1][0], conf_matrix[1][1]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=["Won't Like", 'Will Like']))

print("\n" + "=" * 70)
print("PREDICT FOR NEW USER")
print("=" * 70)
new_user = np.array([[4, 2, 4]])
new_user_scaled = scaler.transform(new_user)
prediction = best_model.predict(new_user_scaled)
prediction_proba = best_model.predict_proba(new_user_scaled)

print("New User Preferences: [Action=4, Comedy=2, Drama=4]")
print("\nPrediction: {}".format("Will Like" if prediction[0] == 1 else "Won't Like"))
print("Probability of Liking: {:.2f}%".format(prediction_proba[0][1] * 100))
print("Probability of Not Liking: {:.2f}%".format(prediction_proba[0][0] * 100))

neighbors = best_model.kneighbors(new_user_scaled, return_distance=True)
print("\nNearest {} Neighbors:".format(best_k))
for i, (dist, idx) in enumerate(zip(neighbors[0][0], neighbors[1][0])):
    neighbor_data = X_train[idx]
    neighbor_label = y_train[idx]
    print("  Neighbor {}: Action={}, Comedy={}, Drama={} → {} (Distance: {:.2f})".format(
        i+1, neighbor_data[0], neighbor_data[1], neighbor_data[2],
        "Like" if neighbor_label == 1 else "Dislike", dist))

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

axes[0, 1].scatter(df[df['Will_Like']==0]['Action_Rating'], 
                   df[df['Will_Like']==0]['Drama_Rating'], 
                   color='red', s=100, label="Won't Like", alpha=0.7, marker='x', linewidths=2)
axes[0, 1].scatter(df[df['Will_Like']==1]['Action_Rating'], 
                   df[df['Will_Like']==1]['Drama_Rating'], 
                   color='green', s=100, label='Will Like', alpha=0.7, marker='o', edgecolors='black')
axes[0, 1].scatter(new_user[0][0], new_user[0][2], color='blue', s=200, marker='*', 
                   label='New User', edgecolors='black', linewidths=2)
axes[0, 1].set_xlabel('Action Rating', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Drama Rating', fontsize=11, weight='bold')
axes[0, 1].set_title('Action vs Drama Ratings', fontsize=12, weight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

cm_display = conf_matrix
im = axes[1, 0].imshow(cm_display, cmap='Blues', aspect='auto')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(["Won't Like", 'Will Like'])
axes[1, 0].set_yticklabels(["Won't Like", 'Will Like'])
axes[1, 0].set_xlabel('Predicted', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Actual', fontsize=11, weight='bold')
axes[1, 0].set_title('Confusion Matrix (K={})'.format(best_k), fontsize=12, weight='bold')

for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, cm_display[i, j], ha="center", va="center", 
                              color="white" if cm_display[i, j] > cm_display.max()/2 else "black",
                              fontsize=20, weight='bold')
plt.colorbar(im, ax=axes[1, 0])

categories = ['Action', 'Comedy', 'Drama']
liked_avg = [df[df['Will_Like']==1]['Action_Rating'].mean(),
             df[df['Will_Like']==1]['Comedy_Rating'].mean(),
             df[df['Will_Like']==1]['Drama_Rating'].mean()]
disliked_avg = [df[df['Will_Like']==0]['Action_Rating'].mean(),
                df[df['Will_Like']==0]['Comedy_Rating'].mean(),
                df[df['Will_Like']==0]['Drama_Rating'].mean()]

x_pos = np.arange(len(categories))
axes[1, 1].bar(x_pos - 0.2, liked_avg, 0.4, label='Will Like', color='green', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x_pos + 0.2, disliked_avg, 0.4, label="Won't Like", color='red', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Movie Aspect', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Average Rating', fontsize=11, weight='bold')
axes[1, 1].set_title('Average Ratings by Preference', fontsize=12, weight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('movie_recommendation_knn/knn_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'knn_analysis.png'")

print("\n" + "=" * 70)
print("HOW CHANGING K AFFECTS PREDICTIONS")
print("=" * 70)
print("K=1 (Low K):")
print("  • Uses only the closest neighbor")
print("  • More sensitive to noise and outliers")
print("  • Can lead to overfitting")
print("  • More complex decision boundaries")

print("\nK=3 (Medium K):")
print("  • Balances between bias and variance")
print("  • Less sensitive to individual outliers")
print("  • Often provides good generalization")

print("\nK=5 (Higher K):")
print("  • Uses more neighbors for voting")
print("  • Smoother decision boundaries")
print("  • Less sensitive to noise")
print("  • May underfit if K is too large")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("1. Users who like action and drama tend to prefer certain movies")
print("2. Comedy preference varies more among users")
print("3. Best K={} provides {:.2f}% accuracy".format(best_k, max(accuracies) * 100))
print("4. Feature scaling is crucial for KNN performance")
print("5. KNN works well for recommendation systems with clear patterns")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
