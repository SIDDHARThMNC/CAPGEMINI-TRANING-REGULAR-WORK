"""
SCENARIO: Movie Streaming Platform User Segmentation

A movie streaming company has collected data on 1,000 users to understand viewing patterns 
and improve personalization.

FEATURES:
- Average watch time per week (hours)
- Number of devices used (TV, phone, tablet)
- Subscription pauses/cancellations frequency
- Genre preferences (Action, Comedy, Drama percentages)

OBJECTIVES:
- Group users into meaningful segments using K-Means clustering
- Use Elbow Method to find optimal number of clusters
- Validate with Silhouette Score for cluster quality
- Design targeted strategies for each user segment

EXPECTED SEGMENTS:
- Weekend Binge-Watchers: High engagement users
- Casual Family Viewers: Occasional viewers
- Genre Loyalists: Strong preference for specific genres
- At-Risk Cancelers: High subscription pause frequency
- Multi-Device Power Users: Active across platforms

BUSINESS APPLICATIONS:
- Personalized movie recommendations
- Loyalty rewards for engaged users
- Re-engagement campaigns for at-risk users
- Content strategy based on genre preferences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic movie streaming user data (1000 users)
n_users = 1000

data = {
    'user_id': range(1, n_users + 1),
    'avg_watch_time_hours': np.random.uniform(2, 40, n_users),  # Hours per week
    'num_devices': np.random.randint(1, 5, n_users),  # 1-4 devices
    'subscription_pauses': np.random.randint(0, 6, n_users),  # Number of pauses
    'action_genre_pct': np.random.uniform(0, 100, n_users),  # Percentage preference
    'comedy_genre_pct': np.random.uniform(0, 100, n_users),
    'drama_genre_pct': np.random.uniform(0, 100, n_users),
}

df = pd.DataFrame(data)

# Normalize genre percentages to sum to 100
total_genre = df['action_genre_pct'] + df['comedy_genre_pct'] + df['drama_genre_pct']
df['action_genre_pct'] = (df['action_genre_pct'] / total_genre * 100).round(2)
df['comedy_genre_pct'] = (df['comedy_genre_pct'] / total_genre * 100).round(2)
df['drama_genre_pct'] = (df['drama_genre_pct'] / total_genre * 100).round(2)

# Save dataset
df.to_csv('movie_streaming_clustering/user_streaming_data.csv', index=False)
print("Dataset created with 1000 users")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# Select features for clustering
features = ['avg_watch_time_hours', 'num_devices', 'subscription_pauses', 
            'action_genre_pct', 'comedy_genre_pct', 'drama_genre_pct']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. ELBOW METHOD - Find optimal number of clusters
print("\n" + "="*60)
print("ELBOW METHOD ANALYSIS")
print("="*60)

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k}: Inertia = {kmeans.inertia_:.2f}")

# 2. SILHOUETTE SCORE - Validate cluster quality
print("\n" + "="*60)
print("SILHOUETTE SCORE ANALYSIS")
print("="*60)

silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")

# Find optimal K based on silhouette score
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on Silhouette Score: {optimal_k}")

# 3. FINAL CLUSTERING with optimal K
print("\n" + "="*60)
print(f"FINAL K-MEANS CLUSTERING (K={optimal_k})")
print("="*60)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_scaled)

# Analyze clusters
print("\nCluster Distribution:")
print(df['cluster'].value_counts().sort_index())

print("\nCluster Characteristics:")
cluster_summary = df.groupby('cluster')[features].mean()
print(cluster_summary)

# Assign meaningful names to clusters
cluster_names = {}
for i in range(optimal_k):
    cluster_data = cluster_summary.loc[i]
    
    if cluster_data['avg_watch_time_hours'] > 25:
        name = "Weekend Binge-Watchers"
    elif cluster_data['subscription_pauses'] > 3:
        name = "At-Risk Cancelers"
    elif cluster_data['num_devices'] >= 3:
        name = "Multi-Device Power Users"
    elif cluster_data['drama_genre_pct'] > 40:
        name = "Drama Loyalists"
    elif cluster_data['action_genre_pct'] > 40:
        name = "Action Enthusiasts"
    elif cluster_data['comedy_genre_pct'] > 40:
        name = "Comedy Fans"
    elif cluster_data['avg_watch_time_hours'] < 10:
        name = "Casual Family Viewers"
    else:
        name = f"Balanced Viewers {i+1}"
    
    cluster_names[i] = name

df['cluster_name'] = df['cluster'].map(cluster_names)

print("\n" + "="*60)
print("CLUSTER SEGMENTS")
print("="*60)
for cluster_id, name in cluster_names.items():
    count = len(df[df['cluster'] == cluster_id])
    print(f"Cluster {cluster_id}: {name} ({count} users)")

# Save clustered data
df.to_csv('movie_streaming_clustering/clustered_users.csv', index=False)
print("\nClustered data saved to 'clustered_users.csv'")

# VISUALIZATION
fig = plt.figure(figsize=(18, 12))

# 1. Elbow Method Plot
plt.subplot(2, 3, 1)
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, fontweight='bold')
plt.title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# 2. Silhouette Score Plot
plt.subplot(2, 3, 2)
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
plt.xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
plt.ylabel('Silhouette Score', fontsize=11, fontweight='bold')
plt.title('Silhouette Score Analysis', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# 3. Cluster Distribution
plt.subplot(2, 3, 3)
cluster_counts = df['cluster'].value_counts().sort_index()
colors = plt.cm.Set3(range(len(cluster_counts)))
plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
plt.xlabel('Cluster', fontsize=11, fontweight='bold')
plt.ylabel('Number of Users', fontsize=11, fontweight='bold')
plt.title('User Distribution Across Clusters', fontsize=13, fontweight='bold')
plt.xticks(cluster_counts.index)
plt.grid(True, alpha=0.3, axis='y')

# 4. Watch Time vs Subscription Pauses
plt.subplot(2, 3, 4)
scatter = plt.scatter(df['avg_watch_time_hours'], df['subscription_pauses'], 
                     c=df['cluster'], cmap='Set3', s=50, alpha=0.6, edgecolors='black')
plt.xlabel('Avg Watch Time (hours/week)', fontsize=11, fontweight='bold')
plt.ylabel('Subscription Pauses', fontsize=11, fontweight='bold')
plt.title('Watch Time vs Subscription Pauses', fontsize=13, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# 5. Genre Preferences by Cluster
plt.subplot(2, 3, 5)
genre_data = df.groupby('cluster')[['action_genre_pct', 'comedy_genre_pct', 'drama_genre_pct']].mean()
x = np.arange(len(genre_data))
width = 0.25

plt.bar(x - width, genre_data['action_genre_pct'], width, label='Action', color='#ff6b6b')
plt.bar(x, genre_data['comedy_genre_pct'], width, label='Comedy', color='#4ecdc4')
plt.bar(x + width, genre_data['drama_genre_pct'], width, label='Drama', color='#45b7d1')

plt.xlabel('Cluster', fontsize=11, fontweight='bold')
plt.ylabel('Genre Preference (%)', fontsize=11, fontweight='bold')
plt.title('Genre Preferences by Cluster', fontsize=13, fontweight='bold')
plt.xticks(x, genre_data.index)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 6. Cluster Characteristics Heatmap
plt.subplot(2, 3, 6)
cluster_normalized = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())
sns.heatmap(cluster_normalized.T, annot=True, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={'label': 'Normalized Value'}, linewidths=0.5)
plt.xlabel('Cluster', fontsize=11, fontweight='bold')
plt.ylabel('Features', fontsize=11, fontweight='bold')
plt.title('Cluster Characteristics (Normalized)', fontsize=13, fontweight='bold')

plt.suptitle('🎬 Movie Streaming Platform - User Segmentation Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('movie_streaming_clustering/streaming_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'streaming_analysis.png'")

plt.show()

# BUSINESS INSIGHTS
print("\n" + "="*60)
print("📊 BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*60)

for cluster_id, name in cluster_names.items():
    cluster_df = df[df['cluster'] == cluster_id]
    avg_watch = cluster_df['avg_watch_time_hours'].mean()
    avg_pauses = cluster_df['subscription_pauses'].mean()
    avg_devices = cluster_df['num_devices'].mean()
    
    print(f"\n{name} (Cluster {cluster_id}):")
    print(f"  • Users: {len(cluster_df)}")
    print(f"  • Avg Watch Time: {avg_watch:.1f} hours/week")
    print(f"  • Avg Subscription Pauses: {avg_pauses:.1f}")
    print(f"  • Avg Devices: {avg_devices:.1f}")
    
    # Recommendations
    if "Binge" in name:
        print(f"  💡 Strategy: Loyalty rewards, early access to new releases")
    elif "Risk" in name or avg_pauses > 3:
        print(f"  💡 Strategy: Re-engagement campaigns, personalized offers")
    elif "Casual" in name or avg_watch < 10:
        print(f"  💡 Strategy: Family bundles, weekend promotions")
    elif "Loyalist" in name or "Enthusiast" in name or "Fans" in name:
        print(f"  💡 Strategy: Genre-specific recommendations, curated playlists")
    else:
        print(f"  💡 Strategy: Balanced content recommendations")

print("\n" + "="*60)
print("✅ Analysis Complete!")
print("="*60)
