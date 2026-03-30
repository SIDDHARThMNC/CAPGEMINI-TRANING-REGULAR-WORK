# DBSCAN Clustering Algorithm
# Density-Based Spatial Clustering of Applications with Noise
#
# WHAT IS DBSCAN?
# DBSCAN is a clustering algorithm that groups together points that are 
# closely packed together, marking points in low-density regions as outliers.
#
# KEY CONCEPTS:
# 1. Epsilon (ε): Maximum distance between two points to be neighbors
# 2. MinPts: Minimum number of points to form a dense region (cluster)
# 3. Core Point: Point with at least MinPts neighbors within epsilon
# 4. Border Point: Point within epsilon of a core point but not core itself
# 5. Noise Point: Point that is neither core nor border (outlier)
#
# HOW TO CHOOSE EPSILON:
# Use K-distance graph (elbow method):
# - Calculate distance to k-th nearest neighbor for each point
# - Sort and plot these distances
# - Look for "elbow" in the graph - that's your epsilon value

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Function to plot k-distance graph
# This helps us choose the right epsilon value
def plot_k_distance_graph(X, k):
    """
    Plot k-distance graph to find optimal epsilon
    
    Parameters:
    X: Dataset (features)
    k: Number of neighbors (usually same as MinPts)
    """
    # Step 1: Fit NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    
    # Step 2: Find k nearest neighbors for each point
    distances, _ = neigh.kneighbors(X)
    
    # Step 3: Sort distances (we want k-th neighbor distance)
    distances = np.sort(distances[:, k-1])
    
    # Step 4: Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.grid(True)
    plt.show()

# Generate sample data for demonstration
print("=" * 60)
print("DBSCAN CLUSTERING DEMONSTRATION")
print("=" * 60)

# Create synthetic dataset with clusters and noise
np.random.seed(42)

# Cluster 1: Around (2, 2)
cluster1 = np.random.randn(50, 2) * 0.5 + [2, 2]

# Cluster 2: Around (8, 8)
cluster2 = np.random.randn(50, 2) * 0.5 + [8, 8]

# Cluster 3: Around (2, 8)
cluster3 = np.random.randn(50, 2) * 0.5 + [2, 8]

# Noise points (outliers)
noise = np.random.uniform(0, 10, (10, 2))

# Combine all data
X = np.vstack([cluster1, cluster2, cluster3, noise])

print(f"\nDataset created:")
print(f"Total points: {len(X)}")
print(f"Cluster 1: 50 points around (2, 2)")
print(f"Cluster 2: 50 points around (8, 8)")
print(f"Cluster 3: 50 points around (2, 8)")
print(f"Noise: 10 random points")

# Step 1: Plot k-distance graph to find epsilon
print("\n" + "=" * 60)
print("STEP 1: FINDING OPTIMAL EPSILON")
print("=" * 60)
print("\nPlotting k-distance graph...")
print("Look for the 'elbow' point in the graph")
print("The y-value at the elbow is your optimal epsilon")

plot_k_distance_graph(X, k=5)

# Step 2: Apply DBSCAN clustering
print("\n" + "=" * 60)
print("STEP 2: APPLYING DBSCAN")
print("=" * 60)

# Parameters for DBSCAN
epsilon = 0.5  # Maximum distance between neighbors
min_samples = 5  # Minimum points to form a cluster

print(f"\nDBSCAN Parameters:")
print(f"Epsilon (ε): {epsilon}")
print(f"MinPts: {min_samples}")

# Create and fit DBSCAN model
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# Step 3: Analyze results
print("\n" + "=" * 60)
print("STEP 3: RESULTS")
print("=" * 60)

# Count clusters (excluding noise which is labeled -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nNumber of clusters found: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Print cluster sizes
for cluster_id in set(labels):
    if cluster_id == -1:
        continue
    cluster_size = list(labels).count(cluster_id)
    print(f"Cluster {cluster_id}: {cluster_size} points")

# Step 4: Visualize clusters
print("\n" + "=" * 60)
print("STEP 4: VISUALIZATION")
print("=" * 60)

plt.figure(figsize=(12, 8))

# Plot each cluster with different color
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points in black
        color = 'black'
        marker = 'x'
        label_name = 'Noise'
    else:
        marker = 'o'
        label_name = f'Cluster {label}'
    
    # Get points belonging to this cluster
    class_member_mask = (labels == label)
    xy = X[class_member_mask]
    
    plt.scatter(xy[:, 0], xy[:, 1], 
               c=[color], 
               marker=marker,
               s=100,
               alpha=0.6,
               edgecolors='black',
               label=label_name)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'DBSCAN Clustering (ε={epsilon}, MinPts={min_samples})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dbscan/dbscan_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved: dbscan_clusters.png")

# Step 5: Detailed explanation
print("\n" + "=" * 60)
print("HOW DBSCAN WORKS")
print("=" * 60)

print("""
1. For each point, find all neighbors within epsilon distance

2. Classify points:
   - Core Point: Has >= MinPts neighbors (including itself)
   - Border Point: Within epsilon of a core point, but not core
   - Noise Point: Neither core nor border (outlier)

3. Form clusters:
   - Start with a core point
   - Add all reachable core points to the cluster
   - Add border points connected to these core points
   - Repeat for all unvisited core points

4. Mark remaining points as noise

ADVANTAGES:
✓ Finds clusters of arbitrary shape
✓ Automatically detects outliers
✓ No need to specify number of clusters
✓ Works well with spatial data

DISADVANTAGES:
✗ Sensitive to epsilon and MinPts parameters
✗ Struggles with varying density clusters
✗ Not suitable for high-dimensional data
""")

# Example with real-world scenario
print("\n" + "=" * 60)
print("REAL-WORLD EXAMPLE: CUSTOMER SEGMENTATION")
print("=" * 60)

# Create customer data
np.random.seed(42)
customer_data = pd.DataFrame({
    'Annual_Income': np.concatenate([
        np.random.normal(50000, 10000, 30),  # Low income
        np.random.normal(100000, 15000, 30), # Medium income
        np.random.normal(150000, 20000, 30), # High income
        np.random.uniform(30000, 180000, 10) # Outliers
    ]),
    'Spending_Score': np.concatenate([
        np.random.normal(30, 10, 30),   # Low spenders
        np.random.normal(50, 10, 30),   # Medium spenders
        np.random.normal(80, 10, 30),   # High spenders
        np.random.uniform(10, 90, 10)   # Outliers
    ])
})

print("\nCustomer Dataset:")
print(customer_data.head())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)

# Apply DBSCAN
dbscan_customers = DBSCAN(eps=0.5, min_samples=5)
customer_labels = dbscan_customers.fit_predict(X_scaled)

# Add labels to dataframe
customer_data['Cluster'] = customer_labels

# Visualize customer segments
plt.figure(figsize=(12, 8))

for label in set(customer_labels):
    if label == -1:
        color = 'black'
        marker = 'x'
        label_name = 'Outliers'
    else:
        color = plt.cm.Set1(label)
        marker = 'o'
        label_name = f'Segment {label}'
    
    mask = customer_labels == label
    plt.scatter(customer_data[mask]['Annual_Income'], 
               customer_data[mask]['Spending_Score'],
               c=[color],
               marker=marker,
               s=100,
               alpha=0.6,
               edgecolors='black',
               label=label_name)

plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using DBSCAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dbscan/customer_segmentation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Customer segmentation visualization saved")

# Summary statistics
print("\n" + "=" * 60)
print("CUSTOMER SEGMENTS SUMMARY")
print("=" * 60)

for label in set(customer_labels):
    if label == -1:
        segment_name = "Outliers"
    else:
        segment_name = f"Segment {label}"
    
    segment_data = customer_data[customer_data['Cluster'] == label]
    print(f"\n{segment_name}:")
    print(f"  Count: {len(segment_data)}")
    print(f"  Avg Income: ${segment_data['Annual_Income'].mean():.2f}")
    print(f"  Avg Spending: {segment_data['Spending_Score'].mean():.2f}")

print("\n" + "=" * 60)
print("✓ DBSCAN CLUSTERING COMPLETE!")
print("=" * 60)
