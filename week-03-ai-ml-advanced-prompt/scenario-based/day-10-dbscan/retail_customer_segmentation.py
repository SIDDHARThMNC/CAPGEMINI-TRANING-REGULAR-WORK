# DBSCAN: Customer Segmentation for a Retail Store
#
# SCENARIO:
# A retail store wants to understand its customers better. They have data on:
# - How much each customer spends per visit (in dollars)
# - How frequently they visit the store (visits per month)
#
# GOAL:
# Group customers into clusters using DBSCAN:
# - "Loyal high spenders" - frequent visitors who spend a lot
# - "Occasional visitors" - infrequent, low spenders
# - "Moderate spenders" - middle ground customers
# - Identify outliers (unusual customer behavior)
#
# WHY DBSCAN?
# - Finds clusters of varying shapes (not just circular)
# - Automatically identifies outliers
# - No need to specify number of clusters beforehand

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

print("=" * 70)
print("RETAIL STORE: CUSTOMER SEGMENTATION USING DBSCAN")
print("=" * 70)

# Step 1: Prepare the customer data
# Each row: [Spending per visit ($), Visits per month]
data = np.array([
    [5000, 50], [5200, 48], [4800, 52],   # Loyal high spenders
    [1500, 10], [1600, 12], [1400, 9],    # Occasional visitors
    [3000, 25], [3100, 27], [2900, 24],   # Moderate spenders
    [8000, 5],                             # Outlier: rare but huge spender
    [200, 2],                              # Outlier: very low engagement
])

print("\nCustomer Data:")
print("Format: [Spending per visit ($), Visits per month]")
print("-" * 70)
for i, customer in enumerate(data):
    print(f"Customer {i+1:2d}: ${customer[0]:5.0f} per visit, {customer[1]:2.0f} visits/month")

print(f"\nTotal customers: {len(data)}")

# Step 2: Standardize the data
# Why? Because spending (in thousands) and visits (in tens) have different scales
# Standardization makes them comparable
print("\n" + "=" * 70)
print("STEP 1: DATA STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print("\nOriginal data has different scales:")
print(f"  Spending range: ${data[:, 0].min():.0f} - ${data[:, 0].max():.0f}")
print(f"  Visits range: {data[:, 1].min():.0f} - {data[:, 1].max():.0f}")
print("\nAfter standardization, both features have:")
print(f"  Mean ≈ 0, Standard deviation ≈ 1")
print("✓ Data is now ready for DBSCAN")

# Step 3: Find optimal epsilon using k-distance graph
print("\n" + "=" * 70)
print("STEP 2: FINDING OPTIMAL EPSILON (ε)")
print("=" * 70)

# Calculate distances to k-th nearest neighbor
k = 2  # MinPts - 1
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(data_scaled)
distances, _ = neigh.kneighbors(data_scaled)

# Sort distances
distances = np.sort(distances[:, k-1])

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances, marker='o', linestyle='-', color='blue')
plt.xlabel('Data Points (sorted by distance)', fontsize=12)
plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph (Elbow Method for Epsilon)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='red', linestyle='--', label='Chosen ε = 0.5')
plt.legend()
plt.tight_layout()
plt.savefig('dbscan/k_distance_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nK-Distance Graph plotted!")
print("Look for the 'elbow' point where the curve bends sharply")
print("The y-value at the elbow is your optimal epsilon")
print("\n✓ Based on the graph, we choose ε = 0.5")

# Step 4: Apply DBSCAN clustering
print("\n" + "=" * 70)
print("STEP 3: APPLYING DBSCAN CLUSTERING")
print("=" * 70)

# DBSCAN parameters
epsilon = 0.5      # Maximum distance between neighbors
min_samples = 2    # Minimum points to form a cluster

print(f"\nDBSCAN Parameters:")
print(f"  Epsilon (ε): {epsilon}")
print(f"    → Maximum distance between two points to be neighbors")
print(f"  MinPts: {min_samples}")
print(f"    → Minimum points needed to form a dense region (cluster)")

# Fit DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels = dbscan.fit_predict(data_scaled)

# Add labels to original data
data_with_labels = np.column_stack((data, labels))

print("\n✓ DBSCAN clustering complete!")

# Step 5: Analyze results
print("\n" + "=" * 70)
print("STEP 4: CLUSTERING RESULTS")
print("=" * 70)

# Count clusters and noise
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(labels).count(-1)

print(f"\nClusters found: {n_clusters}")
print(f"Outliers detected: {n_noise}")

# Analyze each cluster
print("\n" + "-" * 70)
print("CUSTOMER SEGMENTS:")
print("-" * 70)

for label in unique_labels:
    if label == -1:
        segment_name = "OUTLIERS (Unusual Behavior)"
    else:
        segment_name = f"CLUSTER {label}"
    
    # Get customers in this segment
    mask = labels == label
    segment_customers = data[mask]
    
    print(f"\n{segment_name}:")
    print(f"  Number of customers: {len(segment_customers)}")
    
    if len(segment_customers) > 0:
        avg_spending = segment_customers[:, 0].mean()
        avg_visits = segment_customers[:, 1].mean()
        print(f"  Average spending per visit: ${avg_spending:.2f}")
        print(f"  Average visits per month: {avg_visits:.2f}")
        
        # Classify the segment
        if label != -1:
            if avg_spending > 4000 and avg_visits > 40:
                classification = "💎 LOYAL HIGH SPENDERS"
            elif avg_spending > 2500 and avg_visits > 20:
                classification = "⭐ MODERATE SPENDERS"
            elif avg_spending < 2000 and avg_visits < 15:
                classification = "👤 OCCASIONAL VISITORS"
            else:
                classification = "📊 MIXED SEGMENT"
            print(f"  Classification: {classification}")
        
        # Show customer IDs
        customer_ids = np.where(mask)[0] + 1
        print(f"  Customer IDs: {list(customer_ids)}")

# Step 6: Visualize clusters
print("\n" + "=" * 70)
print("STEP 5: VISUALIZATION")
print("=" * 70)

plt.figure(figsize=(14, 8))

# Define colors for clusters
colors = ['red', 'blue', 'green', 'orange', 'purple']

for label in unique_labels:
    if label == -1:
        # Outliers in black with 'X' marker
        color = 'black'
        marker = 'X'
        size = 200
        label_name = 'Outliers'
    else:
        # Clusters with different colors
        color = colors[label % len(colors)]
        marker = 'o'
        size = 150
        label_name = f'Cluster {label}'
    
    # Get points for this cluster
    mask = labels == label
    cluster_points = data[mask]
    
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
               c=color, marker=marker, s=size,
               alpha=0.7, edgecolors='black', linewidth=2,
               label=label_name)
    
    # Add customer numbers
    for i, (x, y) in enumerate(cluster_points):
        customer_id = np.where((data[:, 0] == x) & (data[:, 1] == y))[0][0] + 1
        plt.annotate(f'C{customer_id}', (x, y), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

plt.xlabel('Spending per Visit ($)', fontsize=12, fontweight='bold')
plt.ylabel('Visits per Month', fontsize=12, fontweight='bold')
plt.title('Customer Segmentation using DBSCAN\nRetail Store Analysis', 
         fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dbscan/customer_segments.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved: customer_segments.png")

# Step 7: Business insights and recommendations
print("\n" + "=" * 70)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 70)

print("""
📊 CUSTOMER SEGMENTS IDENTIFIED:

1. LOYAL HIGH SPENDERS (Cluster with high spending + high visits)
   💡 Strategy:
   - VIP treatment and exclusive offers
   - Loyalty rewards program
   - Personal shopping assistance
   - Early access to new products

2. MODERATE SPENDERS (Middle ground customers)
   💡 Strategy:
   - Upselling opportunities
   - Bundle deals to increase spending
   - Frequency incentives to increase visits
   - Targeted promotions

3. OCCASIONAL VISITORS (Low spending + low visits)
   💡 Strategy:
   - Re-engagement campaigns
   - Special discounts to increase frequency
   - Email marketing with personalized offers
   - Referral programs

4. OUTLIERS (Unusual behavior)
   💡 Strategy:
   - Investigate individually
   - Rare high spenders: Special attention, concierge service
   - Very low engagement: Win-back campaigns or let go
   - May indicate data errors or unique circumstances

🎯 KEY TAKEAWAYS:
✓ DBSCAN successfully identified natural customer groups
✓ No need to pre-specify number of segments
✓ Automatically detected unusual customers (outliers)
✓ Can now create targeted marketing strategies for each segment
✓ Better resource allocation based on customer value
""")

# Step 8: Detailed customer report
print("\n" + "=" * 70)
print("DETAILED CUSTOMER REPORT")
print("=" * 70)

print("\n{:<12} {:<20} {:<15} {:<10}".format(
    "Customer", "Spending/Visit", "Visits/Month", "Segment"))
print("-" * 70)

for i, (customer, label) in enumerate(zip(data, labels)):
    if label == -1:
        segment = "Outlier"
    else:
        segment = f"Cluster {label}"
    
    print("{:<12} ${:<19.2f} {:<15.0f} {:<10}".format(
        f"Customer {i+1}", customer[0], customer[1], segment))

print("\n" + "=" * 70)
print("✓ CUSTOMER SEGMENTATION ANALYSIS COMPLETE!")
print("=" * 70)

print("\n📈 NEXT STEPS:")
print("1. Validate segments with business team")
print("2. Create targeted marketing campaigns for each segment")
print("3. Monitor customer movement between segments over time")
print("4. Calculate ROI for each segment")
print("5. Adjust strategies based on segment performance")
