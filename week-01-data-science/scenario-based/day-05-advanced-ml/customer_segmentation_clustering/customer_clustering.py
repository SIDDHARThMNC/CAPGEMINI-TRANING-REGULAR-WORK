import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("=" * 70)
print("CUSTOMER SEGMENTATION - K-MEANS CLUSTERING")
print("=" * 70)

print("\nBusiness Context: Customer Segmentation for a Retail Company 🛍️")
print("\nScenario:")
print("A retail chain wants to understand its customers better.")
print("Instead of treating everyone the same, they want to group customers")
print("into segments (like 'budget shoppers,' 'loyal premium buyers,' etc.)")
print("\nBusiness Goals:")
print("  • Personalize marketing campaigns")
print("  • Recommend products more effectively")
print("  • Improve customer retention")

data = {
    'CustomerID': [1,2,3,4,5,6],
    'Age': [25,45,35,23,52,40],
    'AnnualIncome': [25000,60000,40000,20000,80000,50000],
    'SpendingScore': [30,70,50,20,90,60]
}

df = pd.DataFrame(data)

print("\n" + "=" * 70)
print("CUSTOMER DATASET")
print("=" * 70)
print(df.to_string(index=False))

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Total Customers: {}".format(len(df)))
print("\nFeature Statistics:")
print(df.describe().to_string())

print("\n" + "=" * 70)
print("FEATURE SELECTION FOR CLUSTERING")
print("=" * 70)
print("Selected Features:")
print("  • Annual Income - Customer's purchasing power")
print("  • Spending Score - Customer's purchase behavior")
print("  • Age - Customer's demographic")

X = df[['AnnualIncome', 'SpendingScore', 'Age']].values

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)
print("Why Scaling?")
print("  • Features have different ranges (Income: 20k-80k, Score: 20-90)")
print("  • K-Means uses distance calculations")
print("  • Scaling ensures all features contribute equally")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nBefore Scaling:")
print("  Income range: {} - {}".format(df['AnnualIncome'].min(), df['AnnualIncome'].max()))
print("  Spending Score range: {} - {}".format(df['SpendingScore'].min(), df['SpendingScore'].max()))
print("  Age range: {} - {}".format(df['Age'].min(), df['Age'].max()))

print("\nAfter Scaling:")
print("  All features have mean ≈ 0 and std ≈ 1")

print("\n" + "=" * 70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)
print("Using Elbow Method and Silhouette Score...")

inertias = []
silhouette_scores = []
K_range = range(2, 6)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print("\nK = {}: Inertia = {:.2f}, Silhouette Score = {:.3f}".format(
        k, kmeans.inertia_, silhouette_scores[-1]))

optimal_k = K_range[np.argmax(silhouette_scores)]
print("\nOptimal K (based on Silhouette Score): {}".format(optimal_k))

print("\n" + "=" * 70)
print("TRAINING K-MEANS MODEL")
print("=" * 70)
print("Number of Clusters: {}".format(optimal_k))

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print("\n" + "=" * 70)
print("CUSTOMER SEGMENTS")
print("=" * 70)
print(df.to_string(index=False))

print("\n" + "=" * 70)
print("CLUSTER ANALYSIS")
print("=" * 70)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    print("\nCluster {} ({} customers):".format(cluster_id, len(cluster_data)))
    print("  Average Age: {:.1f} years".format(cluster_data['Age'].mean()))
    print("  Average Income: ${:,.0f}".format(cluster_data['AnnualIncome'].mean()))
    print("  Average Spending Score: {:.1f}".format(cluster_data['SpendingScore'].mean()))
    print("  Customer IDs: {}".format(list(cluster_data['CustomerID'].values)))

print("\n" + "=" * 70)
print("CLUSTER PROFILES")
print("=" * 70)

cluster_profiles = df.groupby('Cluster').agg({
    'Age': 'mean',
    'AnnualIncome': 'mean',
    'SpendingScore': 'mean',
    'CustomerID': 'count'
}).round(2)
cluster_profiles.columns = ['Avg_Age', 'Avg_Income', 'Avg_Spending', 'Count']
print(cluster_profiles.to_string())

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    avg_income = cluster_data['AnnualIncome'].mean()
    avg_spending = cluster_data['SpendingScore'].mean()
    avg_age = cluster_data['Age'].mean()
    
    print("\nCluster {} - ".format(cluster_id), end="")
    
    if avg_income > 60000 and avg_spending > 70:
        print("'Premium Loyal Customers'")
        print("  Strategy: VIP treatment, exclusive offers, loyalty rewards")
    elif avg_income > 60000 and avg_spending < 40:
        print("'High Income, Low Spenders'")
        print("  Strategy: Targeted promotions, personalized recommendations")
    elif avg_income < 30000 and avg_spending > 70:
        print("'Budget Enthusiasts'")
        print("  Strategy: Discount campaigns, value bundles")
    elif avg_income < 30000 and avg_spending < 40:
        print("'Price-Sensitive Shoppers'")
        print("  Strategy: Budget-friendly options, clearance sales")
    else:
        print("'Average Customers'")
        print("  Strategy: Balanced approach, seasonal promotions")

print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(K_range, inertias, marker='o', linewidth=2, markersize=10, color='blue')
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=11, weight='bold')
axes[0, 0].set_ylabel('Inertia', fontsize=11, weight='bold')
axes[0, 0].set_title('Elbow Method', fontsize=13, weight='bold')
axes[0, 0].set_xticks(K_range)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                   label='Optimal K={}'.format(optimal_k))
axes[0, 0].legend(fontsize=10)

colors = ['red', 'blue', 'green', 'orange', 'purple']
for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    axes[0, 1].scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'],
                      s=150, c=colors[cluster_id], label='Cluster {}'.format(cluster_id),
                      alpha=0.7, edgecolors='black', linewidth=1.5)

centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
axes[0, 1].scatter(centers_original[:, 0], centers_original[:, 1],
                  s=300, c='yellow', marker='*', edgecolors='black',
                  linewidth=2, label='Centroids')
axes[0, 1].set_xlabel('Annual Income ($)', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('Spending Score', fontsize=11, weight='bold')
axes[0, 1].set_title('Customer Segments (Income vs Spending)', fontsize=13, weight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    axes[1, 0].scatter(cluster_data['Age'], cluster_data['SpendingScore'],
                      s=150, c=colors[cluster_id], label='Cluster {}'.format(cluster_id),
                      alpha=0.7, edgecolors='black', linewidth=1.5)

axes[1, 0].set_xlabel('Age (years)', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Spending Score', fontsize=11, weight='bold')
axes[1, 0].set_title('Customer Segments (Age vs Spending)', fontsize=13, weight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

cluster_counts = df['Cluster'].value_counts().sort_index()
bars = axes[1, 1].bar(cluster_counts.index, cluster_counts.values,
                     color=[colors[i] for i in cluster_counts.index],
                     edgecolor='black', linewidth=1.5)
axes[1, 1].set_xlabel('Cluster', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Number of Customers', fontsize=11, weight='bold')
axes[1, 1].set_title('Cluster Distribution', fontsize=13, weight='bold')
axes[1, 1].set_xticks(cluster_counts.index)
axes[1, 1].grid(alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   '{}'.format(int(height)), ha='center', va='bottom',
                   fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('customer_segmentation_clustering/customer_clustering_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'customer_clustering_analysis.png'")

print("\n" + "=" * 70)
print("HOW K-MEANS CLUSTERING WORKS")
print("=" * 70)
print("1. Initialize K random centroids")
print("2. Assign each customer to nearest centroid")
print("3. Recalculate centroids based on assigned customers")
print("4. Repeat steps 2-3 until convergence")
print("5. Result: Customers grouped by similarity")

print("\n" + "=" * 70)
print("MARKETING RECOMMENDATIONS")
print("=" * 70)
print("1. Personalized Email Campaigns:")
print("   • Send targeted offers based on cluster characteristics")
print("   • Different messaging for each segment")

print("\n2. Product Recommendations:")
print("   • Premium products for high-income, high-spending clusters")
print("   • Budget options for price-sensitive segments")

print("\n3. Loyalty Programs:")
print("   • Reward high-spending customers with exclusive benefits")
print("   • Incentivize low-spenders with special promotions")

print("\n4. Customer Retention:")
print("   • Identify at-risk segments")
print("   • Proactive engagement strategies")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
