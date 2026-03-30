"""
SCENARIO: Retail Bank Customer Segmentation

A retail bank wants to understand its customers better by analyzing Age and Annual Income.
The goal is to group customers into meaningful segments for targeted financial products.

FEATURES:
- Age: Customer age in years
- Annual Income: Yearly income in rupees

OBJECTIVES:
- Apply Hierarchical Clustering to discover natural customer groups
- Use Dendrogram to visualize cluster hierarchy
- Compare different linkage methods (Ward, Complete, Average, Single)
- Validate with Silhouette Score

EXPECTED SEGMENTS:
- Young Starters: Low income, younger age (entry-level products)
- Growing Professionals: Moderate income, mid-age (career growth products)
- Established Earners: Good income, mature age (wealth building)
- High-Income Segment: High income, senior age (premium services)

BUSINESS APPLICATIONS:
- Targeted loan offers for each segment
- Personalized investment plans
- Marketing campaigns tailored to customer needs
- Risk assessment and credit policies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RETAIL BANK CUSTOMER SEGMENTATION")
print("="*70)
print("\nObjective: Group customers by Age and Annual Income")
print("Purpose: Design targeted loan offers, investment plans, and campaigns\n")

# Customer data: [Age, Annual Income]
data = np.array([
    [25, 15000],
    [28, 16000],
    [30, 18000],
    [35, 22000],
    [40, 25000],
    [45, 60000],
    [50, 65000],
    [55, 70000]
])

# Create DataFrame
df = pd.DataFrame(data, columns=['Age', 'Annual_Income'])
df['Customer_ID'] = range(1, len(df) + 1)
df = df[['Customer_ID', 'Age', 'Annual_Income']]

print("Customer Data:")
print(df)
print("\nDataset Statistics:")
print(df.describe())

# Save original data
df.to_csv('bank_customer_hierarchical_clustering/customer_data.csv', index=False)

# Standardize features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Age', 'Annual_Income']])

# HIERARCHICAL CLUSTERING
print("\n" + "="*70)
print("📊 HIERARCHICAL CLUSTERING ANALYSIS")
print("="*70)

# Perform hierarchical clustering with different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
linkage_results = {}

for method in linkage_methods:
    Z = linkage(X_scaled, method=method)
    linkage_results[method] = Z
    print(f"\n{method.upper()} Linkage computed")

# Use Ward linkage for final clustering (generally best for customer segmentation)
Z_ward = linkage_results['ward']

# Determine optimal number of clusters (try 2, 3, 4 clusters)
print("\n" + "="*70)
print("🎯 CLUSTER EVALUATION")
print("="*70)

silhouette_scores = {}
for n_clusters in range(2, 5):
    clusters = fcluster(Z_ward, n_clusters, criterion='maxclust')
    score = silhouette_score(X_scaled, clusters)
    silhouette_scores[n_clusters] = score
    print(f"Clusters: {n_clusters} | Silhouette Score: {score:.4f}")

# Choose optimal clusters
optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
print(f"\n✅ Optimal number of clusters: {optimal_clusters}")

# Final clustering
final_clusters = fcluster(Z_ward, optimal_clusters, criterion='maxclust')
df['Cluster'] = final_clusters

# Analyze clusters
print("\n" + "="*70)
print("📈 CLUSTER ANALYSIS")
print("="*70)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster Characteristics:")
cluster_summary = df.groupby('Cluster')[['Age', 'Annual_Income']].agg(['mean', 'min', 'max'])
print(cluster_summary)

# Assign meaningful names to clusters
cluster_names = {}
for cluster_id in range(1, optimal_clusters + 1):
    cluster_data = df[df['Cluster'] == cluster_id]
    avg_age = cluster_data['Age'].mean()
    avg_income = cluster_data['Annual_Income'].mean()
    
    if avg_income < 20000:
        name = "Young Starters"
    elif avg_income < 30000:
        name = "Growing Professionals"
    elif avg_income < 50000:
        name = "Established Earners"
    else:
        name = "High-Income Segment"
    
    cluster_names[cluster_id] = name

df['Segment'] = df['Cluster'].map(cluster_names)

print("\n" + "="*70)
print("🎯 CUSTOMER SEGMENTS")
print("="*70)

for cluster_id, name in cluster_names.items():
    cluster_df = df[df['Cluster'] == cluster_id]
    print(f"\n{name} (Cluster {cluster_id}):")
    print(f"  • Customers: {len(cluster_df)}")
    print(f"  • Avg Age: {cluster_df['Age'].mean():.1f} years")
    print(f"  • Avg Income: ₹{cluster_df['Annual_Income'].mean():,.0f}")
    print(f"  • Age Range: {cluster_df['Age'].min()}-{cluster_df['Age'].max()} years")
    print(f"  • Income Range: ₹{cluster_df['Annual_Income'].min():,.0f} - ₹{cluster_df['Annual_Income'].max():,.0f}")

# Save clustered data
df.to_csv('bank_customer_hierarchical_clustering/clustered_customers.csv', index=False)
print("\n✅ Clustered data saved to 'clustered_customers.csv'")

# VISUALIZATION
fig = plt.figure(figsize=(18, 12))

# 1. Dendrogram - Ward Linkage
plt.subplot(2, 3, 1)
dendrogram(Z_ward, labels=df['Customer_ID'].values, leaf_font_size=10)
plt.title('Dendrogram (Ward Linkage)', fontsize=13, fontweight='bold')
plt.xlabel('Customer ID', fontsize=11, fontweight='bold')
plt.ylabel('Distance', fontsize=11, fontweight='bold')
plt.axhline(y=Z_ward[-optimal_clusters+1, 2], color='red', linestyle='--', 
            label=f'{optimal_clusters} Clusters')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 2. Dendrogram - Complete Linkage
plt.subplot(2, 3, 2)
dendrogram(linkage_results['complete'], labels=df['Customer_ID'].values, leaf_font_size=10)
plt.title('Dendrogram (Complete Linkage)', fontsize=13, fontweight='bold')
plt.xlabel('Customer ID', fontsize=11, fontweight='bold')
plt.ylabel('Distance', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 3. Silhouette Scores
plt.subplot(2, 3, 3)
clusters_list = list(silhouette_scores.keys())
scores_list = list(silhouette_scores.values())
plt.plot(clusters_list, scores_list, 'go-', linewidth=2, markersize=10)
plt.axvline(x=optimal_clusters, color='red', linestyle='--', 
            label=f'Optimal: {optimal_clusters} clusters')
plt.xlabel('Number of Clusters', fontsize=11, fontweight='bold')
plt.ylabel('Silhouette Score', fontsize=11, fontweight='bold')
plt.title('Cluster Quality Evaluation', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(clusters_list)

# 4. Customer Scatter Plot with Clusters
plt.subplot(2, 3, 4)
colors = plt.cm.Set3(range(optimal_clusters))
for i, (cluster_id, name) in enumerate(cluster_names.items()):
    cluster_data = df[df['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Age'], cluster_data['Annual_Income'], 
                c=[colors[i]], s=200, alpha=0.7, edgecolors='black', 
                linewidth=2, label=name)
    
    # Add customer IDs
    for _, row in cluster_data.iterrows():
        plt.annotate(f"C{row['Customer_ID']}", 
                    (row['Age'], row['Annual_Income']),
                    fontsize=8, ha='center', va='center', fontweight='bold')

plt.xlabel('Age (years)', fontsize=11, fontweight='bold')
plt.ylabel('Annual Income (₹)', fontsize=11, fontweight='bold')
plt.title('Customer Segmentation', fontsize=13, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)

# 5. Cluster Characteristics Bar Chart
plt.subplot(2, 3, 5)
cluster_means = df.groupby('Cluster')[['Age', 'Annual_Income']].mean()
x = np.arange(len(cluster_means))
width = 0.35

# Normalize for better visualization
age_normalized = cluster_means['Age'] / cluster_means['Age'].max() * 100
income_normalized = cluster_means['Annual_Income'] / cluster_means['Annual_Income'].max() * 100

plt.bar(x - width/2, age_normalized, width, label='Age (normalized)', color='#ff6b6b')
plt.bar(x + width/2, income_normalized, width, label='Income (normalized)', color='#4ecdc4')

plt.xlabel('Cluster', fontsize=11, fontweight='bold')
plt.ylabel('Normalized Value (%)', fontsize=11, fontweight='bold')
plt.title('Cluster Characteristics Comparison', fontsize=13, fontweight='bold')
plt.xticks(x, cluster_means.index)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 6. Cluster Size Distribution
plt.subplot(2, 3, 6)
cluster_counts = df['Cluster'].value_counts().sort_index()
colors_pie = plt.cm.Set3(range(len(cluster_counts)))
wedges, texts, autotexts = plt.pie(cluster_counts.values, 
                                     labels=[cluster_names[i] for i in cluster_counts.index],
                                     autopct='%1.1f%%', startangle=90, 
                                     colors=colors_pie, textprops={'fontsize': 10})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
plt.title('Customer Distribution by Segment', fontsize=13, fontweight='bold')

plt.suptitle('🏦 Retail Bank - Hierarchical Customer Segmentation Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('bank_customer_hierarchical_clustering/hierarchical_clustering_analysis.png', 
            dpi=300, bbox_inches='tight')
print("📊 Visualization saved to 'hierarchical_clustering_analysis.png'")

plt.show()

# BUSINESS RECOMMENDATIONS
print("\n" + "="*70)
print("💡 BUSINESS RECOMMENDATIONS")
print("="*70)

for cluster_id, name in cluster_names.items():
    cluster_df = df[df['Cluster'] == cluster_id]
    avg_age = cluster_df['Age'].mean()
    avg_income = cluster_df['Annual_Income'].mean()
    
    print(f"\n{name}:")
    print(f"  📊 Profile: {len(cluster_df)} customers, Avg Age {avg_age:.0f}, Avg Income ₹{avg_income:,.0f}")
    
    if avg_income < 20000:
        print(f"  🎯 Strategy: Student loans, credit cards with low limits, savings accounts")
        print(f"  📢 Campaign: 'Start Your Financial Journey' - Focus on building credit")
    elif avg_income < 30000:
        print(f"  🎯 Strategy: Personal loans, car loans, basic investment plans")
        print(f"  📢 Campaign: 'Grow With Us' - Career development financial products")
    elif avg_income < 50000:
        print(f"  🎯 Strategy: Home loans, mutual funds, insurance products")
        print(f"  📢 Campaign: 'Secure Your Future' - Long-term wealth building")
    else:
        print(f"  🎯 Strategy: Premium credit cards, wealth management, investment portfolios")
        print(f"  📢 Campaign: 'Elite Banking Experience' - Exclusive benefits & services")

print("\n" + "="*70)
print("✅ HIERARCHICAL CLUSTERING ANALYSIS COMPLETE!")
print("="*70)
