import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("="*70)
print("TELECOM CUSTOMER SEGMENTATION - K-MEANS CLUSTERING")
print("="*70)
print("\nScenario: Telecommunications Company Customer Segmentation 📱")
print("\nBusiness Context:")
print("A telecom company collected data on 500 customers including monthly bill,")
print("call duration, internet usage, and support calls. Goal is to group customers")
print("into segments for targeted marketing and improved customer service.\n")

print("Tasks:")
print("  • Use K-Means clustering to explore customer segments")
print("  • Apply Elbow Method to find optimal clusters")
print("  • Use Silhouette Score to validate cluster quality")

np.random.seed(42)
data = {
    'CustomerID': range(1, 501),
    'MonthlyBill': np.random.randint(20, 200, 500),
    'CallDuration': np.random.randint(50, 500, 500),
    'InternetUsage': np.random.randint(10, 300, 500),
    'SupportCalls': np.random.randint(0, 10, 500)
}
df = pd.DataFrame(data)

print("\n"+"="*70)
print("CUSTOMER DATASET (First 10 rows)")
print("="*70)
print(df.head(10).to_string(index=False))
print("\nTotal Customers: {}".format(len(df)))

print("\n"+"="*70)
print("DATA SUMMARY")
print("="*70)
print(df.describe().to_string())

print("\n"+"="*70)
print("FEATURE SELECTION & SCALING")
print("="*70)
print("Features: MonthlyBill, CallDuration, InternetUsage, SupportCalls")
X = df[['MonthlyBill','CallDuration','InternetUsage','SupportCalls']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling applied: All features normalized (mean≈0, std≈1)")

print("\n"+"="*70)
print("ELBOW METHOD - FINDING OPTIMAL CLUSTERS")
print("="*70)
inertias, silhouette_scores, K_range = [], [], range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print("K={:2d}: Inertia={:8.2f}, Silhouette={:.3f}".format(k, kmeans.inertia_, sil_score))

optimal_k = K_range[np.argmax(silhouette_scores)]
print("\n"+"="*70)
print("OPTIMAL CLUSTER SELECTION")
print("="*70)
print("Based on Silhouette Score: K = {}".format(optimal_k))
print("Silhouette Score: {:.3f}".format(max(silhouette_scores)))
print("\nElbow Method Analysis:")
print("  • Inertia decreases as K increases")
print("  • Look for 'elbow' where improvement slows down")
print("  • Silhouette score helps validate cluster quality")

print("\n"+"="*70)
print("APPLYING K-MEANS CLUSTERING")
print("="*70)
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("Clustering complete with K = {}".format(optimal_k))

print("\n"+"="*70)
print("CLUSTER ANALYSIS")
print("="*70)
for i in range(optimal_k):
    c = df[df['Cluster']==i]
    print("\nCluster {} ({} customers):".format(i, len(c)))
    print("  Avg Monthly Bill: ${:.2f}".format(c['MonthlyBill'].mean()))
    print("  Avg Call Duration: {:.0f} mins".format(c['CallDuration'].mean()))
    print("  Avg Internet Usage: {:.0f} GB".format(c['InternetUsage'].mean()))
    print("  Avg Support Calls: {:.1f}".format(c['SupportCalls'].mean()))

print("\n"+"="*70)
print("CLUSTER PROFILES")
print("="*70)
profiles = df.groupby('Cluster').agg({
    'MonthlyBill':'mean','CallDuration':'mean','InternetUsage':'mean',
    'SupportCalls':'mean','CustomerID':'count'
}).round(2)
profiles.columns = ['Avg_Bill','Avg_CallMin','Avg_InternetGB','Avg_Support','Count']
print(profiles.to_string())

print("\n"+"="*70)
print("CUSTOMER SEGMENT INSIGHTS")
print("="*70)
for i in range(optimal_k):
    c = df[df['Cluster']==i]
    bill, calls, internet, support = c['MonthlyBill'].mean(), c['CallDuration'].mean(), c['InternetUsage'].mean(), c['SupportCalls'].mean()
    
    print("\nCluster {} - ".format(i), end="")
    if bill > 150:
        print("'Premium Customers'")
        print("  Strategy: VIP support, exclusive plans, loyalty rewards")
    elif internet > 200:
        print("'Heavy Data Users'")
        print("  Strategy: Unlimited data plans, streaming bundles")
    elif calls > 350:
        print("'Talk-Heavy Users'")
        print("  Strategy: Unlimited calling plans, family packages")
    elif support > 6:
        print("'High-Maintenance Customers'")
        print("  Strategy: Proactive support, service improvement")
    else:
        print("'Standard Customers'")
        print("  Strategy: Balanced plans, retention campaigns")

print("\n"+"="*70)
print("MARKETING RECOMMENDATIONS")
print("="*70)
print("1. Targeted Campaigns:")
print("   • Customize offers based on cluster characteristics")
print("   • Different messaging for each segment")
print("\n2. Service Optimization:")
print("   • Allocate support resources based on cluster needs")
print("   • Proactive outreach for high-support clusters")
print("\n3. Product Development:")
print("   • Design plans matching cluster usage patterns")
print("   • Bundle services for specific segments")
print("\n4. Retention Strategy:")
print("   • Identify at-risk segments")
print("   • Personalized retention offers")

print("\nCreating visualization...")
fig, axes = plt.subplots(2,3,figsize=(18,12))
colors = ['red','blue','green','orange','purple','cyan','magenta','yellow','brown','pink']

axes[0,0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='blue')
axes[0,0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label='Optimal K={}'.format(optimal_k))
axes[0,0].set_xlabel('Number of Clusters (K)', fontsize=11, weight='bold')
axes[0,0].set_ylabel('Inertia', fontsize=11, weight='bold')
axes[0,0].set_title('Elbow Method', fontsize=13, weight='bold')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(K_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='green')
axes[0,1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label='Optimal K={}'.format(optimal_k))
axes[0,1].set_xlabel('Number of Clusters (K)', fontsize=11, weight='bold')
axes[0,1].set_ylabel('Silhouette Score', fontsize=11, weight='bold')
axes[0,1].set_title('Silhouette Score Analysis', fontsize=13, weight='bold')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

for i in range(optimal_k):
    c = df[df['Cluster']==i]
    axes[0,2].scatter(c['MonthlyBill'], c['InternetUsage'], s=50, c=colors[i], 
                     label='Cluster {}'.format(i), alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0,2].set_xlabel('Monthly Bill ($)', fontsize=11, weight='bold')
axes[0,2].set_ylabel('Internet Usage (GB)', fontsize=11, weight='bold')
axes[0,2].set_title('Bill vs Internet Usage', fontsize=13, weight='bold')
axes[0,2].legend()
axes[0,2].grid(alpha=0.3)

for i in range(optimal_k):
    c = df[df['Cluster']==i]
    axes[1,0].scatter(c['CallDuration'], c['SupportCalls'], s=50, c=colors[i],
                     label='Cluster {}'.format(i), alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1,0].set_xlabel('Call Duration (mins)', fontsize=11, weight='bold')
axes[1,0].set_ylabel('Support Calls', fontsize=11, weight='bold')
axes[1,0].set_title('Call Duration vs Support', fontsize=13, weight='bold')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

counts = df['Cluster'].value_counts().sort_index()
bars = axes[1,1].bar(counts.index, counts.values, color=[colors[i] for i in counts.index], 
                    edgecolor='black', linewidth=1.5)
axes[1,1].set_xlabel('Cluster', fontsize=11, weight='bold')
axes[1,1].set_ylabel('Number of Customers', fontsize=11, weight='bold')
axes[1,1].set_title('Cluster Distribution', fontsize=13, weight='bold')
axes[1,1].grid(alpha=0.3, axis='y')
for bar in bars:
    axes[1,1].text(bar.get_x()+bar.get_width()/2., bar.get_height(), 
                  '{}'.format(int(bar.get_height())), ha='center', va='bottom', fontsize=10, weight='bold')

cluster_means = df.groupby('Cluster')[['MonthlyBill','CallDuration','InternetUsage','SupportCalls']].mean()
x_pos = np.arange(len(cluster_means.columns))
width = 0.8/optimal_k
for i in range(optimal_k):
    axes[1,2].bar(x_pos + i*width, cluster_means.iloc[i], width, 
                 label='Cluster {}'.format(i), color=colors[i], edgecolor='black')
axes[1,2].set_xlabel('Features', fontsize=11, weight='bold')
axes[1,2].set_ylabel('Average Value (normalized)', fontsize=11, weight='bold')
axes[1,2].set_title('Cluster Feature Comparison', fontsize=13, weight='bold')
axes[1,2].set_xticks(x_pos + width*(optimal_k-1)/2)
axes[1,2].set_xticklabels(['Bill','CallMin','InternetGB','Support'], rotation=45)
axes[1,2].legend()
axes[1,2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('telecom_clustering/telecom_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Visualization saved\n")
print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
