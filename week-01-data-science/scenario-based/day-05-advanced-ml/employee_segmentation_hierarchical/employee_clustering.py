import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 💻 Scenario: Employee Segmentation in a Tech Company
print("="*70)
print("💻 TECH COMPANY EMPLOYEE SEGMENTATION")
print("="*70)
print("\nBusiness Problem: Understand employees to design training programs")
print("                  and salary structures")
print("\nExpected Segments:")
print("  • Young, entry-level employees")
print("  • Mid-career professionals")
print("  • Senior, high-earning employees\n")

# Employee data: [Age, Annual Salary]
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
df = pd.DataFrame(data, columns=['Age', 'Annual_Salary'])
df['Employee_ID'] = ['EMP' + str(i).zfill(3) for i in range(1, len(df) + 1)]
df = df[['Employee_ID', 'Age', 'Annual_Salary']]

print("Employee Data:")
print(df)
print("\nDataset Statistics:")
print(df[['Age', 'Annual_Salary']].describe())

# Save original data
df.to_csv('employee_segmentation_hierarchical/employee_data.csv', index=False)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Age', 'Annual_Salary']])

# HIERARCHICAL CLUSTERING
print("\n" + "="*70)
print("📊 HIERARCHICAL CLUSTERING ANALYSIS")
print("="*70)

# Perform hierarchical clustering with different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
linkage_results = {}

print("\nComputing linkage methods:")
for method in linkage_methods:
    Z = linkage(X_scaled, method=method)
    linkage_results[method] = Z
    print(f"  ✓ {method.upper()} linkage")

# Use Ward linkage for final clustering
Z_ward = linkage_results['ward']

# Determine optimal number of clusters
print("\n" + "="*70)
print("🎯 CLUSTER EVALUATION")
print("="*70)

silhouette_scores = {}
for n_clusters in range(2, 6):
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
cluster_summary = df.groupby('Cluster')[['Age', 'Annual_Salary']].agg(['mean', 'min', 'max', 'count'])
print(cluster_summary)

# Assign meaningful names to clusters
cluster_names = {}
for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    avg_age = cluster_data['Age'].mean()
    avg_salary = cluster_data['Annual_Salary'].mean()
    
    if avg_salary < 20000:
        name = "Entry-Level Employees"
    elif avg_salary < 30000:
        name = "Junior Professionals"
    elif avg_salary < 50000:
        name = "Mid-Career Professionals"
    else:
        name = "Senior High-Earners"
    
    cluster_names[cluster_id] = name

df['Segment'] = df['Cluster'].map(cluster_names)

print("\n" + "="*70)
print("👥 EMPLOYEE SEGMENTS")
print("="*70)

for cluster_id in sorted(cluster_names.keys()):
    name = cluster_names[cluster_id]
    cluster_df = df[df['Cluster'] == cluster_id]
    print(f"\n{name} (Cluster {cluster_id}):")
    print(f"  • Employees: {len(cluster_df)}")
    print(f"  • Avg Age: {cluster_df['Age'].mean():.1f} years")
    print(f"  • Avg Salary: ${cluster_df['Annual_Salary'].mean():,.0f}")
    print(f"  • Age Range: {cluster_df['Age'].min()}-{cluster_df['Age'].max()} years")
    print(f"  • Salary Range: ${cluster_df['Annual_Salary'].min():,.0f} - ${cluster_df['Annual_Salary'].max():,.0f}")
    print(f"  • Employee IDs: {', '.join(cluster_df['Employee_ID'].tolist())}")

# Save clustered data
df.to_csv('employee_segmentation_hierarchical/clustered_employees.csv', index=False)
print("\n✅ Clustered data saved to 'clustered_employees.csv'")

# VISUALIZATION
fig = plt.figure(figsize=(18, 12))

# 1. Dendrogram - Ward Linkage (Horizontal)
plt.subplot(2, 3, 1)
dendrogram(Z_ward, labels=df['Employee_ID'].values, leaf_font_size=9, orientation='right')
plt.title('Dendrogram - Ward Linkage', fontsize=13, fontweight='bold')
plt.xlabel('Distance', fontsize=11, fontweight='bold')
plt.ylabel('Employee ID', fontsize=11, fontweight='bold')
plt.axvline(x=Z_ward[-optimal_clusters+1, 2], color='red', linestyle='--', 
            label=f'{optimal_clusters} Clusters', linewidth=2)
plt.legend()
plt.grid(True, alpha=0.3, axis='x')

# 2. Dendrogram - Complete Linkage
plt.subplot(2, 3, 2)
dendrogram(linkage_results['complete'], labels=df['Employee_ID'].values, leaf_font_size=9)
plt.title('Dendrogram - Complete Linkage', fontsize=13, fontweight='bold')
plt.xlabel('Employee ID', fontsize=11, fontweight='bold')
plt.ylabel('Distance', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 3. Silhouette Scores
plt.subplot(2, 3, 3)
clusters_list = list(silhouette_scores.keys())
scores_list = list(silhouette_scores.values())
bars = plt.bar(clusters_list, scores_list, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'], 
               edgecolor='black', linewidth=2)
plt.axhline(y=max(scores_list), color='red', linestyle='--', alpha=0.5, 
            label=f'Best: {optimal_clusters} clusters')
plt.xlabel('Number of Clusters', fontsize=11, fontweight='bold')
plt.ylabel('Silhouette Score', fontsize=11, fontweight='bold')
plt.title('Cluster Quality Evaluation', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(clusters_list)
for bar, score in zip(bars, scores_list):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Employee Scatter Plot with Clusters
plt.subplot(2, 3, 4)
colors = plt.cm.Set3(range(optimal_clusters))
for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
    name = cluster_names[cluster_id]
    cluster_data = df[df['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Age'], cluster_data['Annual_Salary'], 
                c=[colors[i]], s=250, alpha=0.7, edgecolors='black', 
                linewidth=2, label=name, marker='o')
    
    # Add employee IDs
    for _, row in cluster_data.iterrows():
        plt.annotate(row['Employee_ID'], 
                    (row['Age'], row['Annual_Salary']),
                    fontsize=7, ha='center', va='center', fontweight='bold')

plt.xlabel('Age (years)', fontsize=11, fontweight='bold')
plt.ylabel('Annual Salary ($)', fontsize=11, fontweight='bold')
plt.title('Employee Segmentation by Age & Salary', fontsize=13, fontweight='bold')
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)

# 5. Cluster Characteristics Comparison
plt.subplot(2, 3, 5)
cluster_means = df.groupby('Cluster')[['Age', 'Annual_Salary']].mean().sort_index()
x = np.arange(len(cluster_means))
width = 0.35

# Normalize for visualization
age_normalized = cluster_means['Age'] / cluster_means['Age'].max() * 100
salary_normalized = cluster_means['Annual_Salary'] / cluster_means['Annual_Salary'].max() * 100

bars1 = plt.bar(x - width/2, age_normalized, width, label='Age (normalized)', 
                color='#ff6b6b', edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x + width/2, salary_normalized, width, label='Salary (normalized)', 
                color='#4ecdc4', edgecolor='black', linewidth=1.5)

plt.xlabel('Cluster', fontsize=11, fontweight='bold')
plt.ylabel('Normalized Value (%)', fontsize=11, fontweight='bold')
plt.title('Cluster Characteristics Comparison', fontsize=13, fontweight='bold')
plt.xticks(x, cluster_means.index)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 6. Salary Distribution by Segment
plt.subplot(2, 3, 6)
segment_order = [cluster_names[i] for i in sorted(cluster_names.keys())]
colors_box = [colors[i] for i in range(len(segment_order))]

positions = []
data_to_plot = []
for i, segment in enumerate(segment_order):
    segment_data = df[df['Segment'] == segment]['Annual_Salary'].values
    data_to_plot.append(segment_data)
    positions.append(i)

bp = plt.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                 labels=segment_order, showmeans=True)

for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel('Annual Salary ($)', fontsize=11, fontweight='bold')
plt.title('Salary Distribution by Segment', fontsize=13, fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('💻 Tech Company - Employee Segmentation Analysis (Hierarchical Clustering)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('employee_segmentation_hierarchical/employee_clustering_analysis.png', 
            dpi=300, bbox_inches='tight')
print("📊 Visualization saved to 'employee_clustering_analysis.png'")

plt.show()

# BUSINESS RECOMMENDATIONS
print("\n" + "="*70)
print("💡 HR STRATEGY & RECOMMENDATIONS")
print("="*70)

for cluster_id in sorted(cluster_names.keys()):
    name = cluster_names[cluster_id]
    cluster_df = df[df['Cluster'] == cluster_id]
    avg_age = cluster_df['Age'].mean()
    avg_salary = cluster_df['Annual_Salary'].mean()
    
    print(f"\n{name}:")
    print(f"  📊 Profile: {len(cluster_df)} employees, Avg Age {avg_age:.0f}, Avg Salary ${avg_salary:,.0f}")
    
    if avg_salary < 20000:
        print(f"  🎓 Training: Onboarding programs, technical skill development, mentorship")
        print(f"  💰 Compensation: Entry-level packages, performance bonuses, learning stipends")
        print(f"  📈 Growth: Clear career path, skill certifications, junior to mid-level transition")
    elif avg_salary < 30000:
        print(f"  🎓 Training: Advanced technical courses, leadership basics, project management")
        print(f"  💰 Compensation: Competitive mid-range salary, annual increments, health benefits")
        print(f"  📈 Growth: Team lead opportunities, cross-functional projects, specialization")
    elif avg_salary < 50000:
        print(f"  🎓 Training: Leadership development, strategic thinking, team management")
        print(f"  💰 Compensation: Above-market salary, stock options, comprehensive benefits")
        print(f"  📈 Growth: Senior roles, department head positions, strategic initiatives")
    else:
        print(f"  🎓 Training: Executive coaching, industry conferences, innovation workshops")
        print(f"  💰 Compensation: Premium packages, equity, executive benefits, retention bonuses")
        print(f"  📈 Growth: C-level track, board positions, thought leadership, succession planning")

print("\n" + "="*70)
print("🎯 KEY HR INSIGHTS")
print("="*70)
print("\n1. Clear Salary Bands: Distinct compensation tiers identified")
print("2. Career Progression: Natural progression from entry to senior levels")
print("3. Training Needs: Different development programs for each segment")
print("4. Retention Strategy: Targeted retention plans for high-earners")
print("5. Recruitment: Benchmark salaries for hiring at each level")

print("\n" + "="*70)
print("✅ EMPLOYEE SEGMENTATION ANALYSIS COMPLETE!")
print("="*70)
