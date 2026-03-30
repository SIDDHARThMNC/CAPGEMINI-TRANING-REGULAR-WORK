import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("="*70)
print("HOSPITAL PATIENT SEGMENTATION - K-MEANS CLUSTERING")
print("="*70)
print("\nBusiness Context: Hospital Patient Segmentation 🏥")
print("Goals: Personalized treatment, predict high-risk patients, optimize resources\n")

data = {'PatientID':[101,102,103,104,105,106],'Age':[25,60,45,30,70,50],
        'BMI':[22,30,28,24,35,27],'HospitalVisits':[1,5,3,2,7,4],'ChronicConditions':[0,2,1,0,3,1]}
df = pd.DataFrame(data)

print("="*70)
print("PATIENT DATASET")
print("="*70)
print(df.to_string(index=False))
print("\nTotal Patients: {}".format(len(df)))
print("\n"+df.describe().to_string())

print("\n"+"="*70)
print("FEATURE SELECTION & SCALING")
print("="*70)
print("Features: Age, BMI, Hospital Visits, Chronic Conditions")
X = df[['Age','BMI','HospitalVisits','ChronicConditions']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling applied: All features normalized (mean≈0, std≈1)")

print("\n"+"="*70)
print("FINDING OPTIMAL CLUSTERS")
print("="*70)
inertias, silhouette_scores, K_range = [], [], range(2,5)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print("K={}: Inertia={:.2f}, Silhouette={:.3f}".format(k, kmeans.inertia_, silhouette_scores[-1]))

optimal_k = K_range[np.argmax(silhouette_scores)]
print("\nOptimal K: {}".format(optimal_k))

print("\n"+"="*70)
print("APPLYING K-MEANS & INSPECTING RESULTS")
print("="*70)
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print(df.to_string(index=False))

print("\n"+"="*70)
print("CLUSTER ANALYSIS")
print("="*70)
for i in range(optimal_k):
    c = df[df['Cluster']==i]
    print("\nCluster {} ({} patients): Age={:.1f}, BMI={:.1f}, Visits={:.1f}, Conditions={:.1f}".format(
        i, len(c), c['Age'].mean(), c['BMI'].mean(), c['HospitalVisits'].mean(), c['ChronicConditions'].mean()))
    print("  IDs: {}".format(list(c['PatientID'].values)))

print("\n"+df.groupby('Cluster').agg({'Age':'mean','BMI':'mean','HospitalVisits':'mean',
      'ChronicConditions':'mean','PatientID':'count'}).round(2).to_string())

print("\n"+"="*70)
print("HEALTH RISK ASSESSMENT & RECOMMENDATIONS")
print("="*70)
for i in range(optimal_k):
    c = df[df['Cluster']==i]
    age, bmi, visits, cond = c['Age'].mean(), c['BMI'].mean(), c['HospitalVisits'].mean(), c['ChronicConditions'].mean()
    print("\nCluster {}: ".format(i), end="")
    if age>60 and cond>=2:
        print("High-Risk Elderly | Action: Intensive monitoring")
    elif visits>=5 or cond>=2:
        print("Frequent Care | Action: Care coordination")
    elif bmi>=30:
        print("At-Risk | Action: Lifestyle intervention")
    else:
        print("Healthy/Low-Risk | Action: Preventive care")
K_range = range(2, 5)

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
print("APPLYING K-MEANS CLUSTERING")
print("=" * 70)
print("Number of Clusters: {}".format(optimal_k))

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print("\n" + "=" * 70)
print("PATIENT SEGMENTS")
print("=" * 70)
print(df.to_string(index=False))

print("\n" + "=" * 70)
print("INSPECTING RESULTS - CLUSTER ANALYSIS")
print("=" * 70)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    print("\nCluster {} ({} patients):".format(cluster_id, len(cluster_data)))
    print("  Average Age: {:.1f} years".format(cluster_data['Age'].mean()))
    print("  Average BMI: {:.1f}".format(cluster_data['BMI'].mean()))
    print("  Average Hospital Visits: {:.1f} per year".format(cluster_data['HospitalVisits'].mean()))
    print("  Average Chronic Conditions: {:.1f}".format(cluster_data['ChronicConditions'].mean()))
    print("  Patient IDs: {}".format(list(cluster_data['PatientID'].values)))

print("\n" + "=" * 70)
print("CLUSTER PROFILES")
print("=" * 70)

cluster_profiles = df.groupby('Cluster').agg({
    'Age': 'mean',
    'BMI': 'mean',
    'HospitalVisits': 'mean',
    'ChronicConditions': 'mean',
    'PatientID': 'count'
}).round(2)
cluster_profiles.columns = ['Avg_Age', 'Avg_BMI', 'Avg_Visits', 'Avg_Conditions', 'Count']
print(cluster_profiles.to_string())

print("\n" + "=" * 70)
print("HEALTH RISK ASSESSMENT")
print("=" * 70)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    avg_age = cluster_data['Age'].mean()
    avg_bmi = cluster_data['BMI'].mean()
    avg_visits = cluster_data['HospitalVisits'].mean()
    avg_conditions = cluster_data['ChronicConditions'].mean()
    
    print("\nCluster {} - ".format(cluster_id), end="")
    
    if avg_age > 60 and avg_conditions >= 2:
        print("'High-Risk Elderly Patients'")
        print("  Risk Level: HIGH")
        print("  Characteristics: Older age, multiple chronic conditions")
        print("  Action: Intensive monitoring, preventive care programs")
    elif avg_visits >= 5 or avg_conditions >= 2:
        print("'Frequent Care Patients'")
        print("  Risk Level: MEDIUM-HIGH")
        print("  Characteristics: Frequent hospital visits, chronic conditions")
        print("  Action: Care coordination, chronic disease management")
    elif avg_bmi >= 30:
        print("'At-Risk Patients'")
        print("  Risk Level: MEDIUM")
        print("  Characteristics: High BMI, potential health concerns")
        print("  Action: Lifestyle intervention programs, regular checkups")
    else:
        print("'Healthy/Low-Risk Patients'")
        print("  Risk Level: LOW")
        print("  Characteristics: Young, healthy BMI, minimal visits")
        print("  Action: Preventive care, wellness programs")

print("\n" + "=" * 70)
print("RESOURCE ALLOCATION RECOMMENDATIONS")
print("=" * 70)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    avg_visits = cluster_data['HospitalVisits'].mean()
    avg_conditions = cluster_data['ChronicConditions'].mean()
    
    print("\nCluster {} ({} patients):".format(cluster_id, len(cluster_data)))
    
    if avg_visits >= 5:
        print("  • Allocate dedicated care coordinators")
        print("  • Schedule regular follow-up appointments")
        print("  • Provide home healthcare services")
    
    if avg_conditions >= 2:
        print("  • Assign specialist teams")
        print("  • Implement chronic disease management programs")
        print("  • Provide medication management support")
    
    if avg_visits < 3 and avg_conditions == 0:
        print("  • Focus on preventive care")
        print("  • Offer wellness and fitness programs")
        print("  • Provide health education resources")

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
    axes[0, 1].scatter(cluster_data['Age'], cluster_data['BMI'],
                      s=cluster_data['HospitalVisits']*30, c=colors[cluster_id], 
                      label='Cluster {}'.format(cluster_id),
                      alpha=0.7, edgecolors='black', linewidth=1.5)

axes[0, 1].set_xlabel('Age (years)', fontsize=11, weight='bold')
axes[0, 1].set_ylabel('BMI', fontsize=11, weight='bold')
axes[0, 1].set_title('Patient Segments (Age vs BMI)\nBubble size = Hospital Visits', 
                     fontsize=13, weight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    axes[1, 0].scatter(cluster_data['HospitalVisits'], cluster_data['ChronicConditions'],
                      s=150, c=colors[cluster_id], label='Cluster {}'.format(cluster_id),
                      alpha=0.7, edgecolors='black', linewidth=1.5)

axes[1, 0].set_xlabel('Hospital Visits (per year)', fontsize=11, weight='bold')
axes[1, 0].set_ylabel('Chronic Conditions', fontsize=11, weight='bold')
axes[1, 0].set_title('Healthcare Utilization Pattern', fontsize=13, weight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

cluster_counts = df['Cluster'].value_counts().sort_index()
bars = axes[1, 1].bar(cluster_counts.index, cluster_counts.values,
                     color=[colors[i] for i in cluster_counts.index],
                     edgecolor='black', linewidth=1.5)
axes[1, 1].set_xlabel('Cluster', fontsize=11, weight='bold')
axes[1, 1].set_ylabel('Number of Patients', fontsize=11, weight='bold')
axes[1, 1].set_title('Cluster Distribution', fontsize=13, weight='bold')
axes[1, 1].set_xticks(cluster_counts.index)
axes[1, 1].grid(alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   '{}'.format(int(height)), ha='center', va='bottom',
                   fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('hospital_patient_segmentation/patient_clustering_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved as 'patient_clustering_analysis.png'")

print("\n" + "=" * 70)
print("CLINICAL INSIGHTS")
print("=" * 70)
print("1. Patient segmentation enables targeted interventions")
print("2. High-risk patients can be identified proactively")
print("3. Resource allocation can be optimized based on needs")
print("4. Personalized care plans improve patient outcomes")

print("\n" + "=" * 70)
print("IMPLEMENTATION RECOMMENDATIONS")
print("=" * 70)
print("1. Care Coordination:")
print("   • Assign care teams based on cluster characteristics")
print("   • Develop cluster-specific care protocols")

print("\n2. Preventive Programs:")
print("   • Weight management for high BMI clusters")
print("   • Chronic disease prevention for at-risk groups")

print("\n3. Resource Planning:")
print("   • Allocate staff based on cluster size and needs")
print("   • Schedule appointments considering visit patterns")

print("\n4. Monitoring:")
print("   • Track cluster migration over time")
print("   • Measure intervention effectiveness per cluster")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
