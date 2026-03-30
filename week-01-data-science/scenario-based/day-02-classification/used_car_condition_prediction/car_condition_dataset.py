import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

num_samples = 100

mileage = np.random.randint(5000, 150000, num_samples)
engine_performance = np.random.uniform(3.0, 10.0, num_samples)
fuel_efficiency = np.random.uniform(8.0, 25.0, num_samples)
age_of_car = np.random.randint(1, 15, num_samples)

condition = []
for i in range(num_samples):
    score = 0
    if mileage[i] < 50000:
        score += 2
    elif mileage[i] < 100000:
        score += 1
    
    if engine_performance[i] > 7.0:
        score += 2
    elif engine_performance[i] > 5.0:
        score += 1
    
    if fuel_efficiency[i] > 18.0:
        score += 2
    elif fuel_efficiency[i] > 12.0:
        score += 1
    
    if age_of_car[i] < 5:
        score += 2
    elif age_of_car[i] < 10:
        score += 1
    
    condition.append(1 if score >= 5 else 0)

data = {
    'car_id': [f'CAR_{i:03d}' for i in range(1, num_samples + 1)],
    'mileage': mileage,
    'engine_performance_score': np.round(engine_performance, 2),
    'fuel_efficiency_kmpl': np.round(fuel_efficiency, 2),
    'age_years': age_of_car,
    'condition': condition
}

df = pd.DataFrame(data)

print("=" * 70)
print("USED CAR CONDITION PREDICTION - DATASET PREPARATION")
print("=" * 70)
print(f"\nTotal car inspection records: {len(df)}")
print(f"\nCondition Labels:")
print(f"  0 → Needs Repair: {len(df[df['condition'] == 0])} cars")
print(f"  1 → Good Condition: {len(df[df['condition'] == 1])} cars")

print("\n" + "=" * 70)
print("DATASET FEATURES")
print("=" * 70)
print("\n1. Mileage: Distance traveled by the car (km)")
print("2. Engine Performance Score: Rating from 3.0 to 10.0")
print("3. Fuel Efficiency: Kilometers per liter (kmpl)")
print("4. Age of Car: Years since manufacture")
print("5. Condition: 0 = Needs Repair, 1 = Good Condition")

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['condition'])

val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['condition'])

print("\n" + "=" * 70)
print("DATASET SPLIT SUMMARY")
print("=" * 70)
print(f"\nTraining Set: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
print(f"  → Teach the AI model")
print(f"  → Good Condition: {len(train_df[train_df['condition'] == 1])}")
print(f"  → Needs Repair: {len(train_df[train_df['condition'] == 0])}")

print(f"\nValidation Set: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
print(f"  → Tune and improve the model")
print(f"  → Good Condition: {len(val_df[val_df['condition'] == 1])}")
print(f"  → Needs Repair: {len(val_df[val_df['condition'] == 0])}")

print(f"\nTest Set: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
print(f"  → Evaluate final performance")
print(f"  → Good Condition: {len(test_df[test_df['condition'] == 1])}")
print(f"  → Needs Repair: {len(test_df[test_df['condition'] == 0])}")

train_df.to_csv('used_car_condition_prediction/train_set.csv', index=False)
val_df.to_csv('used_car_condition_prediction/validation_set.csv', index=False)
test_df.to_csv('used_car_condition_prediction/test_set.csv', index=False)
df.to_csv('used_car_condition_prediction/complete_dataset.csv', index=False)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)
print("✓ complete_dataset.csv - All 100 car inspection records")
print("✓ train_set.csv - Training data (70%)")
print("✓ validation_set.csv - Validation data (15%)")
print("✓ test_set.csv - Test data (15%)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sizes = [len(train_df), len(val_df), len(test_df)]
labels = [f'Training\n{len(train_df)} records\n(70%)', 
          f'Validation\n{len(val_df)} records\n(15%)', 
          f'Test\n{len(test_df)} records\n(15%)']
colors = ['#4CAF50', '#FFC107', '#2196F3']
explode = (0.05, 0.05, 0.05)

axes[0, 0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 9, 'weight': 'bold'})
axes[0, 0].set_title('Dataset Split Distribution', fontsize=12, weight='bold', pad=15)

condition_data = pd.DataFrame({
    'Training': train_df['condition'].value_counts().sort_index(),
    'Validation': val_df['condition'].value_counts().sort_index(),
    'Test': test_df['condition'].value_counts().sort_index()
})
condition_data.index = ['Needs Repair (0)', 'Good Condition (1)']
condition_data.plot(kind='bar', ax=axes[0, 1], color=['#4CAF50', '#FFC107', '#2196F3'])
axes[0, 1].set_title('Condition Distribution Across Sets', fontsize=12, weight='bold', pad=15)
axes[0, 1].set_xlabel('Condition', fontsize=10, weight='bold')
axes[0, 1].set_ylabel('Number of Cars', fontsize=10, weight='bold')
axes[0, 1].legend(title='Dataset', fontsize=9)
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

axes[1, 0].scatter(df[df['condition']==0]['mileage'], 
                   df[df['condition']==0]['engine_performance_score'],
                   c='red', label='Needs Repair', alpha=0.6, s=50)
axes[1, 0].scatter(df[df['condition']==1]['mileage'], 
                   df[df['condition']==1]['engine_performance_score'],
                   c='green', label='Good Condition', alpha=0.6, s=50)
axes[1, 0].set_xlabel('Mileage (km)', fontsize=10, weight='bold')
axes[1, 0].set_ylabel('Engine Performance Score', fontsize=10, weight='bold')
axes[1, 0].set_title('Mileage vs Engine Performance', fontsize=12, weight='bold', pad=15)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(df[df['condition']==0]['age_years'], 
                   df[df['condition']==0]['fuel_efficiency_kmpl'],
                   c='red', label='Needs Repair', alpha=0.6, s=50)
axes[1, 1].scatter(df[df['condition']==1]['age_years'], 
                   df[df['condition']==1]['fuel_efficiency_kmpl'],
                   c='green', label='Good Condition', alpha=0.6, s=50)
axes[1, 1].set_xlabel('Age (years)', fontsize=10, weight='bold')
axes[1, 1].set_ylabel('Fuel Efficiency (kmpl)', fontsize=10, weight='bold')
axes[1, 1].set_title('Age vs Fuel Efficiency', fontsize=12, weight='bold', pad=15)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('used_car_condition_prediction/dataset_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ dataset_visualization.png - Visual analysis of the dataset")

print("\n" + "=" * 70)
print("SAMPLE DATA FROM TRAINING SET")
print("=" * 70)
print(train_df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print(df.describe())

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE!")
print("=" * 70)
print("\nNext Steps:")
print("1. Use train_set.csv to train your classification model")
print("2. Use validation_set.csv to tune hyperparameters")
print("3. Use test_set.csv for final model evaluation")
print("4. This is a BINARY CLASSIFICATION problem (0 or 1)")
print("=" * 70)
