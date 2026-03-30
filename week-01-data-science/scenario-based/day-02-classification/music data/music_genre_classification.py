import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

genres = ['Rock', 'Pop', 'Classical', 'Jazz', 'Hip-Hop', 'Electronic']

num_samples = 100

data = {
    'clip_id': [f'audio_{i:03d}' for i in range(1, num_samples + 1)],
    'tempo': np.random.randint(60, 180, num_samples),
    'energy': np.random.uniform(0, 1, num_samples),
    'danceability': np.random.uniform(0, 1, num_samples),
    'acousticness': np.random.uniform(0, 1, num_samples),
    'valence': np.random.uniform(0, 1, num_samples),
    'duration_sec': np.random.randint(120, 300, num_samples),
    'genre': np.random.choice(genres, num_samples)
}

df = pd.DataFrame(data)

print("=" * 60)
print("MUSIC GENRE CLASSIFICATION - DATASET PREPARATION")
print("=" * 60)
print(f"\nTotal audio clips collected: {len(df)}")
print(f"\nGenre distribution:")
print(df['genre'].value_counts().sort_index())

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['genre'])

val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['genre'])

print("\n" + "=" * 60)
print("DATASET SPLIT SUMMARY")
print("=" * 60)
print(f"\nTraining Set: {len(train_df)} clips ({len(train_df)/len(df)*100:.1f}%)")
print(f"  → Used to teach the AI model patterns in music")
print(f"\nValidation Set: {len(val_df)} clips ({len(val_df)/len(df)*100:.1f}%)")
print(f"  → Used to tune model parameters and improve accuracy")
print(f"\nTest Set: {len(test_df)} clips ({len(test_df)/len(df)*100:.1f}%)")
print(f"  → Used to evaluate model on completely new songs")

# Display genre distribution in each set
print("\n" + "=" * 60)
print("GENRE DISTRIBUTION BY SET")
print("=" * 60)

print("\nTraining Set:")
print(train_df['genre'].value_counts().sort_index())

print("\nValidation Set:")
print(val_df['genre'].value_counts().sort_index())

print("\nTest Set:")
print(test_df['genre'].value_counts().sort_index())

train_df.to_csv('music data/train_set.csv', index=False)
val_df.to_csv('music data/validation_set.csv', index=False)
test_df.to_csv('music data/test_set.csv', index=False)
df.to_csv('music data/complete_dataset.csv', index=False)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print("✓ complete_dataset.csv - All 100 audio clips")
print("✓ train_set.csv - Training data (70%)")
print("✓ validation_set.csv - Validation data (15%)")
print("✓ test_set.csv - Test data (15%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sizes = [len(train_df), len(val_df), len(test_df)]
labels = [f'Training\n{len(train_df)} clips\n(70%)', 
          f'Validation\n{len(val_df)} clips\n(15%)', 
          f'Test\n{len(test_df)} clips\n(15%)']
colors = ['#4CAF50', '#FFC107', '#2196F3']
explode = (0.05, 0.05, 0.05)

axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
axes[0].set_title('Dataset Split Distribution', fontsize=14, weight='bold', pad=20)

genre_comparison = pd.DataFrame({
    'Training': train_df['genre'].value_counts().sort_index(),
    'Validation': val_df['genre'].value_counts().sort_index(),
    'Test': test_df['genre'].value_counts().sort_index()
})

genre_comparison.plot(kind='bar', ax=axes[1], color=['#4CAF50', '#FFC107', '#2196F3'])
axes[1].set_title('Genre Distribution Across Sets', fontsize=14, weight='bold', pad=20)
axes[1].set_xlabel('Genre', fontsize=11, weight='bold')
axes[1].set_ylabel('Number of Clips', fontsize=11, weight='bold')
axes[1].legend(title='Dataset', fontsize=10)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('music data/dataset_split_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ dataset_split_visualization.png - Visual representation of splits")

print("\n" + "=" * 60)
print("SAMPLE DATA FROM TRAINING SET")
print("=" * 60)
print(train_df.head(10))

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE!")
print("=" * 60)
print("\nNext Steps:")
print("1. Use train_set.csv to train your AI model")
print("2. Use validation_set.csv to tune hyperparameters")
print("3. Use test_set.csv for final model evaluation")
print("=" * 60)
