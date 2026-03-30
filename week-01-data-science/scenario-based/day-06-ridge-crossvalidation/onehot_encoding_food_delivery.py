"""
SCENARIO: Online Food Delivery App - One-Hot Encoding

An online food delivery company wants to build a machine learning model to predict delivery 
time based on the type of cuisine ordered.

FEATURES:
- Cuisine Type: Italian, Chinese, Indian, Mexican

PROBLEM:
Machine learning models can't directly work with text labels.

SOLUTION:
Use One-Hot Encoding to create a new column for each cuisine type.

ENCODING:
- Italian → [1, 0, 0, 0]
- Chinese → [0, 1, 0, 0]
- Indian → [0, 0, 1, 0]
- Mexican → [0, 0, 0, 1]

PARAMETERS:
- sparse=False: Returns regular NumPy array instead of sparse matrix for easier analysis

QUESTIONS:
Part A: Why is One-Hot Encoding more appropriate than Label Encoding for cuisine type?
Part B: What does sparse=False do, and why might it be useful in this scenario?
Part C: If the company adds a new cuisine type (e.g., "Thai"), how will One-Hot Encoding handle it?
Part D: What potential problem could arise if the company has hundreds of cuisine types, 
        and how might they solve it?
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

print("="*70)
print("ONE-HOT ENCODING - ONLINE FOOD DELIVERY APP")
print("="*70)

data = pd.DataFrame({
    'Order_ID': ['O001', 'O002', 'O003', 'O004', 'O005', 'O006', 'O007'],
    'Cuisine_Type': ['Italian', 'Chinese', 'Indian', 'Mexican', 'Italian', 'Chinese', 'Indian']
})

print("\nOriginal Data:")
print(data)

onehot_encoder = OneHotEncoder(sparse_output=False)

cuisine_encoded = onehot_encoder.fit_transform(data[['Cuisine_Type']])

print("\nOne-Hot Encoded Array:")
print(cuisine_encoded)

print("\nEncoding Mappings:")
print("\nCuisine Type → Binary Vector:")
for i, cuisine in enumerate(onehot_encoder.categories_[0]):
    vector = [0] * len(onehot_encoder.categories_[0])
    vector[i] = 1
    print(f"  {cuisine} → {vector}")

encoded_df = pd.DataFrame(
    cuisine_encoded,
    columns=onehot_encoder.get_feature_names_out(['Cuisine_Type'])
)

result_df = pd.concat([data, encoded_df], axis=1)

print("\nFull Dataset with One-Hot Encoding:")
print(result_df)

print("\nColumn Names:")
print(list(encoded_df.columns))

sample_vector = np.array([[0, 1, 0, 0]])
decoded_cuisine = onehot_encoder.inverse_transform(sample_vector)
print(f"\nDecoding Example:")
print(f"  Vector [0, 1, 0, 0] → {decoded_cuisine[0][0]}")

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why One-Hot Encoding is better than Label Encoding?")
print("  - Cuisine types are nominal categories (no natural order)")
print("  - Label Encoding: Chinese (0), Indian (1), Italian (2), Mexican (3)")
print("  - Problem: Model assumes Italian (2) > Indian (1) > Chinese (0) - incorrect!")
print("  - One-Hot Encoding: Each cuisine is independent binary feature")
print("  - No false ordinal relationship created")

print("\nPart B: What does sparse=False do?")
print("  - sparse=False returns regular NumPy array (dense format)")
print("  - sparse=True returns sparse matrix (memory-efficient for large datasets)")
print("  - Benefits of sparse=False:")
print("    1. Easier to read and print for analysis")
print("    2. Simpler to convert to DataFrame")
print("    3. Better for small datasets with few categories")
print("  - Use sparse=True when dealing with thousands of categories")

print("\nPart C: Adding new cuisine 'Thai':")
print("  - If encoder already fitted: Will raise ValueError for unknown category")
print("  - Solution 1: Retrain encoder with updated dataset including 'Thai'")
print("  - Solution 2: Use handle_unknown='ignore' parameter")
print("    - Unknown categories encoded as all zeros [0, 0, 0, 0]")
print("  - New encoding: Italian, Chinese, Indian, Mexican, Thai → 5 columns")

print("\nPart D: Problem with hundreds of cuisines:")
print("  - Issue: Curse of dimensionality")
print("    - 500 cuisines → 500 columns")
print("    - Sparse data (mostly zeros)")
print("    - Increased memory usage and training time")
print("  - Solutions:")
print("    1. Use sparse=True to save memory")
print("    2. Group rare cuisines into 'Other' category")
print("    3. Use feature hashing (FeatureHasher)")
print("    4. Use embedding layers (for deep learning)")
print("    5. Keep only top N most popular cuisines")

result_df.to_csv('Day_6_Mar01_Ridge_CrossValidation/food_delivery_encoded.csv', index=False)
print("\nData saved to 'food_delivery_encoded.csv'")

