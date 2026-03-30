"""
SCENARIO: Package Delivery System - Label Encoding

A logistics company wants to build a machine learning model to optimize package delivery.

FEATURES:
- Size: Small, Medium, Large
- Priority: Low, Medium, High

PROBLEM:
Machine learning models work with numbers, not text labels.

SOLUTION:
Use Label Encoding to convert categories into numeric values.

ENCODING:
- Size → Small = 2, Medium = 1, Large = 0
- Priority → Low = 1, Medium = 2, High = 0

QUESTIONS:
Part A: Why convert categorical values to numbers before training?
Part B: Does the order (0, 1, 2) matter for all models? Why or why not?
Part C: How to decode numeric values back to original labels after prediction?
Part D: What happens if we add "Extra Large" after model is trained?
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("="*70)
print("LABEL ENCODING - PACKAGE DELIVERY SYSTEM")
print("="*70)

data = pd.DataFrame({
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
    'Priority': ['Low', 'High', 'Medium', 'Low', 'High']
})

print("\nOriginal Data:")
print(data)

label_encoder_size = LabelEncoder()
label_encoder_priority = LabelEncoder()

data['Size_Encoded'] = label_encoder_size.fit_transform(data['Size'])
data['Priority_Encoded'] = label_encoder_priority.fit_transform(data['Priority'])

print("\nEncoded Data:")
print(data)

print("\nEncoding Mappings:")
print("\nSize Mapping:")
for i, label in enumerate(label_encoder_size.classes_):
    print(f"  {label} → {i}")

print("\nPriority Mapping:")
for i, label in enumerate(label_encoder_priority.classes_):
    print(f"  {label} → {i}")

decoded_size = label_encoder_size.inverse_transform(data['Size_Encoded'])
decoded_priority = label_encoder_priority.inverse_transform(data['Priority_Encoded'])

print("\nDecoded Values (verification):")
print(f"Size: {decoded_size}")
print(f"Priority: {decoded_priority}")

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why convert to numbers?")
print("  - ML algorithms perform mathematical operations (addition, multiplication)")
print("  - Text labels cannot be used in equations")
print("  - Numeric encoding enables model training")

print("\nPart B: Does order matter?")
print("  - For tree-based models (Decision Tree, Random Forest): Order doesn't matter")
print("  - For linear models (Linear Regression, Logistic Regression): Order matters!")
print("  - Problem: Model might assume Large (0) < Medium (1) < Small (2)")
print("  - Solution: Use One-Hot Encoding for nominal categories")

print("\nPart C: How to decode?")
print("  - Use inverse_transform() method")
print("  - Example: label_encoder.inverse_transform([0, 1, 2])")

print("\nPart D: Adding 'Extra Large'?")
print("  - Challenge: LabelEncoder hasn't seen this category during training")
print("  - Error: Will raise ValueError for unknown category")
print("  - Solution: Retrain encoder with new data or use handle_unknown parameter")

data.to_csv('Day_6_Mar01_Ridge_CrossValidation/package_delivery_encoded.csv', index=False)
print("\nData saved to 'package_delivery_encoded.csv'")
