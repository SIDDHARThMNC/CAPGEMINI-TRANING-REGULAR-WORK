"""
SCENARIO: Restaurant Ordering System - Label Encoding

A restaurant wants to build a machine learning model to predict order preparation time.

FEATURES:
- Meal Type: Breakfast, Lunch, Dinner
- Spice Level: Mild, Medium, Hot

PROBLEM:
Models can't directly work with text labels.

SOLUTION:
Use Label Encoding to convert categories into numbers.

QUESTIONS:
Part A: Why is Label Encoding necessary for this restaurant system?
Part B: What are the encoded values for each meal type and spice level?
Part C: Can we use these encodings for a linear regression model? What issues might arise?
Part D: How would you handle a new category like "Brunch" that appears after training?
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("="*70)
print("LABEL ENCODING - RESTAURANT ORDERING SYSTEM")
print("="*70)

data = pd.DataFrame({
    'Meal_Type': ['Breakfast', 'Lunch', 'Dinner', 'Breakfast', 'Lunch', 'Dinner', 'Breakfast'],
    'Spice_Level': ['Mild', 'Hot', 'Medium', 'Medium', 'Mild', 'Hot', 'Hot']
})

print("\nOriginal Data:")
print(data)

label_encoder_meal = LabelEncoder()
label_encoder_spice = LabelEncoder()

data['Meal_Type_Encoded'] = label_encoder_meal.fit_transform(data['Meal_Type'])
data['Spice_Level_Encoded'] = label_encoder_spice.fit_transform(data['Spice_Level'])

print("\nEncoded Data:")
print(data)

print("\nEncoding Mappings:")
print("\nMeal Type Mapping:")
for i, label in enumerate(label_encoder_meal.classes_):
    print(f"  {label} → {i}")

print("\nSpice Level Mapping:")
for i, label in enumerate(label_encoder_spice.classes_):
    print(f"  {label} → {i}")

decoded_meal = label_encoder_meal.inverse_transform(data['Meal_Type_Encoded'])
decoded_spice = label_encoder_spice.inverse_transform(data['Spice_Level_Encoded'])

print("\nDecoded Values (verification):")
print(f"Meal Type: {decoded_meal}")
print(f"Spice Level: {decoded_spice}")

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why is Label Encoding necessary?")
print("  - ML models require numerical input for mathematical computations")
print("  - Text categories cannot be processed in algorithms")
print("  - Encoding converts categorical data into model-compatible format")

print("\nPart B: Encoded values:")
print(f"  Meal Type: {dict(zip(label_encoder_meal.classes_, range(len(label_encoder_meal.classes_))))}")
print(f"  Spice Level: {dict(zip(label_encoder_spice.classes_, range(len(label_encoder_spice.classes_))))}")

print("\nPart C: Issues with linear regression:")
print("  - Label Encoding creates ordinal relationship (0 < 1 < 2)")
print("  - Model assumes Breakfast (0) < Dinner (1) < Lunch (2)")
print("  - This is incorrect for nominal categories (no natural order)")
print("  - Better approach: Use One-Hot Encoding for nominal data")

print("\nPart D: Handling new category 'Brunch':")
print("  - LabelEncoder will raise ValueError for unseen category")
print("  - Solutions:")
print("    1. Retrain encoder with updated dataset including 'Brunch'")
print("    2. Map unknown categories to a default value")
print("    3. Use handle_unknown='ignore' in production pipelines")

data.to_csv('Day_6_Mar01_Ridge_CrossValidation/restaurant_orders_encoded.csv', index=False)
print("\nData saved to 'restaurant_orders_encoded.csv'")

