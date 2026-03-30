"""
SCENARIO: Employee Training & Satisfaction Survey - Ordinal Encoding

A company conducts a survey to understand how employee education level and job satisfaction 
affect performance.

FEATURES:
- Education: High School, Bachelor, Master, PhD
- Satisfaction: Poor, Average, Good, Excellent

PROBLEM:
These categories have a natural order (PhD > Master > Bachelor, Excellent > Good > Average).

SOLUTION:
Use Ordinal Encoding to convert them into numbers that respect this ranking.

CUSTOM ORDERING:
- Education → High School (0), Bachelor (1), Master (2), PhD (3)
- Satisfaction → Poor (0), Average (1), Good (2), Excellent (3)

QUESTIONS:
Part A: Why is Ordinal Encoding better than Label Encoding for this scenario?
Part B: What happens if we use Label Encoding instead? What problem arises?
Part C: How does the model interpret the numeric values in Ordinal Encoding?
Part D: If we add a new category "Associate Degree" between High School and Bachelor, 
        how would we update the encoding?
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

print("="*70)
print("ORDINAL ENCODING - EMPLOYEE TRAINING & SATISFACTION SURVEY")
print("="*70)

data = pd.DataFrame({
    'Education': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor', 'Master'],
    'Satisfaction': ['Good', 'Excellent', 'Poor', 'Good', 'Average', 'Excellent']
})

print("\nOriginal Data:")
print(data)

education_order = [['High School', 'Bachelor', 'Master', 'PhD']]
satisfaction_order = [['Poor', 'Average', 'Good', 'Excellent']]

ordinal_encoder_education = OrdinalEncoder(categories=education_order)
ordinal_encoder_satisfaction = OrdinalEncoder(categories=satisfaction_order)

data['Education_Encoded'] = ordinal_encoder_education.fit_transform(data[['Education']])
data['Satisfaction_Encoded'] = ordinal_encoder_satisfaction.fit_transform(data[['Satisfaction']])

print("\nEncoded Data:")
print(data)

print("\nOrdinal Encoding Mappings:")
print("\nEducation Mapping (ordered by level):")
for i, label in enumerate(education_order[0]):
    print(f"  {label} → {i}")

print("\nSatisfaction Mapping (ordered by quality):")
for i, label in enumerate(satisfaction_order[0]):
    print(f"  {label} → {i}")

manual_education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
manual_satisfaction_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}

data['Education_Manual'] = data['Education'].map(manual_education_map)
data['Satisfaction_Manual'] = data['Satisfaction'].map(manual_satisfaction_map)

print("\nComparison with Manual Dictionary Mapping:")
print(data[['Education', 'Education_Encoded', 'Education_Manual']])
print("\n")
print(data[['Satisfaction', 'Satisfaction_Encoded', 'Satisfaction_Manual']])

print("\nVerification:")
print(f"Education encoding matches manual: {(data['Education_Encoded'] == data['Education_Manual']).all()}")
print(f"Satisfaction encoding matches manual: {(data['Satisfaction_Encoded'] == data['Satisfaction_Manual']).all()}")

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why Ordinal Encoding is better?")
print("  - Preserves natural ordering: PhD (3) > Master (2) > Bachelor (1) > High School (0)")
print("  - Model understands that higher values mean higher education/satisfaction")
print("  - Label Encoding assigns arbitrary numbers without respecting order")
print("  - Ordinal Encoding maintains meaningful relationships between categories")

print("\nPart B: Problem with Label Encoding:")
print("  - Label Encoding sorts alphabetically: Bachelor (0), High School (1), Master (2), PhD (3)")
print("  - This creates incorrect ordering: Bachelor < High School (wrong!)")
print("  - Model learns incorrect relationships")
print("  - Predictions become unreliable")

print("\nPart C: Model interpretation:")
print("  - Model treats encoded values as continuous scale")
print("  - Difference between PhD (3) and Master (2) = 1 unit")
print("  - Assumes equal intervals between categories")
print("  - Higher values indicate higher education/satisfaction level")

print("\nPart D: Adding 'Associate Degree':")
print("  - New ordering: High School (0), Associate Degree (1), Bachelor (2), Master (3), PhD (4)")
print("  - All existing encodings shift up by 1 after Associate Degree")
print("  - Must retrain model with updated encoding")
print("  - Alternative: Use gaps (0, 2, 4, 6, 8) to allow future insertions")

data.to_csv('Day_6_Mar01_Ridge_CrossValidation/employee_survey_encoded.csv', index=False)
print("\nData saved to 'employee_survey_encoded.csv'")

