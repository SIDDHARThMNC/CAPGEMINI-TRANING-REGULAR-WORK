"""
SCENARIO: Patient Health Monitoring - Ordinal Encoding

A hospital wants to analyze patient records to understand how disease severity and recovery 
satisfaction affect treatment outcomes.

FEATURES:
- Disease Severity: Mild, Moderate, Severe, Critical
- Recovery Satisfaction: Poor, Average, Good, Excellent

PROBLEM:
These categories have a natural order (Critical > Severe > Moderate, Excellent > Good > Average).

SOLUTION:
Use Ordinal Encoding to convert them into numbers that respect this ranking.

CUSTOM ORDERING:
- Disease Severity → Mild (0), Moderate (1), Severe (2), Critical (3)
- Recovery Satisfaction → Poor (0), Average (1), Good (2), Excellent (3)

QUESTIONS:
Part A: Why is maintaining order important for medical data analysis?
Part B: How does Ordinal Encoding help doctors make better decisions?
Part C: What would happen if we used One-Hot Encoding instead?
Part D: If a patient's severity changes from Moderate to Severe, how does the model capture this?
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

print("="*70)
print("ORDINAL ENCODING - PATIENT HEALTH MONITORING")
print("="*70)

data = pd.DataFrame({
    'Patient_ID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006'],
    'Disease_Severity': ['Moderate', 'Severe', 'Mild', 'Critical', 'Moderate', 'Severe'],
    'Recovery_Satisfaction': ['Good', 'Average', 'Excellent', 'Poor', 'Good', 'Average']
})

print("\nOriginal Data:")
print(data)

severity_order = [['Mild', 'Moderate', 'Severe', 'Critical']]
satisfaction_order = [['Poor', 'Average', 'Good', 'Excellent']]

ordinal_encoder_severity = OrdinalEncoder(categories=severity_order)
ordinal_encoder_satisfaction = OrdinalEncoder(categories=satisfaction_order)

data['Severity_Encoded'] = ordinal_encoder_severity.fit_transform(data[['Disease_Severity']])
data['Satisfaction_Encoded'] = ordinal_encoder_satisfaction.fit_transform(data[['Recovery_Satisfaction']])

print("\nEncoded Data:")
print(data)

print("\nOrdinal Encoding Mappings:")
print("\nDisease Severity Mapping (ordered by intensity):")
for i, label in enumerate(severity_order[0]):
    print(f"  {label} → {i}")

print("\nRecovery Satisfaction Mapping (ordered by quality):")
for i, label in enumerate(satisfaction_order[0]):
    print(f"  {label} → {i}")

manual_severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2, 'Critical': 3}
manual_satisfaction_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}

data['Severity_Manual'] = data['Disease_Severity'].map(manual_severity_map)
data['Satisfaction_Manual'] = data['Recovery_Satisfaction'].map(manual_satisfaction_map)

print("\nComparison with Manual Dictionary Mapping:")
print(data[['Disease_Severity', 'Severity_Encoded', 'Severity_Manual']])
print("\n")
print(data[['Recovery_Satisfaction', 'Satisfaction_Encoded', 'Satisfaction_Manual']])

print("\nVerification:")
print(f"Severity encoding matches manual: {(data['Severity_Encoded'] == data['Severity_Manual']).all()}")
print(f"Satisfaction encoding matches manual: {(data['Satisfaction_Encoded'] == data['Satisfaction_Manual']).all()}")

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why maintaining order is important:")
print("  - Medical severity has clear progression: Mild → Moderate → Severe → Critical")
print("  - Treatment decisions depend on severity level")
print("  - Model needs to understand that Critical (3) is worse than Mild (0)")
print("  - Ordinal encoding preserves clinical meaning")

print("\nPart B: How it helps doctors:")
print("  - Predictive models can identify high-risk patients (Severity = 2 or 3)")
print("  - Resource allocation: Critical patients need more attention")
print("  - Track recovery trends: Satisfaction improving from Poor (0) to Good (2)")
print("  - Early intervention for patients with low satisfaction scores")

print("\nPart C: Using One-Hot Encoding instead:")
print("  - Creates separate columns: Mild, Moderate, Severe, Critical")
print("  - Loses ordinal relationship between categories")
print("  - Model can't understand that Severe is worse than Moderate")
print("  - Increases dimensionality (4 columns instead of 1)")
print("  - Not suitable for ordered medical data")

print("\nPart D: Capturing severity changes:")
print("  - Moderate (1) → Severe (2): Numeric increase by 1")
print("  - Model interprets this as worsening condition")
print("  - Can track disease progression over time")
print("  - Enables trend analysis and early warning systems")

data.to_csv('Day_6_Mar01_Ridge_CrossValidation/patient_health_encoded.csv', index=False)
print("\nData saved to 'patient_health_encoded.csv'")

