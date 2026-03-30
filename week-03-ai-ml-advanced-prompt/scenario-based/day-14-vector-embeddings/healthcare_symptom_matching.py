# Scenario: Healthcare Chatbot - Patient Symptom Matching
# Uses cosine similarity to match patient symptoms with medical conditions

import math

print("=" * 60)
print("HEALTHCARE CHATBOT - PATIENT SYMPTOM MATCHING")
print("=" * 60)

# Feature labels
features = ["Fever", "Cough", "Fatigue", "Headache"]

# Medical conditions symptom profiles
conditions = {
    "Common Cold":      [0.1, 0.9, -0.2,  0.4],
    "Flu (Influenza)":  [0.8, 0.7,  0.9,  0.6],
    "COVID-19":         [0.6, 0.8,  0.7,  0.5],
    "Migraine":         [0.0, 0.0, -0.1,  1.0],
    "Pneumonia":        [0.7, 0.9,  0.8, -0.1],
    "Allergies":        [0.1, 0.6, -0.3,  0.2],
}

# Patient symptom vectors
patients = {
    "Patient A": [0.2,  0.8, -0.3,  0.5],   # mild fever, strong cough, no fatigue, moderate headache
    "Patient B": [0.9,  0.6,  0.8,  0.5],   # high fever, cough, fatigue, headache
    "Patient C": [0.0, -0.1, -0.2,  0.9],   # no fever, no cough, severe headache
}

def cosine_similarity(v1, v2):
    dot  = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

def describe_symptoms(vector):
    desc = []
    labels = ["fever", "cough", "fatigue", "headache"]
    intensity = {True: ["mild", "moderate", "strong"], False: ["no", "low"]}
    for i, val in enumerate(vector):
        if val >= 0.7:
            desc.append(f"strong {labels[i]}")
        elif val >= 0.3:
            desc.append(f"moderate {labels[i]}")
        elif val > 0:
            desc.append(f"mild {labels[i]}")
        else:
            desc.append(f"no {labels[i]}")
    return ", ".join(desc)

# Main analysis
print(f"\nSymptom Features: {features}\n")

for patient_name, symptoms in patients.items():
    print("=" * 60)
    print(f"{patient_name} Symptoms: {symptoms}")
    print(f"Description: {describe_symptoms(symptoms)}")
    print("-" * 60)

    results = []
    for condition, profile in conditions.items():
        sim = cosine_similarity(symptoms, profile)
        results.append((condition, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    print("Condition Match Results:")
    for condition, score in results:
        bar = "█" * int(score * 20)
        print(f"  {condition:20} | {score:.4f} | {bar}")

    best, best_score = results[0]
    print(f"\nDiagnosis Suggestion: {best} (similarity: {best_score:.4f})")

    if best_score >= 0.95:
        print("Confidence: VERY HIGH - Strongly matches this condition")
    elif best_score >= 0.80:
        print("Confidence: HIGH - Likely matches this condition")
    elif best_score >= 0.60:
        print("Confidence: MODERATE - Possible match, consult a doctor")
    else:
        print("Confidence: LOW - Please visit a doctor for proper diagnosis")

# Detailed example from scenario
print("\n" + "=" * 60)
print("SCENARIO EXAMPLE (from problem statement):")
print("=" * 60)
patient_vec   = [0.2,  0.8, -0.3,  0.5]
condition_vec = [0.1,  0.9, -0.2,  0.4]

sim = cosine_similarity(patient_vec, condition_vec)
print(f"\nPatient vector   : {patient_vec}")
print(f"  → mild fever, strong cough, no fatigue, moderate headache")
print(f"\nCondition vector : {condition_vec}")
print(f"  → mild fever, strong cough, little fatigue, moderate headache")
print(f"\nCosine Similarity: {sim:.6f}")
print(f"Interpretation   : {sim*100:.2f}% match → Very high alignment!")
print("=" * 60)
