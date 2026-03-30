# Healthcare Chatbot - Patient Symptom Matching using Multiple Models
# Compares: Cosine Similarity, Euclidean Distance, Manhattan Distance, Dot Product

import math

print("=" * 60)
print("HEALTHCARE CHATBOT - MULTI-MODEL SYMPTOM MATCHING")
print("=" * 60)

features = ["Fever", "Cough", "Fatigue", "Headache"]

conditions = {
    "Common Cold":     [0.1,  0.9, -0.2,  0.4],
    "Flu (Influenza)": [0.8,  0.7,  0.9,  0.6],
    "COVID-19":        [0.6,  0.8,  0.7,  0.5],
    "Migraine":        [0.0,  0.0, -0.1,  1.0],
    "Pneumonia":       [0.7,  0.9,  0.8, -0.1],
    "Allergies":       [0.1,  0.6, -0.3,  0.2],
}

patient = {
    "name": "Patient A",
    "vector": [0.2, 0.8, -0.3, 0.5]
}

# ── Model 1: Cosine Similarity ──────────────────────────────
def cosine_similarity(v1, v2):
    dot  = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(b**2 for b in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

# ── Model 2: Euclidean Distance (converted to similarity) ───
def euclidean_similarity(v1, v2):
    dist = math.sqrt(sum((a - b)**2 for a, b in zip(v1, v2)))
    return 1 / (1 + dist)   # higher = more similar

# ── Model 3: Manhattan Distance (converted to similarity) ───
def manhattan_similarity(v1, v2):
    dist = sum(abs(a - b) for a, b in zip(v1, v2))
    return 1 / (1 + dist)

# ── Model 4: Dot Product (normalized) ───────────────────────
def dot_product_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    # normalize to 0-1 range using sigmoid
    return 1 / (1 + math.exp(-dot))

models = {
    "Cosine Similarity":    cosine_similarity,
    "Euclidean Similarity": euclidean_similarity,
    "Manhattan Similarity": manhattan_similarity,
    "Dot Product (sigmoid)":dot_product_similarity,
}

pv = patient["vector"]
print(f"\nPatient: {patient['name']}")
print(f"Symptoms Vector: {pv}")
print(f"Description: mild fever, strong cough, no fatigue, moderate headache\n")

# Run all models
all_results = {}
for model_name, model_fn in models.items():
    scores = {cond: model_fn(pv, vec) for cond, vec in conditions.items()}
    all_results[model_name] = scores

# Display results per model
for model_name, scores in all_results.items():
    print("=" * 60)
    print(f"MODEL: {model_name}")
    print("-" * 60)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for cond, score in sorted_scores:
        bar = "█" * int(score * 20)
        print(f"  {cond:20} | {score:.4f} | {bar}")
    best = sorted_scores[0]
    print(f"\n  >> Best Match: {best[0]} ({best[1]:.4f})")

# Comparison summary table
print("\n" + "=" * 60)
print("COMPARISON SUMMARY - TOP DIAGNOSIS PER MODEL")
print("=" * 60)
print(f"{'Model':<25} | {'Top Diagnosis':<20} | {'Score'}")
print("-" * 60)
for model_name, scores in all_results.items():
    best_cond = max(scores, key=scores.get)
    best_score = scores[best_cond]
    print(f"{model_name:<25} | {best_cond:<20} | {best_score:.4f}")

# Agreement check
top_diagnoses = [max(s, key=s.get) for s in all_results.values()]
if len(set(top_diagnoses)) == 1:
    print(f"\nAll models AGREE: '{top_diagnoses[0]}' is the best match!")
else:
    print(f"\nModels DISAGREE on top diagnosis: {set(top_diagnoses)}")

print("\n" + "=" * 60)
print("Key Takeaways:")
print("  Cosine    -> Best for direction/angle between vectors")
print("  Euclidean -> Best for absolute distance in space")
print("  Manhattan -> Robust to outliers, city-block distance")
print("  Dot Prod  -> Considers both magnitude and direction")
print("=" * 60)

