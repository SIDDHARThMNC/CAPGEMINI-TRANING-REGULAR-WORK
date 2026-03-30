# Scenario: Fitness & Wellness Chatbot Training
# Classifies sentences into exercise, nutrition, or wellness categories

import math

print("=" * 60)
print("FITNESS & WELLNESS CHATBOT - SENTENCE EMBEDDINGS")
print("=" * 60)

# Training sentences
sentences = [
    "Push-ups strengthen chest and triceps",
    "Yoga improves flexibility and balance",
    "Apples are rich in fiber",
    "Cardio exercises boost heart health"
]

# Category keywords
exercise_keywords = ["push-ups", "yoga", "cardio", "exercises", "strengthen", "flexibility", "balance", "workout", "gym", "training"]
nutrition_keywords = ["apples", "fiber", "protein", "vitamins", "diet", "calories", "fruits", "vegetables", "nutrients"]
wellness_keywords = ["health", "heart", "boost", "improves", "benefits", "mental", "sleep", "stress"]

def get_embedding(text):
    words = text.lower().replace("-", " ").split()
    ex_score  = sum(1 for w in words if w in exercise_keywords)
    nut_score = sum(1 for w in words if w in nutrition_keywords)
    wel_score = sum(1 for w in words if w in wellness_keywords)
    total = max(len(words), 1)
    return [ex_score / total, nut_score / total, wel_score / total]

def cosine_similarity(v1, v2):
    dot  = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

def classify(emb):
    scores = {"Exercise": emb[0], "Nutrition": emb[1], "Wellness": emb[2]}
    return max(scores, key=scores.get)

# Generate embeddings
print("\nTraining Sentences & Embeddings:")
print("-" * 60)

embeddings = {}
for sent in sentences:
    emb = get_embedding(sent)
    embeddings[sent] = emb
    cat = classify(emb)
    print(f"\n\"{sent}\"")
    print(f"  Embedding : [Exercise={emb[0]:.2f}, Nutrition={emb[1]:.2f}, Wellness={emb[2]:.2f}]")
    print(f"  Category  : {cat}")

# Chatbot response mapping
responses = {
    "Exercise": "Great! Here's a workout tip: Start with 3 sets of 10 reps and gradually increase.",
    "Nutrition": "Nutrition tip: Include more whole foods and stay hydrated throughout the day.",
    "Wellness": "Wellness tip: Combine regular exercise with good sleep for optimal health benefits."
}

# Test new queries
test_queries = [
    "Squats build leg muscles and core strength",
    "Oranges are packed with vitamin C",
    "Meditation reduces stress and anxiety"
]

print("\n" + "=" * 60)
print("CHATBOT RESPONSE DEMO:")
print("=" * 60)

for query in test_queries:
    emb = get_embedding(query)
    cat = classify(emb)
    print(f"\nUser: \"{query}\"")
    print(f"Category Detected: {cat}")
    print(f"Chatbot: {responses[cat]}")

# Similarity analysis
print("\n" + "=" * 60)
print("SIMILARITY ANALYSIS:")
print("=" * 60)

pairs = [
    ("Push-ups strengthen chest and triceps", "Cardio exercises boost heart health"),
    ("Push-ups strengthen chest and triceps", "Apples are rich in fiber"),
    ("Yoga improves flexibility and balance", "Cardio exercises boost heart health"),
]

for s1, s2 in pairs:
    sim = cosine_similarity(embeddings[s1], embeddings[s2])
    label = "HIGH - same category" if sim > 0.5 else "LOW - different category"
    print(f"\n\"{s1[:35]}...\"")
    print(f"\"{s2[:35]}...\"")
    print(f"  Similarity: {sim:.4f} ({label})")

print("\n" + "=" * 60)
print("Chatbot can now tailor responses:")
print("  Exercise  -> Suggest workouts")
print("  Nutrition -> Offer diet tips")
print("  Wellness  -> Explain health benefits")
print("=" * 60)
