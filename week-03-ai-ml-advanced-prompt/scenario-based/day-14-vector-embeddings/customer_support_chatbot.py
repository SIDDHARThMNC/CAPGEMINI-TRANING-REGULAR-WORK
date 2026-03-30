# Scenario: Customer Support Chatbot Training
# Classifies sentences by semantic category using embeddings

import math

print("=" * 60)
print("CUSTOMER SUPPORT CHATBOT - SENTENCE EMBEDDINGS")
print("=" * 60)

# Training sentences
sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Bananas are yellow fruits",
    "Artificial intelligence powers chatbots"
]

# Simple keyword-based embedding
ai_keywords = ["machine", "learning", "deep", "neural", "networks", "artificial", "intelligence", "ai", "chatbots", "subset"]
food_keywords = ["bananas", "yellow", "fruits", "food", "eat"]

def get_embedding(text):
    words = text.lower().split()
    ai_score = sum(1 for w in words if w in ai_keywords)
    food_score = sum(1 for w in words if w in food_keywords)
    total = len(words)
    return [ai_score / total, food_score / total, total / 10]

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

print("\nTraining Sentences & Embeddings:")
print("-" * 60)

embeddings = {}
for sent in sentences:
    emb = get_embedding(sent)
    embeddings[sent] = emb
    print(f"\n\"{sent}\"")
    print(f"  Embedding: {[round(e, 3) for e in emb]}")
    category = "AI/Tech" if emb[0] > emb[1] else "Other"
    print(f"  Category: {category}")

print("\n" + "=" * 60)
print("SIMILARITY ANALYSIS:")
print("=" * 60)

pairs = [
    ("Machine learning is a subset of AI", "Deep learning uses neural networks"),
    ("Machine learning is a subset of AI", "Bananas are yellow fruits"),
    ("Deep learning uses neural networks", "Artificial intelligence powers chatbots"),
]

for s1, s2 in pairs:
    sim = cosine_similarity(embeddings[s1], embeddings[s2])
    print(f"\n\"{s1[:30]}...\"")
    print(f"\"{s2[:30]}...\"")
    print(f"  Similarity: {sim:.4f} {'(HIGH - same topic)' if sim > 0.5 else '(LOW - different topic)'}")

print("\n" + "=" * 60)
print("Chatbot can now route queries to correct department!")
print("=" * 60)
