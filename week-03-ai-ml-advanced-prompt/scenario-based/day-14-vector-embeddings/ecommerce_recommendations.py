# Scenario: E-Commerce Product Recommendations
# Smart product search using semantic embeddings

import math

print("=" * 60)
print("E-COMMERCE SMART RECOMMENDATION ENGINE")
print("=" * 60)

# Product catalog
products = [
    {"id": 1, "name": "lightweight sneakers for jogging", "price": 2999},
    {"id": 2, "name": "heavy duty boots for hiking", "price": 4999},
    {"id": 3, "name": "casual running shoes for gym", "price": 1999},
    {"id": 4, "name": "formal leather office shoes", "price": 3499},
    {"id": 5, "name": "comfortable sports sandals", "price": 1499},
]

# Synonym mapping for semantic understanding
synonyms = {
    "comfortable": "casual", "running": "jogging", "shoes": "sneakers",
    "sneakers": "sneakers", "jogging": "jogging", "lightweight": "lightweight",
    "fast": "jogging", "sport": "sports", "athletic": "jogging"
}

def normalize(text):
    words = text.lower().split()
    return [synonyms.get(w, w) for w in words]

def get_vector(text, vocab):
    words = normalize(text)
    vec = [0] * len(vocab)
    for word in words:
        if word in vocab:
            vec[vocab.index(word)] += 1
    return vec

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

# Build vocabulary
all_text = " ".join(p["name"] for p in products)
vocab = list(set(normalize(all_text)))

# Customer queries
queries = [
    "comfortable running shoes",
    "lightweight sneakers for jogging",
    "office formal wear"
]

print("\nProduct Catalog:")
for p in products:
    print(f"  [{p['id']}] {p['name']} - Rs.{p['price']}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

for query in queries:
    print(f"\nCustomer Search: \"{query}\"")
    q_vec = get_vector(query, vocab)

    results = []
    for p in products:
        p_vec = get_vector(p["name"], vocab)
        sim = cosine_similarity(q_vec, p_vec)
        results.append((p, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    print("Recommended Products:")
    for p, score in results[:3]:
        print(f"  [{score:.2f}] {p['name']} - Rs.{p['price']}")

print("\n" + "=" * 60)
print("Semantic search finds relevant products even with different words!")
print("=" * 60)
