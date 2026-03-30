# Scenario: E-Commerce Product Recommendations
# Uses numpy embeddings + cosine similarity to recommend products

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("E-COMMERCE SMART RECOMMENDATION ENGINE")
print("=" * 60)

# Customer query embedding
query_embedding = np.array([0.1, 0.8, 0.5])

# Product catalog embeddings
product_embeddings = {
    "lightweight sneakers for jogging": np.array([0.2, 0.7, 0.6]),
    "formal leather shoes":             np.array([0.9, 0.1, 0.2]),
    "sandals for summer":               np.array([0.3, 0.4, 0.5]),
    "running shoes for gym":            np.array([0.1, 0.9, 0.5]),
    "heavy duty hiking boots":          np.array([0.6, 0.3, 0.2]),
}

print(f"\nCustomer Query  : 'comfortable running shoes'")
print(f"Query Embedding : {query_embedding}\n")
print("-" * 60)
print(f"{'Product':<35} | {'Dot Product':>11} | {'Cosine Sim':>10}")
print("-" * 60)

results = []
for product, embedding in product_embeddings.items():
    dot_sim    = np.dot(query_embedding, embedding)
    cosine_sim = cosine_similarity([query_embedding], [embedding])[0][0]
    results.append((product, dot_sim, cosine_sim))
    print(f"{product:<35} | {dot_sim:>11.3f} | {cosine_sim:>10.3f}")

# Best match
best_match = max(product_embeddings.items(), key=lambda x: np.dot(query_embedding, x[1]))

print("\n" + "=" * 60)
print(f"Recommended Product: {best_match[0]}")
print("=" * 60)

# Rankings
print("\nFull Rankings (by Cosine Similarity):")
for i, (product, dot, cos) in enumerate(sorted(results, key=lambda x: x[2], reverse=True), 1):
    bar = "█" * int(cos * 20)
    print(f"  {i}. {product:<35} | {cos:.3f} | {bar}")

print("\n" + "=" * 60)
print("Why Cosine Similarity?")
print("  - Dot Product   : fast but affected by vector magnitude")
print("  - Cosine Sim    : normalized, captures semantic direction")
print("  - Best for NLP  : meaning > magnitude")
print("=" * 60)
