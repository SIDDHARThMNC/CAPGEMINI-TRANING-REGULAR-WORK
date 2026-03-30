# Scenario: Student Course Recommendations
# Simulates Chroma/Pinecone vector search using numpy + sklearn
# (chromadb/sentence-transformers not installed due to disk space)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("STUDENT COURSE RECOMMENDATION SYSTEM")
print("(Chroma-style local vector search)")
print("=" * 60)

# ── Simulated sentence embeddings (all-MiniLM-L6-v2 style) ──
# Each vector captures semantic meaning of the document
documents = [
    "Python is a high-level programming language",
    "Machine learning models need training data",
    "Dogs are loyal and friendly animals",
    "Cats are independent and curious pets",
]

# Manually crafted embeddings (dims: tech, ml, animals, pets, language, data)
doc_embeddings = np.array([
    [0.9, 0.2, 0.0, 0.0, 0.8, 0.3],   # Python programming
    [0.5, 0.9, 0.0, 0.0, 0.2, 0.9],   # Machine learning
    [0.0, 0.0, 0.9, 0.7, 0.0, 0.0],   # Dogs
    [0.0, 0.0, 0.8, 0.9, 0.0, 0.0],   # Cats
], dtype="float32")

# Normalize (like Chroma does internally)
norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
doc_embeddings = doc_embeddings / norms

print(f"\nDocuments indexed: {len(documents)}")
print(f"Embedding dims   : {doc_embeddings.shape[1]}")
print("\nIndexed Documents:")
for i, doc in enumerate(documents):
    print(f"  [doc{i}] {doc}")

# ── Chroma-style collection.add() simulation ─────────────────
collection = {
    "ids":       [f"doc{i}" for i in range(len(documents))],
    "documents": documents,
    "embeddings": doc_embeddings,
    "metadatas": [{"source": "demo", "idx": i} for i in range(len(documents))]
}

# ── Query ─────────────────────────────────────────────────────
def query_collection(collection, query_text, query_emb, n_results=3):
    q = np.array(query_emb, dtype="float32").reshape(1, -1)
    q = q / np.linalg.norm(q)
    sims = cosine_similarity(q, collection["embeddings"])[0]
    # Chroma returns distances (lower = better), convert: dist = 1 - sim
    distances = 1 - sims
    top_idx = np.argsort(distances)[:n_results]
    return {
        "documents": [[collection["documents"][i] for i in top_idx]],
        "distances": [[distances[i] for i in top_idx]],
        "ids":       [[collection["ids"][i] for i in top_idx]],
    }

# ── Test Queries ──────────────────────────────────────────────
queries = {
    "What animals make good companions?":     [0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
    "beginner-friendly Python tutorials":     [0.8, 0.2, 0.0, 0.0, 0.9, 0.2],
    "how do AI models learn from data?":      [0.4, 0.9, 0.0, 0.0, 0.1, 0.8],
}

print("\n" + "=" * 60)
print("SEMANTIC SEARCH RESULTS:")
print("=" * 60)

for query_text, q_emb in queries.items():
    print(f"\nStudent Query: \"{query_text}\"")
    print("-" * 50)

    results = query_collection(collection, query_text, q_emb, n_results=3)

    for doc, dist, doc_id in zip(
        results["documents"][0],
        results["distances"][0],
        results["ids"][0]
    ):
        sim = 1 - dist
        bar = "█" * int(sim * 15)
        print(f"  [{dist:.3f} dist | {sim:.3f} sim] [{doc_id}] {doc}  {bar}")

    print(f"\n  ✅ Best Match: {results['documents'][0][0]}")

# ── Chroma vs Pinecone comparison ────────────────────────────
print("\n" + "=" * 60)
print("CHROMA (Local) vs PINECONE (Cloud)")
print("=" * 60)
print(f"\n{'Feature':<25} | {'Chroma':>15} | {'Pinecone':>15}")
print("-" * 60)
rows = [
    ("Setup",          "No API key",    "API key needed"),
    ("Scale",          "Local/small",   "Billions of items"),
    ("Speed",          "Fast locally",  "Cloud-optimized"),
    ("Best for",       "Demos/dev",     "Production"),
    ("Persistence",    "Local disk",    "Cloud storage"),
]
for feat, chroma, pinecone in rows:
    print(f"{feat:<25} | {chroma:>15} | {pinecone:>15}")

print("\n" + "=" * 60)
print("Key Insight:")
print("  Traditional search: 'Python tutorials' ≠ 'Programming'")
print("  Semantic search   : embeddings connect meaning, not words!")
print("=" * 60)
