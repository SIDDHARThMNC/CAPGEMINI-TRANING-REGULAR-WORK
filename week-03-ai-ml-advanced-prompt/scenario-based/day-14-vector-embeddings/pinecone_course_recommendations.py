# Scenario: Student Course Recommendations - OPTION B: Pinecone (Cloud)
# Simulates Pinecone vector search using numpy + sklearn
# (pinecone library not installed, simulating the API behavior)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("STUDENT COURSE RECOMMENDATION - PINECONE STYLE")
print("(Cloud Vector Database Simulation)")
print("=" * 60)

# ── Simulated Pinecone Index ──────────────────────────────────
class PineconeIndex:
    """Simulates Pinecone index behavior locally"""
    def __init__(self, name, dimension, metric="cosine"):
        self.name      = name
        self.dimension = dimension
        self.metric    = metric
        self._vectors  = {}   # id -> {"values": vec, "metadata": {}}
        print(f"\n[Pinecone] Index '{name}' created")
        print(f"  Dimension : {dimension}")
        print(f"  Metric    : {metric}")
        print(f"  Cloud     : aws  |  Region: us-east-1")

    def upsert(self, vectors):
        for item in vectors:
            self._vectors[item["id"]] = {
                "values":   np.array(item["values"], dtype="float32"),
                "metadata": item.get("metadata", {})
            }
        print(f"[Pinecone] Upserted {len(vectors)} vectors → Total: {len(self._vectors)}")

    def query(self, vector, top_k=3, include_metadata=True):
        q = np.array(vector, dtype="float32").reshape(1, -1)
        q = q / np.linalg.norm(q)
        results = []
        for vid, vdata in self._vectors.items():
            v = vdata["values"].reshape(1, -1)
            v = v / np.linalg.norm(v)
            score = cosine_similarity(q, v)[0][0]
            results.append({
                "id":       vid,
                "score":    float(score),
                "metadata": vdata["metadata"] if include_metadata else {}
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": results[:top_k]}

    def describe_index_stats(self):
        return {"total_vector_count": len(self._vectors), "dimension": self.dimension}


# ── Setup ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Initialize Pinecone & Create Index")
print("=" * 60)

# Simulated Pinecone init
print("\n# pc = Pinecone(api_key='YOUR_API_KEY')")
print("# index = pc.Index('course-recommendations')")
print("\n[Simulating Pinecone locally...]")

index = PineconeIndex(name="course-recommendations", dimension=6, metric="cosine")

# ── Documents & Embeddings ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Upsert Documents into Pinecone Index")
print("=" * 60)

documents = [
    "Python is a high-level programming language",
    "Machine learning models need training data",
    "Dogs are loyal and friendly animals",
    "Cats are independent and curious pets",
]

# Embeddings: [tech, ml, animals, pets, language, data]
raw_embeddings = [
    [0.9, 0.2, 0.0, 0.0, 0.8, 0.3],
    [0.5, 0.9, 0.0, 0.0, 0.2, 0.9],
    [0.0, 0.0, 0.9, 0.7, 0.0, 0.0],
    [0.0, 0.0, 0.8, 0.9, 0.0, 0.0],
]

vectors = [
    {
        "id":       f"doc{i}",
        "values":   raw_embeddings[i],
        "metadata": {"text": doc, "source": "course-catalog", "idx": i}
    }
    for i, doc in enumerate(documents)
]

index.upsert(vectors)
stats = index.describe_index_stats()
print(f"[Pinecone] Stats: {stats}")

# ── Query ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Query Pinecone Index")
print("=" * 60)

queries = {
    "What animals make good companions?":  [0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
    "beginner-friendly Python tutorials":  [0.8, 0.2, 0.0, 0.0, 0.9, 0.2],
    "how do AI models learn from data?":   [0.4, 0.9, 0.0, 0.0, 0.1, 0.8],
}

for query_text, q_emb in queries.items():
    print(f"\nStudent Query : \"{query_text}\"")
    print("-" * 55)

    results = index.query(vector=q_emb, top_k=3, include_metadata=True)

    for match in results["matches"]:
        bar  = "█" * int(match["score"] * 15)
        text = match["metadata"].get("text", match["id"])
        print(f"  [{match['score']:.3f}] [{match['id']}] {text}  {bar}")

    best = results["matches"][0]
    print(f"\n  ✅ Pinecone Recommends: {best['metadata']['text']}")

# ── Chroma vs Pinecone ────────────────────────────────────────
print("\n" + "=" * 60)
print("CHROMA (Local) vs PINECONE (Cloud)")
print("=" * 60)
print(f"\n{'Feature':<25} | {'Chroma':>15} | {'Pinecone':>15}")
print("-" * 60)
rows = [
    ("API Key",        "Not needed",    "Required"),
    ("Scale",          "Local/small",   "Billions of items"),
    ("Latency",        "~1ms local",    "~10ms cloud"),
    ("Best for",       "Demos/dev",     "Production"),
    ("Persistence",    "Local disk",    "Cloud storage"),
    ("Cost",           "Free",          "Pay per use"),
]
for feat, chroma, pinecone in rows:
    print(f"{feat:<25} | {chroma:>15} | {pinecone:>15}")

print("\n" + "=" * 60)
print("Production Code (Real Pinecone):")
print("  from pinecone import Pinecone")
print("  pc = Pinecone(api_key='YOUR_KEY')")
print("  index = pc.Index('course-recommendations')")
print("  index.upsert(vectors=[...])")
print("  results = index.query(vector=q_emb, top_k=3)")
print("=" * 60)
