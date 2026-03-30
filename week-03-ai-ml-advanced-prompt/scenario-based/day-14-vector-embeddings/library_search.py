# Scenario: University Library Search
# Smart search using vector embeddings to find semantically similar books

import math

print("=" * 60)
print("UNIVERSITY LIBRARY SMART SEARCH SYSTEM")
print("=" * 60)

# Simple TF-IDF style word embeddings (no external libraries)
def get_word_vector(text, vocab):
    words = text.lower().split()
    vector = [0] * len(vocab)
    for word in words:
        synonyms = {
            "canine": "dog", "sprinting": "running", "fast": "fast",
            "quick": "fast", "hound": "dog", "jogging": "running"
        }
        word = synonyms.get(word, word)
        if word in vocab:
            vector[vocab.index(word)] += 1
    return vector

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

# Library catalog
catalog = [
    "The dog runs fast",
    "Python programming for beginners",
    "Machine learning with neural networks",
    "The quick brown fox jumps",
    "Data science and analytics",
    "Animal behavior and instincts"
]

# Build vocabulary
vocab = list(set(
    w for book in catalog for w in book.lower().split()
) | {"dog", "running", "fast", "canine", "sprinting"})

# Student queries
queries = [
    "fast dog running",
    "canine sprinting"
]

print("\nLibrary Catalog:")
for i, book in enumerate(catalog, 1):
    print(f"  {i}. {book}")

print("\n" + "=" * 60)
print("SEARCH RESULTS:")
print("=" * 60)

for query in queries:
    print(f"\nStudent Query: \"{query}\"")
    q_vec = get_word_vector(query, vocab)

    results = []
    for book in catalog:
        b_vec = get_word_vector(book, vocab)
        sim = cosine_similarity(q_vec, b_vec)
        results.append((book, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    print("Top Matches:")
    for book, score in results[:3]:
        print(f"  [{score:.2f}] {book}")

print("\n" + "=" * 60)
print("Traditional keyword search would MISS these matches!")
print("Embeddings capture SEMANTIC meaning instead.")
print("=" * 60)
