# Scenario: Online Learning Platform Recommendations
# Uses numpy embeddings + cosine similarity to recommend courses

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("ONLINE LEARNING PLATFORM - COURSE RECOMMENDER")
print("=" * 60)

# Course catalog embeddings
# Dimensions: [python, beginner, data, web, ai, math]
course_embeddings = {
    "Introduction to Programming with Python":  np.array([0.9, 0.8, 0.2, 0.1, 0.2, 0.3]),
    "Advanced Machine Learning with Python":    np.array([0.7, 0.1, 0.9, 0.1, 0.9, 0.7]),
    "Web Development with HTML and CSS":        np.array([0.1, 0.7, 0.1, 0.9, 0.1, 0.1]),
    "Data Science for Beginners":               np.array([0.5, 0.8, 0.9, 0.1, 0.5, 0.6]),
    "Deep Learning and Neural Networks":        np.array([0.4, 0.1, 0.8, 0.1, 1.0, 0.8]),
    "Python for Data Analysis":                 np.array([0.8, 0.5, 0.9, 0.1, 0.4, 0.5]),
    "Mathematics for AI":                       np.array([0.2, 0.3, 0.6, 0.1, 0.7, 1.0]),
}

# Student queries
student_queries = {
    "beginner-friendly Python tutorials":       np.array([0.9, 0.9, 0.2, 0.1, 0.1, 0.2]),
    "learn data science from scratch":          np.array([0.4, 0.8, 0.9, 0.1, 0.4, 0.5]),
    "AI and deep learning course":              np.array([0.3, 0.2, 0.7, 0.1, 1.0, 0.7]),
    "build websites for beginners":             np.array([0.1, 0.8, 0.1, 0.9, 0.1, 0.1]),
}

def get_recommendations(query_emb, catalog, top_n=3):
    results = []
    for course, emb in catalog.items():
        dot  = np.dot(query_emb, emb)
        cos  = cosine_similarity([query_emb], [emb])[0][0]
        results.append((course, dot, cos))
    return sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

# Run recommendations for each query
for query, q_emb in student_queries.items():
    print("\n" + "=" * 60)
    print(f"Student Search : \"{query}\"")
    print(f"Query Embedding: {q_emb}")
    print("-" * 60)
    print(f"{'Course':<45} | {'Dot':>6} | {'Cosine':>7}")
    print("-" * 60)

    recs = get_recommendations(q_emb, course_embeddings)
    for course, dot, cos in recs:
        bar = "█" * int(cos * 15)
        print(f"{course:<45} | {dot:>6.3f} | {cos:>7.3f} {bar}")

    print(f"\n  ✅ Top Recommendation: {recs[0][0]}")

# Full similarity matrix
print("\n" + "=" * 60)
print("SIMILARITY MATRIX (All Queries vs All Courses)")
print("=" * 60)

q_names = list(student_queries.keys())
c_names = list(course_embeddings.keys())
q_matrix = np.array(list(student_queries.values()))
c_matrix = np.array(list(course_embeddings.values()))

sim_matrix = cosine_similarity(q_matrix, c_matrix)

# Print header
print(f"\n{'Query':<35}", end="")
for c in c_names:
    print(f" | {c[:10]:>10}", end="")
print()
print("-" * (35 + 13 * len(c_names)))

for i, q in enumerate(q_names):
    print(f"{q[:35]:<35}", end="")
    for j in range(len(c_names)):
        print(f" | {sim_matrix[i][j]:>10.3f}", end="")
    print()

print("\n" + "=" * 60)
print("Key Insight:")
print("  Semantic embeddings find relevant courses")
print("  even when exact keywords don't match!")
print("=" * 60)
