# Scenario: Hospital FAQ Assistant
# Converts FAQs and patient queries into embeddings,
# then uses cosine similarity to find the closest match.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("HOSPITAL FAQ ASSISTANT - SEMANTIC SEARCH")
print("=" * 60)

# FAQ embeddings
# Dimensions: [appointment, fees, emergency, reports, visiting, insurance]
faqs = {
    "How do I book an appointment?":       np.array([0.9, 0.1, 0.1, 0.1, 0.2, 0.1]),
    "What are the visiting hours?":        np.array([0.1, 0.1, 0.2, 0.1, 0.9, 0.1]),
    "How do I get my test reports?":       np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1]),
    "What insurance plans are accepted?":  np.array([0.1, 0.3, 0.1, 0.1, 0.1, 0.9]),
    "Is emergency care available 24/7?":   np.array([0.1, 0.1, 0.9, 0.1, 0.2, 0.1]),
    "How do I pay hospital fees?":         np.array([0.2, 0.9, 0.1, 0.1, 0.1, 0.3]),
}

faq_answers = {
    "How do I book an appointment?":       "Book online at our website or call 1800-XXX-XXXX.",
    "What are the visiting hours?":        "Visiting hours are 9 AM to 8 PM daily.",
    "How do I get my test reports?":       "Reports are available on the patient portal within 24 hours.",
    "What insurance plans are accepted?":  "We accept Star Health, HDFC Ergo, and government schemes.",
    "Is emergency care available 24/7?":   "Yes, our emergency department is open 24/7.",
    "How do I pay hospital fees?":         "Pay via the patient dashboard using card or net banking.",
}

# Patient queries with different phrasing
patient_queries = {
    "How do I book an appointment?":              np.array([0.9, 0.1, 0.1, 0.1, 0.2, 0.1]),
    "What's the process to schedule a doctor visit?": np.array([0.8, 0.1, 0.1, 0.1, 0.3, 0.1]),
    "Can I see a physician tomorrow?":            np.array([0.7, 0.1, 0.1, 0.1, 0.4, 0.1]),
    "When can I visit the hospital?":             np.array([0.2, 0.1, 0.2, 0.1, 0.8, 0.1]),
    "How to get my lab results?":                 np.array([0.1, 0.1, 0.1, 0.8, 0.1, 0.1]),
    "Is there 24 hour emergency service?":        np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
}

faq_keys    = list(faqs.keys())
faq_matrix  = np.array(list(faqs.values()))

print(f"\nFAQ Database: {len(faqs)} questions loaded")
print(f"Embedding Dimensions: {faq_matrix.shape[1]}")
print(f"[appointment, fees, emergency, reports, visiting, insurance]\n")

print("=" * 60)
print("PATIENT QUERY MATCHING:")
print("=" * 60)

for query, q_emb in patient_queries.items():
    sims = cosine_similarity([q_emb], faq_matrix)[0]
    best_idx   = np.argmax(sims)
    best_faq   = faq_keys[best_idx]
    best_score = sims[best_idx]

    print(f"\nPatient : \"{query}\"")
    print(f"Matched : \"{best_faq}\"  [{best_score:.4f}]")
    print(f"Answer  : {faq_answers[best_faq]}")

# Full similarity matrix
print("\n" + "=" * 60)
print("SIMILARITY MATRIX")
print("=" * 60)

q_names    = list(patient_queries.keys())
q_matrix   = np.array(list(patient_queries.values()))
sim_matrix = cosine_similarity(q_matrix, faq_matrix)

print(f"\n{'Patient Query':<45}", end="")
for faq in faq_keys:
    print(f" | {faq[:8]:>8}", end="")
print()
print("-" * (45 + 11 * len(faq_keys)))

for i, q in enumerate(q_names):
    print(f"{q[:45]:<45}", end="")
    for j in range(len(faq_keys)):
        print(f" | {sim_matrix[i][j]:>8.3f}", end="")
    print()

print("\n" + "=" * 60)
print("Key Insight:")
print("  'Schedule a doctor visit' → matches 'Book an appointment'")
print("  Semantic embeddings bridge the vocabulary gap!")
print("=" * 60)
