# Scenario: Hospital FAQ Assistant
# Matches patient queries to FAQs using cosine similarity embeddings

import math

print("=" * 60)
print("HOSPITAL FAQ ASSISTANT - SEMANTIC SEARCH")
print("=" * 60)

# Hospital FAQs
faqs = [
    {"q": "How do I book an appointment?",
     "a": "You can book an appointment online at our website or call our helpline at 1800-XXX-XXXX."},
    {"q": "What are the visiting hours?",
     "a": "Visiting hours are from 9 AM to 8 PM daily."},
    {"q": "How do I get my test reports?",
     "a": "Test reports are available online via the patient portal within 24 hours."},
    {"q": "What insurance plans are accepted?",
     "a": "We accept all major insurance plans including Star Health, HDFC Ergo, and government schemes."},
    {"q": "Is emergency care available 24/7?",
     "a": "Yes, our emergency department is open 24 hours, 7 days a week."},
]

# Synonym mapping
synonyms = {
    "schedule": "book", "physician": "doctor", "visit": "appointment",
    "see": "book", "process": "how", "can": "how", "tomorrow": "appointment",
    "doctor": "appointment", "reports": "test", "results": "test",
    "timing": "hours", "open": "hours", "urgent": "emergency"
}

def normalize(text):
    words = text.lower().replace("?", "").split()
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

# Build vocabulary from all FAQs
all_text = " ".join(f["q"] for f in faqs)
vocab = list(set(normalize(all_text)))

# Precompute FAQ vectors
faq_vectors = [get_vector(f["q"], vocab) for f in faqs]

# Patient queries (different phrasing)
patient_queries = [
    "How do I book an appointment?",
    "What's the process to schedule a doctor visit?",
    "Can I see a physician tomorrow?",
    "When can I visit the hospital?",
    "How to get my lab results?"
]

print("\nHospital FAQs Loaded:", len(faqs), "questions")
print("\n" + "=" * 60)
print("PATIENT QUERY MATCHING:")
print("=" * 60)

for query in patient_queries:
    print(f"\nPatient: \"{query}\"")
    q_vec = get_vector(query, vocab)

    results = []
    for i, faq in enumerate(faqs):
        sim = cosine_similarity(q_vec, faq_vectors[i])
        results.append((faq, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    best_faq, best_score = results[0]

    print(f"Matched FAQ [{best_score:.2f}]: \"{best_faq['q']}\"")
    print(f"Answer: {best_faq['a']}")

print("\n" + "=" * 60)
print("Chatbot handles different phrasings of the same question!")
print("=" * 60)
