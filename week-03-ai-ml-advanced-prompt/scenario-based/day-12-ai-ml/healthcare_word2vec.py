# Healthcare Research Insights - Word2Vec Analysis
# Explores semantic relationships between medical concepts

from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# Healthcare corpus
corpus = [
    "Diagnosis of patients requires careful examination of symptoms and medical history",
    "Treatment plans include medication and therapy for patient recovery",
    "Symptoms like fever and cough indicate respiratory infections",
    "Patients receive medication based on their diagnosis and condition",
    "Medical diagnosis involves analyzing symptoms and test results",
    "Effective treatment combines medication with lifestyle changes",
    "Patients report symptoms to doctors during clinical examinations",
    "Medication dosage depends on patient age and diagnosis severity",
    "Symptoms guide doctors in making accurate diagnosis decisions",
    "Treatment outcomes improve when patients follow medication schedules"
]

print("=" * 60)
print("HEALTHCARE RESEARCH - WORD2VEC ANALYSIS")
print("=" * 60)

# Tokenize and preprocess
sentences = [word_tokenize(sent.lower()) for sent in corpus]

print(f"\nCorpus size: {len(sentences)} sentences")
print(f"Sample sentence: {sentences[0]}\n")

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4, epochs=100)

print("-" * 60)
print("MEDICAL CONCEPT SIMILARITY ANALYSIS:")
print("-" * 60)

# Analyze word similarities
test_pairs = [
    ("diagnosis", "symptoms"),
    ("treatment", "medication"),
    ("patients", "diagnosis"),
    ("symptoms", "medication")
]

for word1, word2 in test_pairs:
    similarity = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

print("\n" + "-" * 60)
print("MOST SIMILAR MEDICAL TERMS:")
print("-" * 60)

# Find similar words
key_concepts = ["diagnosis", "treatment", "symptoms", "patients", "medication"]

for concept in key_concepts:
    similar = model.wv.most_similar(concept, topn=3)
    print(f"\n'{concept}' is similar to:")
    for word, score in similar:
        print(f"  - {word}: {score:.4f}")

print("\n" + "=" * 60)
