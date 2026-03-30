# Smart Manufacturing Insights - Word2Vec Analysis
# Explores semantic relationships between manufacturing concepts

from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# Manufacturing corpus
corpus = [
    "Predictive maintenance helps reduce equipment downtime and improve efficiency",
    "Telemetry data from sensors monitors machine performance in real time",
    "Quality control ensures products meet manufacturing standards",
    "Supply chain optimization reduces costs and improves delivery times",
    "Equipment maintenance schedules prevent unexpected failures",
    "Sensor telemetry provides insights into production processes",
    "Quality assurance teams inspect products for defects",
    "Supply chain management coordinates material flow and inventory",
    "Maintenance teams use data analytics to predict equipment issues",
    "Real time telemetry enables proactive quality control measures"
]

print("=" * 60)
print("SMART MANUFACTURING - WORD2VEC ANALYSIS")
print("=" * 60)

# Tokenize and preprocess
sentences = [word_tokenize(sent.lower()) for sent in corpus]

print(f"\nCorpus size: {len(sentences)} sentences")
print(f"Sample sentence: {sentences[0]}\n")

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4, epochs=100)

print("-" * 60)
print("WORD SIMILARITY ANALYSIS:")
print("-" * 60)

# Analyze word similarities
test_words = [
    ("maintenance", "equipment"),
    ("telemetry", "sensors"),
    ("quality", "products"),
    ("supply", "chain")
]

for word1, word2 in test_words:
    similarity = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

print("\n" + "-" * 60)
print("MOST SIMILAR WORDS:")
print("-" * 60)

# Find similar words
key_concepts = ["maintenance", "telemetry", "quality", "supply"]

for concept in key_concepts:
    similar = model.wv.most_similar(concept, topn=3)
    print(f"\n'{concept}' is similar to:")
    for word, score in similar:
        print(f"  - {word}: {score:.4f}")

print("\n" + "=" * 60)
