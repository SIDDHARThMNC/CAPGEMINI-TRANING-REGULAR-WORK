# Transformers and Large Language Models (LLMs)
# Understanding the Architecture Behind ChatGPT, BERT, and GPT
#
# WHAT ARE TRANSFORMERS?
# Transformers are a type of neural network architecture introduced in 2017
# that revolutionized Natural Language Processing (NLP). They use a mechanism
# called "attention" to understand context and relationships in text.
#
# KEY CONCEPTS:
# 1. Self-Attention: Model learns which words are important to each other
# 2. Positional Encoding: Understands word order in sentences
# 3. Encoder-Decoder: Processes input and generates output
# 4. Multi-Head Attention: Looks at text from multiple perspectives
#
# FAMOUS TRANSFORMER MODELS:
# - BERT: Bidirectional Encoder (Google)
# - GPT: Generative Pre-trained Transformer (OpenAI)
# - T5: Text-to-Text Transfer Transformer (Google)
# - LLaMA: Large Language Model (Meta)

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("TRANSFORMERS & LARGE LANGUAGE MODELS")
print("Understanding the Architecture Behind Modern AI")
print("=" * 70)

# Step 1: Understanding Attention Mechanism
print("\n" + "=" * 70)
print("STEP 1: ATTENTION MECHANISM")
print("=" * 70)

print("""
WHAT IS ATTENTION?
Attention allows the model to focus on relevant parts of the input when
processing each word. It's like highlighting important words in a sentence.

EXAMPLE:
Sentence: "The cat sat on the mat"

When processing "sat":
- High attention to: "cat" (who sat?)
- Medium attention to: "mat" (where?)
- Low attention to: "the" (less important)

ATTENTION FORMULA:
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Where:
- Q (Query): What we're looking for
- K (Key): What we're looking at
- V (Value): The actual information
- d_k: Dimension of keys (for scaling)
""")

# Simulate simple attention
def simple_attention_demo():
    """Demonstrate attention mechanism with a simple example"""
    
    # Example sentence: "The cat sat"
    words = ["The", "cat", "sat"]
    
    # Simulate word embeddings (in reality, these are 768 or more dimensions)
    embeddings = np.array([
        [0.1, 0.2, 0.3],  # The
        [0.5, 0.6, 0.7],  # cat
        [0.8, 0.9, 1.0]   # sat
    ])
    
    # Calculate attention scores (simplified)
    # In reality: Q·K^T / √d_k
    attention_scores = np.dot(embeddings, embeddings.T)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(attention_scores) / np.exp(attention_scores).sum(axis=1, keepdims=True)
    
    print("\nAttention Weights Matrix:")
    print("(Each row shows how much attention a word pays to other words)")
    print("\n       The    cat    sat")
    for i, word in enumerate(words):
        print(f"{word:5s}", end="  ")
        for j in range(len(words)):
            print(f"{attention_weights[i, j]:.3f}", end="  ")
        print()
    
    return attention_weights

attention_weights = simple_attention_demo()

# Visualize attention
plt.figure(figsize=(8, 6))
plt.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xticks(range(3), ["The", "cat", "sat"])
plt.yticks(range(3), ["The", "cat", "sat"])
plt.xlabel('Key (attending to)')
plt.ylabel('Query (attending from)')
plt.title('Attention Weights Visualization')
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{attention_weights[i, j]:.2f}', 
                ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig('attention_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Attention visualization saved")

# Step 2: Transformer Architecture
print("\n" + "=" * 70)
print("STEP 2: TRANSFORMER ARCHITECTURE")
print("=" * 70)

print("""
TRANSFORMER COMPONENTS:

1. INPUT EMBEDDING:
   - Converts words to vectors (numbers)
   - Example: "cat" → [0.5, 0.6, 0.7, ...]

2. POSITIONAL ENCODING:
   - Adds position information
   - Helps model understand word order
   - "cat sat" vs "sat cat" have different meanings

3. ENCODER (Left side):
   - Processes input text
   - Multiple layers of:
     * Multi-Head Self-Attention
     * Feed-Forward Neural Network
     * Layer Normalization
     * Residual Connections

4. DECODER (Right side):
   - Generates output text
   - Similar to encoder but with:
     * Masked Self-Attention (can't see future words)
     * Cross-Attention (looks at encoder output)

5. OUTPUT:
   - Probability distribution over vocabulary
   - Picks most likely next word

ARCHITECTURE DIAGRAM:

Input Text
    ↓
[Embedding + Positional Encoding]
    ↓
┌─────────────────────┐
│   ENCODER BLOCK     │
│  ┌───────────────┐  │
│  │ Multi-Head    │  │
│  │ Attention     │  │
│  └───────────────┘  │
│         ↓           │
│  ┌───────────────┐  │
│  │ Feed Forward  │  │
│  └───────────────┘  │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   DECODER BLOCK     │
│  ┌───────────────┐  │
│  │ Masked        │  │
│  │ Attention     │  │
│  └───────────────┘  │
│         ↓           │
│  ┌───────────────┐  │
│  │ Cross         │  │
│  │ Attention     │  │
│  └───────────────┘  │
│         ↓           │
│  ┌───────────────┐  │
│  │ Feed Forward  │  │
│  └───────────────┘  │
└─────────────────────┘
    ↓
Output Text
""")

# Step 3: Different Types of Transformers
print("\n" + "=" * 70)
print("STEP 3: TYPES OF TRANSFORMER MODELS")
print("=" * 70)

print("""
1. BERT (Bidirectional Encoder Representations from Transformers)
   Architecture: Encoder-only
   Training: Masked Language Modeling
   Use Cases:
   - Text classification
   - Named entity recognition
   - Question answering
   - Sentiment analysis
   
   Example:
   Input: "The [MASK] sat on the mat"
   BERT predicts: "cat"

2. GPT (Generative Pre-trained Transformer)
   Architecture: Decoder-only
   Training: Next word prediction
   Use Cases:
   - Text generation
   - Chatbots (ChatGPT)
   - Code generation (GitHub Copilot)
   - Creative writing
   
   Example:
   Input: "Once upon a time"
   GPT generates: "there was a brave knight..."

3. T5 (Text-to-Text Transfer Transformer)
   Architecture: Encoder-Decoder
   Training: Text-to-text format
   Use Cases:
   - Translation
   - Summarization
   - Question answering
   - Any NLP task
   
   Example:
   Input: "translate English to French: Hello"
   T5 outputs: "Bonjour"

4. BART (Bidirectional and Auto-Regressive Transformers)
   Architecture: Encoder-Decoder
   Training: Denoising autoencoder
   Use Cases:
   - Text summarization
   - Text generation
   - Translation
""")

# Step 4: How LLMs are Trained
print("\n" + "=" * 70)
print("STEP 4: TRAINING LARGE LANGUAGE MODELS")
print("=" * 70)

print("""
TRAINING PROCESS:

1. PRE-TRAINING (Unsupervised):
   - Train on massive text corpus (billions of words)
   - Learn language patterns, grammar, facts
   - Takes weeks/months on powerful GPUs
   - Cost: Millions of dollars
   
   Data Sources:
   - Books
   - Wikipedia
   - Web pages
   - Code repositories
   - Scientific papers

2. FINE-TUNING (Supervised):
   - Train on specific task data
   - Adapt model for particular use case
   - Takes hours/days
   - Cost: Thousands of dollars
   
   Examples:
   - Medical text → Medical chatbot
   - Legal documents → Legal assistant
   - Customer reviews → Sentiment analyzer

3. REINFORCEMENT LEARNING (RLHF):
   - Human feedback improves responses
   - Makes model more helpful and safe
   - Used in ChatGPT
   
   Process:
   - Generate multiple responses
   - Humans rank them
   - Model learns from rankings

MODEL SIZES:
- Small: 125M parameters (BERT-base)
- Medium: 1.5B parameters (GPT-2)
- Large: 175B parameters (GPT-3)
- Huge: 540B parameters (PaLM)
- Massive: 1.7T parameters (GPT-4 estimated)
""")

# Step 5: Practical Example with Transformers
print("\n" + "=" * 70)
print("STEP 5: USING TRANSFORMERS (CONCEPTUAL)")
print("=" * 70)

print("""
USING HUGGING FACE TRANSFORMERS LIBRARY:

Installation:
```bash
pip install transformers torch
```

Example 1: Text Classification (Sentiment Analysis)
```python
from transformers import pipeline

# Load pre-trained model
classifier = pipeline("sentiment-analysis")

# Analyze sentiment
result = classifier("I love this product!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

Example 2: Text Generation
```python
from transformers import pipeline

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")

# Generate text
result = generator("Once upon a time", max_length=50)
print(result[0]['generated_text'])
```

Example 3: Question Answering
```python
from transformers import pipeline

# Load QA model
qa = pipeline("question-answering")

# Ask question
context = "Paris is the capital of France. It has a population of 2 million."
question = "What is the capital of France?"

result = qa(question=question, context=context)
print(result['answer'])  # Output: Paris
```

Example 4: Translation
```python
from transformers import pipeline

# Load translation model
translator = pipeline("translation_en_to_fr")

# Translate
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
# Output: Bonjour, comment allez-vous?
```

Example 5: Named Entity Recognition
```python
from transformers import pipeline

# Load NER model
ner = pipeline("ner")

# Extract entities
text = "Apple Inc. is located in Cupertino, California."
result = ner(text)

for entity in result:
    print(f"{entity['word']}: {entity['entity']}")
# Output:
# Apple: ORG
# Cupertino: LOC
# California: LOC
```
""")

# Step 6: Applications and Use Cases
print("\n" + "=" * 70)
print("STEP 6: REAL-WORLD APPLICATIONS")
print("=" * 70)

print("""
1. CHATBOTS & VIRTUAL ASSISTANTS:
   - ChatGPT, Google Bard, Claude
   - Customer service bots
   - Personal assistants

2. CONTENT CREATION:
   - Article writing
   - Code generation (GitHub Copilot)
   - Email drafting
   - Social media posts

3. TRANSLATION:
   - Google Translate
   - DeepL
   - Real-time conversation translation

4. SEARCH & INFORMATION RETRIEVAL:
   - Semantic search
   - Document summarization
   - Question answering systems

5. HEALTHCARE:
   - Medical diagnosis assistance
   - Drug discovery
   - Patient record analysis

6. EDUCATION:
   - Personalized tutoring
   - Essay grading
   - Learning content generation

7. BUSINESS:
   - Market analysis
   - Report generation
   - Email automation
   - Meeting summarization

8. CREATIVE ARTS:
   - Story writing
   - Poetry generation
   - Song lyrics
   - Script writing
""")

# Step 7: Limitations and Challenges
print("\n" + "=" * 70)
print("STEP 7: LIMITATIONS & CHALLENGES")
print("=" * 70)

print("""
LIMITATIONS:

1. HALLUCINATIONS:
   - Models can generate false information
   - Confidently state incorrect facts
   - Need fact-checking

2. BIAS:
   - Reflects biases in training data
   - Can perpetuate stereotypes
   - Requires careful monitoring

3. CONTEXT LENGTH:
   - Limited memory (2K-32K tokens)
   - Can't process very long documents
   - Loses context in long conversations

4. COMPUTATIONAL COST:
   - Expensive to train and run
   - Requires powerful hardware
   - High energy consumption

5. LACK OF REASONING:
   - No true understanding
   - Pattern matching, not thinking
   - Struggles with complex logic

6. PRIVACY CONCERNS:
   - May memorize training data
   - Risk of data leakage
   - Need for data protection

FUTURE DIRECTIONS:
- Multimodal models (text + images + audio)
- More efficient architectures
- Better reasoning capabilities
- Reduced bias and hallucinations
- Lower computational costs
""")

# Step 8: Key Metrics
print("\n" + "=" * 70)
print("STEP 8: EVALUATION METRICS")
print("=" * 70)

print("""
COMMON METRICS FOR LLMs:

1. PERPLEXITY:
   - Measures how well model predicts text
   - Lower is better
   - Formula: exp(average negative log-likelihood)

2. BLEU SCORE:
   - Measures translation quality
   - Compares to reference translations
   - Range: 0-100 (higher is better)

3. ROUGE SCORE:
   - Measures summarization quality
   - Compares overlap with reference
   - Types: ROUGE-1, ROUGE-2, ROUGE-L

4. ACCURACY:
   - For classification tasks
   - Percentage of correct predictions

5. F1 SCORE:
   - Balance of precision and recall
   - Good for imbalanced datasets

6. HUMAN EVALUATION:
   - Fluency: How natural is the text?
   - Coherence: Does it make sense?
   - Relevance: Is it on-topic?
   - Factuality: Is it correct?
""")

print("\n" + "=" * 70)
print("✓ TRANSFORMERS & LLM OVERVIEW COMPLETE!")
print("=" * 70)

print("""
KEY TAKEAWAYS:

1. Transformers use attention mechanism to understand context
2. Different architectures for different tasks (BERT, GPT, T5)
3. Pre-training on massive data, then fine-tuning for specific tasks
4. Wide range of applications across industries
5. Still have limitations (hallucinations, bias, cost)
6. Future is multimodal and more efficient

NEXT STEPS:
- Try Hugging Face Transformers library
- Experiment with different models
- Fine-tune models for your use case
- Stay updated with latest research
""")
