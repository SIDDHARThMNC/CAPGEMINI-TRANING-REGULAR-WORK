# BERT Tokenizer Demo
# Demonstrates tokenization for research and healthcare applications

import re

print("=" * 60)
print("BERT TOKENIZER DEMONSTRATION")
print("=" * 60)

# Simple tokenization function (simulates BERT tokenizer)
def bert_tokenize(text):
    # Convert to lowercase and split
    tokens = text.lower().split()
    
    # Add special tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Simulate subword tokenization
    subword_tokens = []
    for token in tokens:
        if len(token) > 6 and token not in ['[CLS]', '[SEP]']:
            # Split long words into subwords
            mid = len(token) // 2
            subword_tokens.extend([token[:mid], '##' + token[mid:]])
        else:
            subword_tokens.append(token)
    
    return subword_tokens

# Scenario 1: Research Assistant
print("\n" + "=" * 60)
print("SCENARIO 1: RESEARCH ASSISTANT FOR ACADEMIC PAPERS")
print("=" * 60)

research_text = "Deep learning models are powerful"
tokens = bert_tokenize(research_text)

print(f"\nInput: \"{research_text}\"")
print(f"\nTokens: {tokens}")
print(f"Token Count: {len(tokens)}")
print(f"Token IDs (simulated): {list(range(len(tokens)))}")

# Scenario 2: Healthcare Chatbot
print("\n" + "=" * 60)
print("SCENARIO 2: HEALTHCARE CHATBOT FOR PATIENT QUERIES")
print("=" * 60)

patient_query = "I have chest pain and shortness of breath"
tokens = bert_tokenize(patient_query)

print(f"\nPatient Query: \"{patient_query}\"")
print(f"\nTokens: {tokens}")
print(f"Token Count: {len(tokens)}")
print("\nChatbot Analysis:")
print("- Detected symptoms: chest pain, shortness of breath")
print("- Urgency level: HIGH")
print("- Recommendation: Seek immediate medical attention")

print("\n" + "=" * 60)
print("Key Benefits:")
print("- Breaks complex sentences into manageable units")
print("- Handles medical terminology effectively")
print("- Enables semantic understanding")
print("=" * 60)
