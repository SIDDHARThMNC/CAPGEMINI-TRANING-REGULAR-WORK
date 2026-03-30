# University AI Helpdesk Chatbot
# Demonstrates Generative AI, LLMs, Tokenization, and Prompt Engineering
# Simple rule-based implementation (no external libraries needed)

import re

print("=" * 60)
print("UNIVERSITY AI HELPDESK CHATBOT")
print("=" * 60)

# Knowledge base
knowledge_base = {
    "hostel": {
        "keywords": ["hostel", "admission", "documents", "requirements"],
        "response": """For hostel admission, you need:
- Admission confirmation letter
- Government ID proof
- Two passport size photographs"""
    },
    "fees": {
        "keywords": ["fees", "payment", "pay", "money"],
        "response": """Fee payment options:
- Pay through the student dashboard
- Online methods: debit card, credit card, net banking"""
    },
    "exams": {
        "keywords": ["exam", "examination", "test", "schedule"],
        "response": """Examination Schedule:
- Semester exams usually start in December and May"""
    },
    "registration": {
        "keywords": ["register", "registration", "course", "enroll"],
        "response": """Course Registration:
- Registration happens online through the university portal
- Students must register before the semester deadline"""
    },
    "library": {
        "keywords": ["library", "book", "access"],
        "response": """Library Access:
- Students must carry their university ID card"""
    }
}

print("\nKnowledge Base Loaded")
print(f"Topics: {len(knowledge_base)}")
print("-" * 60)

# Tokenization function
def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    return tokens

# Intent matching function
def get_response(query):
    tokens = tokenize(query)
    
    for topic, data in knowledge_base.items():
        for keyword in data["keywords"]:
            if keyword in tokens:
                return data["response"]
    
    return "I'm sorry, I don't have information about that. Please contact the helpdesk."

# Sample student queries
queries = [
    "What documents do I need for hostel admission?",
    "How can I pay my university fees?",
    "When do semester exams start?",
    "How do I register for courses?"
]

print("\n" + "=" * 60)
print("CHATBOT RESPONSES:")
print("=" * 60)

for query in queries:
    print(f"\nStudent Query: {query}")
    
    # Tokenize query
    tokens = tokenize(query)
    print(f"Tokens ({len(tokens)}): {tokens[:5]}...")
    
    # Get response using prompt engineering
    response = get_response(query)
    print(f"\nChatbot Response:\n{response}")
    print("-" * 60)

print("\n" + "=" * 60)
print("CHATBOT DEMO COMPLETE")
print("=" * 60)
print("\nKey Concepts Demonstrated:")
print("- Tokenization: Breaking text into words")
print("- Intent Recognition: Matching keywords to topics")
print("- Prompt Engineering: Structured knowledge base")
print("- Response Generation: Rule-based AI assistant")
print("=" * 60)
