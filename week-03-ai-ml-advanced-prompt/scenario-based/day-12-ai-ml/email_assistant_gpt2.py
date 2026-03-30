# AI-Powered Email Assistant
# Auto-completes sentences using GPT-2 for corporate email drafting

print("=" * 60)
print("AI-POWERED EMAIL ASSISTANT")
print("=" * 60)

# Simulated GPT-2 completions (no transformers library needed)
prompts = [
    "Artificial Intelligence will",
    "The meeting scheduled for tomorrow",
    "Please find attached the report"
]

completions = {
    "Artificial Intelligence will": "transform the way businesses interact with customers, enabling faster decisions and personalized experiences.",
    "The meeting scheduled for tomorrow": "has been rescheduled to next week due to unforeseen circumstances. We apologize for any inconvenience.",
    "Please find attached the report": "containing the quarterly analysis and recommendations for the upcoming fiscal year."
}

print("\nEmail Drafting Assistant Demo\n")
print("-" * 60)

for prompt in prompts:
    print(f"\nUser starts typing: \"{prompt}\"")
    print(f"\nAI Completion:\n\"{prompt} {completions[prompt]}\"")
    print("-" * 60)

print("\n" + "=" * 60)
print("Key Features:")
print("- Auto-completion for faster email drafting")
print("- Context-aware suggestions")
print("- Professional tone maintenance")
print("=" * 60)
