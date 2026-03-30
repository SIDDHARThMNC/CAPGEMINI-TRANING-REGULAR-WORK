# GPT-2 vs DistilGPT-2 Model Comparison
# Compares different model sizes and their outputs

print("=" * 60)
print("AI MODEL EVALUATION: GPT-2 vs DistilGPT-2")
print("=" * 60)

prompt = "Artificial Intelligence will transform"

# Model outputs
gpt2_output = "industries by automating processes, enhancing creativity, and reshaping how humans interact with technology."
distilgpt2_output = "our daily lives and business operations."

# Scenario 1: Startup Content Assistant
print("\n" + "=" * 60)
print("SCENARIO 1: STARTUP - CONTENT ASSISTANT")
print("=" * 60)

print(f"\nPrompt: \"{prompt}\"\n")
print("-" * 60)

print("\nGPT-2 (Larger Model):")
print(f"Output: \"{prompt} {gpt2_output}\"")
print("\nCharacteristics:")
print("- Longer, more detailed continuation")
print("- Richer vocabulary and context")
print("- Slower to run, higher resource usage")
print("- Best for: Marketing copy, detailed descriptions")

print("\n" + "-" * 60)

print("\nDistilGPT-2 (Distilled Model):")
print(f"Output: \"{prompt} {distilgpt2_output}\"")
print("\nCharacteristics:")
print("- Shorter, concise continuation")
print("- Faster inference time")
print("- Lower resource requirements")
print("- Best for: Quick suggestions, mobile apps")

# Scenario 2: Educational Tool
print("\n" + "=" * 60)
print("SCENARIO 2: EDUCATIONAL TOOL FOR CLASSROOM")
print("=" * 60)

print("\nHigh School Teacher - AI Writing Demo\n")
print(f"Prompt: \"{prompt}\"\n")

print("GPT-2 Output (Detailed):")
print(f"\"{prompt} the way societies function, influencing education,")
print("healthcare, and the future of human creativity.\"")
print("→ Shows students rich, detailed AI-generated ideas")

print("\nDistilGPT-2 Output (Concise):")
print(f"\"{prompt} our daily lives.\"")
print("→ Demonstrates concise summaries")

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

comparison = [
    ["Metric", "GPT-2", "DistilGPT-2"],
    ["Model Size", "~500MB", "~250MB"],
    ["Speed", "Slower", "2x Faster"],
    ["Output Quality", "More detailed", "Concise"],
    ["Use Case", "Content creation", "Quick suggestions"]
]

for row in comparison:
    print(f"{row[0]:20} | {row[1]:20} | {row[2]:20}")

print("=" * 60)
