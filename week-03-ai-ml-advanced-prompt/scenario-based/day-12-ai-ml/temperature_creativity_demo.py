# Temperature Effects on AI Creativity
# Demonstrates how temperature affects text generation

print("=" * 60)
print("EXPLORING CREATIVITY IN AI WRITING")
print("=" * 60)

prompt = "The future of Artificial Intelligence"

# Different temperature outputs
outputs = {
    0.2: "will bring advancements in healthcare, education, and business.",
    0.5: "holds promise for transforming industries and improving quality of life.",
    0.8: "dances with uncertainty, weaving stories of machines dreaming and societies reborn.",
    1.0: "sparkles with infinite possibilities, where silicon minds paint tomorrow's canvas."
}

print(f"\nPrompt: \"{prompt}\"\n")
print("-" * 60)

for temp, continuation in outputs.items():
    print(f"\nTemperature = {temp}")
    if temp <= 0.3:
        print("(Low randomness - Predictable & Conservative)")
    elif temp <= 0.6:
        print("(Medium randomness - Balanced)")
    else:
        print("(High randomness - Creative & Imaginative)")
    
    print(f"\nOutput: \"{prompt} {continuation}\"")
    print("-" * 60)

print("\n" + "=" * 60)
print("SCENARIO: SCRIPTWRITING ASSISTANT FOR FILM PRODUCTION")
print("=" * 60)

print("\nTemperature = 0.2 (Documentary Script):")
print("\"The future of Artificial Intelligence will improve industries,")
print("enhance productivity, and reshape education.\"")
print("→ Useful for serious, factual narration")

print("\nTemperature = 0.8 (Sci-Fi Creative Dialogue):")
print("\"The future of Artificial Intelligence dances with uncertainty,")
print("weaving stories of machines dreaming and societies reborn.\"")
print("→ Perfect for creative dialogue or speculative storytelling")

print("\n" + "=" * 60)
