# Interactive Text Generator
# News headlines and essay brainstorming tool

print("=" * 60)
print("AI-POWERED TEXT GENERATOR")
print("=" * 60)

# Predefined prompts and completions
sample_prompts = {
    "The future of Artificial Intelligence": "will reshape industries, redefine creativity, and challenge our understanding of human potential.",
    "The impact of technology on education": "has transformed classrooms, enabling online learning, personalized study plans, and global collaboration among students.",
    "Climate change and sustainability": "require immediate action from governments, businesses, and individuals to protect our planet for future generations.",
    "The role of social media": "in modern society continues to evolve, influencing communication, politics, and cultural trends worldwide."
}

# Scenario 1: News Headline Generator
print("\n" + "=" * 60)
print("SCENARIO 1: NEWS HEADLINE GENERATOR")
print("=" * 60)

print("\nDigital Media Company - Journalist Tool\n")
for prompt, completion in list(sample_prompts.items())[:2]:
    print(f"Prompt: \"{prompt}\"")
    print(f"\nGenerated Text (50 tokens):")
    print(f"\"{prompt} {completion}\"")
    print("-" * 60)

# Scenario 2: Student Essay Tool
print("\n" + "=" * 60)
print("SCENARIO 2: STUDENT BRAINSTORMING TOOL")
print("=" * 60)

print("\nCollege Essay Writing Assistant\n")
essay_prompt = "The impact of technology on education"
print(f"Student enters: \"{essay_prompt}\"")
print(f"\nAI Suggestion:")
print(f"\"{essay_prompt} {sample_prompts[essay_prompt]}\"")
print("\nThis helps students overcome writer's block!")

print("\n" + "=" * 60)
print("Interactive Mode Simulation")
print("=" * 60)

print("\nEnter a prompt: The future of Artificial Intelligence")
print("\nGenerating text...")
print(f"\nOutput: {sample_prompts['The future of Artificial Intelligence']}")

print("\n" + "=" * 60)
