# Game Storyline Generator
# Generates quest ideas and dialogue for fantasy RPG games

print("=" * 60)
print("GAME STORYLINE GENERATOR")
print("=" * 60)

# Story prompts and continuations
story_prompts = [
    "The hero enters the ancient forest and discovers",
    "In the depths of the dragon's lair",
    "The mysterious stranger whispers"
]

continuations = {
    "The hero enters the ancient forest and discovers": "a hidden village where the people whisper of a cursed artifact buried beneath the old ruins.",
    "In the depths of the dragon's lair": "lies a magical sword that glows with an ethereal light, waiting for a worthy champion to claim it.",
    "The mysterious stranger whispers": "secrets of an ancient prophecy that foretells the return of the Shadow King and the chosen one who will stop him."
}

print("\nFantasy RPG Quest Generator\n")
print("-" * 60)

for prompt in story_prompts:
    print(f"\nDeveloper Prompt: \"{prompt}\"")
    print(f"\nGPT-2 Story Continuation:\n\"{prompt} {continuations[prompt]}\"")
    print("-" * 60)

print("\n" + "=" * 60)
print("Use Cases:")
print("- Quick quest idea generation")
print("- NPC dialogue brainstorming")
print("- Plot twist suggestions")
print("=" * 60)
