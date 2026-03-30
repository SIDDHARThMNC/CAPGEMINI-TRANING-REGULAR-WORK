# Scenario: Basketball Player Evaluation
# Coach uses dot product & cosine similarity to match players to team needs

import math

print("=" * 60)
print("BASKETBALL PLAYER EVALUATION SYSTEM")
print("=" * 60)

features = ["Shooting", "Defense", "Passing"]

# Team requirements
team_needs = [2, 5, 1]

# Player roster
players = {
    "Player A (Scenario)": [3, 4, 2],
    "Player B - Shooter":  [5, 2, 1],
    "Player C - Defender": [1, 5, 2],
    "Player D - Playmaker":[2, 3, 5],
    "Player E - All-Round":[4, 4, 3],
}

# ── Models ───────────────────────────────────────────────────
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def magnitude(v):
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(a, b):
    dp = dot_product(a, b)
    mag = magnitude(a) * magnitude(b)
    return dp / mag if mag else 0.0

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

# ── Scenario Example ─────────────────────────────────────────
print(f"\nTeam Needs Vector  : {team_needs}")
print(f"  Shooting={team_needs[0]}, Defense={team_needs[1]}, Passing={team_needs[2]}")

pa = players["Player A (Scenario)"]
print(f"\nPlayer A Skill Vector: {pa}")
print(f"  Shooting={pa[0]}, Defense={pa[1]}, Passing={pa[2]}")

dp = dot_product(team_needs, pa)
print(f"\nDot Product Calculation:")
print(f"  a · b = (2×3) + (5×4) + (1×2)")
print(f"       = 6 + 20 + 2")
print(f"       = {dp}")
print(f"\nInterpretation: Score of {dp} → Player A is a {'GOOD' if dp >= 28 else 'MODERATE'} fit!")

# ── Full Evaluation ───────────────────────────────────────────
print("\n" + "=" * 60)
print("FULL ROSTER EVALUATION")
print("=" * 60)
print(f"\nTeam Needs: Shooting={team_needs[0]}, Defense={team_needs[1]}, Passing={team_needs[2]}\n")
print(f"{'Player':<25} | {'Vector':<14} | {'Dot Product':>11} | {'Cosine Sim':>10} | {'Euclidean':>9}")
print("-" * 80)

results = []
for name, skills in players.items():
    dp  = dot_product(team_needs, skills)
    cos = cosine_similarity(team_needs, skills)
    euc = euclidean_distance(team_needs, skills)
    results.append((name, skills, dp, cos, euc))
    print(f"{name:<25} | {str(skills):<14} | {dp:>11} | {cos:>10.4f} | {euc:>9.4f}")

# ── Rankings ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PLAYER RANKINGS")
print("=" * 60)

print("\nBy Dot Product (overall fit score):")
for i, (name, _, dp, _, _) in enumerate(sorted(results, key=lambda x: x[2], reverse=True), 1):
    print(f"  {i}. {name:<25} → Score: {dp}")

print("\nBy Cosine Similarity (directional alignment):")
for i, (name, _, _, cos, _) in enumerate(sorted(results, key=lambda x: x[3], reverse=True), 1):
    bar = "█" * int(cos * 20)
    print(f"  {i}. {name:<25} → {cos:.4f} {bar}")

print("\nBy Euclidean Distance (closest skill profile):")
for i, (name, _, _, _, euc) in enumerate(sorted(results, key=lambda x: x[4]), 1):
    print(f"  {i}. {name:<25} → Distance: {euc:.4f}")

# ── Best Pick ─────────────────────────────────────────────────
best_dp  = max(results, key=lambda x: x[2])
best_cos = max(results, key=lambda x: x[3])
best_euc = min(results, key=lambda x: x[4])

print("\n" + "=" * 60)
print("COACH'S RECOMMENDATION:")
print("=" * 60)
print(f"  Dot Product winner  : {best_dp[0]}  (score: {best_dp[2]})")
print(f"  Cosine Sim winner   : {best_cos[0]}  (score: {best_cos[3]:.4f})")
print(f"  Euclidean winner    : {best_euc[0]}  (distance: {best_euc[4]:.4f})")

names = {best_dp[0], best_cos[0], best_euc[0]}
if len(names) == 1:
    print(f"\n  ✅ All models agree: SIGN {best_dp[0]}!")
else:
    print(f"\n  Models suggest: {', '.join(names)}")
    print(f"  Coach's final pick (Dot Product): {best_dp[0]}")

print("\n" + "=" * 60)
print("Model Insights:")
print("  Dot Product    -> Weighted fit score (higher = better match)")
print("  Cosine Sim     -> Skill direction alignment (0 to 1)")
print("  Euclidean Dist -> Closest overall skill profile")
print("=" * 60)
