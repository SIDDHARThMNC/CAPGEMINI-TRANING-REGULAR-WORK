# Scenario: Warehouse Robot Navigation
# Calculates distances between robot and items using multiple distance metrics

import math

print("=" * 60)
print("WAREHOUSE ROBOT NAVIGATION SYSTEM")
print("=" * 60)

# ── Distance Models ──────────────────────────────────────────
def euclidean(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def manhattan(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))

def chebyshev(p1, p2):
    return max(abs(a - b) for a, b in zip(p1, p2))

# ── Warehouse Setup ──────────────────────────────────────────
robot = (1, 2)

items = {
    "Item A - Electronics": (4, 6),
    "Item B - Groceries":   (2, 5),
    "Item C - Furniture":   (8, 1),
    "Item D - Clothing":    (3, 3),
    "Item E - Books":       (6, 7),
}

print(f"\nRobot Position : {robot}")
print(f"Warehouse Grid : 10x10")
print(f"\nItems to collect: {len(items)}")

# ── Scenario Example ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SCENARIO: Robot at (1,2) → Item at (4,6)")
print("=" * 60)

p1, p2 = (1, 2), (4, 6)
euc = euclidean(p1, p2)
man = manhattan(p1, p2)
che = chebyshev(p1, p2)

print(f"\nRobot Position : {p1}")
print(f"Item Position  : {p2}")
print(f"\nDistance Calculations:")
print(f"  Euclidean (straight-line) : √((4-1)² + (6-2)²) = √(9+16) = √25 = {euc:.4f} units")
print(f"  Manhattan (grid path)     : |4-1| + |6-2| = 3 + 4 = {man} units")
print(f"  Chebyshev (diagonal moves): max(|4-1|, |6-2|) = max(3,4) = {che} units")

# ── All Items Navigation ──────────────────────────────────────
print("\n" + "=" * 60)
print("FULL WAREHOUSE - NAVIGATION TO ALL ITEMS")
print("=" * 60)
print(f"\n{'Item':<28} | {'Position':<10} | {'Euclidean':>10} | {'Manhattan':>10} | {'Chebyshev':>10}")
print("-" * 75)

results = []
for item, pos in items.items():
    euc = euclidean(robot, pos)
    man = manhattan(robot, pos)
    che = chebyshev(robot, pos)
    results.append((item, pos, euc, man, che))
    print(f"{item:<28} | {str(pos):<10} | {euc:>10.4f} | {man:>10} | {che:>10}")

# ── Optimal Path (nearest item first) ────────────────────────
print("\n" + "=" * 60)
print("OPTIMAL PICKUP ORDER (Euclidean - nearest first)")
print("=" * 60)

sorted_items = sorted(results, key=lambda x: x[2])
current = robot
total_distance = 0

print(f"\nStart: Robot at {current}")
for i, (item, pos, *_) in enumerate(sorted_items, 1):
    dist = euclidean(current, pos)
    total_distance += dist
    print(f"  Step {i}: → {item} at {pos}  (distance: {dist:.4f})")
    current = pos

print(f"\nTotal Travel Distance: {total_distance:.4f} units")

# ── Model Comparison ──────────────────────────────────────────
print("\n" + "=" * 60)
print("DISTANCE MODEL COMPARISON")
print("=" * 60)
print("  Euclidean  -> Straight-line (flying drone path)")
print("  Manhattan  -> Grid-based (robot on warehouse floor)")
print("  Chebyshev  -> Diagonal moves allowed (chess king moves)")
print("=" * 60)
