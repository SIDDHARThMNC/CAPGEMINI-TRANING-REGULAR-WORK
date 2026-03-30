import random

heads = 0
total = 10000

for i in range(total):
    result = random.randint(0, 1)
    if result == 1:
        heads = heads + 1

tails = total - heads

print("=" * 70)
print("COIN FLIP PROBABILITY")
print("=" * 70)
print()

print(f"Total Flips: {total}")
print(f"Heads: {heads}")
print(f"Tails: {tails}")
print()

print(f"P(Head) = {heads/total}")
print(f"P(Tail) = {tails/total}")
print()

print(f"P(Head) in percentage = {(heads/total)*100:.2f}%")
print(f"P(Tail) in percentage = {(tails/total)*100:.2f}%")
