from scipy.stats import binom

n = 10
p = 0.25

print("=" * 70)
print("MCQ TEST PROBABILITY USING BINOMIAL DISTRIBUTION")
print("=" * 70)
print()

print("Problem:")
print("- 10 questions MCQ test")
print("- 4 options each (A, B, C, D)")
print("- Only 1 correct answer per question")
print("- You guess randomly")
print()

print("=" * 70)
print("BINOMIAL DISTRIBUTION PARAMETERS")
print("=" * 70)
print()

print(f"n = {n} (number of trials/questions)")
print(f"p = {p} (probability of success/correct guess)")
print()

print("=" * 70)
print("QUESTION 1: EXACTLY 3 CORRECT")
print("=" * 70)
print()

k_exactly = 3
prob_exactly_3 = binom.pmf(k_exactly, n, p)

print(f"P(X = {k_exactly}) = {prob_exactly_3:.4f}")
print(f"P(X = {k_exactly}) = {prob_exactly_3*100:.2f}%")
print()
print(f"Answer: Your chances of getting exactly 3 correct: {prob_exactly_3*100:.2f}%")
print()

print("=" * 70)
print("QUESTION 2: AT LEAST 5 CORRECT")
print("=" * 70)
print()

k_atleast = 5
prob_atleast_5 = 1 - binom.cdf(k_atleast - 1, n, p)

print(f"P(X ≥ {k_atleast}) = 1 - P(X ≤ {k_atleast - 1})")
print(f"P(X ≥ {k_atleast}) = 1 - {binom.cdf(k_atleast - 1, n, p):.4f}")
print(f"P(X ≥ {k_atleast}) = {prob_atleast_5:.4f}")
print(f"P(X ≥ {k_atleast}) = {prob_atleast_5*100:.2f}%")
print()
print(f"Answer: Your chances of getting at least 5 correct: {prob_atleast_5*100:.2f}%")
print()

print("=" * 70)
print("DETAILED BREAKDOWN")
print("=" * 70)
print()

print("At least 5 correct means: 5, 6, 7, 8, 9, or 10 correct")
print()
for i in range(5, n+1):
    prob = binom.pmf(i, n, p)
    print(f"P(X = {i:2d}) = {prob*100:5.2f}%")
print(f"                -------")
print(f"Total (≥5)  = {prob_atleast_5*100:5.2f}%")
print()

print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print()

print(f"Exactly 3 correct: {prob_exactly_3*100:.2f}% (approximately 1 in {int(1/prob_exactly_3)} chance)")
print(f"At least 5 correct: {prob_atleast_5*100:.2f}% (approximately 1 in {int(1/prob_atleast_5)} chance)")
print()

print("=" * 70)
print("PROBABILITY DISTRIBUTION FOR ALL OUTCOMES")
print("=" * 70)
print()

for i in range(n+1):
    prob = binom.pmf(i, n, p)
    bar = '█' * int(prob * 200)
    print(f"{i:2d} correct: {prob*100:5.2f}% {bar}")
print()

cumulative_3_or_less = binom.cdf(3, n, p)
cumulative_5_or_more = 1 - binom.cdf(4, n, p)

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print(f"P(X = 3)  = {prob_exactly_3*100:.2f}% (exactly 3 correct)")
print(f"P(X ≥ 5)  = {prob_atleast_5*100:.2f}% (at least 5 correct)")
print(f"P(X ≤ 3)  = {cumulative_3_or_less*100:.2f}% (3 or fewer correct)")
print()

expected_value = n * p
variance = n * p * (1 - p)
std_dev = variance ** 0.5

print("=" * 70)
print("STATISTICAL MEASURES")
print("=" * 70)
print()

print(f"Expected Value (Mean): {expected_value:.2f} correct answers")
print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
