import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
exam_scores = np.random.normal(75, 10, 1000)

print("=" * 70)
print("EXAM SCORES DISTRIBUTION ANALYSIS")
print("=" * 70)
print()

print("Dataset Statistics:")
print(f"Total Students: {len(exam_scores)}")
print(f"Average Score: {np.mean(exam_scores):.2f}")
print(f"Standard Deviation: {np.std(exam_scores):.2f}")
print(f"Minimum Score: {np.min(exam_scores):.2f}")
print(f"Maximum Score: {np.max(exam_scores):.2f}")
print(f"Median Score: {np.median(exam_scores):.2f}")
print()

plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(exam_scores, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(exam_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(exam_scores):.2f}')
plt.axvline(np.median(exam_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(exam_scores):.2f}')
plt.xlabel('Exam Scores (Marks)', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.title('Distribution of Exam Scores (1,000 Students)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('exam_scores_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("SCORE DISTRIBUTION BREAKDOWN")
print("=" * 70)
print()

ranges = [
    (0, 50, "Below 50 (Fail)"),
    (50, 60, "50-60 (Pass)"),
    (60, 70, "60-70 (Average)"),
    (70, 80, "70-80 (Good)"),
    (80, 90, "80-90 (Very Good)"),
    (90, 100, "90-100 (Excellent)")
]

for low, high, label in ranges:
    count = np.sum((exam_scores >= low) & (exam_scores < high))
    percentage = (count / len(exam_scores)) * 100
    print(f"{label}: {count} students ({percentage:.1f}%)")
print()

print("=" * 70)
print("CHART INTERPRETATION")
print("=" * 70)
print()

print("1. Distribution Shape:")
print("   - The histogram shows a bell-shaped (normal) distribution")
print("   - Scores are symmetrically distributed around the mean (75)")
print("   - This indicates a well-balanced exam difficulty level")
print()

print("2. Student Performance Clustering:")
print("   - Most students scored between 65-85 marks")
print("   - Strong clustering around the average (75 marks)")
print("   - Standard deviation of ~10 shows moderate variation")
print("   - Few outliers on both extremes")
print()

print("3. Performance Categories:")
within_1_std = np.sum((exam_scores >= 65) & (exam_scores <= 85))
within_2_std = np.sum((exam_scores >= 55) & (exam_scores <= 95))
print(f"   - Within 1 std dev (65-85): {within_1_std} students ({(within_1_std/1000)*100:.1f}%)")
print(f"   - Within 2 std dev (55-95): {within_2_std} students ({(within_2_std/1000)*100:.1f}%)")
print("   - Follows 68-95-99.7 rule (normal distribution)")
print()

print("=" * 70)
print("RECOMMENDATIONS FOR TEACHING METHODS")
print("=" * 70)
print()

print("1. For High Performers (>85):")
print("   - Provide advanced materials and challenges")
print("   - Offer enrichment activities and research projects")
print("   - Consider peer tutoring opportunities")
print()

print("2. For Average Performers (65-85):")
print("   - Current teaching methods are effective")
print("   - Maintain current curriculum difficulty")
print("   - Provide optional practice materials")
print()

print("3. For Struggling Students (<65):")
print("   - Identify specific weak areas")
print("   - Offer additional tutoring sessions")
print("   - Provide supplementary learning resources")
print("   - Consider one-on-one mentoring")
print()

print("4. Overall Teaching Strategy:")
print("   - Normal distribution indicates fair assessment")
print("   - Exam difficulty is appropriate for the class")
print("   - Consider differentiated instruction for extremes")
print("   - Use formative assessments to track progress")
print("   - Implement early intervention for low performers")
print()

print("Chart saved as 'exam_scores_histogram.png'")
