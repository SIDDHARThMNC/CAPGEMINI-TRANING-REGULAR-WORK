import matplotlib.pyplot as plt

sections = ['North Wing', 'South Wing', 'East Wing', 'West Wing']
books_checked_out = [2000, 1500, 1800, 2200]

print("=" * 70)
print("LIBRARY BOOKS CHECKOUT ANALYSIS")
print("=" * 70)
print()

print("Books Checked Out by Section:")
for section, count in zip(sections, books_checked_out):
    print(f"{section}: {count} books")
print()

plt.figure(figsize=(10, 6))
bars = plt.bar(sections, books_checked_out, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], width=0.6)
plt.xlabel('Library Section', fontsize=12)
plt.ylabel('Number of Books Checked Out', fontsize=12)
plt.title('Library Books Checkout by Section (Last Semester)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('library_books_checkout_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("CHART INTERPRETATION")
print("=" * 70)
print()

highest_section = sections[books_checked_out.index(max(books_checked_out))]
highest_count = max(books_checked_out)
lowest_section = sections[books_checked_out.index(min(books_checked_out))]
lowest_count = min(books_checked_out)

print(f"Highest Checkouts: {highest_section} with {highest_count} books")
print(f"Lowest Checkouts: {lowest_section} with {lowest_count} books")
print()

difference = highest_count - lowest_count
print(f"Difference between highest and lowest: {difference} books")
print(f"Percentage difference: {(difference/lowest_count)*100:.1f}%")
print()

total_books = sum(books_checked_out)
print("Section-wise Distribution:")
for section, count in zip(sections, books_checked_out):
    percentage = (count/total_books)*100
    print(f"{section}: {percentage:.1f}% of total checkouts")
print()

print("=" * 70)
print("INSIGHTS FOR RESOURCE ALLOCATION")
print("=" * 70)
print()

print("1. West Wing Performance:")
print("   - Highest activity with 2,200 checkouts")
print("   - Consider expanding collection in this section")
print("   - May need additional staff during peak hours")
print()

print("2. South Wing Concerns:")
print("   - Lowest activity with 1,500 checkouts")
print("   - Review collection relevance and quality")
print("   - Consider relocating popular books here")
print("   - Improve signage and accessibility")
print()

print("3. North Wing & East Wing:")
print("   - Moderate activity (2,000 and 1,800 respectively)")
print("   - Maintain current resource levels")
print("   - Monitor trends for future adjustments")
print()

print("4. Overall Recommendations:")
print("   - Redistribute staff based on section traffic")
print("   - Allocate more budget to high-performing sections")
print("   - Investigate reasons for South Wing underperformance")
print("   - Consider student surveys for collection improvement")
print()

print("Chart saved as 'library_books_checkout_chart.png'")