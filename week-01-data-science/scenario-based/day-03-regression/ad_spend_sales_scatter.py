import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
ad_spend = np.random.uniform(2000, 18000, 50)
sales_revenue = ad_spend * 2.5 + np.random.normal(0, 5000, 50)

print("=" * 70)
print("ADVERTISING SPEND VS SALES REVENUE ANALYSIS")
print("=" * 70)
print()

print("Dataset Statistics:")
print(f"Total Campaigns: {len(ad_spend)}")
print(f"Ad Spend Range: ${np.min(ad_spend):.2f} - ${np.max(ad_spend):.2f}")
print(f"Average Ad Spend: ${np.mean(ad_spend):.2f}")
print(f"Sales Revenue Range: ${np.min(sales_revenue):.2f} - ${np.max(sales_revenue):.2f}")
print(f"Average Sales Revenue: ${np.mean(sales_revenue):.2f}")
print()

correlation = np.corrcoef(ad_spend, sales_revenue)[0, 1]
print(f"Correlation Coefficient: {correlation:.3f}")
print()

z = np.polyfit(ad_spend, sales_revenue, 1)
p = np.poly1d(z)
trend_line = p(ad_spend)

plt.figure(figsize=(12, 7))
plt.scatter(ad_spend, sales_revenue, color='#3498db', s=100, alpha=0.6, edgecolors='black', label='Campaign Data')
plt.plot(sorted(ad_spend), p(sorted(ad_spend)), color='red', linewidth=2, linestyle='--', label=f'Trend Line (y = {z[0]:.2f}x + {z[1]:.2f})')
plt.xlabel('Advertising Spend ($)', fontsize=12)
plt.ylabel('Sales Revenue ($)', fontsize=12)
plt.title('Relationship Between Advertising Spend and Sales Revenue', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ad_spend_sales_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("TREND LINE ANALYSIS")
print("=" * 70)
print()

print(f"Trend Line Equation: Sales = {z[0]:.2f} × Ad_Spend + {z[1]:.2f}")
print(f"Slope: {z[0]:.2f} (For every $1 increase in ad spend, sales increase by ${z[0]:.2f})")
print(f"Intercept: ${z[1]:.2f}")
print()

roi = (z[0] - 1) * 100
print(f"Return on Investment (ROI): {roi:.1f}%")
print()

print("=" * 70)
print("CHART INTERPRETATION")
print("=" * 70)
print()

print("1. Relationship Pattern:")
if correlation > 0.7:
    print("   - Strong positive correlation between ad spend and sales")
elif correlation > 0.4:
    print("   - Moderate positive correlation between ad spend and sales")
else:
    print("   - Weak positive correlation between ad spend and sales")
print(f"   - Correlation coefficient: {correlation:.3f}")
print("   - Higher ad spend generally leads to higher sales")
print()

print("2. Trend Analysis:")
print("   - The red trend line shows the overall pattern")
print("   - Data points cluster around the trend line")
print("   - Some variation exists due to other factors")
print("   - Overall upward trend is clearly visible")
print()

print("3. Business Insights:")
print(f"   - Every $1,000 spent on ads generates ~${z[0]*1000:.2f} in sales")
print(f"   - ROI of {roi:.1f}% indicates profitable advertising")
print("   - Consistent pattern across different spending levels")
print()

print("=" * 70)
print("RECOMMENDATIONS FOR MARKETING STRATEGY")
print("=" * 70)
print()

print("1. Budget Allocation:")
print("   - Increase ad spend for higher revenue potential")
print("   - Current ROI justifies continued investment")
print("   - Consider scaling up successful campaigns")
print()

print("2. Optimization Opportunities:")
print("   - Analyze outliers (points far from trend line)")
print("   - Identify what makes high-performing campaigns successful")
print("   - Replicate strategies from campaigns above the trend line")
print("   - Investigate underperforming campaigns below the line")
print()

print("3. Risk Management:")
print("   - Diversify ad channels to reduce risk")
print("   - Test different spending levels systematically")
print("   - Monitor ROI continuously for changes")
print("   - Set maximum spend limits based on diminishing returns")
print()

print("4. Future Actions:")
print("   - Collect more data on campaign types and channels")
print("   - Track seasonal variations in effectiveness")
print("   - A/B test different ad creatives and messaging")
print("   - Consider non-linear models for better predictions")
print()

low_spend_avg = np.mean(sales_revenue[ad_spend < 10000])
high_spend_avg = np.mean(sales_revenue[ad_spend >= 10000])
print("Performance Comparison:")
print(f"   - Low Spend (<$10k): Average Sales = ${low_spend_avg:.2f}")
print(f"   - High Spend (≥$10k): Average Sales = ${high_spend_avg:.2f}")
print(f"   - Difference: ${high_spend_avg - low_spend_avg:.2f}")
print()

print("Chart saved as 'ad_spend_sales_scatter.png'")
