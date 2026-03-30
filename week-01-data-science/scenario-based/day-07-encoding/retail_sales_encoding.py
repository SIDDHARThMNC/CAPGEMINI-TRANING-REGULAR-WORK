"""
SCENARIO: Retail Sales Data Encoding

You are working as a data analyst for a retail company. The company wants to build 
a machine learning model to predict sales performance based on product type and region.

PROBLEM:
The dataset contains categorical variables (Product and Region) that machine learning 
algorithms cannot directly process - they need numerical inputs.

FEATURES:
- Product: Electronics, Clothing, Furniture
- Region: North, South, East, West

SOLUTION:
Apply different encoding techniques to convert categorical data into numerical format.

ENCODING TECHNIQUES:
1. Label Encoding - Assigns numeric codes to categories
2. One-Hot Encoding - Creates binary columns for each category
3. Ordinal Encoding - For ordered categories (if applicable)

QUESTIONS:
Part A: Why can't machine learning models work directly with text categories?
Part B: When should you use Label Encoding vs One-Hot Encoding?
Part C: What is the "dummy variable trap" and how do we avoid it?
Part D: How does encoding affect model performance and interpretation?
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("RETAIL SALES DATA ENCODING")
print("="*70)

data = pd.DataFrame({
    'Product': ['Electronics', 'Clothing', 'Furniture', 'Electronics', 'Clothing', 
                'Furniture', 'Electronics', 'Clothing', 'Furniture', 'Electronics'],
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 
               'North', 'South'],
    'Sales': [25000, 15000, 30000, 22000, 18000, 28000, 26000, 16000, 32000, 24000]
})

print("\nOriginal Dataset:")
print(data)

print("\nDataset Info:")
print(f"Total Records: {len(data)}")
print(f"Product Categories: {data['Product'].unique()}")
print(f"Region Categories: {data['Region'].unique()}")

print("\n" + "="*70)
print("METHOD 1: LABEL ENCODING")
print("="*70)

label_encoder_product = LabelEncoder()
label_encoder_region = LabelEncoder()

data['Product_Label'] = label_encoder_product.fit_transform(data['Product'])
data['Region_Label'] = label_encoder_region.fit_transform(data['Region'])

print("\nLabel Encoded Data:")
print(data[['Product', 'Product_Label', 'Region', 'Region_Label', 'Sales']])

print("\nLabel Encoding Mappings:")
print("\nProduct Mapping:")
for i, label in enumerate(label_encoder_product.classes_):
    print(f"  {label} → {i}")

print("\nRegion Mapping:")
for i, label in enumerate(label_encoder_region.classes_):
    print(f"  {label} → {i}")

print("\n" + "="*70)
print("METHOD 2: ONE-HOT ENCODING")
print("="*70)

onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')

product_region_encoded = onehot_encoder.fit_transform(data[['Product', 'Region']])

onehot_columns = onehot_encoder.get_feature_names_out(['Product', 'Region'])
onehot_df = pd.DataFrame(product_region_encoded, columns=onehot_columns)

print("\nOne-Hot Encoded Features:")
print(onehot_df)

print("\nOne-Hot Encoding Mappings:")
print(f"Original Features: Product, Region")
print(f"Encoded Features: {list(onehot_columns)}")
print(f"Note: drop='first' removes one category to avoid multicollinearity")

data_with_onehot = pd.concat([data[['Product', 'Region', 'Sales']], onehot_df], axis=1)

print("\nFull Dataset with One-Hot Encoding:")
print(data_with_onehot)

print("\n" + "="*70)
print("METHOD 3: PANDAS GET_DUMMIES")
print("="*70)

dummies_df = pd.get_dummies(data[['Product', 'Region']], drop_first=True)

print("\nPandas get_dummies Result:")
print(dummies_df)

data_with_dummies = pd.concat([data[['Sales']], dummies_df], axis=1)

print("\nDataset Ready for ML Model:")
print(data_with_dummies)

print("\n" + "="*70)
print("COMPARISON: LABEL VS ONE-HOT ENCODING")
print("="*70)

comparison_data = {
    'Encoding Type': ['Label Encoding', 'One-Hot Encoding'],
    'Features Created': [2, len(onehot_columns)],
    'Preserves Independence': ['No', 'Yes'],
    'Best For': ['Tree-based models', 'Linear models'],
    'Dimensionality': ['Low', 'Higher']
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

fig = plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
product_counts = data['Product'].value_counts()
plt.bar(product_counts.index, product_counts.values, color=['#3498db', '#e74c3c', '#2ecc71'], 
        edgecolor='black', linewidth=2)
plt.xlabel('Product Category', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Product Distribution', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 2)
region_counts = data['Region'].value_counts()
colors_region = ['#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
plt.bar(region_counts.index, region_counts.values, color=colors_region, 
        edgecolor='black', linewidth=2)
plt.xlabel('Region', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Region Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 3)
avg_sales_product = data.groupby('Product')['Sales'].mean().sort_values(ascending=False)
plt.barh(avg_sales_product.index, avg_sales_product.values, 
         color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
plt.xlabel('Average Sales', fontweight='bold')
plt.ylabel('Product', fontweight='bold')
plt.title('Average Sales by Product', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 3, 4)
avg_sales_region = data.groupby('Region')['Sales'].mean().sort_values(ascending=False)
plt.barh(avg_sales_region.index, avg_sales_region.values, 
         color=['#9b59b6', '#f39c12', '#1abc9c', '#e67e22'], edgecolor='black', linewidth=2)
plt.xlabel('Average Sales', fontweight='bold')
plt.ylabel('Region', fontweight='bold')
plt.title('Average Sales by Region', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 3, 5)
label_data = data[['Product_Label', 'Region_Label']].head(5)
x = np.arange(len(label_data))
width = 0.35
plt.bar(x - width/2, label_data['Product_Label'], width, label='Product', 
        color='steelblue', edgecolor='black')
plt.bar(x + width/2, label_data['Region_Label'], width, label='Region', 
        color='coral', edgecolor='black')
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Encoded Value', fontweight='bold')
plt.title('Label Encoding Example (First 5 Rows)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 6)
encoding_comparison = pd.DataFrame({
    'Method': ['Label\nEncoding', 'One-Hot\nEncoding'],
    'Features': [2, len(onehot_columns)]
})
plt.bar(encoding_comparison['Method'], encoding_comparison['Features'], 
        color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
plt.ylabel('Number of Features', fontweight='bold')
plt.title('Feature Count Comparison', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(encoding_comparison['Features']):
    plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold', fontsize=12)

plt.suptitle('Retail Sales Data Encoding Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Day_7_Mar02_Encoding/retail_sales_encoding_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved")

plt.show()

data.to_csv('Day_7_Mar02_Encoding/retail_sales_label_encoded.csv', index=False)
data_with_onehot.to_csv('Day_7_Mar02_Encoding/retail_sales_onehot_encoded.csv', index=False)

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

print("\nPart A: Why can't ML models work with text categories?")
print("  - ML algorithms perform mathematical operations (addition, multiplication)")
print("  - Text cannot be used in equations: 'Electronics' + 'Clothing' = undefined")
print("  - Models need numerical features to calculate distances, weights, gradients")
print("  - Encoding converts text into numbers while preserving information")

print("\nPart B: When to use Label vs One-Hot Encoding?")
print("  Label Encoding:")
print("    - Tree-based models (Decision Tree, Random Forest, XGBoost)")
print("    - When categories have natural order (Low, Medium, High)")
print("    - Reduces dimensionality (1 column per feature)")
print("  One-Hot Encoding:")
print("    - Linear models (Linear Regression, Logistic Regression)")
print("    - When categories are independent (no natural order)")
print("    - Prevents false ordinal relationships")

print("\nPart C: What is the dummy variable trap?")
print("  - Problem: Perfect multicollinearity in linear models")
print("  - Example: If Product_Electronics=0 and Product_Clothing=0, then Product_Furniture=1")
print("  - One category is perfectly predictable from others")
print("  - Solution: Drop one category (drop='first' or drop_first=True)")
print("  - Result: n categories → n-1 encoded columns")

print("\nPart D: How encoding affects model performance?")
print("  - Correct encoding improves accuracy and training speed")
print("  - Wrong encoding (Label for linear models) creates false relationships")
print("  - One-Hot increases dimensionality - may cause overfitting with many categories")
print("  - Interpretation: One-Hot coefficients show impact of each category")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"\n1. Original Features: 2 categorical (Product, Region)")
print(f"2. Label Encoding: 2 numerical features")
print(f"3. One-Hot Encoding: {len(onehot_columns)} binary features (with drop='first')")
print(f"4. Average Sales by Product: {avg_sales_product.index[0]} (${avg_sales_product.values[0]:,.0f})")
print(f"5. Average Sales by Region: {avg_sales_region.index[0]} (${avg_sales_region.values[0]:,.0f})")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
