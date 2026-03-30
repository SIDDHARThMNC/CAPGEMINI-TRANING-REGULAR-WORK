# Day 7 - Data Encoding Techniques (March 2, 2026)

## Overview
This folder contains projects focused on categorical data encoding techniques for machine learning.

## Project

### Retail Sales Data Encoding
**File:** `retail_sales_encoding.py`

**Scenario:**
A retail company wants to build a machine learning model to predict sales performance based on product type and region. The dataset contains categorical variables that need to be converted to numerical format.

**Features:**
- Product: Electronics, Clothing, Furniture
- Region: North, South, East, West
- Sales: Numerical target variable

**Encoding Techniques Covered:**
1. **Label Encoding** - Assigns numeric codes to categories
2. **One-Hot Encoding** - Creates binary columns for each category
3. **Pandas get_dummies** - Alternative one-hot encoding method

**Key Concepts:**
- Why ML models need numerical inputs
- Label Encoding vs One-Hot Encoding
- Dummy variable trap and multicollinearity
- Impact of encoding on model performance

**Questions Answered:**
- Part A: Why can't ML models work with text categories?
- Part B: When to use Label vs One-Hot Encoding?
- Part C: What is the dummy variable trap?
- Part D: How does encoding affect model performance?

**Visualizations:**
- Product and Region distribution
- Average sales by Product and Region
- Label encoding example
- Feature count comparison

**Output Files:**
- `retail_sales_label_encoded.csv` - Dataset with label encoding
- `retail_sales_onehot_encoded.csv` - Dataset with one-hot encoding
- `retail_sales_encoding_analysis.png` - Comprehensive visualization

## Learning Objectives
- Understand different categorical encoding techniques
- Learn when to apply each encoding method
- Avoid common pitfalls like dummy variable trap
- Prepare categorical data for machine learning models

## Technologies Used
- Python
- Pandas
- Scikit-learn (LabelEncoder, OneHotEncoder)
- Matplotlib & Seaborn
- NumPy

## How to Run
```bash
python retail_sales_encoding.py
```

## Key Takeaways
1. Label Encoding is best for tree-based models
2. One-Hot Encoding is best for linear models
3. Always drop one category to avoid multicollinearity
4. Correct encoding improves model accuracy and interpretability
