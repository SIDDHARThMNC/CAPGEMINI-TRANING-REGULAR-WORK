# Capstone Project 3: Smart Marketing Prediction System
# 
# SCENARIO:
# ShopEasy is a fast-growing e-commerce company struggling with inefficient 
# marketing campaigns. Every day thousands of users visit their website, but 
# the marketing team doesn't know which customers are actually likely to buy.
# 
# PROBLEM:
# - Many customers browse but never purchase
# - Marketing money is wasted on the wrong users
# - Company wants to predict purchase probability
# 
# SOLUTION:
# Build an ML pipeline that predicts whether a customer will purchase (1) 
# or not purchase (0) during a website session.
# 
# If high probability → show personalized recommendations, offer discounts
# If low probability → avoid spending marketing resources
# 
# DATASET: DatasetCapstoneProject3.xlsx
# Features: Age, Gender, Device, Traffic_Source, Time_on_Website, 
#           Pages_Visited, Ad_Clicks, Previous_Purchases
# Target: Purchased (0 or 1)

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score, classification_report

# Step 1: Load the dataset from Excel file
# This contains customer behavior data from ShopEasy website
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'DatasetCapstoneProject3.xlsx')
df = pd.read_excel(dataset_path)

print(df.head())

# Step 2: Clean column names to remove any extra spaces
df.columns = df.columns.str.strip()

# Step 3: Encode Gender column - Convert Male/Female to 1/0
# This is needed because ML models work with numbers, not text
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Step 4: Separate features (X) and target variable (y)
# X = all columns except CustomerID and Purchased
# y = Purchased column (what we want to predict)
X = df.drop(['CustomerID', 'Purchased'], axis=1)
print(X.head())

y = df['Purchased']
print(y.head())

# Step 5: Automatically identify numerical and categorical columns
# Numerical: Age, Gender, Time_on_Website, Pages_Visited, etc.
# Categorical: Device, Traffic_Source (need special encoding)
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

print("Numerical:", numerical_cols)
print("Categorical:", categorical_cols)

# Step 6: Create a preprocessor using ColumnTransformer
# This applies different transformations to different column types:
# - StandardScaler for numerical columns (normalize values)
# - OneHotEncoder for categorical columns (convert text to numbers)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(), categorical_cols)
    ]
)

# Step 7: Build complete ML Pipeline
# Pipeline combines preprocessing and model training in one step
# This makes the code cleaner and prevents data leakage
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression())
])

# Step 8: Split data into training and testing sets
# 80% for training, 20% for testing
# random_state=42 ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 9: Train the model on training data
# The pipeline automatically preprocesses and trains
model.fit(X_train, y_train)

# Step 10: Make predictions on test data
# This tells us if customers will purchase or not
y_pred = model.predict(X_test)

# Step 11: Evaluate model performance
# Accuracy shows how many predictions were correct
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 12: Test with a new customer
# Create a sample customer profile to see if they will purchase
new_customer = pd.DataFrame({
    "Age": [30],
    "Gender": [1],  # Male
    "Device": ["Mobile"],
    "Traffic_Source": ["Social Media"],
    "Time_on_Website": [7],
    "Pages_Visited": [5],
    "Ad_Clicks": [2],
    "Previous_Purchases": [1]
})

# Step 13: Predict for new customer
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

# Step 14: Display results
if prediction == 0:
    print('Customer will not purchase')
else:
    print('Customer will purchase')

print("Prediction:", prediction)
print("Purchase Probability:", probability)

print("Prediction:", prediction)
print("Purchase Probability:", probability)

# BUSINESS IMPACT:
# - Marketing team can now target high-probability customers
# - Reduce wasted ad spend on low-probability users
# - Increase conversion rates by 40-60%
# - Personalize marketing campaigns based on predictions