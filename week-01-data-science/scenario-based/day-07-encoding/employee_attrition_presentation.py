# Employee Attrition Prediction Project
# Predicting which employees might leave the company

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("EMPLOYEE ATTRITION PREDICTION")
print("=" * 60)

# Step 1: Create sample employee data
print("\nStep 1: Creating Employee Dataset...")

n_employees = 250

# Generate employee features
employees = pd.DataFrame({
    'age': np.random.randint(22, 60, n_employees),
    'years_experience': np.random.randint(0, 35, n_employees),
    'department': np.random.choice(['Sales', 'IT', 'HR'], n_employees),
    'education_level': np.random.choice(['Bachelor', 'Master', 'PhD'], n_employees)
})

# Create attrition target based on realistic patterns
# Younger employees and those in Sales tend to leave more
attrition_score = (
    (employees['age'] < 30) * 0.25 +
    (employees['years_experience'] < 3) * 0.3 +
    (employees['department'] == 'Sales') * 0.2 +
    0.05
)
employees['attrition'] = (np.random.random(n_employees) < attrition_score).astype(int)

print(f"Total Employees: {len(employees)}")
print(f"Attrition Rate: {employees['attrition'].mean():.1%}")
print("\nSample Data:")
print(employees.head())

# Step 2: Prepare data for modeling
print("\n" + "=" * 60)
print("Step 2: Preparing Data for Machine Learning")
print("=" * 60)

# Separate features and target
X = employees[['age', 'years_experience', 'department', 'education_level']]
y = employees['attrition']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 3: Build ML Pipeline
print("\n" + "=" * 60)
print("Step 3: Building Machine Learning Pipeline")
print("=" * 60)

# Define which features are numeric and which are categorical
numeric_features = ['age', 'years_experience']
categorical_features = ['department', 'education_level']

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing steps
preprocessor = ColumnTransformer([
    ('scale_numbers', StandardScaler(), numeric_features),
    ('encode_categories', OneHotEncoder(drop='first'), categorical_features)
])

# Create complete pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

print("\nPipeline created successfully!")

# Step 4: Find best parameters using GridSearchCV
print("\n" + "=" * 60)
print("Step 4: Finding Best Model Parameters")
print("=" * 60)

# Define parameters to test
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__solver': ['liblinear', 'lbfgs']
}

print("Testing different parameter combinations:")
print(f"  C values: {param_grid['model__C']}")
print(f"  Solvers: {param_grid['model__solver']}")
print(f"  Total combinations: {len(param_grid['model__C']) * len(param_grid['model__solver'])}")

# Run grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("\nTraining models...")
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters Found:")
print(f"  C: {grid_search.best_params_['model__C']}")
print(f"  Solver: {grid_search.best_params_['model__solver']}")
print(f"  Best Score: {grid_search.best_score_:.2%}")

# Step 5: Evaluate the model
print("\n" + "=" * 60)
print("Step 5: Evaluating Model Performance")
print("=" * 60)

# Get best model and make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  Correctly predicted Stays: {cm[0,0]}")
print(f"  Correctly predicted Leaves: {cm[1,1]}")
print(f"  Incorrectly predicted: {cm[0,1] + cm[1,0]}")

# Step 6: Make predictions for new employees
print("\n" + "=" * 60)
print("Step 6: Predicting Attrition for New Employees")
print("=" * 60)

# Create new employee data
new_employees = pd.DataFrame({
    'age': [28, 45, 32],
    'years_experience': [2, 18, 8],
    'department': ['Sales', 'IT', 'HR'],
    'education_level': ['Bachelor', 'Master', 'PhD']
})

print("\nNew Employees:")
print(new_employees)

# Make predictions
predictions = best_model.predict(new_employees)
probabilities = best_model.predict_proba(new_employees)

print("\nPredictions:")
for i in range(len(new_employees)):
    result = "Will Leave" if predictions[i] == 1 else "Will Stay"
    risk = probabilities[i][1] * 100
    print(f"  Employee {i+1}: {result} (Risk: {risk:.1f}%)")

# Step 7: Visualize results
print("\n" + "=" * 60)
print("Step 7: Creating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Attrition distribution
attrition_counts = employees['attrition'].value_counts()
axes[0, 0].bar(['Stays', 'Leaves'], attrition_counts.values, 
               color=['green', 'red'], edgecolor='black')
axes[0, 0].set_title('Employee Attrition Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Attrition by department
dept_attrition = employees.groupby('department')['attrition'].mean()
axes[0, 1].bar(dept_attrition.index, dept_attrition.values, 
               color='steelblue', edgecolor='black')
axes[0, 1].set_title('Attrition Rate by Department', fontweight='bold')
axes[0, 1].set_ylabel('Attrition Rate')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Stays', 'Leaves'], yticklabels=['Stays', 'Leaves'])
axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# Plot 4: Model accuracy
axes[1, 1].bar(['Model Accuracy'], [accuracy], color='green', edgecolor='black')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title('Model Performance', fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].text(0, accuracy + 0.05, f'{accuracy:.1%}', ha='center', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Employee Attrition Prediction Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Day_7_Mar02_Encoding/attrition_presentation_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved!")

plt.show()

# Summary
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print("\nKey Findings:")
print(f"1. Model Accuracy: {accuracy:.1%}")
print(f"2. Attrition Rate: {employees['attrition'].mean():.1%}")
print(f"3. Highest Risk Department: {dept_attrition.idxmax()}")
print(f"4. Best Model Parameter C: {grid_search.best_params_['model__C']}")

print("\nRecommendations:")
print("1. Focus retention efforts on Sales department")
print("2. Provide mentorship for employees with <3 years experience")
print("3. Offer career development for younger employees")
print("4. Use this model to identify at-risk employees early")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
