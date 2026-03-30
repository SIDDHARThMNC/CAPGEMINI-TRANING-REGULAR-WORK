# Titanic Survival Prediction - Model Comparison

## Scenario
Researchers studying the Titanic disaster want to build models that predict passenger survival based on demographic and ticket information.

## Dataset
Using Seaborn's built-in Titanic dataset with features:
- **pclass**: Passenger class (1st, 2nd, 3rd)
- **sex**: Gender (male/female)
- **age**: Age in years
- **sibsp**: Number of siblings/spouses aboard
- **parch**: Number of parents/children aboard
- **fare**: Ticket fare price

## Target Variable
- **1**: Survived
- **0**: Died

## Models Compared

### 1. Logistic Regression
- Linear classification model
- Outputs probability of survival
- Requires feature scaling
- Interpretable coefficients

### 2. K-Nearest Neighbors (k=5)
- Instance-based learning
- Predicts based on 5 nearest neighbors
- Requires feature scaling
- Less interpretable

### 3. Decision Tree (max_depth=4)
- Rule-based classification
- Creates decision rules
- No scaling required
- Highly interpretable

## Key Questions Answered

### Which model performs best?
Compare accuracy scores and classification reports to determine the best performer.

### Interpretability Comparison
- **Logistic Regression**: Shows feature coefficients (positive = increases survival chance)
- **Decision Tree**: Provides clear if-then rules (e.g., "if female and 1st class, then survived")

### Why Scaling?
- **Logistic Regression & KNN**: Distance-based algorithms sensitive to feature scales
- **Decision Tree**: Uses splits, not distances, so scale-invariant

### Precision vs Recall
- **Precision**: Of those predicted to survive, how many actually survived?
  - High precision = Few false survival predictions
- **Recall**: Of all who truly survived, how many were correctly predicted?
  - High recall = Few missed survivors

### Historical Analysis
**Decision Tree is best for historians** because:
- Provides clear, interpretable rules
- Shows which factors mattered most
- Easy to explain survival patterns
- Mimics human decision-making

## Files
- `titanic_model_comparison.py`: Complete analysis with 3 models
- `model_comparison_results.csv`: Accuracy and prediction counts
- `titanic_model_comparison.png`: Comprehensive visualizations

## How to Run
```bash
python titanic_model_comparison.py
```

## Expected Insights
- Gender and passenger class are strongest predictors
- Women and children had higher survival rates
- First-class passengers had better survival chances
- Age and fare also influence predictions

## Requirements
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
