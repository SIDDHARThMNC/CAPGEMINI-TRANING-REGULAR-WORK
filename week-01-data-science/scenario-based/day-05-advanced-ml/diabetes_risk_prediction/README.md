# Diabetes Risk Prediction - Logistic Regression 🏥

## Scenario
A hospital wants to predict whether patients are at risk of developing diabetes based on their BMI (Body Mass Index).

## Problem Type
Binary Classification using Logistic Regression

## Target Variable
- **1** = Diabetes
- **0** = No Diabetes

## Features
- **BMI** - Body Mass Index (weight in kg / height in m²)

## BMI Categories (WHO Classification)
- **Underweight**: BMI < 18.5
- **Normal weight**: BMI 18.5 - 24.9
- **Overweight**: BMI 25.0 - 29.9
- **Obese**: BMI ≥ 30.0

## Dataset
- 20 patient records
- BMI range: 18.5 - 43.5
- Binary outcome: Diabetes or No Diabetes

## Files
- `logistic_regression_diabetes.py` - Complete logistic regression model
- `diabetes_dataset.csv` - Complete dataset
- `diabetes_risk_analysis.png` - Visualization (Logistic curve with BMI categories + Confusion matrix)

## How to Run
```bash
python logistic_regression_diabetes.py
```

## Model Output
- Dataset summary and statistics
- BMI category information
- Train-test split information
- Model parameters (coefficient and intercept)
- Logistic regression equation
- Predictions on test data with probabilities
- Model evaluation (Accuracy, Confusion Matrix, Classification Report)
- Key insight: BMI at which diabetes probability crosses 50%
- Sample predictions for different BMI values
- Clinical recommendations
- Comprehensive visualizations

## Key Questions Answered
1. How does diabetes risk change with BMI?
2. At what BMI does the probability of diabetes cross 50%?
3. What is the model's accuracy in predicting diabetes?
4. How can hospitals use this for early intervention?

## Interpretation
- Diabetes risk increases significantly with higher BMI
- Patients with BMI ≥ 30 (Obese) are at high risk
- Model helps identify high-risk patients for preventive care
- Early intervention can prevent or delay diabetes onset

## Clinical Recommendations
- **BMI < 25**: Low risk - Maintain healthy lifestyle
- **BMI 25-30**: Moderate risk - Weight management recommended
- **BMI ≥ 30**: High risk - Medical intervention and lifestyle changes needed

## Real-World Application
This technique is used in:
- Healthcare risk assessment
- Preventive medicine
- Patient screening programs
- Public health initiatives
- Clinical decision support systems
