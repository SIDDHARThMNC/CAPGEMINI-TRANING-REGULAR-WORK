# Insurance Claim Prediction - Logistic Regression 🚗

## Scenario
A car insurance company wants to predict whether a driver is likely to file a claim in the next year based on their age.

## Problem Type
Binary Classification using Logistic Regression

## Target Variable
- **1** = Claim Filed
- **0** = No Claim

## Features
- **Age** - Driver's age in years

## Dataset
- 20 driver records
- Age range: 18-65 years
- Binary outcome: Claim Filed or No Claim

## Files
- `logistic_regression_insurance.py` - Complete logistic regression model
- `insurance_claim_dataset.csv` - Complete dataset
- `logistic_regression_analysis.png` - Visualization (Logistic curve + Confusion matrix)

## How to Run
```bash
python logistic_regression_insurance.py
```

## Model Output
- Dataset summary and statistics
- Train-test split information
- Model parameters (coefficient and intercept)
- Logistic regression equation
- Predictions on test data with probabilities
- Model evaluation (Accuracy, Confusion Matrix, Classification Report)
- Key insight: Age at which claim probability crosses 50%
- Sample predictions for new drivers
- Comprehensive visualizations

## Key Questions Answered
1. How does claim probability change with age?
2. At what age does the probability of filing a claim cross 50%?
3. What is the model's accuracy in predicting claims?
4. How can insurance companies use this for risk management?

## Interpretation
- The logistic curve shows probability decreases with age
- Younger drivers have higher claim probability
- Model helps insurance companies adjust premiums based on risk
- Business application: Risk management and pricing strategy

## Real-World Application
This technique is used in:
- Insurance risk assessment
- Credit scoring
- Medical diagnosis
- Customer churn prediction
- Fraud detection
