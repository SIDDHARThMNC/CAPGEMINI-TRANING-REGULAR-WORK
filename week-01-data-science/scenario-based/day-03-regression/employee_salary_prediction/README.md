# Employee Salary Prediction - Multiple Linear Regression

## Scenario
A company wants to predict employee salary based on several important factors using Multiple Linear Regression.

## Features Used
1. **Years of Experience** - Work experience in years
2. **Education Level** - 1 = Bachelor, 2 = Master, 3 = PhD
3. **Number of Skills Known** - Technical skills count
4. **Performance Rating** - Rating from 1 to 5

## Formula
```
Salary = b0 + b1(Experience) + b2(Education) + b3(Skills) + b4(Performance)
```

Where:
- b0 = Intercept
- b1, b2, b3, b4 = Coefficients for each feature

## Dataset
- 15 employee records with multiple features
- Target variable: Salary in lakhs

## Files
- `salary_prediction.py` - Main prediction model
- `employee_salary_dataset.csv` - Complete dataset
- `salary_analysis.png` - Visualization charts

## How to Run
```bash
python salary_prediction.py
```

## Model Output
- Model parameters (coefficients and intercept)
- Predictions on test data
- Model evaluation metrics (MAE, RMSE, R²)
- New employee salary predictions
- Feature importance analysis
- Visualization charts

## Key Insights
Multiple Linear Regression considers all features simultaneously to make accurate salary predictions based on experience, education, skills, and performance.
