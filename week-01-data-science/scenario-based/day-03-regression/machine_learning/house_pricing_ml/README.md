# House Price Prediction - Multiple Linear Regression

## Scenario
A real estate company wants to predict the price of a house based on several important features using Multiple Linear Regression.

## Features Used
1. **Area** - Square feet of the house
2. **Bedrooms** - Number of bedrooms
3. **Age** - Age of the house in years
4. **Distance** - Distance from city center in km

## Formula
```
Price = b0 + b1(Area) + b2(Bedrooms) + b3(Age) + b4(Distance)
```

Where:
- b0 = Intercept
- b1, b2, b3, b4 = Coefficients for each feature

## Dataset
- 12 house records with multiple features
- Target variable: Price in lakhs

## Files
- `multilinear_regression.py` - Main prediction model
- `house_dataset.csv` - Complete dataset
- `house_price_analysis.png` - Visualization charts

## How to Run
```bash
python multilinear_regression.py
```

## Model Output
- Model parameters (coefficients and intercept)
- Predictions on test data
- Model evaluation metrics (MAE, RMSE, R²)
- New house price predictions
- Feature importance analysis
- Visualization charts

## Key Insights
Multiple Linear Regression considers all features simultaneously to make more accurate predictions compared to simple linear regression.
