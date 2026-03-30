# Used Car Price Prediction with EDA - Multiple Linear Regression

## Scenario
A used car dealership wants to predict the selling price of cars based on multiple factors. They perform Exploratory Data Analysis (EDA) to understand patterns before building the model.

## Features Used
1. ⚙️ **Engine Size** - Engine capacity in liters
2. 🛣️ **Mileage** - Thousand kilometers driven
3. 📅 **Car Age** - Age in years
4. 💨 **Horsepower** - Engine power

## Formula
```
Price = b0 + b1(Engine_Size) + b2(Mileage) + b3(Age) + b4(Horsepower)
```

## EDA Components
- Statistical summary of all features
- Correlation analysis
- Feature vs Price scatter plots
- Correlation heatmap
- Residual analysis
- Feature importance visualization
- Price distribution histogram

## Dataset
- 15 used car records
- Target variable: Price in lakhs

## Files
- `car_price_eda.py` - Complete EDA and prediction model
- `car_price_dataset.csv` - Complete dataset
- `car_price_eda_analysis.png` - Comprehensive EDA visualization (9 charts)

## How to Run
```bash
python car_price_eda.py
```

## Model Output
- Complete EDA insights
- Statistical summary
- Correlation analysis
- Model parameters and equation
- Predictions on test data
- Model evaluation metrics (MAE, RMSE, R²)
- New car price predictions
- Feature importance ranking
- Comprehensive visualizations

## Key Insights
- Horsepower has the strongest impact on price
- Mileage and Age negatively affect price
- Engine Size positively correlates with price
- EDA helps understand data patterns before modeling
