# E-commerce Delivery Time Prediction - Multiple Linear Regression

## Scenario
An e-commerce company wants to predict how long an order will take to deliver based on multiple factors using Multiple Linear Regression.

## Features Used
1. **Distance to Customer** - Distance in kilometers
2. **Number of Items** - Count of items in the order
3. **Traffic Level** - 1 = Low, 2 = Medium, 3 = High
4. **Warehouse Processing Time** - Time in hours

## Formula
```
DeliveryTime = b0 + b1(Distance) + b2(Items) + b3(Traffic) + b4(ProcessingTime)
```

Where:
- b0 = Intercept
- b1, b2, b3, b4 = Coefficients for each feature

## Dataset
- 15 order records with multiple features
- Target variable: Delivery Time in hours

## Files
- `delivery_time_prediction.py` - Main prediction model
- `delivery_dataset.csv` - Complete dataset
- `delivery_analysis.png` - Visualization charts

## How to Run
```bash
python delivery_time_prediction.py
```

## Model Output
- Model parameters (coefficients and intercept)
- Predictions on test data
- Model evaluation metrics (MAE, RMSE, R²)
- New order delivery time predictions
- Feature importance analysis
- Visualization charts

## Key Insights
Multiple Linear Regression helps the e-commerce company estimate delivery times accurately by considering distance, order size, traffic conditions, and processing time simultaneously.
