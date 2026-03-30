# Used Car Condition Prediction - AI System

## Scenario
A car resale company wants to build an AI model that predicts whether a used car is in good condition or needs repairs based on inspection data.

## Dataset Details

### Total Records: 100 car inspection records

### Features:
1. **Mileage** - Distance traveled by the car (km)
2. **Engine Performance Score** - Rating from 3.0 to 10.0
3. **Fuel Efficiency** - Kilometers per liter (kmpl)
4. **Age of Car** - Years since manufacture

### Target Variable (Condition):
- **0** → Needs Repair
- **1** → Good Condition

## Dataset Split

The dataset is divided into three parts for reliable AI model training:

- **Training Set (70%)** - 70 records to teach the AI model
- **Validation Set (15%)** - 15 records to tune and improve the model
- **Test Set (15%)** - 15 records to evaluate final performance

## Files Generated

1. `complete_dataset.csv` - All 100 car inspection records
2. `train_set.csv` - Training data (70%)
3. `validation_set.csv` - Validation data (15%)
4. `test_set.csv` - Test data (15%)
5. `dataset_visualization.png` - Visual analysis charts

## How to Run

```bash
python car_condition_dataset.py
```

## Problem Type
This is a **Binary Classification** problem where the model predicts one of two classes:
- Class 0: Needs Repair
- Class 1: Good Condition

## Recommended Algorithms
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)
- Neural Networks

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score
