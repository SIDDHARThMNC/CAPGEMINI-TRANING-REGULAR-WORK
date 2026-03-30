# K-Nearest Neighbors (KNN) - Heart Disease Risk Prediction

## Scenario
A fitness app wants to predict whether a person is at risk of heart disease based on three lifestyle indicators:
- Exercise Level (hours of physical activity per week)
- Diet Quality (rating from 1-5, higher = healthier)
- Stress Level (rating from 1-5, higher = more stress)

## Dataset
- 10 users with lifestyle data
- Features: Exercise, Diet, Stress
- Target: AtRisk (1 = At Risk, 0 = Not at Risk)
- 50% risk rate in the dataset

## Model Performance
- Algorithm: K-Nearest Neighbors (KNN)
- Best K value: 1
- Accuracy: 100.00%
- Train-Test Split: 70-30

## Key Findings
1. People at risk tend to have:
   - Lower exercise hours
   - Lower diet quality
   - Higher stress levels

2. Feature scaling is crucial for KNN to ensure all features contribute equally to distance calculations

3. Different K values (1, 3, 5) were tested:
   - K=1: Uses only closest neighbor - sensitive to noise
   - K=3: Majority vote from 3 neighbors - balanced approach
   - K=5: Majority vote from 5 neighbors - smoother boundaries

## New User Prediction
For a user with [Exercise=4 hrs/week, Diet=3, Stress=4]:
- Prediction: At Risk
- Probability: 100%

## Health Recommendations
To reduce heart disease risk:
- Increase exercise to 5+ hours per week
- Improve diet quality (aim for 4-5 rating)
- Manage stress levels (aim for 1-2 rating)

## Files
- `knn_heart_disease.py` - Main KNN implementation
- `Fitness_app_dataset - Sheet1.csv` - Dataset
- `heart_disease_knn_analysis.png` - Visualization
- `README.md` - Documentation
