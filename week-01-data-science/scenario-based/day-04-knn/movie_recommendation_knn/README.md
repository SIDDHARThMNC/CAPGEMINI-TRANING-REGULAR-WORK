# Movie Recommendation System - K-Nearest Neighbors (KNN) 🎬

## Scenario
A streaming platform wants to recommend movies to users based on their preferences using the K-Nearest Neighbors algorithm.

## Problem Type
Binary Classification using K-Nearest Neighbors (KNN)

## Target Variable
- **1** = Will Like
- **0** = Won't Like

## Features
1. **Action Rating** - How action-packed the movie is (1-5)
2. **Comedy Rating** - How funny the movie is (1-5)
3. **Drama Rating** - How emotional the movie is (1-5)

## Dataset
- 10 movie records
- 3 rating features per movie
- Binary outcome: Will Like or Won't Like

## Files
- `knn_movie_recommendation.py` - Complete KNN recommendation model
- `movie_dataset.csv` - Complete dataset
- `knn_analysis.png` - Comprehensive visualization (4 charts)

## How to Run
```bash
python knn_movie_recommendation.py
```

## Model Output
- Dataset summary and statistics
- Train-test split information
- Feature scaling confirmation
- Model training with different K values (1, 3, 5)
- Best K selection based on accuracy
- Predictions on test data
- Model evaluation (Accuracy, Confusion Matrix, Classification Report)
- Prediction for new user [Action=4, Comedy=2, Drama=4]
- Nearest neighbors analysis
- Comprehensive visualizations
- Discussion on how K affects predictions

## Key Questions Answered
1. How to recommend movies based on user preferences?
2. What is the best K value for this dataset?
3. How does changing K affect model predictions?
4. Will a user with [Action=4, Comedy=2, Drama=4] like the movie?

## How K Affects Predictions

### K=1 (Low K)
- Uses only the closest neighbor
- More sensitive to noise and outliers
- Can lead to overfitting
- More complex decision boundaries

### K=3 (Medium K)
- Balances between bias and variance
- Less sensitive to individual outliers
- Often provides good generalization

### K=5 (Higher K)
- Uses more neighbors for voting
- Smoother decision boundaries
- Less sensitive to noise
- May underfit if K is too large

## Why Feature Scaling is Important
KNN uses distance calculations (Euclidean distance) to find nearest neighbors. Features with larger scales can dominate the distance calculation, so standardization ensures all features contribute equally.

## Real-World Application
This technique is used in:
- Netflix movie recommendations
- Spotify music recommendations
- Amazon product recommendations
- YouTube video suggestions
- Content-based filtering systems
