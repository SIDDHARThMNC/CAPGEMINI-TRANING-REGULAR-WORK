# Student Success & Career Path Prediction - Capstone Project

## Overview
Comprehensive machine learning analysis to help universities understand student performance, identify at-risk students, and recommend targeted interventions.

## Dataset
Source: [GitHub - Student Success and Career Path Prediction](https://github.com/himanshusar123/Datasets)

## Project Objectives

### 1. Regression: Predict Exam Scores
Build models to forecast student exam performance based on various factors.

**Models Used:**
- Linear Regression
- Random Forest Regression

**Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 2. Classification: Identify At-Risk Students
Classify students into "At Risk" vs "On Track" categories for early intervention.

**Models Used:**
- Logistic Regression
- Random Forest Classifier

**Metrics:**
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 3. Clustering: Group Students by Study Habits
Discover natural groupings of students with similar study patterns and behaviors.

**Method:**
- K-Means Clustering
- Elbow Method for optimal K
- Silhouette Score validation

**Expected Clusters:**
- High Achievers
- Steady Performers
- Needs Support
- Critical Intervention

## Intervention Recommendations

### Critical Intervention Group
- Immediate one-on-one tutoring
- Academic counseling sessions
- Study skills workshop
- Weekly progress monitoring

### Needs Support Group
- Small group tutoring sessions
- Time management workshop
- Peer mentoring program
- Bi-weekly check-ins

### Steady Performers
- Optional study groups
- Advanced topic workshops
- Career guidance sessions
- Monthly progress reviews

### High Achievers
- Leadership opportunities
- Peer tutoring roles
- Advanced placement programs
- Scholarship information

## Key Features Analyzed
- Study hours
- Attendance rate
- Participation level
- Previous academic performance
- Demographic factors
- Extracurricular involvement

## Business Impact
- **Early Warning System**: Identify struggling students before it's too late
- **Resource Allocation**: Target interventions where they're needed most
- **Personalized Support**: Tailor programs to student group needs
- **Improved Outcomes**: Increase graduation rates and student success

## Files Generated
- `student_data.csv`: Original dataset
- `student_analysis_results.csv`: Complete results with predictions and clusters
- `student_analysis_visualization.png`: Comprehensive visualizations

## How to Run
```bash
python student_analysis.py
```

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Expected Insights
- Identify key factors affecting exam performance
- Determine which students need immediate support
- Understand different student learning patterns
- Provide data-driven intervention strategies
