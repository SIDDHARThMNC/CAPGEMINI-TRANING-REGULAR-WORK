# Capstone Projects - Machine Learning

This folder contains two capstone projects converted from Jupyter notebooks to Python scripts.

## Project 1: Titanic Survival Prediction
**File:** `titanic_survival_prediction.py`

### Objective
Predict whether a passenger would survive the Titanic disaster based on their information.

### Features Used
- Passenger class (pclass)
- Gender (sex)
- Age (age)
- Number of siblings/spouses aboard (sibsp)
- Number of parents/children aboard (parch)
- Ticket fare (fare)

### Models Implemented
1. Logistic Regression
2. K-Nearest Neighbors (KNN) with k=5
3. Decision Tree with max depth = 4

### Key Questions Explored
- Which model performs best at predicting survival?
- How does Logistic Regression differ from Decision Tree in interpretability?
- Why is scaling applied before training Logistic Regression and KNN?

---

## Project 2: Student Success & Career Path Prediction
**File:** `student_success_career_path.py`

### Objective
Analyze student performance data to:
- Predict exam scores (Regression)
- Classify students into "At Risk" vs. "On Track" categories (Classification)
- Cluster students into groups with similar study habits (Clustering)
- Recommend interventions (tutoring, workshops, counseling)

### Features Used
- Hours_Studied
- Attendance (%)
- Assignments_Submitted
- Previous_Sem_GPA
- Participation_Score
- Career_Readiness_Score
- Age
- Gender

### Models Implemented
1. Linear Regression (for exam score prediction)
2. Logistic Regression (for pass/fail classification)
3. K-Means Clustering (for grouping students)

### Evaluation Metrics
- Regression: MAE, R² Score
- Classification: Precision, Recall, F1-Score
- Clustering: Silhouette Score

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Run Titanic Project
```bash
python titanic_survival_prediction.py
```

### Run Student Success Project
```bash
python student_success_career_path.py
```

**Note:** For the student success project, you'll need the `Student_Success_Career_Path.csv` dataset file.

---

## Date
Created: February 28, 2026
