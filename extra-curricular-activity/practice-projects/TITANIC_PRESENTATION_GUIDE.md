# Titanic Survival Prediction - Presentation Guide

## 🚢 Project Overview

**Historical Context:**
- RMS Titanic sank on April 15, 1912
- 2,224 passengers and crew aboard
- Only 710 survived (32% survival rate)

**Problem Statement:**
Researchers want to build ML models to predict passenger survival based on their characteristics.

**Goal:** Compare three classification algorithms and understand survival patterns.

---

## 📊 Dataset Features

### Input Features:
- **pclass**: Passenger class (1st, 2nd, 3rd)
  - 1 = Upper class (expensive tickets)
  - 2 = Middle class
  - 3 = Lower class (cheapest tickets)

- **sex**: Gender (male/female)
  - Critical survival factor ("Women and children first")

- **age**: Passenger age in years
  - Children had priority in lifeboats

- **sibsp**: Number of siblings/spouses aboard
  - Family connections

- **parch**: Number of parents/children aboard
  - Family size indicator

- **fare**: Ticket price in British pounds
  - Wealth indicator, correlates with class

### Target Variable:
- **survived**: 
  - 1 = Survived
  - 0 = Died

---

## 🔧 Step-by-Step Code Explanation

### Step 1: Import Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
```

**Why these libraries?**
- `seaborn`: Provides built-in Titanic dataset
- `sklearn`: Machine learning algorithms
- `SimpleImputer`: Handle missing age values
- `StandardScaler`: Normalize features

---

### Step 2: Load Dataset
```python
df = sns.load_dataset('titanic')
print(df.head())
```

**Dataset Info:**
- 891 passengers in training data
- 15 columns total (we use 6 features)
- Real historical data from Titanic manifest

**Sample Output:**
```
   survived  pclass     sex   age  sibsp  parch     fare
0         0       3    male  22.0      1      0   7.2500
1         1       1  female  38.0      1      0  71.2833
2         1       3  female  26.0      0      0   7.9250
```

---

### Step 3: Select Features and Target
```python
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
y = df['survived']
```

**Why these features?**
- **pclass**: Wealth/social status (upper decks closer to lifeboats)
- **sex**: Gender (women prioritized)
- **age**: Children prioritized
- **sibsp/parch**: Family size (helped or hindered escape)
- **fare**: Another wealth indicator

---

### Step 4: Encode Gender
```python
X['sex'] = X['sex'].map({'male': 0, 'female': 1})
```

**Why encode?**
- ML models need numerical input
- Text → Numbers conversion
- male = 0, female = 1 (arbitrary assignment)

**Before:** `['male', 'female', 'male', ...]`
**After:** `[0, 1, 0, ...]`

---

### Step 5: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Split Breakdown:**
- **Training set**: 712 passengers (80%)
  - Used to teach the model patterns
- **Test set**: 179 passengers (20%)
  - Used to evaluate model on unseen data
- **random_state=42**: Ensures same split every time

**Why split?**
- Prevents overfitting
- Tests model on "new" passengers
- Simulates real-world prediction

---

### Step 6: Handle Missing Values
```python
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

**Problem:** Age column has missing values (~20% missing)

**Solution:** Replace missing ages with median age
- Median age ≈ 28 years
- More robust than mean (not affected by outliers)

**Why median?**
- Mean can be skewed by very old/young passengers
- Median represents "typical" passenger age

---

### Step 7: Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why scale?**
Features have different ranges:
- age: 0-80
- fare: 0-500
- pclass: 1-3
- sex: 0-1

**What StandardScaler does:**
- Transforms to mean=0, std=1
- Formula: `(value - mean) / std_deviation`

**Example:**
```
Before scaling:
age=30, fare=50, pclass=2

After scaling:
age=0.15, fare=0.23, pclass=-0.5
```

**Important:** Only needed for Logistic Regression and KNN!

---

## 🎯 Model 1: Logistic Regression

### Step 8: Train Logistic Regression
```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
lr_pred = model.predict(X_test_scaled)
```

**What is Logistic Regression?**
- Despite name, it's for CLASSIFICATION (not regression)
- Predicts probability of survival (0 to 1)
- Uses sigmoid function: S-shaped curve

**How it works:**
1. Calculates weighted sum: `z = w1*pclass + w2*sex + w3*age + ...`
2. Applies sigmoid: `probability = 1 / (1 + e^(-z))`
3. If probability > 0.5 → Survived (1)
4. If probability ≤ 0.5 → Died (0)

**Interpretability:**
- Each feature gets a coefficient (weight)
- Positive weight → increases survival chance
- Negative weight → decreases survival chance

**Example coefficients:**
- sex (female): +2.5 (strong positive effect)
- pclass: -1.2 (higher class = lower number = better survival)
- age: -0.03 (younger = better survival)

---

### Step 9: Evaluate Logistic Regression
```python
print("Accuracy:", accuracy_score(y_test, lr_pred))
```

**Result: 79.89% accuracy**

**What does this mean?**
- Out of 179 test passengers, correctly predicted ~143
- Missed ~36 passengers
- Better than random guessing (50%)

---

## 🎯 Model 2: K-Nearest Neighbors (KNN)

### Step 10: Train KNN
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
```

**What is KNN?**
- "Tell me who your neighbors are, I'll tell you who you are"
- Looks at 5 most similar passengers
- Majority vote determines prediction

**How it works:**
1. For new passenger, find 5 most similar passengers in training data
2. Similarity based on Euclidean distance
3. Count survivors among those 5
4. If 3+ survived → Predict survived
5. If 3+ died → Predict died

**Example:**
```
New passenger: Female, 1st class, age 30
5 nearest neighbors:
- Female, 1st class, age 28 → Survived ✓
- Female, 1st class, age 32 → Survived ✓
- Female, 2nd class, age 29 → Survived ✓
- Male, 1st class, age 30 → Died ✗
- Female, 1st class, age 35 → Survived ✓

Vote: 4 survived, 1 died → Predict SURVIVED
```

**Why k=5?**
- Odd number avoids ties
- Not too small (k=1 is noisy)
- Not too large (k=100 loses local patterns)

---

### Step 11: Evaluate KNN
```python
print("Accuracy:", accuracy_score(y_test, knn_pred))
```

**Result: 78.77% accuracy**

**Slightly lower than Logistic Regression**
- More sensitive to feature scaling
- Can be affected by irrelevant features
- Works well when similar passengers have similar outcomes

---

## 🎯 Model 3: Decision Tree

### Step 12: Train Decision Tree
```python
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)  # Note: No scaling needed!
tree_pred = tree.predict(X_test)
```

**What is Decision Tree?**
- Series of yes/no questions
- Creates a flowchart-like structure
- Easy to visualize and explain

**How it works:**
```
                    Is sex = female?
                   /              \
                 YES               NO
                  |                 |
            Is pclass ≤ 2?     Is age ≤ 13?
            /          \        /         \
          YES          NO     YES         NO
           |            |      |           |
       SURVIVED      DIED   SURVIVED     DIED
```

**max_depth=4:**
- Limits tree to 4 levels of questions
- Prevents overfitting
- Keeps tree interpretable

**Why no scaling?**
- Decision trees use splits, not distances
- Only cares about "greater than" or "less than"
- Scale doesn't matter for comparisons

---

### Step 13: Evaluate Decision Tree
```python
print("Accuracy:", accuracy_score(y_test, tree_pred))
```

**Result: 79.89% accuracy**

**Same as Logistic Regression!**
- Both models perform equally well
- Different approaches, similar results

---

## 📊 Model Comparison

### Performance Summary:
| Model | Accuracy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| Logistic Regression | 79.89% | Fast, interpretable coefficients | Assumes linear relationships |
| KNN (k=5) | 78.77% | No training needed, flexible | Slow prediction, needs scaling |
| Decision Tree | 79.89% | Easy to visualize, no scaling | Can overfit, unstable |

---

## 🎤 Answering the Key Questions

### Q1: Which model performs best?

**Answer:**
- **Logistic Regression** and **Decision Tree** tie at 79.89%
- KNN slightly behind at 78.77%

**Best choice depends on goal:**
- **For explanation**: Decision Tree (visual rules)
- **For speed**: Logistic Regression (fast predictions)
- **For flexibility**: KNN (no assumptions about data)

**In this case:** Logistic Regression or Decision Tree are best

---

### Q2: How does Logistic Regression differ from Decision Tree in interpretability?

**Logistic Regression:**
- ✅ Provides numerical coefficients
- ✅ Shows feature importance quantitatively
- ✅ Statistical significance testing
- ❌ Harder to explain to non-technical audience
- ❌ Assumes linear relationships

**Example:**
"Being female increases survival odds by 2.5x"

**Decision Tree:**
- ✅ Visual flowchart
- ✅ Easy to explain with rules
- ✅ No math background needed
- ✅ Shows feature interactions
- ❌ Can be complex with many branches

**Example:**
"If female AND 1st/2nd class → 95% survived"

**For historians:** Decision Tree is better
- Clear rules: "Women and children first"
- Shows class-based survival patterns
- No statistical jargon needed

---

### Q3: Why is scaling needed for Logistic Regression and KNN, but not Decision Trees?

**Logistic Regression:**
- Uses gradient descent optimization
- Features with larger ranges dominate
- Example: fare (0-500) vs pclass (1-3)
- Without scaling, fare gets too much weight

**KNN:**
- Uses distance calculations
- Euclidean distance: `√[(x1-x2)² + (y1-y2)² + ...]`
- Large-scale features dominate distance
- Example: fare difference of 100 >> age difference of 10

**Decision Tree:**
- Uses splits: "Is age > 30?"
- Only cares about order, not magnitude
- Splitting at 30 vs 31 doesn't depend on scale
- Works with any range

**Analogy:**
- Logistic/KNN: Measuring distances (need same units)
- Decision Tree: Sorting items (order matters, not size)

---

### Q4: What do Precision and Recall mean?

**Confusion Matrix:**
```
                Predicted
              Died  Survived
Actual Died    TN      FP
     Survived  FN      TP
```

**Precision:** "Of those we predicted survived, how many actually survived?"
- Formula: `TP / (TP + FP)`
- Answers: "Are our survival predictions reliable?"
- High precision = Few false alarms

**Recall:** "Of those who actually survived, how many did we predict?"
- Formula: `TP / (TP + FN)`
- Answers: "Did we catch all survivors?"
- High recall = Few missed survivors

**Example:**
```
Predicted 100 survived:
- 80 actually survived (TP)
- 20 actually died (FP)
Precision = 80/100 = 80%

Actually 120 survived:
- 80 predicted correctly (TP)
- 40 missed (FN)
Recall = 80/120 = 67%
```

**Trade-off:**
- High precision → Conservative predictions
- High recall → Aggressive predictions
- F1-score balances both

---

## 🎤 Presentation Structure

### Slide 1: Title & Context
- **Title:** Predicting Titanic Survival with Machine Learning
- **Context:** Historical disaster, 32% survival rate
- **Goal:** Compare 3 ML algorithms

### Slide 2: Dataset Overview
- 891 passengers
- 6 features: class, gender, age, family, fare
- Target: Survived (1) or Died (0)
- Show sample data table

### Slide 3: Data Preprocessing
- **Missing values:** Imputed age with median
- **Encoding:** Male=0, Female=1
- **Scaling:** StandardScaler for Logistic/KNN
- **Split:** 80% train, 20% test

### Slide 4: Model 1 - Logistic Regression
- **How it works:** Probability-based classification
- **Accuracy:** 79.89%
- **Strengths:** Fast, interpretable coefficients
- **Key insight:** Gender most important feature

### Slide 5: Model 2 - K-Nearest Neighbors
- **How it works:** Majority vote of 5 neighbors
- **Accuracy:** 78.77%
- **Strengths:** No assumptions, flexible
- **Limitation:** Requires feature scaling

### Slide 6: Model 3 - Decision Tree
- **How it works:** Series of yes/no questions
- **Accuracy:** 79.89%
- **Strengths:** Visual, no scaling needed
- **Key rules:** "Women and children first" clearly visible

### Slide 7: Model Comparison
- Show comparison table
- All models ~80% accurate
- Different strengths for different use cases

### Slide 8: Key Findings
- **Gender:** Strongest predictor (women 3x more likely to survive)
- **Class:** 1st class had better survival (closer to lifeboats)
- **Age:** Children prioritized
- **Family:** Small families had advantage

### Slide 9: Answering Research Questions
- Best model: Tie between Logistic Regression and Decision Tree
- Interpretability: Decision Tree wins for non-technical audience
- Scaling: Needed for distance-based models only

### Slide 10: Conclusion & Lessons
- ML can uncover historical patterns
- Multiple models provide different insights
- 80% accuracy shows clear survival patterns existed
- Validates "women and children first" policy

---

## 🔑 Key Takeaways

1. **Three different approaches, similar accuracy**
   - Shows problem is well-suited for ML
   - Multiple valid solutions exist

2. **Feature engineering matters**
   - Encoding, imputation, scaling all crucial
   - Preprocessing affects model performance

3. **Interpretability vs Performance trade-off**
   - Decision Tree: Easy to explain
   - Logistic Regression: Statistical rigor
   - KNN: Flexible but "black box"

4. **Historical validation**
   - Models confirm known survival patterns
   - Gender and class were critical factors
   - Data science validates historical accounts

---

## 💡 Advanced Discussion Points

### Why 80% and not 100%?
- Some survival was random (luck, location on ship)
- Missing features (swimming ability, cabin location)
- Human decisions not captured in data

### Real-world applications:
- Disaster preparedness planning
- Risk assessment in emergencies
- Historical research validation
- Educational tool for ML concepts

### Ethical considerations:
- Models reflect historical biases (class-based survival)
- Should we use such models today?
- Importance of fairness in ML

---

## 📝 Common Questions & Answers

**Q: Why not use all 15 columns?**
A: Some columns have too many missing values or are not predictive (e.g., ticket number, cabin).

**Q: Can we improve accuracy beyond 80%?**
A: Yes, with:
- Feature engineering (family size, title from name)
- Ensemble methods (Random Forest, XGBoost)
- More data (test set from Kaggle)

**Q: Which model would you deploy?**
A: Logistic Regression - fast, interpretable, good accuracy.

**Q: Why random_state=42?**
A: Ensures reproducible results. 42 is arbitrary (Hitchhiker's Guide reference).

**Q: What if we had more data?**
A: Deep learning (neural networks) might perform better, but current models are sufficient.

---

## 🚀 Results Summary

### Model Performance:
```
Logistic Regression: 79.89% ⭐
KNN (k=5):          78.77%
Decision Tree:      79.89% ⭐
```

### Key Survival Factors (in order):
1. **Gender** (female >> male)
2. **Class** (1st > 2nd > 3rd)
3. **Age** (children > adults)
4. **Family size** (small families better)
5. **Fare** (higher fare = better survival)

### Historical Validation:
✅ "Women and children first" policy confirmed
✅ Class-based survival differences evident
✅ Family connections mattered
✅ Wealth (fare) correlated with survival

---

Good luck with your presentation! 🎉🚢
