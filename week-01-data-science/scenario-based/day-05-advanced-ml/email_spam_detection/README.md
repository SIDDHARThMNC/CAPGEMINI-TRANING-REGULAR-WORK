# Email Spam Detection - Logistic Regression 📧

## Scenario
An email service provider wants to build a model to detect whether an incoming email is spam or not spam.

## Problem Type
Binary Classification using Logistic Regression

## Target Variable
- **1** = Spam
- **0** = Not Spam

## Features
1. **Word Count** - Total number of words in the email
2. **Has Link** - Whether the email contains a hyperlink (1=yes, 0=no)
3. **Caps Ratio** - Proportion of words written in ALL CAPS

## Dataset
- 15 email records
- 3 features per email
- Binary outcome: Spam or Not Spam

## Files
- `logistic_regression_spam.py` - Complete spam detection model
- `spam_dataset.csv` - Complete dataset
- `spam_detection_analysis.png` - Comprehensive visualization (4 charts)

## How to Run
```bash
python logistic_regression_spam.py
```

## Model Output
- Dataset summary and statistics
- Feature statistics
- Train-test split information
- Model parameters (coefficients and intercept)
- Logistic regression equation
- Predictions on test data with probabilities
- Model evaluation (Accuracy, Confusion Matrix, Classification Report)
- Probability interpretation
- Predictions on new emails
- Feature importance analysis
- Comprehensive visualizations

## Key Questions Answered
1. How to classify emails as spam or not spam?
2. What does a 70% spam probability mean?
3. Which features are most important for spam detection?
4. How accurate is the model?

## Interpreting Probabilities
- **70% spam probability** means:
  - Model is 70% confident it's spam
  - 30% chance it's legitimate
  - Typically classified as spam (>50% threshold)
  - Higher probability = Higher confidence

## Feature Importance
- **Caps Ratio**: Strong indicator (high caps = likely spam)
- **Has Link**: Moderate indicator (links common in spam)
- **Word Count**: Weak indicator alone

## Practical Recommendations
- Flag emails with >50% spam probability for review
- Auto-filter emails with >80% probability
- Monitor false positives to avoid blocking legitimate emails
- Regularly update model with new spam patterns

## Real-World Application
This technique is used in:
- Email filtering systems
- Spam detection services
- Phishing detection
- Content moderation
- Fraud detection
