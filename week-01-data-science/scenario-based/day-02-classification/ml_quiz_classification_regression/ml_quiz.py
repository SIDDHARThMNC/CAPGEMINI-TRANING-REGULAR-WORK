import pandas as pd
import numpy as np

print("=" * 70)
print("MACHINE LEARNING QUIZ: CLASSIFICATION OR REGRESSION?")
print("=" * 70)
print("\nInstructions: For each scenario, determine whether it requires")
print("Classification or Regression algorithm.\n")

quiz_data = [
    {
        "Question": "Predicting house prices based on size, location, and number of rooms",
        "Answer": "Regression",
        "Explanation": "House price is a continuous numerical value"
    },
    {
        "Question": "Identifying whether an email is spam or not spam",
        "Answer": "Classification",
        "Explanation": "Binary classification with two categories: spam/not spam"
    },
    {
        "Question": "Estimating a student's exam score based on study hours",
        "Answer": "Regression",
        "Explanation": "Exam score is a continuous numerical value"
    },
    {
        "Question": "Categorizing customer reviews as positive, neutral, or negative",
        "Answer": "Classification",
        "Explanation": "Multi-class classification with three categories"
    },
    {
        "Question": "Forecasting tomorrow's temperature in degrees Celsius",
        "Answer": "Regression",
        "Explanation": "Temperature is a continuous numerical value"
    },
    {
        "Question": "Detecting whether a tumor is benign or malignant",
        "Answer": "Classification",
        "Explanation": "Binary classification with two categories"
    },
    {
        "Question": "Predicting stock prices for the next day",
        "Answer": "Regression",
        "Explanation": "Stock price is a continuous numerical value"
    },
    {
        "Question": "Recognizing handwritten digits (0-9)",
        "Answer": "Classification",
        "Explanation": "Multi-class classification with 10 categories"
    },
    {
        "Question": "Estimating the age of a person from their photo",
        "Answer": "Regression",
        "Explanation": "Age is a continuous numerical value"
    },
    {
        "Question": "Identifying the breed of a dog from an image",
        "Answer": "Classification",
        "Explanation": "Multi-class classification with multiple breed categories"
    },
    {
        "Question": "Predicting the number of units sold next month",
        "Answer": "Regression",
        "Explanation": "Sales quantity is a continuous numerical value"
    },
    {
        "Question": "Determining if a credit card transaction is fraudulent",
        "Answer": "Classification",
        "Explanation": "Binary classification: fraudulent/legitimate"
    },
    {
        "Question": "Estimating the salary of an employee based on experience",
        "Answer": "Regression",
        "Explanation": "Salary is a continuous numerical value"
    },
    {
        "Question": "Classifying music into genres (Rock, Pop, Jazz, etc.)",
        "Answer": "Classification",
        "Explanation": "Multi-class classification with genre categories"
    },
    {
        "Question": "Predicting the distance a car can travel on remaining fuel",
        "Answer": "Regression",
        "Explanation": "Distance is a continuous numerical value"
    }
]

df = pd.DataFrame(quiz_data)

for idx, row in df.iterrows():
    print(f"\nQuestion {idx + 1}:")
    print(f"Scenario: {row['Question']}")
    print(f"Answer: {row['Answer']}")
    print(f"Explanation: {row['Explanation']}")
    print("-" * 70)

df.to_csv('ml_quiz_classification_regression/quiz_data.csv', index=False)

print("\n" + "=" * 70)
print("KEY DIFFERENCES")
print("=" * 70)

differences = {
    "Aspect": [
        "Output Type",
        "Prediction",
        "Examples",
        "Algorithms",
        "Evaluation Metrics"
    ],
    "Classification": [
        "Discrete categories/labels",
        "Predicts class membership",
        "Spam detection, Image recognition, Disease diagnosis",
        "Logistic Regression, Decision Trees, SVM, Random Forest, Neural Networks",
        "Accuracy, Precision, Recall, F1-Score, Confusion Matrix"
    ],
    "Regression": [
        "Continuous numerical values",
        "Predicts quantity/amount",
        "Price prediction, Temperature forecasting, Sales prediction",
        "Linear Regression, Polynomial Regression, Ridge, Lasso, Neural Networks",
        "MSE, RMSE, MAE, R-squared, Adjusted R-squared"
    ]
}

diff_df = pd.DataFrame(differences)
print("\n", diff_df.to_string(index=False))

diff_df.to_csv('ml_quiz_classification_regression/key_differences.csv', index=False)

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"\nTotal Questions: {len(df)}")
print(f"Classification Questions: {len(df[df['Answer'] == 'Classification'])}")
print(f"Regression Questions: {len(df[df['Answer'] == 'Regression'])}")

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)
print("✓ quiz_data.csv - Complete quiz with answers and explanations")
print("✓ key_differences.csv - Classification vs Regression comparison")
print("=" * 70)
