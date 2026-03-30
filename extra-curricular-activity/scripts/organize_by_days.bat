@echo off
echo Creating day-wise folder structure...

REM Create day folders
mkdir "Day_1_Feb23-24_Data_Wrangling"
mkdir "Day_2_Feb25_Classification"
mkdir "Day_3_Feb26_Regression"
mkdir "Day_4_Feb27_KNN"
mkdir "Day_5_Feb28_Advanced_ML"

echo.
echo Moving Day 1 projects...
move "e commerce" "Day_1_Feb23-24_Data_Wrangling\"

echo.
echo Moving Day 2 projects...
move "music data" "Day_2_Feb25_Classification\"
move "ml_quiz_classification_regression" "Day_2_Feb25_Classification\"
move "used_car_condition_prediction" "Day_2_Feb25_Classification\"
move "test_marks" "Day_2_Feb25_Classification\"

echo.
echo Moving Day 3 projects...
move "employee_salary_prediction" "Day_3_Feb26_Regression\"
move "ecommerce_delivery_prediction" "Day_3_Feb26_Regression\"
move "house pricing with dataset ml" "Day_3_Feb26_Regression\"
move "used_car_price_prediction_eda" "Day_3_Feb26_Regression\"
move linear_regression_simple.py "Day_3_Feb26_Regression\"
move linear_regretion.py "Day_3_Feb26_Regression\"
move linear_regression_output.png "Day_3_Feb26_Regression\"

echo.
echo Moving Day 4 projects...
move "fitness_app.py" "Day_4_Feb27_KNN\"
move "movie_recommendation_knn" "Day_4_Feb27_KNN\"

echo.
echo Moving Day 5 projects...
move "university.py" "Day_5_Feb28_Advanced_ML\"
move "loan_approval_decision_tree" "Day_5_Feb28_Advanced_ML\"
move "email_spam_detection" "Day_5_Feb28_Advanced_ML\"
move "email_spam_evaluation" "Day_5_Feb28_Advanced_ML\"
move "diabetes_risk_prediction" "Day_5_Feb28_Advanced_ML\"
move "insurance_claim_prediction" "Day_5_Feb28_Advanced_ML\"
move "customer_segmentation_clustering" "Day_5_Feb28_Advanced_ML\"
move "hospital_patient_segmentation" "Day_5_Feb28_Advanced_ML\"
move "telecom_clustering" "Day_5_Feb28_Advanced_ML\"
move "movie_streaming_clustering" "Day_5_Feb28_Advanced_ML\"
move "bank_customer_hierarchical_clustering" "Day_5_Feb28_Advanced_ML\"
move "employee_segmentation_hierarchical" "Day_5_Feb28_Advanced_ML\"
move "titanic_survival_prediction_capstone 1" "Day_5_Feb28_Advanced_ML\"
move "student_success_capstone 2" "Day_5_Feb28_Advanced_ML\"

echo.
echo ========================================
echo Day-wise organization complete!
echo ========================================
echo.
echo IMPORTANT: Review the changes before committing to Git
echo Run: git status
echo.
pause
