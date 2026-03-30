@echo off
echo ========================================
echo Pushing Capstone Projects to GitHub
echo ========================================

echo.
echo Adding capstone_new folder...
git add capstone_new/

echo.
echo Committing changes...
git commit -m "Add capstone projects: Titanic Survival and Student Success prediction with detailed presentation guides"

echo.
echo Pushing to GitHub...
git push origin main

echo.
echo ========================================
echo Push completed successfully!
echo ========================================
pause
