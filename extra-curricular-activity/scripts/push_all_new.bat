@echo off
echo ========================================
echo Pushing All New Files to GitHub
echo ========================================

echo.
echo Adding all new and modified files...
git add .

echo.
echo Committing changes...
git commit -m "Add capstone projects, Day 7 encoding files, and other updates"

echo.
echo Pushing to GitHub...
git push origin main

echo.
echo ========================================
echo Push completed successfully!
echo ========================================
pause
