@echo off
echo ========================================
echo PUSHING EVERYTHING TO GITHUB
echo ========================================
echo.
echo This will push ALL changes to your repository:
echo - capstone_new folder (6 files)
echo - Day_7_Mar02_Encoding updates (2 files)
echo - Modified Capstone_Projects files (2 files)
echo - New root files (5 files)
echo.
echo Total: 15+ files will be uploaded
echo ========================================
echo.

echo Step 1: Adding all files...
git add -A
echo ✓ All files added
echo.

echo Step 2: Committing changes...
git commit -m "Complete update: Add capstone projects with presentation guides, Day 7 encoding files, and other improvements"
echo ✓ Changes committed
echo.

echo Step 3: Pushing to GitHub...
git push origin main
echo ✓ Pushed to GitHub
echo.

echo ========================================
echo SUCCESS! All files uploaded to GitHub
echo ========================================
echo.
echo Check your repository at:
echo https://github.com/SIDDHARThMNC/DATA-SCIENCE-
echo.
pause
