@echo off
echo 🏥 Kidney Stone Detection - Setup Script
echo ===============================================

echo.
echo 1. Creating virtual environment...
python -m venv kidney_stone_env

echo.
echo 2. Activating virtual environment...
call kidney_stone_env\Scripts\activate.bat

echo.
echo 3. Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 4. Installing dependencies...
pip install -r requirements.txt

echo.
echo 5. Checking data structure...
python -c "from utils import validate_data_structure; valid, msg = validate_data_structure('data/train'); print(f'Data validation: {msg}')"

echo.
echo ✅ Setup complete! 
echo.
echo Next steps:
echo   1. Run: kidney_stone_env\Scripts\activate
echo   2. Run: python train.py
echo   3. Run: python app.py
echo.
pause
