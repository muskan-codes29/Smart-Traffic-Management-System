@echo off
REM One-click launcher for the Smart Traffic Management System (Windows)
cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt

echo Starting dashboard at http://localhost:8501 ...
streamlit run app.py
pause
