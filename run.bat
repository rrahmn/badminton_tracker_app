@echo off
setlocal

if not exist .venv\Scripts\activate (
    echo Virtual environment not found.
    echo Run setup.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate
if errorlevel 1 goto :error

if not exist .streamlit\secrets.toml (
    echo secrets.toml not found.
    echo Creating it from example now...
    if not exist .streamlit mkdir .streamlit
    copy .streamlit\secrets.example.toml .streamlit\secrets.toml >nul
    echo Please edit .streamlit\secrets.toml if needed.
)

echo Starting Streamlit app...
streamlit run app.py
exit /b %errorlevel%

:error
echo.
echo Failed to activate the virtual environment.
pause
exit /b 1
