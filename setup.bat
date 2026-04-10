@echo off
setlocal

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 goto :error

echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist .streamlit mkdir .streamlit
if not exist .streamlit\secrets.toml (
    echo Creating .streamlit\secrets.toml from example...
    copy .streamlit\secrets.example.toml .streamlit\secrets.toml >nul
) else (
    echo .streamlit\secrets.toml already exists.
)

echo.
echo Setup complete.
echo Edit .streamlit\secrets.toml before first real use.
echo.
pause
exit /b 0

:error
echo.
echo Setup failed.
pause
exit /b 1
