@echo off
echo 啟動 RAG 後端伺服器...
echo.

cd /d %~dp0

echo 檢查虛擬環境...
if not exist "venv\Scripts\activate.bat" (
    echo 請先運行 setup.bat 安裝依賴
    pause
    exit /b 1
)

echo 激活虛擬環境...
call venv\Scripts\activate

echo 啟動伺服器...
python -m app.main

pause