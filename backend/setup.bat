@echo off
echo 安裝 RAG 後端依賴...
echo.

cd /d %~dp0

echo 創建虛擬環境...
python -m venv venv

echo 啟動虛擬環境...
call venv\Scripts\activate

echo 安裝依賴...
pip install --upgrade pip
pip install fastapi uvicorn python-multipart pydantic
pip install langchain langchain-community chromadb
pip install sentence-transformers duckduckgo-search requests pypdf

echo.
echo ✅ 安裝完成！
echo.
echo 啟動伺服器: run.bat
pause