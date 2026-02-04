AI智能問答系統安裝手冊
-------------------------
1. git clone https://github.com/lancewu80/main-rag-react.git

2. 設定後端backend
A. cmd: cd D:\project\ai\ollama\src\main-rag-react\backend
B. run setup.bat (這個要半小時)

3. 設定前端frontend
A. cmd: cd D:\project\ai\ollama\src\main-rag-react\frontend
B. npm install

4. 跑前端及後端
A. vs code打開目錄D:\project\ai\ollama\src\main-rag-react\
B. 按下Control+Shift+D,選擇啟動全端開發(Python+React)
C. 瀏覽器會自動開啟http://localhost:3000/

========================
# 建立專案資料夾
mkdir main-rag-react
cd main-rag-react

# 建立前端 React 專案
npx create-react-app frontend
cd frontend

# 安裝必要套件
npm install axios react-router-dom @mui/material @mui/icons-material @emotion/react @emotion/styled recharts


第二步：建立 FastAPI 後端
1. 後端專案結構
bash
# 在專案根目錄建立後端
cd main-rag-react
mkdir backend
cd backend

# 建立虛擬環境
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安裝依賴
pip install fastapi uvicorn python-multipart langchain langchain-community chromadb sentence-transformers pydantic duckduckgo-search requests pypdf


=====================
啟動後端伺服器

bash
# 終端機 1
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python -m app.main
啟動 React 前端

bash
# 終端機 2
cd frontend
npm install
npm start



後端啟動cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn
D:\project\ai\ollama\src\main-rag-react\backend\python -m app.main