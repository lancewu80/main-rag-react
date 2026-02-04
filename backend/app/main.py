from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 創建 FastAPI 應用
app = FastAPI(
    title="🤖 智能 RAG 問答系統",
    description="基於本地文件和網路搜尋的智能問答系統 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發時允許所有來源，生產環境應限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 導入路由
from .routers import system, documents, qa, knowledge

# 註冊路由
app.include_router(system.router, prefix="/api/system", tags=["系統管理"])
app.include_router(documents.router, prefix="/api/documents", tags=["文件管理"])
app.include_router(qa.router, prefix="/api/qa", tags=["智能問答"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["知識庫管理"])

@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "🤖 歡迎使用智能 RAG 問答系統 API",
        "docs": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "system": "/api/system",
            "documents": "/api/documents",
            "qa": "/api/qa",
            "knowledge": "/api/knowledge"
        }
    }

@app.get("/api")
async def api_info():
    """API 信息"""
    return {
        "name": "RAG QA System API",
        "version": "1.0.0",
        "description": "智能問答系統後端 API",
        "routes": [
            {"path": "/api/system/health", "method": "GET", "description": "健康檢查"},
            {"path": "/api/system/info", "method": "GET", "description": "系統信息"},
            {"path": "/api/documents/list", "method": "GET", "description": "列出文檔"},
            {"path": "/api/documents/upload", "method": "POST", "description": "上傳文檔"},
            {"path": "/api/qa/ask", "method": "POST", "description": "提問"},
            {"path": "/api/knowledge/build", "method": "POST", "description": "建立知識庫"},
            {"path": "/api/knowledge/status", "method": "GET", "description": "知識庫狀態"}
        ]
    }

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 啟動智能 RAG 問答系統後端")
    print("=" * 50)
    print("📚 API 文檔: http://localhost:8000/docs")
    print("🌐 前端應用: http://localhost:3000")
    print("🔧 健康檢查: http://localhost:8000/api/system/health")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )