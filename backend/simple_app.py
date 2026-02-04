"""
簡化版本的後端，用於快速測試
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "後端運行正常"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/test")
def test():
    return {
        "fastapi": "已成功導入",
        "message": "所有路由都正常工作",
        "endpoints": ["/", "/health", "/api/test", "/docs", "/redoc"]
    }

if __name__ == "__main__":
    print("🚀 啟動簡化版後端...")
    uvicorn.run(app, host="0.0.0.0", port=8000)