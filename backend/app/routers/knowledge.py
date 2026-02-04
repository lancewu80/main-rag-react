from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import os
import json
from datetime import datetime

router = APIRouter()

class BuildRequest(BaseModel):
    force: bool = False
    incremental: bool = True

class BuildResponse(BaseModel):
    task_id: str
    status: str
    message: str
    timestamp: str

# 模擬的知識庫建置任務
build_tasks = {}

async def mock_build_knowledge(task_id: str, force: bool = False):
    """模擬知識庫建置過程"""
    try:
        build_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "正在初始化..."
        }
        
        # 模擬處理步驟
        steps = [
            ("正在檢查文檔...", 10),
            ("正在載入文件...", 20),
            ("正在分割文本...", 40),
            ("正在計算嵌入...", 60),
            ("正在建立向量索引...", 80),
            ("正在保存資料庫...", 95),
            ("完成！", 100)
        ]
        
        for message, progress in steps:
            await asyncio.sleep(1)  # 模擬處理時間
            build_tasks[task_id] = {
                "status": "processing",
                "progress": progress,
                "message": message,
                "updated_at": datetime.now().isoformat()
            }
        
        build_tasks[task_id]["status"] = "completed"
        build_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        build_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }

@router.post("/build")
async def build_knowledge_base(
    request: BuildRequest,
    background_tasks: BackgroundTasks
):
    """建立或重建知識庫"""
    try:
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        # 啟動背景任務
        background_tasks.add_task(
            mock_build_knowledge,
            task_id=task_id,
            force=request.force
        )
        
        return BuildResponse(
            task_id=task_id,
            status="started",
            message="知識庫建置任務已啟動",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/task/{task_id}")
async def get_build_task_status(task_id: str):
    """獲取建置任務狀態"""
    try:
        if task_id not in build_tasks:
            raise HTTPException(status_code=404, detail="任務不存在")
        
        return build_tasks[task_id]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def knowledge_base_status():
    """獲取知識庫狀態"""
    try:
        db_dir = "./vectordb"
        config_file = "./vectordb/config.json"
        
        exists = os.path.exists(db_dir)
        config_exists = os.path.exists(config_file)
        
        status = {
            "database_exists": exists,
            "config_exists": config_exists,
            "path": db_dir
        }
        
        if config_exists:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                status["config"] = config
            except:
                status["config"] = {"error": "無法讀取配置"}
        
        if exists:
            # 計算資料庫大小
            total_size = 0
            for root, dirs, files in os.walk(db_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            status["size"] = total_size
            status["size_mb"] = round(total_size / 1024 / 1024, 2)
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update")
async def update_knowledge_base(background_tasks: BackgroundTasks):
    """增量更新知識庫"""
    try:
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        # 啟動背景任務（與建置類似但訊息不同）
        async def mock_update_knowledge(task_id: str):
            try:
                build_tasks[task_id] = {
                    "status": "processing",
                    "progress": 0,
                    "message": "正在檢查更新..."
                }
                
                steps = [
                    ("正在檢查文檔變更...", 20),
                    ("正在載入新文件...", 40),
                    ("正在處理變更...", 60),
                    ("正在更新向量索引...", 80),
                    ("完成更新！", 100)
                ]
                
                for message, progress in steps:
                    await asyncio.sleep(0.5)
                    build_tasks[task_id] = {
                        "status": "processing",
                        "progress": progress,
                        "message": message
                    }
                
                build_tasks[task_id]["status"] = "completed"
                
            except Exception as e:
                build_tasks[task_id] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        background_tasks.add_task(mock_update_knowledge, task_id)
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "知識庫更新任務已啟動",
            "type": "incremental"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))