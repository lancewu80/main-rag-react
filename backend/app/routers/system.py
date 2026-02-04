from fastapi import APIRouter, HTTPException
from datetime import datetime
import os
import sys
import json
from pathlib import Path

router = APIRouter()

@router.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "rag-qa-system"
    }

@router.get("/info")
async def system_info():
    """系統資訊"""
    try:
        # 基本系統資訊
        info = {
            "system": {
                "python_version": sys.version,
                "platform": sys.platform,
                "current_time": datetime.now().isoformat()
            },
            "paths": {
                "working_directory": str(Path.cwd()),
                "python_path": sys.path
            },
            "environment": dict(os.environ)
        }
        
        # 嘗試讀取配置
        config_path = Path("./vectordb/config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    info["config"] = json.load(f)
            except:
                info["config"] = {"error": "無法讀取配置檔案"}
        else:
            info["config"] = {"status": "未找到配置檔案"}
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_config():
    """獲取系統配置"""
    try:
        config_path = Path("./vectordb/config.json")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"message": "未找到配置檔案"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))