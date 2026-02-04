from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import json
from pathlib import Path
import shutil

router = APIRouter()

DOCS_DIR = "./docs"
METADATA_FILE = "./vectordb/document_metadata.json"

@router.get("/status")
async def document_status():
    """獲取文件狀態"""
    try:
        if not os.path.exists(METADATA_FILE):
            return {"documents": [], "total": 0}
        
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 檢查文件是否存在
        documents = []
        for filename, info in metadata.items():
            file_exists = os.path.exists(info.get('path', ''))
            documents.append({
                "filename": filename,
                "exists": file_exists,
                "size": info.get('size', 0),
                "last_processed": info.get('last_processed'),
                "hash": info.get('hash', '')[:8] + '...' if info.get('hash') else None
            })
        
        return {
            "documents": documents,
            "total": len(documents),
            "docs_dir": DOCS_DIR,
            "metadata_file": METADATA_FILE
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上傳文件"""
    try:
        # 確保文檔目錄存在
        os.makedirs(DOCS_DIR, exist_ok=True)
        
        # 檢查文件類型
        filename = file.filename
        if not (filename.endswith('.txt') or filename.endswith('.pdf')):
            raise HTTPException(status_code=400, detail="只支持 .txt 和 .pdf 文件")
        
        # 保存文件
        file_path = os.path.join(DOCS_DIR, filename)
        
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # 讀取文件大小
        file_size = os.path.getsize(file_path)
        
        return {
            "filename": filename,
            "size": file_size,
            "path": file_path,
            "message": "文件上傳成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{filename}")
async def delete_document(filename: str):
    """刪除文件"""
    try:
        # 安全檢查：防止路徑遍歷攻擊
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="無效的文件名")
        
        file_path = os.path.join(DOCS_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 刪除文件
        os.remove(file_path)
        
        # 更新元數據
        metadata_file = METADATA_FILE
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if filename in metadata:
                    del metadata[filename]
                    
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            except:
                pass
        
        return {
            "filename": filename,
            "message": "文件刪除成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_documents():
    """列出所有文件"""
    try:
        if not os.path.exists(DOCS_DIR):
            return {"documents": [], "total": 0}
        
        documents = []
        for filename in os.listdir(DOCS_DIR):
            if filename.endswith('.txt') or filename.endswith('.pdf'):
                file_path = os.path.join(DOCS_DIR, filename)
                if os.path.isfile(file_path):
                    documents.append({
                        "filename": filename,
                        "size": os.path.getsize(file_path),
                        "type": "txt" if filename.endswith('.txt') else "pdf"
                    })
        
        return {
            "documents": documents,
            "total": len(documents),
            "directory": DOCS_DIR
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))