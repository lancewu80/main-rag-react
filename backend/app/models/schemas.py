from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class DocumentUpload(BaseModel):
    filename: str
    size: int
    upload_time: datetime = datetime.now()

class Question(BaseModel):
    text: str
    type: str = "rag"  # rag, web, hybrid
    timestamp: datetime = datetime.now()

class Answer(BaseModel):
    text: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.0
    processing_time: Optional[float] = None

class SystemStatus(BaseModel):
    status: str
    version: str
    uptime: Optional[float] = None
    memory_usage: Optional[Dict[str, Any]] = None
    database_status: str