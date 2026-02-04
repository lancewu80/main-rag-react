from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Query, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import os
import json
import csv
import shutil
import hashlib
from datetime import datetime
import uuid
from pathlib import Path
import re

# 匯入真實的文本處理庫
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

# Chroma 向量資料庫
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import chromadb
    from chromadb.config import Settings
    VECTOR_DB_AVAILABLE = True
    print("✓ Chroma 向量資料庫可用")
except ImportError as e:
    VECTOR_DB_AVAILABLE = False
    print(f"✗ Chroma 向量資料庫不可用: {e}")

router = APIRouter()

class BuildRequest(BaseModel):
    force: bool = False
    incremental: bool = True

class DocumentInfo(BaseModel):
    filename: str
    size: int
    size_mb: float
    modified_time: str
    file_type: str
    status: str
    path: str
    last_processed: Optional[str] = None
    chunks: Optional[int] = 0

class KnowledgeSearchRequest(BaseModel):
    query: str
    limit: int = 10

class CSVImportRequest(BaseModel):
    has_headers: bool = True
    delimiter: str = ","
    content_column: Optional[str] = None

# 實際的知識庫建置任務存儲
build_tasks = {}

# 獲取項目根目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"BASE_DIR: {BASE_DIR}")

# 設定正確的目錄路徑
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
UPLOAD_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "processed")
DOCS_DIR = os.path.join(BASE_DIR, "docs")  # 原始文檔目錄
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb")  # 向量資料庫目錄

# Chroma 設置 - 使用獨立的 chroma_db 目錄
CHROMA_DB_PATH = os.path.join(VECTOR_DB_DIR, "chroma_db")
CHROMA_COLLECTION_NAME = "knowledge_base"

METADATA_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "metadata.json")
DOCUMENT_METADATA_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "document_metadata.json")

print(f"KNOWLEDGE_BASE_DIR: {KNOWLEDGE_BASE_DIR}")
print(f"VECTOR_DB_DIR: {VECTOR_DB_DIR}")
print(f"CHROMA_DB_PATH: {CHROMA_DB_PATH}")
print(f"DOCS_DIR: {DOCS_DIR}")

# 創建必要的目錄
for directory in [KNOWLEDGE_BASE_DIR, UPLOAD_DIR, PROCESSED_DIR, VECTOR_DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# 文本分割配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def get_text_splitter():
    """獲取文本分割器"""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", ",", " ", ""],
        length_function=len
    )

def get_embeddings():
    """獲取嵌入模型"""
    if not VECTOR_DB_AVAILABLE:
        print("✗ 向量資料庫不可用，無法獲取嵌入模型")
        return None

    try:
        print("正在載入嵌入模型...")
        # 使用輕量級的中文友好嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ 嵌入模型載入成功")
        return embeddings
    except Exception as e:
        print(f"✗ 載入嵌入模型失敗: {e}")
        return None

def get_vector_store(embeddings=None, create_new=False):
    """獲取向量資料庫實例"""
    if not VECTOR_DB_AVAILABLE:
        print("✗ 向量資料庫不可用: VECTOR_DB_AVAILABLE = False")
        return None

    try:
        if embeddings is None:
            print("正在獲取嵌入模型...")
            embeddings = get_embeddings()
            if embeddings is None:
                print("✗ 無法獲取嵌入模型")
                return None
            else:
                print("✓ 嵌入模型獲取成功")

        print(f"Chroma 資料庫路徑: {CHROMA_DB_PATH}")
        print(f"路徑是否存在: {os.path.exists(CHROMA_DB_PATH)}")

        # 詳細檢查目錄內容
        if os.path.exists(CHROMA_DB_PATH):
            print(f"Chroma 目錄內容:")
            items = os.listdir(CHROMA_DB_PATH)
            if items:
                for item in items:
                    item_path = os.path.join(CHROMA_DB_PATH, item)
                    if os.path.isfile(item_path):
                        print(f"  文件: {item} ({os.path.getsize(item_path)} bytes)")
                    else:
                        print(f"  目錄: {item}")
            else:
                print("  目錄為空")

        # 檢查 Chroma 資料庫文件是否存在
        chroma_db_exists = os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0
        print(f"Chroma 資料庫文件存在: {chroma_db_exists}")

        if create_new or not chroma_db_exists:
            print(f"{'創建新的' if create_new else '沒有現有文件，創建新的'} Chroma 資料庫")

            # 如果目錄已存在，先清理
            if os.path.exists(CHROMA_DB_PATH):
                try:
                    print(f"清理舊的 Chroma 資料庫: {CHROMA_DB_PATH}")
                    shutil.rmtree(CHROMA_DB_PATH)
                    print("✓ 清理完成")
                except Exception as e:
                    print(f"清理 Chroma 資料庫失敗: {e}")

            # 創建目錄
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            print(f"創建目錄: {CHROMA_DB_PATH}")

            # 創建新的 Chroma 資料庫 - 使用新 API
            try:
                print("正在創建新的 Chroma 資料庫實例...")

                # 新版本的 Chroma 初始化方式
                vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=CHROMA_DB_PATH,
                    collection_name=CHROMA_COLLECTION_NAME
                )

                print("✓ Chroma 資料庫創建成功")
                return vector_store

            except Exception as e:
                print(f"✗ 創建 Chroma 資料庫失敗: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("正在載入現有 Chroma 資料庫...")
            # 載入現有 Chroma 資料庫 - 使用新 API
            try:
                vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=CHROMA_DB_PATH,
                    collection_name=CHROMA_COLLECTION_NAME
                )

                print("✓ Chroma 資料庫載入成功")

                # 測試是否能讀取
                try:
                    collection = vector_store._collection
                    count = collection.count()
                    print(f"✓ Chroma 資料庫包含 {count} 個文檔")
                except Exception as e:
                    print(f"讀取 Chroma 資料庫信息失敗: {e}")
                    # 如果讀取失敗，資料庫可能損壞
                    print("資料庫可能損壞，嘗試重新創建...")
                    return get_vector_store(embeddings, create_new=True)

                return vector_store

            except Exception as e:
                print(f"✗ 載入 Chroma 資料庫失敗: {e}")
                import traceback
                traceback.print_exc()
                # 如果載入失敗，嘗試重新創建
                print("嘗試重新創建資料庫...")
                return get_vector_store(embeddings, create_new=True)

    except Exception as e:
        print(f"✗ 獲取向量資料庫失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_file_hash(filepath: str) -> str:
    """計算文件哈希值"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""

def load_document_metadata():
    """載入文檔元數據"""
    if os.path.exists(DOCUMENT_METADATA_FILE):
        try:
            with open(DOCUMENT_METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_document_metadata(metadata):
    """保存文檔元數據"""
    with open(DOCUMENT_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata():
    """載入系統元數據"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        "documents": {},
        "build_history": [],
        "vector_db": {
            "created_at": None,
            "last_updated": None,
            "total_documents": 0,
            "total_chunks": 0,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "vector_db_available": VECTOR_DB_AVAILABLE,
            "vector_db_path": CHROMA_DB_PATH
        }
    }

def save_metadata(metadata):
    """保存系統元數據"""
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def get_supported_extensions():
    """獲取支援的文件擴展名"""
    return ['.txt', '.pdf', '.md', '.csv', '.json']

def is_supported_file(filename: str) -> bool:
    """檢查文件是否支援"""
    return any(filename.lower().endswith(ext) for ext in get_supported_extensions())

def load_document(filepath: str):
    """載入文檔內容"""
    filename = os.path.basename(filepath)

    try:
        if filename.endswith('.txt'):
            loader = TextLoader(filepath, encoding='utf-8')
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif filename.endswith('.md'):
            loader = UnstructuredMarkdownLoader(filepath)
        elif filename.endswith('.csv'):
            loader = CSVLoader(filepath, encoding='utf-8')
        else:
            # 嘗試作為文本文件讀取
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                return [{
                    "page_content": content,
                    "metadata": {"source": filename}
                }]
            except:
                return []

        return loader.load()
    except Exception as e:
        print(f"載入文件 {filename} 失敗: {e}")
        return []

async def process_documents_task(task_id: str, file_paths: List[str], force: bool = False):
    """實際處理文檔的任務"""
    try:
        build_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "正在初始化...",
            "type": "build",
            "started_at": datetime.now().isoformat(),
            "processed_files": [],
            "total_chunks": 0,
            "errors": []
        }

        metadata = load_metadata()
        doc_metadata = load_document_metadata()

        print(f"\n=== 開始處理 {len(file_paths)} 個文件，強制重建: {force} ===")

        # 獲取向量資料庫實例
        print("正在初始化向量資料庫...")
        vector_store = get_vector_store(create_new=force)

        # 正確的判斷邏輯
        if vector_store is not None:
            print("✓ 向量資料庫初始化成功")
            # 檢查資料庫中的文檔數量
            try:
                count = vector_store._collection.count()
                print(f"✓ 向量資料庫目前包含 {count} 個文檔")
            except Exception as e:
                print(f"⚠ 無法獲取向量資料庫文檔數量: {e}")
                print(f"⚠ 但仍然可以使用向量資料庫")
        else:
            print("✗ 向量資料庫初始化失敗，vector_store 為 None")

        total_files = len(file_paths)
        processed_count = 0

        # 處理每個文件
        all_chunks = []

        for i, file_path in enumerate(file_paths):
            filename = os.path.basename(file_path)

            # 更新進度
            progress = int((i / total_files) * 100)
            build_tasks[task_id].update({
                "progress": progress,
                "message": f"正在處理: {filename}",
                "updated_at": datetime.now().isoformat()
            })

            try:
                print(f"\n--- 處理文件 {i+1}/{total_files}: {filename} ---")
                print(f"文件路徑: {file_path}")

                # 檢查文件是否存在
                if not os.path.exists(file_path):
                    error_msg = f"文件不存在: {file_path}"
                    build_tasks[task_id]["errors"].append(error_msg)
                    print(error_msg)
                    continue

                # 檢查文件是否已處理（增量模式）
                file_hash = calculate_file_hash(file_path)
                file_mtime = os.path.getmtime(file_path)

                if not force and filename in doc_metadata:
                    existing_meta = doc_metadata[filename]
                    if existing_meta.get("hash") == file_hash and existing_meta.get("mtime") == file_mtime:
                        print(f"文件 {filename} 未修改，跳過處理")
                        continue

                # 載入文檔
                print(f"載入文件: {filename}")
                documents = load_document(file_path)
                if not documents:
                    error_msg = f"無法載入文件: {filename}"
                    build_tasks[task_id]["errors"].append(error_msg)
                    print(error_msg)
                    continue

                print(f"成功載入，原始文件數: {len(documents)}")

                # 分割文本
                splitter = get_text_splitter()
                chunks = splitter.split_documents(documents)
                print(f"分割後塊數: {len(chunks)}")

                # 為每個塊添加元數據
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": filename,
                        "file_type": filename.split('.')[-1] if '.' in filename else "unknown",
                        "processed_at": datetime.now().isoformat(),
                        "chunk_id": hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
                    })

                all_chunks.extend(chunks)

                # 更新文檔元數據
                doc_metadata[filename] = {
                    "hash": file_hash,
                    "mtime": file_mtime,
                    "last_processed": datetime.now().isoformat(),
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "chunks": len(chunks)
                }

                # 更新系統元數據
                metadata["documents"][filename] = {
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2),
                    "modified_time": datetime.fromtimestamp(file_mtime).isoformat(),
                    "file_type": filename.split('.')[-1] if '.' in filename else "unknown",
                    "status": "processed",
                    "last_processed": datetime.now().isoformat(),
                    "chunks": len(chunks),
                    "path": file_path
                }

                processed_count += 1
                build_tasks[task_id]["processed_files"].append(filename)

                # 移動文件到已處理目錄（如果是上傳的文件）
                if file_path.startswith(UPLOAD_DIR):
                    processed_path = os.path.join(PROCESSED_DIR, filename)
                    print(f"移動文件到已處理目錄: {processed_path}")
                    shutil.move(file_path, processed_path)

                await asyncio.sleep(0.1)  # 避免阻塞

            except Exception as e:
                error_msg = f"處理文件 {filename} 失敗: {str(e)}"
                build_tasks[task_id]["errors"].append(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()

        # 保存元數據
        save_document_metadata(doc_metadata)

        # 更新向量資料庫 - 修正邏輯
        if all_chunks:
            print(f"\n=== 準備更新向量資料庫，共 {len(all_chunks)} 個 chunks ===")

            # 正確檢查 vector_store 是否可用
            if vector_store is not None and VECTOR_DB_AVAILABLE:
                print(f"✓ 使用向量資料庫 (vector_store: {vector_store is not None}, VECTOR_DB_AVAILABLE: {VECTOR_DB_AVAILABLE})")

                try:
                    # 添加到向量資料庫
                    print("正在添加文檔到向量資料庫...")

                    # 檢查當前數量
                    before_count = 0
                    try:
                        before_count = vector_store._collection.count()
                        print(f"添加前文檔數量: {before_count}")
                    except Exception as e:
                        print(f"⚠ 無法獲取添加前數量: {e}")

                    # 批量添加文檔
                    batch_size = 50
                    success_count = 0
                    for i in range(0, len(all_chunks), batch_size):
                        batch = all_chunks[i:i + batch_size]
                        print(f"添加批次 {i//batch_size + 1}，共 {len(batch)} 個文檔")
                        try:
                            vector_store.add_documents(batch)
                            success_count += len(batch)
                            print(f"批次 {i//batch_size + 1} 添加成功")

                            # 持久化
                            try:
                                vector_store.persist()
                                print(f"批次 {i//batch_size + 1} 持久化完成")
                            except Exception as e:
                                print(f"⚠ 批次 {i//batch_size + 1} 持久化失敗: {e}")

                        except Exception as e:
                            print(f"✗ 批次 {i//batch_size + 1} 添加失敗: {e}")

                    print(f"✓ 向量資料庫更新完成，成功添加 {success_count}/{len(all_chunks)} 個文檔")

                    # 檢查更新後的數量
                    try:
                        after_count = vector_store._collection.count()
                        print(f"✓ 向量資料庫中的文檔數量: {after_count}")
                        if before_count > 0:
                            print(f"✓ 新增了 {after_count - before_count} 個文檔")
                    except Exception as e:
                        print(f"無法獲取向量資料庫數量: {e}")

                    build_tasks[task_id]["message"] += " (已更新向量資料庫)"
                    build_tasks[task_id]["vector_db_updated"] = True

                except Exception as e:
                    error_msg = f"更新向量資料庫失敗: {str(e)}"
                    build_tasks[task_id]["errors"].append(error_msg)
                    print(f"✗ {error_msg}")
                    import traceback
                    traceback.print_exc()
                    build_tasks[task_id]["vector_db_updated"] = False

                    # 如果向量資料庫更新失敗，保存到 chunks.json 作為備份
                    print("正在保存到備份文件...")
                    chunks_file = os.path.join(VECTOR_DB_DIR, "chunks.json")
                    chunks_data = []

                    for chunk in all_chunks:
                        chunks_data.append({
                            "content": chunk.page_content,
                            "source": chunk.metadata.get("source", "unknown"),
                            "chunk_id": chunk.metadata.get("chunk_id", ""),
                            "file_type": chunk.metadata.get("file_type", "unknown"),
                            "processed_at": chunk.metadata.get("processed_at", "")
                        })

                    print(f"保存 {len(chunks_data)} 個 chunks 到備份文件")
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                    print("✓ 備份文件保存完成")
            else:
                print(f"⚠ 向量資料庫不可用 (vector_store: {vector_store is not None}, VECTOR_DB_AVAILABLE: {VECTOR_DB_AVAILABLE})")
                # 保存塊信息到文件（備用）
                chunks_file = os.path.join(VECTOR_DB_DIR, "chunks.json")
                chunks_data = []

                # 載入現有chunks
                existing_chunks = []
                if os.path.exists(chunks_file):
                    try:
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            existing_chunks = json.load(f)
                        print(f"載入現有 {len(existing_chunks)} 個 chunks")
                    except:
                        print("無法載入現有 chunks")

                # 添加新chunks
                for chunk in all_chunks:
                    chunks_data.append({
                        "content": chunk.page_content,
                        "source": chunk.metadata.get("source", "unknown"),
                        "chunk_id": chunk.metadata.get("chunk_id", ""),
                        "file_type": chunk.metadata.get("file_type", "unknown"),
                        "processed_at": chunk.metadata.get("processed_at", "")
                    })

                # 合併並保存（避免重複）
                all_chunks_data = existing_chunks + chunks_data
                # 去重
                seen = set()
                unique_chunks = []
                for chunk in all_chunks_data:
                    chunk_id = chunk.get("chunk_id", "")
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        unique_chunks.append(chunk)

                print(f"保存 {len(unique_chunks)} 個唯一 chunks 到 {chunks_file}")
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_chunks, f, ensure_ascii=False, indent=2)
                print("✓ chunks.json 保存完成")

                build_tasks[task_id]["message"] += " (使用關鍵詞搜索)"
                build_tasks[task_id]["vector_db_updated"] = False

            # 更新向量資料庫信息
            metadata["vector_db"].update({
                "last_updated": datetime.now().isoformat(),
                "total_documents": len(metadata["documents"]),
                "total_chunks": len(all_chunks),
                "vector_db_available": VECTOR_DB_AVAILABLE,
                "vector_db_path": CHROMA_DB_PATH,
                "collection_name": CHROMA_COLLECTION_NAME,
                "vector_store_available": vector_store is not None,
                "vector_store_updated": build_tasks[task_id].get("vector_db_updated", False)
            })

        # 添加建置歷史
        metadata["build_history"].append({
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "type": "full" if force else "incremental",
            "processed_files": processed_count,
            "total_chunks": len(all_chunks),
            "errors": build_tasks[task_id]["errors"],
            "vector_db_updated": build_tasks[task_id].get("vector_db_updated", False)
        })

        save_metadata(metadata)

        # 完成任務
        build_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"處理完成！成功處理 {processed_count}/{total_files} 個文件",
            "completed_at": datetime.now().isoformat(),
            "total_files": total_files,
            "successful_files": processed_count,
            "total_chunks": len(all_chunks),
            "vector_db_updated": build_tasks[task_id].get("vector_db_updated", False)
        })

        print(f"\n=== 任務完成: {build_tasks[task_id]['message']} ===")

    except Exception as e:
        build_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
        import traceback
        traceback.print_exc()

async def import_csv_task(task_id: str, csv_filepath: str, has_headers: bool = True,
                         delimiter: str = ",", content_column: Optional[str] = None):
    """實際處理 CSV 導入任務"""
    try:
        build_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "正在讀取 CSV 文件...",
            "type": "csv_import",
            "started_at": datetime.now().isoformat(),
            "imported_rows": 0,
            "errors": []
        }

        filename = os.path.basename(csv_filepath)

        # 讀取 CSV 文件
        rows = []
        try:
            with open(csv_filepath, 'r', encoding='utf-8') as f:
                if has_headers:
                    csv_reader = csv.DictReader(f, delimiter=delimiter)
                    rows = list(csv_reader)
                else:
                    csv_reader = csv.reader(f, delimiter=delimiter)
                    rows = list(csv_reader)

                    # 如果沒有標題，創建通用欄位名
                    if rows:
                        headers = [f"column_{i+1}" for i in range(len(rows[0]))]
                        rows = [dict(zip(headers, row)) for row in rows]
        except Exception as e:
            build_tasks[task_id] = {
                "status": "failed",
                "error": f"讀取 CSV 文件失敗: {str(e)}",
                "failed_at": datetime.now().isoformat()
            }
            return

        total_rows = len(rows)

        if total_rows == 0:
            build_tasks[task_id] = {
                "status": "failed",
                "error": "CSV 文件為空",
                "failed_at": datetime.now().isoformat()
            }
            return

        # 更新進度
        build_tasks[task_id].update({
            "progress": 25,
            "message": f"正在解析 {total_rows} 行數據...",
            "updated_at": datetime.now().isoformat()
        })

        # 確定內容欄位
        if not content_column and rows:
            # 自動尋找可能的內容欄位
            possible_columns = ['content', 'text', 'description', '摘要', '內容', '說明']
            for col in possible_columns:
                if col in rows[0]:
                    content_column = col
                    break

            # 如果沒找到，使用第一個欄位
            if not content_column:
                content_column = list(rows[0].keys())[0]

        # 處理數據
        processed_rows = []
        for i, row in enumerate(rows):
            # 更新進度
            if i % 10 == 0:  # 每10行更新一次進度
                progress = 25 + int((i / total_rows) * 70)
                build_tasks[task_id].update({
                    "progress": progress,
                    "message": f"正在處理第 {i+1}/{total_rows} 行...",
                    "updated_at": datetime.now().isoformat()
                })

            try:
                # 提取內容
                if content_column and content_column in row:
                    content = row[content_column]
                else:
                    # 合併所有欄位作為內容
                    content = " | ".join([f"{k}: {v}" for k, v in row.items()])

                processed_rows.append({
                    "content": content,
                    "metadata": row,
                    "row_number": i + 1
                })

                await asyncio.sleep(0.01)  # 避免阻塞

            except Exception as e:
                error_msg = f"處理第 {i+1} 行失敗: {str(e)}"
                build_tasks[task_id]["errors"].append(error_msg)

        # 保存處理後的數據
        import_data = {
            "source_file": filename,
            "imported_at": datetime.now().isoformat(),
            "total_rows": total_rows,
            "processed_rows": len(processed_rows),
            "has_headers": has_headers,
            "delimiter": delimiter,
            "content_column": content_column,
            "sample_data": processed_rows[:3] if processed_rows else [],
            "all_data": [row["content"] for row in processed_rows[:50]]  # 只保存前50行內容
        }

        import_filename = f"imported_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import_filepath = os.path.join(PROCESSED_DIR, import_filename)

        with open(import_filepath, 'w', encoding='utf-8') as f:
            json.dump(import_data, f, ensure_ascii=False, indent=2)

        # 將 CSV 內容轉換為文本文件供後續處理
        text_content = []
        for row in processed_rows:
            text_content.append(row["content"])

        text_filename = f"csv_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        text_filepath = os.path.join(PROCESSED_DIR, text_filename)

        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(text_content))

        # 更新元數據
        metadata = load_metadata()
        metadata["documents"][text_filename] = {
            "filename": text_filename,
            "size": os.path.getsize(text_filepath),
            "size_mb": round(os.path.getsize(text_filepath) / 1024 / 1024, 2),
            "modified_time": datetime.now().isoformat(),
            "file_type": "txt",
            "status": "processed",
            "last_processed": datetime.now().isoformat(),
            "chunks": len(processed_rows),
            "source_csv": filename,
            "path": text_filepath
        }

        save_metadata(metadata)

        # 移動原始 CSV 文件
        processed_csv_path = os.path.join(PROCESSED_DIR, filename)
        shutil.move(csv_filepath, processed_csv_path)

        # 完成任務
        build_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"CSV 導入完成！處理了 {len(processed_rows)}/{total_rows} 行",
            "completed_at": datetime.now().isoformat(),
            "imported_rows": len(processed_rows),
            "output_file": import_filepath,
            "text_file": text_filepath
        })

    except Exception as e:
        build_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }

async def search_knowledge_task(task_id: str, query: str, limit: int = 10):
    """實際知識檢索任務"""
    try:
        build_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "正在解析查詢...",
            "type": "search",
            "query": query,
            "started_at": datetime.now().isoformat(),
            "results": [],
            "total_found": 0
        }

        print(f"\n=== 知識搜索: {query} ===")

        # 首先嘗試使用向量資料庫搜索
        print("正在初始化向量資料庫...")
        vector_store = get_vector_store()

        if vector_store:
            try:
                print(f"使用向量資料庫搜索: {query}")
                # 向量相似度搜索
                results_docs = vector_store.similarity_search_with_score(query, k=limit)
                print(f"向量搜索結果數: {len(results_docs)}")

                results = []

                for doc, score in results_docs:
                    results.append({
                        "id": doc.metadata.get("chunk_id", str(uuid.uuid4())[:8]),
                        "content": doc.page_content,
                        "source_file": doc.metadata.get("source", "unknown"),
                        "similarity": max(0.0, 1.0 - float(score)),  # 轉換為相似度
                        "metadata": {
                            "score": score,
                            "match_type": "vector_similarity",
                            "file_type": doc.metadata.get("file_type", "unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", "")
                        }
                    })

                build_tasks[task_id]["search_type"] = "vector"
                build_tasks[task_id]["vector_db_used"] = True
                print("✓ 使用向量搜索完成")

            except Exception as e:
                print(f"向量搜索失敗: {e}")
                # 回退到關鍵詞搜索
                results = await keyword_search(query, limit)
                build_tasks[task_id]["search_type"] = "keyword_fallback"
                build_tasks[task_id]["vector_db_used"] = False
                print("✓ 使用關鍵詞搜索完成 (向量搜索失敗回退)")
        else:
            print("向量資料庫不可用，使用關鍵詞搜索")
            # 使用關鍵詞搜索
            results = await keyword_search(query, limit)
            build_tasks[task_id]["search_type"] = "keyword"
            build_tasks[task_id]["vector_db_available"] = VECTOR_DB_AVAILABLE
            print("✓ 使用關鍵詞搜索完成")

        # 按分數排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # 更新進度
        build_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"找到 {len(results)} 個相關結果",
            "completed_at": datetime.now().isoformat(),
            "results": results[:limit],
            "total_found": len(results)
        })

        print(f"=== 搜索完成，找到 {len(results)} 個結果 ===")

    except Exception as e:
        build_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }

async def keyword_search(query: str, limit: int = 10):
    """關鍵詞搜索備用方案"""
    # 載入所有塊數據
    chunks_file = os.path.join(VECTOR_DB_DIR, "chunks.json")
    chunks = []
    if os.path.exists(chunks_file):
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except:
            pass

    query_lower = query.lower()
    results = []

    for chunk in chunks:
        content = chunk.get("content", "").lower()
        source = chunk.get("source", "")

        # 計算匹配分數
        score = 0

        # 關鍵詞匹配
        keywords = query_lower.split()
        for keyword in keywords:
            if len(keyword) > 2:  # 忽略太短的詞
                if keyword in content:
                    score += 1
                if keyword in source.lower():
                    score += 2

        if score > 0:
            results.append({
                "id": chunk.get("chunk_id", str(uuid.uuid4())[:8]),
                "content": chunk.get("content", ""),
                "source_file": source,
                "similarity": min(score / 5, 0.99),  # 正規化分數
                "metadata": {
                    "score": score,
                    "match_type": "keyword"
                }
            })

    return results

@router.post("/build")
async def build_knowledge_base(
    request: BuildRequest,
    background_tasks: BackgroundTasks
):
    """建立或重建知識庫"""
    try:
        task_id = str(uuid.uuid4())[:8]

        # 收集要處理的文件
        file_paths = []

        # 檢查上傳目錄
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                filepath = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(filepath) and is_supported_file(filename):
                    file_paths.append(filepath)
                    print(f"找到上傳文件: {filename}")

        # 檢查 docs 目錄
        if not request.incremental or not file_paths:
            if os.path.exists(DOCS_DIR):
                for filename in os.listdir(DOCS_DIR):
                    filepath = os.path.join(DOCS_DIR, filename)
                    if os.path.isfile(filepath) and is_supported_file(filename):
                        file_paths.append(filepath)
                        print(f"找到 docs 文件: {filename}")

        if not file_paths:
            print("沒有找到可處理的文件")
            raise HTTPException(status_code=400, detail="沒有可處理的文件")

        print(f"\n=== 知識庫建置任務啟動 ===")
        print(f"處理文件數: {len(file_paths)}")
        print(f"增量模式: {request.incremental}")
        print(f"強制重建: {request.force}")
        print(f"向量資料庫路徑: {CHROMA_DB_PATH}")
        print(f"向量資料庫可用: {VECTOR_DB_AVAILABLE}")

        # 啟動背景任務
        background_tasks.add_task(
            process_documents_task,
            task_id=task_id,
            file_paths=file_paths,
            force=request.force
        )

        return {
            "task_id": task_id,
            "status": "started",
            "message": f"知識庫建置任務已啟動，將處理 {len(file_paths)} 個文件",
            "type": "full" if request.force else "incremental",
            "timestamp": datetime.now().isoformat(),
            "vector_db_available": VECTOR_DB_AVAILABLE,
            "vector_db_path": CHROMA_DB_PATH
        }

    except Exception as e:
        print(f"建置任務啟動失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...)
):
    """上傳文件到知識庫"""
    try:
        uploaded_files = []

        for file in files:
            # 檢查文件類型
            filename = file.filename
            if not is_supported_file(filename):
                continue

            # 保存文件
            file_path = os.path.join(UPLOAD_DIR, filename)
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)

            uploaded_files.append({
                "filename": filename,
                "size": len(content),
                "size_mb": round(len(content) / 1024 / 1024, 2),
                "saved_path": file_path
            })

        return {
            "status": "success",
            "message": f"成功上傳 {len(uploaded_files)} 個文件",
            "uploaded_files": uploaded_files,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/csv/import")
async def import_csv_file(
    file: UploadFile = File(...),
    has_headers: bool = Form(True),
    delimiter: str = Form(","),
    content_column: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """導入 CSV 文件到知識庫"""
    try:
        task_id = str(uuid.uuid4())[:8]

        # 保存上傳的 CSV 文件
        csv_filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(csv_filepath, 'wb') as f:
            content = await file.read()
            f.write(content)

        # 啟動背景任務
        background_tasks.add_task(
            import_csv_task,
            task_id=task_id,
            csv_filepath=csv_filepath,
            has_headers=has_headers,
            delimiter=delimiter,
            content_column=content_column
        )

        return {
            "task_id": task_id,
            "status": "started",
            "message": f"CSV 導入任務已啟動: {file.filename}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_knowledge(
    request: KnowledgeSearchRequest,
    background_tasks: BackgroundTasks
):
    """搜索知識庫內容"""
    try:
        # 先立即執行搜索（同步）
        print(f"\n=== 立即執行知識搜索: {request.query} ===")

        # 直接執行搜索邏輯
        vector_store = get_vector_store()
        results = []

        if vector_store:
            try:
                print(f"使用向量資料庫搜索: {request.query}")
                # 向量相似度搜索
                results_docs = vector_store.similarity_search_with_score(request.query, k=request.limit)
                print(f"向量搜索結果數: {len(results_docs)}")

                for doc, score in results_docs:
                    results.append({
                        "id": doc.metadata.get("chunk_id", str(uuid.uuid4())[:8]),
                        "content": doc.page_content,
                        "source_file": doc.metadata.get("source", "unknown"),
                        "similarity": max(0.0, 1.0 - float(score)),  # 轉換為相似度
                        "metadata": {
                            "score": score,
                            "match_type": "vector_similarity",
                            "file_type": doc.metadata.get("file_type", "unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", "")
                        }
                    })

                search_type = "vector"
                vector_db_used = True
                print("✓ 使用向量搜索完成")

            except Exception as e:
                print(f"向量搜索失敗: {e}")
                # 回退到關鍵詞搜索
                results = await keyword_search(request.query, request.limit)
                search_type = "keyword_fallback"
                vector_db_used = False
                print("✓ 使用關鍵詞搜索完成 (向量搜索失敗回退)")
        else:
            print("向量資料庫不可用，使用關鍵詞搜索")
            # 使用關鍵詞搜索
            results = await keyword_search(request.query, request.limit)
            search_type = "keyword"
            vector_db_used = False
            print("✓ 使用關鍵詞搜索完成")

        # 按分數排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # 創建任務ID
        task_id = str(uuid.uuid4())[:8]

        # 立即返回結果
        response_data = {
            "task_id": task_id,
            "status": "completed",
            "message": f"找到 {len(results)} 個相關結果",
            "timestamp": datetime.now().isoformat(),
            "search_type": search_type,
            "vector_db_used": vector_db_used,
            "results": results[:request.limit],
            "total_found": len(results)
        }

        # 同時啟動背景任務記錄
        build_tasks[task_id] = response_data.copy()
        build_tasks[task_id]["type"] = "search"
        build_tasks[task_id]["query"] = request.query
        build_tasks[task_id]["started_at"] = datetime.now().isoformat()
        build_tasks[task_id]["completed_at"] = datetime.now().isoformat()

        return response_data

    except Exception as e:
        print(f"搜索失敗: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
      
@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """獲取任務狀態"""
    try:
        if task_id not in build_tasks:
            raise HTTPException(status_code=404, detail="任務不存在")

        return build_tasks[task_id]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def list_documents(
    status: str = Query("all", description="文件狀態: all, uploaded, processed"),
    file_type: str = Query(None, description="文件類型過濾")
):
    """獲取所有文件列表"""
    try:
        documents = []
        metadata = load_metadata()

        # 獲取已處理的文件
        if status in ["all", "processed"]:
            for filename, doc_info in metadata.get("documents", {}).items():
                if file_type and doc_info.get("file_type") != file_type:
                    continue

                documents.append(doc_info)

        # 獲取上傳的文件
        if status in ["all", "uploaded"]:
            if os.path.exists(UPLOAD_DIR):
                for filename in os.listdir(UPLOAD_DIR):
                    filepath = os.path.join(UPLOAD_DIR, filename)
                    if os.path.isfile(filepath):
                        if file_type and not filename.endswith(f".{file_type}"):
                            continue

                        stat = os.stat(filepath)
                        documents.append({
                            "filename": filename,
                            "size": stat.st_size,
                            "size_mb": round(stat.st_size / 1024 / 1024, 2),
                            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "file_type": filename.split('.')[-1] if '.' in filename else "unknown",
                            "status": "uploaded",
                            "path": filepath
                        })

        return {
            "total": len(documents),
            "documents": documents
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_knowledge_status():
    """獲取知識庫狀態"""
    try:
        metadata = load_metadata()

        # 檢查 Chroma 資料庫是否存在
        vector_db_exists = False
        chroma_files = [
            os.path.join(CHROMA_DB_PATH, "chroma.sqlite3"),
            os.path.join(CHROMA_DB_PATH, "chroma.sqlite3.wal"),
            os.path.join(CHROMA_DB_PATH, "index"),
            os.path.join(CHROMA_DB_PATH, "chroma.sqlite3-shm")
        ]

        # 檢查是否有任何 Chroma 文件
        for file_path in chroma_files:
            if os.path.exists(file_path):
                vector_db_exists = True
                break

        # 如果沒有找到標準文件，檢查是否有 .parquet 文件
        if not vector_db_exists and os.path.exists(CHROMA_DB_PATH):
            for file in os.listdir(CHROMA_DB_PATH):
                if file.endswith('.parquet'):
                    vector_db_exists = True
                    break

        # 檢查目錄
        status = {
            "knowledge_base_exists": os.path.exists(KNOWLEDGE_BASE_DIR),
            "upload_dir_exists": os.path.exists(UPLOAD_DIR),
            "processed_dir_exists": os.path.exists(PROCESSED_DIR),
            "vector_db_exists": vector_db_exists,
            "vector_db_available": VECTOR_DB_AVAILABLE,
            "docs_dir_exists": os.path.exists(DOCS_DIR),
            "paths": {
                "knowledge_base": KNOWLEDGE_BASE_DIR,
                "uploads": UPLOAD_DIR,
                "processed": PROCESSED_DIR,
                "vector_db": CHROMA_DB_PATH,
                "docs": DOCS_DIR
            }
        }

        # 添加統計信息
        if metadata:
            status.update({
                "total_documents": metadata["vector_db"].get("total_documents", 0),
                "total_chunks": metadata["vector_db"].get("total_chunks", 0),
                "last_updated": metadata["vector_db"].get("last_updated"),
                "chunk_size": metadata["vector_db"].get("chunk_size", CHUNK_SIZE),
                "chunk_overlap": metadata["vector_db"].get("chunk_overlap", CHUNK_OVERLAP),
                "vector_db_available": metadata["vector_db"].get("vector_db_available", False),
                "vector_store_available": metadata["vector_db"].get("vector_store_available", False)
            })

        # 計算檔案大小
        total_size = 0
        for dir_path in [KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR, DOCS_DIR]:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)

        status["total_size"] = total_size
        status["total_size_mb"] = round(total_size / 1024 / 1024, 2)

        # 檢查各目錄的文件數量
        for dir_name, dir_path in [
            ("uploads", UPLOAD_DIR),
            ("processed", PROCESSED_DIR),
            ("docs", DOCS_DIR)
        ]:
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                status[f"{dir_name}_file_count"] = len(files)
            else:
                status[f"{dir_name}_file_count"] = 0

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/document/{filename}")
async def delete_document(filename: str):
    """刪除文件"""
    try:
        deleted = False
        message = ""

        # 檢查上傳目錄
        upload_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(upload_path):
            os.remove(upload_path)
            deleted = True
            message = f"已刪除上傳文件: {filename}"

        # 檢查處理目錄
        processed_path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(processed_path):
            os.remove(processed_path)
            deleted = True
            message = f"已刪除已處理文件: {filename}"

        # 更新元數據
        if deleted:
            metadata = load_metadata()
            if filename in metadata.get("documents", {}):
                del metadata["documents"][filename]
                save_metadata(metadata)

            return {
                "status": "success",
                "message": message
            }
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/build-history")
async def get_build_history(limit: int = Query(10, ge=1, le=100)):
    """獲取建置歷史"""
    try:
        metadata = load_metadata()
        history = metadata.get("build_history", [])

        # 按時間排序
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return {
            "total": len(history),
            "history": history[:limit]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
