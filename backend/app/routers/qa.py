from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import os
import json
from datetime import datetime
import requests
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun

router = APIRouter()

# ============ 安全處理函數 ============
def safe_round(value, decimals=2):
    """安全地進行四捨五入"""
    if value is None:
        return 0.0
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return 0.0

def safe_float(value):
    """安全地轉換為 float"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
# =====================================

# 配置
DB_DIR = "./vectordb"
DOCS_DIR = "./docs"
OLLAMA_HOST = "http://localhost:11434"  # Ollama 預設地址

# RAG 配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# 初始化 DuckDuckGo 搜尋
def init_duckduckgo_search():
    """初始化 DuckDuckGo 搜尋工具"""
    try:
        search_tool = DuckDuckGoSearchRun()
        # 測試搜尋
        test_result = search_tool.run("test")[:100]
        print(f"✅ DuckDuckGo 搜尋工具初始化成功")
        return search_tool
    except Exception as e:
        print(f"❌ DuckDuckGo 搜尋工具初始化失敗: {e}")
        return None

# 初始化 RAG 向量資料庫
def init_rag_vector_db():
    """初始化 RAG 向量資料庫"""
    try:
        if not os.path.exists(DB_DIR):
            print(f"❌ 向量資料庫不存在: {DB_DIR}")
            return None

        # 載入嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 載入向量資料庫
        vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        # 測試資料庫
        test_docs = vectordb.similarity_search("測試", k=1)
        print(f"✅ RAG 向量資料庫初始化成功，文件數: {vectordb._collection.count()}")
        return vectordb
    except Exception as e:
        print(f"❌ RAG 向量資料庫初始化失敗: {e}")
        return None

# 初始化搜尋工具和向量資料庫
DUCKDUCKGO_SEARCH = init_duckduckgo_search()
RAG_VECTORDB = init_rag_vector_db()

class QuestionRequest(BaseModel):
    question: str
    type: str = "rag"  # rag, web, hybrid
    options: Optional[Dict[str, Any]] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

# 檢查 Ollama 是否可用
def check_ollama_available():
    """檢查 Ollama 服務是否可用"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama 連接成功，可用模型: {[m['name'] for m in models]}")
            return True, models
    except Exception as e:
        print(f"❌ 無法連接到 Ollama: {e}")
        print(f"   請確保 Ollama 服務正在運行: ollama serve")
    return False, []

# 初始化時檢查 Ollama
OLLAMA_AVAILABLE, AVAILABLE_MODELS = check_ollama_available()

# 預設使用 DeepSeek R1，如果沒有則使用其他可用模型
def get_preferred_model():
    """獲取首選模型"""
    preferred_models = [
        "deepseek-r1:8b",
        "qwen2.5:7b",
        "llama3.2:3b",
        "mistral:7b",
        "gemma:2b"
    ]

    for model in preferred_models:
        for available in AVAILABLE_MODELS:
            if model in available.get('name', ''):
                print(f"✅ 使用模型: {model}")
                return model

    if AVAILABLE_MODELS:
        first_model = AVAILABLE_MODELS[0].get('name', '')
        print(f"✅ 使用可用模型: {first_model}")
        return first_model

    print("⚠️  沒有可用的 Ollama 模型")
    return None

PREFERRED_MODEL = get_preferred_model()

async def call_ollama_api(prompt: str, model: str = None) -> str:
    """調用 Ollama API 生成回答"""
    if not OLLAMA_AVAILABLE or not PREFERRED_MODEL:
        return "⚠️ Ollama 服務未連接，請確保 Ollama 正在運行。"

    model_to_use = model or PREFERRED_MODEL

    try:
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }

        start_time = datetime.now()
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            processing_time = (datetime.now() - start_time).total_seconds()

            answer = result.get("response", "").strip()

            # 清理回答
            if answer.startswith("。"):
                answer = answer[1:]
            if answer.startswith("，"):
                answer = answer[1:]

            print(f"✅ Ollama 回答生成成功，耗時: {processing_time:.2f}秒")
            return answer, processing_time
        else:
            return f"❌ Ollama API 錯誤: {response.status_code}", 0

    except requests.exceptions.Timeout:
        return "❌ Ollama 請求超時，請稍後再試。", 0
    except Exception as e:
        return f"❌ Ollama 調用失敗: {str(e)}", 0

# DuckDuckGo 搜尋函數
async def search_duckduckgo(query: str, max_results: int = 5) -> Dict[str, Any]:
    """使用 DuckDuckGo 進行搜尋"""
    if not DUCKDUCKGO_SEARCH:
        return {
            "status": "error",
            "message": "DuckDuckGo 搜尋工具未初始化",
            "results": [],
            "query": query
        }

    try:
        print(f"🔍 DuckDuckGo 搜尋: {query}")

        # 執行搜尋
        search_result = DUCKDUCKGO_SEARCH.run(query)

        # 解析搜尋結果（DuckDuckGoSearchRun 返回的是文本，需要解析）
        results = []

        if search_result:
            # 將搜尋結果分割成段落
            paragraphs = search_result.split('\n\n')
            for i, paragraph in enumerate(paragraphs[:max_results]):
                if paragraph.strip():
                    results.append({
                        "index": i + 1,
                        "content": paragraph.strip()[:500],  # 限制長度
                        "relevance": 1.0 - (i * 0.1),  # 簡單相關性評分
                        "type": "web_search"
                    })

        print(f"✅ 找到 {len(results)} 個搜尋結果")

        return {
            "status": "success",
            "message": f"找到 {len(results)} 個相關結果",
            "results": results,
            "query": query,
            "search_engine": "DuckDuckGo"
        }

    except Exception as e:
        print(f"❌ DuckDuckGo 搜尋失敗: {e}")
        return {
            "status": "error",
            "message": str(e),
            "results": [],
            "query": query
        }

# RAG 檢索函數
async def search_rag(query: str, k: int = 4) -> Dict[str, Any]:
    """使用 RAG 檢索本地知識庫"""
    if not RAG_VECTORDB:
        return {
            "status": "error",
            "message": "RAG 向量資料庫未初始化",
            "results": [],
            "query": query
        }

    try:
        print(f"📚 RAG 檢索: {query}")

        # 執行相似度搜尋
        docs = RAG_VECTORDB.similarity_search(query, k=k)

        results = []
        for i, doc in enumerate(docs):
            content = doc.page_content[:400]  # 限制長度
            source = doc.metadata.get('source', '未知')
            results.append({
                "index": i + 1,
                "content": content,
                "source": source,
                "relevance": 1.0 - (i * 0.15),  # 簡單相關性評分
                "type": "rag_document"
            })

        print(f"✅ 找到 {len(results)} 個本地知識庫結果")

        return {
            "status": "success",
            "message": f"找到 {len(results)} 個本地知識庫結果",
            "results": results,
            "query": query
        }

    except Exception as e:
        print(f"❌ RAG 檢索失敗: {e}")
        return {
            "status": "error",
            "message": str(e),
            "results": [],
            "query": query
        }

async def rag_qa_internal(question: str, k: int = 4) -> QuestionResponse:
    """RAG 問答內部實現（使用本地知識庫）"""
    start_time = datetime.now()

    try:
        # 先檢索本地知識庫
        rag_results = await search_rag(question, k)

        # 構建上下文
        if rag_results["status"] == "success" and rag_results["results"]:
            context = "【本地知識庫資訊】\n\n"
            for result in rag_results["results"]:
                context += f"來源: {result['source']}\n"
                context += f"內容: {result['content']}\n\n"
        else:
            context = "本地知識庫中沒有找到相關資訊。"

        # 構建提示詞
        prompt = f"""請根據以下本地知識庫資訊回答問題：

{context}

【問題】
{question}

請根據本地知識庫提供：
1. 準確、有用的資訊
2. 具體的細節和數據
3. 實用的建議
4. 相關的注意事項

如果知識庫中沒有足夠資訊，請基於您的知識補充說明。
用繁體中文回答，保持專業且易於理解。

回答："""

        # 調用 Ollama
        answer, llm_time = await call_ollama_api(prompt)

        # 計算總處理時間
        total_time = (datetime.now() - start_time).total_seconds()

        # 構建來源資訊
        sources = []
        if rag_results["status"] == "success" and rag_results["results"]:
            for result in rag_results["results"]:
                sources.append({
                    "source": f"本地知識庫: {result['source']}",
                    "relevance": result["relevance"],
                    "type": "rag",
                    "content_preview": result["content"][:100]
                })
        else:
            sources.append({
                "source": "AI 知識庫",
                "relevance": 0.9,
                "type": "ai"
            })

        sources.append({
            "source": "Ollama AI 分析",
            "relevance": 0.95,
            "type": "ai",
            "model": PREFERRED_MODEL
        })

        return QuestionResponse(
            answer=answer,
            sources=sources,
            metadata={
                "type": "rag",
                "model_used": PREFERRED_MODEL or "ai_model",
                "ollama_available": OLLAMA_AVAILABLE,
                "processing_time": safe_round(total_time, 2),
                "llm_time": safe_round(llm_time, 2),
                "rag_results_count": len(rag_results["results"]),
                "rag_status": rag_results["status"],
                "answer_source": "ollama_ai_with_rag"
            }
        )

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        return QuestionResponse(
            answer=f"處理問題時發生錯誤：{str(e)[:100]}",
            sources=[],
            metadata={
                "type": "rag",
                "error": str(e),
                "processing_time": safe_round(error_time, 2)
            }
        )

async def web_qa_internal(question: str) -> QuestionResponse:
    """網路問答內部實現（使用 DuckDuckGo 搜尋）"""
    start_time = datetime.now()

    try:
        # 使用 DuckDuckGo 進行搜尋
        search_results = await search_duckduckgo(question, max_results=5)

        # 構建搜尋結果上下文
        if search_results["status"] == "success" and search_results["results"]:
            search_context = "【網路搜尋結果】\n\n"
            for i, result in enumerate(search_results["results"], 1):
                content = result["content"]
                search_context += f"{i}. {content}\n\n"

            print(f"✅ 使用 {len(search_results['results'])} 個搜尋結果")
        else:
            # 如果搜尋失敗，使用模擬搜尋
            search_context = f"搜尋關鍵字：{question}\n\n搜尋結果：\n1. 相關網路資訊\n2. 新聞報導\n3. 用戶討論\n4. 官方資訊"
            print("⚠️ 使用模擬搜尋結果")

        # 構建給 Ollama 的提示詞
        prompt = f"""請根據以下搜尋結果回答問題：

問題：{question}

{search_context}

請基於搜尋結果提供：
1. 關鍵資訊摘要
2. 實用建議
3. 進一步查詢的方向

用繁體中文回答，註明資訊來源為網路搜尋。

回答："""

        # 調用 Ollama
        answer, llm_time = await call_ollama_api(prompt)

        total_time = (datetime.now() - start_time).total_seconds()

        # 構建來源資訊
        sources = []
        if search_results["status"] == "success" and search_results["results"]:
            for result in search_results["results"][:3]:  # 只取前3個
                sources.append({
                    "source": f"DuckDuckGo 搜尋結果 #{result['index']}",
                    "relevance": result["relevance"],
                    "type": "web",
                    "content_preview": result["content"][:100]
                })
        else:
            sources.append({
                "source": "模擬網路搜尋",
                "relevance": 0.8,
                "type": "web",
                "note": "實際搜尋未啟用或失敗"
            })

        sources.append({
            "source": "Ollama AI 分析",
            "relevance": 0.9,
            "type": "ai",
            "model": PREFERRED_MODEL
        })

        return QuestionResponse(
            answer=answer,
            sources=sources,
            metadata={
                "type": "web",
                "model_used": PREFERRED_MODEL or "simulation",
                "processing_time": round(total_time, 2),
                "llm_time": round(llm_time, 2) if llm_time else 0,
                "search_engine": search_results.get("search_engine", "simulated"),
                "search_status": search_results["status"],
                "search_results_count": len(search_results["results"])
            }
        )

    except Exception as e:
        return QuestionResponse(
            answer=f"網路搜尋失敗：{str(e)[:100]}",
            sources=[],
            metadata={
                "type": "web",
                "error": str(e),
                "processing_time": round((datetime.now() - start_time).total_seconds(), 2)
            }
        )

async def hybrid_qa_internal(question: str) -> QuestionResponse:
    """混合問答內部實現（結合 RAG 和 Web）"""
    start_time = datetime.now()

    try:
        print(f"🔀 開始混合問答: {question}")

        # 同時進行 RAG 檢索和 Web 搜尋
        rag_task = search_rag(question, k=4)
        web_task = search_duckduckgo(question, max_results=5)

        # 等待兩個任務完成
        rag_results, web_results = await asyncio.gather(rag_task, web_task)

        # 評估兩種來源的相關性
        rag_has_content = rag_results["status"] == "success" and rag_results["results"]
        web_has_content = web_results["status"] == "success" and web_results["results"]

        print(f"📊 檢索結果: RAG={len(rag_results['results']) if rag_has_content else 0} 個, Web={len(web_results['results']) if web_has_content else 0} 個")

        # 構建整合的上下文
        context_parts = []

        if rag_has_content:
            rag_context = "【本地知識庫資訊】\n\n"
            for result in rag_results["results"]:
                rag_context += f"來源: {result['source']}\n"
                rag_context += f"相關性: {result['relevance']:.2f}\n"
                rag_context += f"內容: {result['content']}\n\n"
            context_parts.append(rag_context)

        if web_has_content:
            web_context = "【網路搜尋資訊】\n\n"
            for i, result in enumerate(web_results["results"], 1):
                web_context += f"結果 {i} (相關性: {result['relevance']:.2f}):\n"
                web_context += f"{result['content']}\n\n"
            context_parts.append(web_context)

        if not rag_has_content and not web_has_content:
            context = "⚠️ 沒有找到相關的本地知識庫或網路資訊。"
        else:
            context = "\n".join(context_parts)

        # 根據可用的資訊類型構建不同的提示詞
        if rag_has_content and web_has_content:
            prompt = f"""請綜合以下本地知識庫和網路搜尋的資訊，提供一個全面、準確的回答：

{context}

【問題】
{question}

請根據以上資訊提供：
1. 核心資訊摘要（綜合兩方面資訊）
2. 具體細節和數據（優先使用本地知識庫的權威資訊）
3. 實用建議和操作步驟
4. 注意事項和風險提示

如果資訊有衝突：
- 技術性、專業性資訊以本地知識庫為準
- 時效性、新聞性資訊以網路搜尋為準
- 註明資訊來源（本地知識庫/網路搜尋）

用繁體中文回答，保持專業、客觀且易於理解。

綜合回答："""

        elif rag_has_content:
            prompt = f"""請根據以下本地知識庫資訊回答問題：

{context}

【問題】
{question}

請主要使用本地知識庫資訊回答，如果資訊不足可以補充您的通用知識。
註明資訊來源為本地知識庫。

回答："""

        elif web_has_content:
            prompt = f"""請根據以下網路搜尋結果回答問題：

{context}

【問題】
{question}

請主要使用網路搜尋結果回答，注意資訊的時效性。
註明資訊來源為網路搜尋。

回答："""

        else:
            prompt = f"""請根據您的知識回答以下問題：

【問題】
{question}

請提供準確、有用的資訊，並註明這是基於通用知識的回答。

回答："""

        # 調用 Ollama 生成整合回答
        print("🤖 正在生成綜合回答...")
        integrated_answer, llm_time = await call_ollama_api(prompt)

        # 合併來源資訊
        all_sources = []

        if rag_has_content:
            for result in rag_results["results"][:3]:  # 只取前3個
                all_sources.append({
                    "source": f"本地知識庫: {result['source']}",
                    "relevance": result["relevance"],
                    "type": "rag",
                    "content_preview": result["content"][:100]
                })

        if web_has_content:
            for result in web_results["results"][:5]:  # 只取前5個
                all_sources.append({
                    "source": f"DuckDuckGo 搜尋結果 #{result['index']}",
                    "relevance": result["relevance"],
                    "type": "web",
                    "content_preview": result["content"][:100]
                })

        # 添加 AI 分析來源
        all_sources.append({
            "source": "Ollama AI 綜合分析",
            "relevance": 0.95,
            "type": "ai",
            "model": PREFERRED_MODEL
        })

        total_time = (datetime.now() - start_time).total_seconds()

        return QuestionResponse(
            answer=integrated_answer,
            sources=all_sources,
            metadata={
                "type": "hybrid",
                "model_used": PREFERRED_MODEL or "integration",
                "ollama_available": OLLAMA_AVAILABLE,
                "processing_time": round(total_time, 2),
                "llm_time": round(llm_time, 2) if llm_time else 0,
                "rag_status": rag_results["status"],
                "rag_results_count": len(rag_results["results"]),
                "web_status": web_results["status"],
                "web_results_count": len(web_results["results"]),
                "search_engine": web_results.get("search_engine", "simulated"),
                "integration_method": "ollama_ai_integration",
                "sources_used": {
                    "rag": rag_has_content,
                    "web": web_has_content,
                    "ai": True
                }
            }
        )

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        return QuestionResponse(
            answer=f"混合問答失敗：{str(e)[:100]}",
            sources=[],
            metadata={
                "type": "hybrid",
                "error": str(e),
                "processing_time": round(error_time, 2)
            }
        )

@router.get("/status")
async def get_qa_status():
    """獲取問答系統狀態"""
    return {
        "status": "running",
        "service": "qa_system",
        "version": "1.0.0",
        "ollama": {
            "available": OLLAMA_AVAILABLE,
            "host": OLLAMA_HOST,
            "preferred_model": PREFERRED_MODEL,
            "available_models": [m.get('name', '') for m in AVAILABLE_MODELS]
        },
        "rag": {
            "available": RAG_VECTORDB is not None,
            "db_dir": DB_DIR,
            "embedding_model": EMBEDDING_MODEL,
            "document_count": RAG_VECTORDB._collection.count() if RAG_VECTORDB else 0
        },
        "search": {
            "duckduckgo_available": DUCKDUCKGO_SEARCH is not None,
            "search_tool": "DuckDuckGoSearchRun"
        },
        "capabilities": {
            "rag_enabled": RAG_VECTORDB is not None,
            "web_search_enabled": DUCKDUCKGO_SEARCH is not None,
            "hybrid_qa_enabled": (RAG_VECTORDB is not None) or (DUCKDUCKGO_SEARCH is not None),
            "ai_powered": OLLAMA_AVAILABLE
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/test")
async def test_endpoint():
    """測試端點"""
    return {
        "status": "ok",
        "message": "QA 系統工作正常",
        "ollama_status": "connected" if OLLAMA_AVAILABLE else "disconnected",
        "rag_status": "connected" if RAG_VECTORDB else "disconnected",
        "duckduckgo_status": "connected" if DUCKDUCKGO_SEARCH else "disconnected",
        "preferred_model": PREFERRED_MODEL,
        "test_suggestion": "請嘗試 POST /api/qa/hybrid 進行綜合問答測試"
    }

@router.get("/search-test")
async def test_search():
    """測試搜尋功能"""
    try:
        search_results = await search_duckduckgo("台灣天氣", max_results=3)
        return {
            "status": "ok",
            "search_test": "completed",
            "duckduckgo_available": DUCKDUCKGO_SEARCH is not None,
            "search_results": search_results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "duckduckgo_available": DUCKDUCKGO_SEARCH is not None
        }

@router.get("/rag-test")
async def test_rag():
    """測試 RAG 功能"""
    try:
        rag_results = await search_rag("測試", k=2)
        return {
            "status": "ok",
            "rag_test": "completed",
            "rag_available": RAG_VECTORDB is not None,
            "rag_results": rag_results,
            "document_count": RAG_VECTORDB._collection.count() if RAG_VECTORDB else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "rag_available": RAG_VECTORDB is not None
        }

@router.get("/models")
async def get_available_models():
    """獲取可用模型列表"""
    return {
        "available_models": [m.get('name', '') for m in AVAILABLE_MODELS],
        "preferred_model": PREFERRED_MODEL,
        "ollama_host": OLLAMA_HOST,
        "ollama_status": "connected" if OLLAMA_AVAILABLE else "disconnected"
    }

@router.post("/rag", response_model=QuestionResponse)
async def rag_qa(request: QuestionRequest):
    """RAG 問答端點"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="問題不能為空")
    return await rag_qa_internal(request.question.strip())

@router.post("/web", response_model=QuestionResponse)
async def web_qa(request: QuestionRequest):
    """網路問答端點"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="問題不能為空")
    return await web_qa_internal(request.question.strip())

@router.post("/hybrid", response_model=QuestionResponse)
async def hybrid_qa(request: QuestionRequest):
    """混合問答端點"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="問題不能為空")
    return await hybrid_qa_internal(request.question.strip())
