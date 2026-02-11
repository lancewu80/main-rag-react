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

import ctypes
import os
import time
import json

from typing import Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加載 PyTorch 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 PyTorch 正在使用設備: {device}")
rerank_model_name = "BAAI/bge-reranker-base"
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
rerank_model.eval() # 設定為推論模式

def torch_rerank(query, documents, top_n=3):
    if not documents:
        return []

    # 確保模型在正確的設備 (GPU/CPU)
    rerank_model.to(device)

    pairs = [[query, doc.page_content] for doc in documents]

    with torch.no_grad():
        # 將數據移至設備
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

        # 模型推論
        logits = rerank_model(**inputs).logits
        scores = logits.view(-1,).float()

        # 排序並取出前 top_n 名
        scored_pairs = zip(scores.cpu().tolist(), documents)
        sorted_docs = sorted(scored_pairs, key=lambda x: x[0], reverse=True)

        # 這裡會用到傳入的 top_n
        return [doc for score, doc in sorted_docs[:top_n]]

# --- C 函式庫初始化 ---
# 取得目前檔案的絕對路徑，並指向 ../c/io_writer.dll
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "c", "io_writer.dll")

try:
    c_lib = ctypes.CDLL(DLL_PATH)
    c_lib.fast_write.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    c_lib.fast_write.restype = ctypes.c_double
    print(f"✅ 成功載入 C 擴展: {DLL_PATH}")
except Exception as e:
    print(f"❌ 無法載入 C 擴展: {e}")
    c_lib = None

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

# 配置 - 修正：統一使用絕對路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(BASE_DIR, "vectordb")  # 改成絕對路徑
DOCS_DIR = os.path.join(BASE_DIR, "docs")    # 改成絕對路徑
OLLAMA_HOST = "http://localhost:11434"  # Ollama 預設地址

print(f"QA系統配置:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"DB_DIR: {DB_DIR}")
print(f"DOCS_DIR: {DOCS_DIR}")

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
            embedding_function=embeddings,
            collection_name="default"
        )

        # 測試資料庫
        count = vectordb._collection.count()
        print(f"✅ RAG 向量資料庫初始化成功")
        print(f"   資料庫路徑: {DB_DIR}")
        print(f"   文件數量: {count}")

        if count == 0:
            print(f"⚠️  警告: 向量資料庫為空，請先使用 knowledge API 建置知識庫")

        return vectordb
    except Exception as e:
        print(f"❌ RAG 向量資料庫初始化失敗: {e}")
        import traceback
        traceback.print_exc()
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

async def call_ollama_api(
    prompt: str,
    model: Optional[str] = None
) -> Tuple[str, float]:
    """調用 Ollama API 生成回答"""

    if not OLLAMA_AVAILABLE or not PREFERRED_MODEL:
        return "⚠️ Ollama 服務未連接，請確保 Ollama 正在運行。", 0.0

    model_to_use: str = model or PREFERRED_MODEL

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

            if answer.startswith("。"):
                answer = answer[1:]
            if answer.startswith("，"):
                answer = answer[1:]

            print(f"✅ Ollama 回答生成成功，耗時: {processing_time:.2f}秒")
            return answer, processing_time
        else:
            return f"❌ Ollama API 錯誤: {response.status_code}", 0.0

    except requests.exceptions.Timeout:
        return "❌ Ollama 請求超時，請稍後再試。", 0.0
    except Exception as e:
        return f"❌ Ollama 調用失敗: {str(e)}", 0.0


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
            metadata = doc.metadata

            results.append({
                "index": i + 1,
                "content": content,
                "source": metadata.get("source", "未知來源"),
                "relevance": 1.0 - (i * 0.15),  # 簡單相關性評分
                "type": "rag"
            })

        print(f"✅ 找到 {len(results)} 個相關文檔")

        return {
            "status": "success",
            "message": f"找到 {len(results)} 個相關結果",
            "results": results,
            "query": query
        }

    except Exception as e:
        print(f"❌ RAG 檢索失敗: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "results": [],
            "query": query
        }

# RAG 問答 (已整合 PyTorch Rerank 優化版)
async def rag_qa_internal(question: str) -> QuestionResponse:
    """執行 RAG 問答流程，並透過 PyTorch 進行重排優化"""
    try:
        start_time = datetime.now()

        # 1. 初始檢索：擴大範圍至 k=10，讓 Reranker 有挑選空間
        rag_results = await search_rag(question, k=10)

        if rag_results["status"] == "error":
            raise Exception(rag_results["message"])

        if not rag_results["results"]:
            return QuestionResponse(
                answer="⚠️ 沒有找到相關的本地知識庫資訊。",
                sources=[],
                metadata={
                    "type": "rag",
                    "processing_time": round((datetime.now() - start_time).total_seconds(), 2),
                    "message": "知識庫中沒有相關內容"
                }
            )

        # 2. PyTorch 重排邏輯
        # 將 search_rag 的結果轉換為 torch_rerank 需要的 Document 格式物件
        class SimpleDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        initial_docs = [SimpleDoc(r['content'], {'source': r['source'], 'relevance': r['relevance']}) for r in rag_results["results"]]

        # 調用您定義的 PyTorch Rerank 函數 (取出分數最高的前 4 名)
        # [註] 此處會使用您剛寫好的 torch_rerank，且內部已處理 sorted_docs 定義問題
        rerank_start = time.time()
        final_docs = torch_rerank(question, initial_docs, top_n=4)
        rerank_time = time.time() - rerank_start

        # 3. 構建上下文 (使用重排後的精選內容)
        context = "【本地知識庫資訊 (已通過 PyTorch Rerank 優化)】\n\n"
        for i, doc in enumerate(final_docs):
            context += f"來源: {doc.metadata['source']}\n"
            context += f"內容: {doc.page_content}\n\n"

        # 生成提示詞
        prompt = f"""請根據以下本地知識庫資訊回答問題：

{context}

【問題】
{question}

請提供：
1. 核心資訊摘要
2. 具體細節和數據
3. 實用建議（如適用）

用繁體中文回答，保持專業、客觀且易於理解。註明資訊來源為本地知識庫。

回答："""

        # 調用 Ollama 生成回答
        print(f"🤖 正在生成回答 (模型: {PREFERRED_MODEL})...")
        answer, llm_time = await call_ollama_api(prompt)

        # 準備來源資訊 (反映重排後的順序)
        sources = []
        for doc in final_docs:
            sources.append({
                "source": f"本地知識庫: {doc.metadata['source']}",
                "relevance": "High (Reranked)",
                "type": "rag",
                "content_preview": doc.page_content[:100]
            })

        total_time = (datetime.now() - start_time).total_seconds()

        return QuestionResponse(
            answer=answer,
            sources=sources,
            metadata={
                "type": "rag_reranked",
                "model_used": PREFERRED_MODEL or "unknown",
                "ollama_available": OLLAMA_AVAILABLE,
                "processing_time": round(total_time, 2),
                "llm_time": round(llm_time, 2) if llm_time else 0,
                "rerank_time": round(rerank_time, 4),
                "results_count": len(final_docs),
                "device": str(device) # 顯示是用 CPU 還是 CUDA
            }
        )

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        return QuestionResponse(
            answer=f"RAG 問答失敗：{str(e)}",
            sources=[],
            metadata={
                "type": "rag",
                "error": str(e),
                "processing_time": round(error_time, 2)
            }
        )

# 網路問答
async def web_qa_internal(question: str) -> QuestionResponse:
    """執行網路問答流程"""
    try:
        start_time = datetime.now()

        # 網路搜尋
        web_results = await search_duckduckgo(question, max_results=5)

        if web_results["status"] == "error":
            raise Exception(web_results["message"])

        if not web_results["results"]:
            return QuestionResponse(
                answer="⚠️ 沒有找到相關的網路搜尋結果。",
                sources=[],
                metadata={
                    "type": "web",
                    "processing_time": round((datetime.now() - start_time).total_seconds(), 2),
                    "message": "網路搜尋沒有結果"
                }
            )

        # 構建上下文
        context = "【網路搜尋資訊】\n\n"
        for i, result in enumerate(web_results["results"], 1):
            context += f"結果 {i} (相關性: {result['relevance']:.2f}):\n"
            context += f"{result['content']}\n\n"

        # 生成提示詞
        prompt = f"""請根據以下網路搜尋結果回答問題：

{context}

【問題】
{question}

請提供：
1. 核心資訊摘要
2. 最新動態和趨勢
3. 實用建議

用繁體中文回答，注意資訊的時效性。註明資訊來源為網路搜尋。

回答："""

        # 調用 Ollama 生成回答
        print("🤖 正在生成回答...")
        answer, llm_time = await call_ollama_api(prompt)

        # 準備來源資訊
        sources = []
        for result in web_results["results"]:
            sources.append({
                "source": f"DuckDuckGo 搜尋結果 #{result['index']}",
                "relevance": result["relevance"],
                "type": "web",
                "content_preview": result["content"][:100]
            })

        total_time = (datetime.now() - start_time).total_seconds()

        return QuestionResponse(
            answer=answer,
            sources=sources,
            metadata={
                "type": "web",
                "model_used": PREFERRED_MODEL or "unknown",
                "ollama_available": OLLAMA_AVAILABLE,
                "processing_time": round(total_time, 2),
                "llm_time": round(llm_time, 2) if llm_time else 0,
                "search_engine": web_results.get("search_engine", "DuckDuckGo"),
                "results_count": len(web_results["results"])
            }
        )

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        return QuestionResponse(
            answer=f"網路問答失敗：{str(e)}",
            sources=[],
            metadata={
                "type": "web",
                "error": str(e),
                "processing_time": round(error_time, 2)
            }
        )

# 混合問答
async def hybrid_qa_internal(question: str) -> QuestionResponse:
    """執行混合問答流程（RAG + Web Search）"""
    try:
        start_time = datetime.now()

        # 並行執行 RAG 檢索和網路搜尋
        print("🔄 正在執行混合檢索...")
        rag_results, web_results = await asyncio.gather(
            search_rag(question, k=4),
            search_duckduckgo(question, max_results=5)
        )

        print(f"RAG 狀態: {rag_results['status']}, 結果數: {len(rag_results['results'])}")
        print(f"Web 狀態: {web_results['status']}, 結果數: {len(web_results['results'])}")

        # 檢查是否有可用的結果
        rag_has_content = rag_results["status"] == "success" and len(rag_results["results"]) > 0
        web_has_content = web_results["status"] == "success" and len(web_results["results"]) > 0

# 準備要寫入的數據
        log_data = json.dumps({
            "question": question,
            "rag": rag_results,
            "web": web_results,
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False)
        data_bytes = log_data.encode('utf-8')

        # 1. Python 寫入測試
        py_start = time.perf_counter()
        with open("perf_python.json", "w", encoding="utf-8") as f:
            f.write(log_data*10000)
        py_duration = time.perf_counter() - py_start

# --- 2. C 寫入測試 (修正變數範圍問題) ---
# 修正後的 C 寫入測試
        c_duration = -1.0
        filename = "perf_c.json"
        target_path = os.path.join(BASE_DIR, filename)
        abs_target_path = os.path.normpath(os.path.abspath(target_path))

        if c_lib:
            try:
                # 💡 嘗試將路徑轉為 Windows 系統原生編碼 (重要！)
                # 如果 utf-8 會報 Errno 22 (無效參數)，請改用 'mbcs'
                try:
                    c_path_bytes = abs_target_path.encode('mbcs')
                except:
                    c_path_bytes = abs_target_path.encode('utf-8')

                # 確保 data_bytes 也是正確的 bytes
                if isinstance(log_data, str):
                    data_bytes = (log_data*10000).encode('utf-8')
                else:
                    data_bytes = log_data*10000

                c_duration = c_lib.fast_write(c_path_bytes, data_bytes)
            except Exception as e:
                print(f"❌ 呼叫 C DLL 時發生異常: {e}")

        # --- 3. 顯示結果 ---
        print(f"--- I/O Performance Analysis ---")
        print(f"Target Path:  {abs_target_path}")
        print(f"Python Write: {py_duration:.6f} s")
        print(f"C Write:      {c_duration:.6f} s")

        if c_duration == -1.0:
            print(f"❌ 錯誤提示: C 語言無法開啟檔案。原因可能是權限不足、路徑錯誤或 DLL 載入失敗。")
        elif c_duration > 0:
            print(f"Speedup:      {py_duration / (c_duration if c_duration > 0 else 0.000001):.2f}x")
        print(f"--------------------------------")

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
