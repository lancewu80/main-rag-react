from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import os
import json
from datetime import datetime
import requests
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

router = APIRouter()

# 配置
DB_DIR = "./vectordb"
DOCS_DIR = "./docs"

class QuestionRequest(BaseModel):
    question: str
    type: str = "rag"  # rag, web, hybrid
    options: Optional[Dict[str, Any]] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

# 初始化 LLM
def init_llm():
    """初始化 Ollama LLM"""
    models = ["qwen2.5:latest", "deepseek-r1:8b", "llama3.2:3b", "mistral:latest"]
    for m in models:
        try:
            llm = Ollama(model=m)
            # 測試連接
            llm.invoke("hello")
            print(f"✅ 使用模型：{m}")
            return llm
        except Exception as e:
            print(f"❌ 模型 {m} 載入失敗: {e}")
            continue
    # 如果都沒有，返回模擬 LLM
    print("⚠️  使用模擬 LLM")
    return None

# 初始化向量資料庫
def init_vector_db():
    """初始化向量資料庫"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'}
        )

        if os.path.exists(DB_DIR):
            vectordb = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
            print("✅ 向量資料庫載入成功")
            return vectordb
        else:
            print("⚠️  向量資料庫不存在")
            return None
    except Exception as e:
        print(f"❌ 向量資料庫載入失敗: {e}")
        return None

# 初始化搜尋工具
def init_search_tool():
    """初始化搜尋工具"""
    try:
        search = DuckDuckGoSearchRun()
        return search
    except Exception as e:
        print(f"❌ 搜尋工具初始化失敗: {e}")
        return None

# 全域變數
llm = init_llm()
vectordb = init_vector_db()
search_tool = init_search_tool()

def extract_keywords(text: str) -> List[str]:
    """簡單的關鍵詞提取"""
    # 移除常見停用詞
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '嗎', '？', '什麼'}
    words = [word for word in text if word not in stop_words and len(word.strip()) > 0]
    return list(set(words))

async def rag_qa_internal(question: str, k: int = 4) -> QuestionResponse:
    """RAG 問答內部實現"""
    if not vectordb:
        raise HTTPException(status_code=500, detail="向量資料庫未初始化")

    try:
        # 從向量資料庫檢索相關文檔
        docs = vectordb.similarity_search(question, k=k)

        if not docs:
            return QuestionResponse(
                answer="抱歉，在本地文檔中沒有找到相關資訊。",
                sources=[],
                metadata={"type": "rag", "documents_found": 0}
            )

        # 準備上下文
        context_parts = []
        sources = []

        for i, doc in enumerate(docs, 1):
            source_info = {
                "source": doc.metadata.get('source', '未知文件'),
                "content_preview": doc.page_content[:100] + "...",
                "relevance_score": 0.8 + (i * 0.05),  # 模擬相關度分數
                "chunk_id": doc.metadata.get('chunk_id', '')
            }
            sources.append(source_info)
            context_parts.append(f"[文檔{i}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # 如果沒有 LLM，使用模擬回答
        if not llm:
            answer = f"根據您的文檔，關於「{question}」：\n\n"
            answer += "相關文檔摘要：\n"
            for i, doc in enumerate(docs[:2], 1):
                answer += f"{i}. {doc.page_content[:150]}...\n\n"
            answer += "\n（這是模擬回答，實際會使用 AI 模型生成）"

            return QuestionResponse(
                answer=answer,
                sources=sources,
                metadata={
                    "type": "rag",
                    "documents_found": len(docs),
                    "model_used": "simulated",
                    "processing_time": 0.5
                }
            )

        # 使用真實 LLM 生成回答
        prompt = f"""請根據以下文檔內容回答問題：

問題：{question}

相關文檔：
{context}

請基於以上文檔回答，如果文檔中沒有相關資訊，請明確說明。
回答要簡潔、準確，並可以引用文檔編號。

回答："""

        # 生成回答
        start_time = datetime.now()
        answer = llm.invoke(prompt)
        processing_time = (datetime.now() - start_time).total_seconds()

        return QuestionResponse(
            answer=answer,
            sources=sources,
            metadata={
                "type": "rag",
                "documents_found": len(docs),
                "model_used": llm.model if hasattr(llm, 'model') else "ollama",
                "processing_time": round(processing_time, 2)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 問答失敗: {str(e)}")

async def web_qa_internal(question: str) -> QuestionResponse:
    """網路問答內部實現"""
    if not search_tool:
        raise HTTPException(status_code=500, detail="搜尋工具未初始化")

    try:
        # 使用 DuckDuckGo 搜尋
        search_results = search_tool.run(question)

        # 如果沒有 LLM，使用模擬回答
        if not llm:
            answer = f"關於「{question}」的網路搜尋結果：\n\n"
            answer += f"搜尋摘要：{search_results[:300]}...\n\n"
            answer += "（這是模擬回答，實際會整合多個搜尋結果並使用 AI 分析）"

            return QuestionResponse(
                answer=answer,
                sources=[
                    {
                        "source": "DuckDuckGo 搜尋",
                        "query": question,
                        "result_count": 1
                    }
                ],
                metadata={
                    "type": "web",
                    "search_engine": "DuckDuckGo",
                    "processing_time": 1.2
                }
            )

        # 使用 LLM 整理搜尋結果
        prompt = f"""請根據以下網路搜尋結果回答問題：

問題：{question}

搜尋結果：
{search_results[:2000]}  # 限制長度

請基於搜尋結果提供準確回答，註明資訊來源是網路搜尋。
如果搜尋結果中沒有足夠資訊，請明確說明。

回答："""

        start_time = datetime.now()
        answer = llm.invoke(prompt)
        processing_time = (datetime.now() - start_time).total_seconds()

        return QuestionResponse(
            answer=answer,
            sources=[
                {
                    "source": "網路搜尋",
                    "engine": "DuckDuckGo",
                    "query": question,
                    "result_preview": search_results[:200]
                }
            ],
            metadata={
                "type": "web",
                "search_engine": "DuckDuckGo",
                "processing_time": round(processing_time, 2),
                "model_used": llm.model if hasattr(llm, 'model') else "ollama"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"網路問答失敗: {str(e)}")

async def hybrid_qa_internal(question: str) -> QuestionResponse:
    """混合問答內部實現（簡化版）"""
    try:
        # 初始化變數
        rag_answer = None
        web_answer = None
        rag_sources = []
        web_sources = []
        errors = []

        # 嘗試 RAG 問答
        try:
            rag_response = await rag_qa_internal(question)
            if rag_response and hasattr(rag_response, 'answer'):
                rag_answer = rag_response.answer
                if hasattr(rag_response, 'sources'):
                    rag_sources = rag_response.sources
        except Exception as e:
            errors.append(f"RAG錯誤: {str(e)}")

        # 嘗試 Web 問答
        try:
            web_response = await web_qa_internal(question)
            if web_response and hasattr(web_response, 'answer'):
                web_answer = web_response.answer
                if hasattr(web_response, 'sources'):
                    web_sources = web_response.sources
        except Exception as e:
            errors.append(f"Web錯誤: {str(e)}")

        # 如果兩個都失敗
        if not rag_answer and not web_answer:
            error_msg = "；".join(errors) if errors else "未知錯誤"
            return QuestionResponse(
                answer=f"抱歉，無法回答您的問題。\n\n錯誤信息：{error_msg}",
                sources=[],
                metadata={
                    "type": "hybrid",
                    "rag_success": False,
                    "web_success": False,
                    "errors": errors
                }
            )

        # 構建回答
        if rag_answer and web_answer:
            # 兩個都有，進行整合
            answer = f"綜合回答「{question}」：\n\n"
            answer += "【本地文檔資訊】\n"
            answer += rag_answer[:400] + "\n\n"
            answer += "【網路搜尋資訊】\n"
            answer += web_answer[:400] + "\n\n"
            answer += "以上綜合了本地文檔和網路搜尋的資訊。"

            # 合併來源
            all_sources = rag_sources + web_sources

            return QuestionResponse(
                answer=answer,
                sources=all_sources,
                metadata={
                    "type": "hybrid",
                    "rag_success": True,
                    "web_success": True,
                    "rag_answer_length": len(rag_answer),
                    "web_answer_length": len(web_answer)
                }
            )
        elif rag_answer:
            # 只有 RAG
            return QuestionResponse(
                answer=f"【基於本地文檔】\n{rag_answer}\n\n（網路搜尋失敗）",
                sources=rag_sources,
                metadata={
                    "type": "hybrid",
                    "rag_success": True,
                    "web_success": False,
                    "note": "僅使用本地文檔"
                }
            )
        else:
            # 只有 Web
            return QuestionResponse(
                answer=f"【基於網路搜尋】\n{web_answer}\n\n（本地文檔搜尋失敗）",
                sources=web_sources,
                metadata={
                    "type": "hybrid",
                    "rag_success": False,
                    "web_success": True,
                    "note": "僅使用網路搜尋"
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"混合問答處理失敗: {str(e)}")

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """通用問答接口"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="問題不能為空")

        if request.type == "rag":
            return await rag_qa_internal(request.question)
        elif request.type == "web":
            return await web_qa_internal(request.question)
        elif request.type == "hybrid":
            return await hybrid_qa_internal(request.question)
        else:
            raise HTTPException(status_code=400, detail="不支持的問答類型")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"問答失敗: {str(e)}")

@router.post("/rag", response_model=QuestionResponse)
async def rag_qa(request: QuestionRequest):
    """RAG 問答"""
    return await ask_question(request)

@router.post("/web", response_model=QuestionResponse)
async def web_qa(request: QuestionRequest):
    """網路問答"""
    return await ask_question(request)

@router.post("/hybrid", response_model=QuestionResponse)
async def hybrid_qa(request: QuestionRequest):
    """混合問答"""
    return await ask_question(request)

@router.get("/models")
async def get_available_models():
    """獲取可用模型"""
    try:
        # 嘗試獲取 Ollama 模型列表
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {"models": [model["name"] for model in models]}
        else:
            return {"models": ["模擬模式"]}
    except:
        return {"models": ["模擬模式"]}
