# 功能说明：實現使用langchain框架，使用LCEL構建一個完整的LLM應用程序用於RAG知識庫的查詢，並使用fastapi進行發布
# 包含：langchain框架的使用，langsmith跟踪檢測

# 相關依賴庫
# pip install langchain langchain-openai langchain-chroma

# 引入相關庫
import os
import re
import json
import asyncio
import uuid
import time
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
# prompt模版
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
# 部署REST API相關
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
# 向量數據庫chroma相關
from langchain_chroma import Chroma
# openai的向量模型
from langchain_openai import OpenAIEmbeddings
# RAG相關
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# HuggingFace相關
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import AutoTokenizer
# 環境變數相關
import os
from dotenv import load_dotenv  # pip install python-dotenv
# torch
import torch
torch.cuda.empty_cache()

# 設置langsmith環境變量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# 設置日誌模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 載入 .env 檔案
load_dotenv()

# 指定向量數據庫chromaDB的存儲位置和集合，根據自己的實際情況進行調整
CHROMADB_DIRECTORY = os.getenv("CHROMADB_DIRECTORY") 
CHROMADB_COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME")
logger.info(f"CHROMADB_DIRECTORY: {CHROMADB_DIRECTORY}")
logger.info(f"CHROMADB_COLLECTION_NAME: {CHROMADB_COLLECTION_NAME}")

# prompt模版設置相關，根據自己的實際情況進行調整
PROMPT_PDF_TEMPLATE_TXT = "promt/"+os.getenv("PROMPT_PDF_TEMPLATE_TXT") 
logger.info(f"PROMPT_PDF_TEMPLATE_TXT: {PROMPT_PDF_TEMPLATE_TXT}")

# 模型設置相關，根據自己的實際情況進行調整
API_TYPE = os.getenv("API_TYPE") 
logger.info(f"API_TYPE: {API_TYPE}")

# openai模型相關配置，根據自己的實際情況進行調整
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") 
logger.info(f"OPENAI_API_BASE: {OPENAI_API_BASE}")
OPENAI_CHAT_API_KEY = os.getenv("OPENAI_CHAT_API_KEY") 
logger.info(f"OPENAI_CHAT_API_KEY: {OPENAI_CHAT_API_KEY}")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") 
logger.info(f"OPENAI_CHAT_MODEL: {OPENAI_CHAT_MODEL}")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY") 
logger.info(f"OPENAI_EMBEDDING_API_KEY: {OPENAI_EMBEDDING_API_KEY}")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") 
logger.info(f"OPENAI_EMBEDDING_MODEL: {OPENAI_EMBEDDING_MODEL}")

# LMSTUDIO模型相關配置，根據自己的實際情況進行調整
LMSTUDIO_API_BASE = os.getenv("LMSTUDIO_API_BASE") 
logger.info(f"LMSTUDIO_API_BASE: {LMSTUDIO_API_BASE}")
LMSTUDIO_CHAT_API_KEY = os.getenv("LMSTUDIO_CHAT_API_KEY") 
logger.info(f"LMSTUDIO_CHAT_API_KEY: {LMSTUDIO_CHAT_API_KEY}")
LMSTUDIO_CHAT_MODEL = os.getenv("LMSTUDIO_CHAT_MODEL") 
logger.info(f"LMSTUDIO_CHAT_MODEL: {LMSTUDIO_CHAT_MODEL}")
LMSTUDIO_EMBEDDING_API_KEY = os.getenv("LMSTUDIO_EMBEDDING_API_KEY") 
logger.info(f"LMSTUDIO_EMBEDDING_API_KEY: {LMSTUDIO_EMBEDDING_API_KEY}")
LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL") 
logger.info(f"LMSTUDIO_EMBEDDING_MODEL: {LMSTUDIO_EMBEDDING_MODEL}")

# OLLAMA模型相關配置，根據自己的實際情況進行調整
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE") 
logger.info(f"OLLAMA_API_BASE: {OLLAMA_API_BASE}")
OLLAMA_CHAT_API_KEY = os.getenv("OLLAMA_CHAT_API_KEY") 
logger.info(f"OLLAMA_CHAT_API_KEY: {OLLAMA_CHAT_API_KEY}")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL") 
logger.info(f"OLLAMA_CHAT_MODEL: {OLLAMA_CHAT_MODEL}")
OLLAMA_EMBEDDING_API_KEY = os.getenv("OLLAMA_EMBEDDING_API_KEY") 
logger.info(f"OLLAMA_EMBEDDING_API_KEY: {OLLAMA_EMBEDDING_API_KEY}")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL") 
logger.info(f"OLLAMA_EMBEDDING_MODEL: {OLLAMA_EMBEDDING_MODEL}")

# HUGGINGFACE模型相關配置，根據自己的實際情況進行調整
HUGGINGFACE_CHAT_MODEL=os.getenv("HUGGINGFACE_CHAT_MODEL") 
HUGGINGFACE_EMBEDDING_MODEL= os.getenv("HUGGINGFACE_EMBEDDING_MODEL") 

# API服務設置相關，根據自己的實際情況進行調整
PORT = int(os.getenv("PORT"))   # 服務訪問的端口
logger.info(f"PORT: {PORT}")

# 申明全局變量，全局調用
# query_content = ''   # 將chain中傳遞的用戶輸入的信息賦值到query_content
model = None  # 使用的LLM模型
embeddings = None  # 使用的Embedding模型
vectorstore = None  # 向量數據庫實例
prompt = None  # prompt內容
chain = None  # 定義的chain

# 定義Message類，用於封裝用戶或系統發送的消息
class Message(BaseModel):
    role: str
    content: str

# 定義ChatCompletionRequest類，用於處理聊天請求
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False

# 定義ChatCompletionResponseChoice類，用於封裝每個聊天回應的選項
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# 定義ChatCompletionResponse類，用於封裝聊天回應
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None

# 獲取prompt在chain中傳遞的prompt最終的內容
def getPrompt(prompt):
    logger.info(f"最後給到LLM的prompt的內容: {prompt}")
    return prompt

# 格式化回應，對輸入的文本進行段落分隔、添加適當的換行符，以及在代碼塊中增加標記，以便生成更具可讀性的輸出
def format_response(response):
    """
    格式化回應內容，優化段落分隔與代碼塊標註
    """
    paragraphs = re.split(r'\n{2,}', response)
    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # 代碼塊
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')
        formatted_paragraphs.append(para.strip())
    return '\n\n'.join(formatted_paragraphs)

# 定義一個異步函數 lifespan，用於管理應用生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    應用生命周期管理，包含啟動初始化與關閉清理
    """
    global model, embeddings, vectorstore, prompt, chain, API_TYPE, CHROMADB_DIRECTORY, CHROMADB_COLLECTION_NAME, PROMPT_TEMPLATE_TXT
    global OPENAI_CHAT_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    try:
        # 初始化模型判斷
        logger.info("正在初始化模型、Chroma對象、提取prompt模版、定義chain...")
        # HUGGINGFACE的維度只支援384、768、1024，如果不是這兩種維度，請用opeai的EMBEDDING
        if API_TYPE == "lmstudio":
            model = ChatOpenAI(
                base_url=LMSTUDIO_API_BASE,
                api_key=LMSTUDIO_CHAT_API_KEY,
                model=LMSTUDIO_CHAT_MODEL
            )
            #embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
            embeddings = OpenAIEmbeddings(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY,
                model=OPENAI_EMBEDDING_MODEL,
                )
        elif API_TYPE == "ollama":
            model = ChatOpenAI(
                base_url=OLLAMA_API_BASE,
                api_key=OLLAMA_CHAT_API_KEY,
                model=OLLAMA_CHAT_MODEL
            )
            embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
            # embeddings = OpenAIEmbeddings(
            #     base_url=OPENAI_API_BASE,
            #     api_key=OPENAI_EMBEDDING_API_KEY,
            #     model=OPENAI_EMBEDDING_MODEL,
            #     )
        elif API_TYPE == "huggingface":
            hf_pipeline = pipeline(
                "text-generation",
                model=HUGGINGFACE_CHAT_MODEL,
                device=-1,
                max_length=4096,  # 增加 max_length
                max_new_tokens=256  # 限制新生成的 tokens 長度
            )
            model = HuggingFacePipeline(pipeline=hf_pipeline)
            embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
            # embeddings = OpenAIEmbeddings(
            #     base_url=OPENAI_API_BASE,
            #     api_key=OPENAI_EMBEDDING_API_KEY,
            #     model=OPENAI_EMBEDDING_MODEL,
            #     )
        elif API_TYPE == "openai":
            model = ChatOpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_CHAT_API_KEY,
                model=OPENAI_CHAT_MODEL
            )
            #embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
            embeddings = OpenAIEmbeddings(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY,
                model=OPENAI_EMBEDDING_MODEL,
            )
        # 讀取向量資料庫
        vectorstore = Chroma(persist_directory=CHROMADB_DIRECTORY,
                             collection_name=CHROMADB_COLLECTION_NAME,
                             embedding_function=embeddings)
        
        # 讀取提示詞模板
        with open(PROMPT_PDF_TEMPLATE_TXT, 'r', encoding='utf-8') as f:
            template_content = f.read()
        prompt_template = PromptTemplate(input_variables=["query", "context"], template=template_content)
        prompt = ChatPromptTemplate.from_messages([("human", str(prompt_template.template))])

        # 最大邊際相關算法 mmr: Maximal Marginal Relevance
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 20, 'lambda_mult': 0.5}
        )

        # 文本相似度算法 cosine: Cosine Similarity
        # retriever = vectorstore.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={"k": 5}
        # )

        # 串連各個組件
        chain = {
                    "query": RunnablePassthrough(),
                    "context": retriever
                } | prompt | getPrompt | model

        logger.info("初始化完成")

    except Exception as e:
        logger.error(f"初始化過程中出錯: {str(e)}")
        raise

    yield

    logger.info("正在關閉...")

# lifespan 參數用於管理應用程序生命周期
app = FastAPI(lifespan=lifespan)

# 定義POST接口，與LLM進行知識問答
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    提供知識問答功能的POST接口
    """
    if not model or not embeddings or not vectorstore or not prompt or not chain:
        logger.error("服務未初始化")
        raise HTTPException(status_code=500, detail="服務未初始化")
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages不能為空")
    query_prompt = request.messages[-1].content
    for message in request.messages:
        if "role" not in message.model_dump() or "content" not in message.model_dump():
            raise HTTPException(status_code=400, detail="每個Message必須包含'role'和'content'")
        if message.role not in ["user", "assistant", "system"]:
            raise HTTPException(status_code=400, detail=f"無效的角色: {message.role}")
    try:
        logger.info(f"收到聊天請求: {request}")
        result = chain.invoke(query_prompt)
        formatted_response = str(format_response(result.content))
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                lines = formatted_response.split('\n')
                for line in lines:
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + '\n'},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"{json.dumps(chunk)}\n"
                    await asyncio.sleep(0.5)
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"{json.dumps(final_chunk)}\n"
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            response = ChatCompletionResponse(
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )]
            )
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"處理聊天請求時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上啟動服務")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
