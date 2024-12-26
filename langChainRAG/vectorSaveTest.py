# 功能说明：將PDF文件進行向量計算並持久化存儲到向量數據庫（chroma）

# 相關依賴庫
# pip install openai chromadb

# 引入相關庫
import logging
from openai import OpenAI
import chromadb
import uuid
import numpy as np
from tools import pdfSplitTest_Ch  # 引入處理中文PDF文件的工具模塊
from tools import pdfSplitTest_En  # 引入處理英文PDF文件的工具模塊
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv  # pip install python-dotenv

# 設置日誌模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 載入 .env 檔案
load_dotenv()

# 模型設置相關，根據自己的實際情況進行調整
API_TYPE = os.getenv("API_TYPE")
logger.info(f"API_TYPE: {API_TYPE}")

# OpenAI模型相關配置，根據自己的實際情況進行調整
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_CHAT_API_KEY=os.getenv("OPENAI_CHAT_API_KEY")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
logger.info(f"OPENAI_API_BASE: {OPENAI_API_BASE}")
logger.info(f"OPENAI_EMBEDDING_API_KEY: {OPENAI_EMBEDDING_API_KEY}")
logger.info(f"OPENAI_EMBEDDING_MODEL: {OPENAI_EMBEDDING_MODEL}")

# HUGGINGFACE模型相關配置，根據自己的實際情況進行調整
HUGGINGFACE_EMBEDDING_MODEL= os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
logger.info(f"HUGGINGFACE_EMBEDDING_MODEL: {HUGGINGFACE_EMBEDDING_MODEL}")


# 設置測試文本類型
TEXT_LANGUAGE = os.getenv("TEXT_LANGUAGE")
logger.info(f"TEXT_LANGUAGE: {TEXT_LANGUAGE}")

# 測試的PDF文件路徑
INPUT_PDF = "../input/"+os.getenv("INPUT_PDF")
logger.info(f"INPUT_PDF: {INPUT_PDF}")

# 指定文件中待處理的頁碼，全部頁碼則填None
PAGE_NUMBERS = None
logger.info(f"PAGE_NUMBERS: {PAGE_NUMBERS}")

# 指定向量數據庫chromaDB的存儲位置和集合，根據自己的實際情況進行調整
CHROMADB_DIRECTORY = os.getenv("CHROMADB_DIRECTORY") 
CHROMADB_COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME")
logger.info(f"CHROMADB_DIRECTORY: {CHROMADB_DIRECTORY}")
logger.info(f"CHROMADB_COLLECTION_NAME: {CHROMADB_COLLECTION_NAME}")

# get_embeddings方法計算向量
def get_embeddings(texts):
    """
    根據文本內容生成向量表示
    """
    global API_TYPE, OPENAI_API_BASE, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    if API_TYPE == 'ollama':
        try:
            # 初始化ollama的Embedding模型
            client = OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts, model=OPENAI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量時出錯: {e}")
            return []

    elif API_TYPE == 'lmstudio':
        try:
            # 初始化lmstudio的Embedding模型
            client = OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts, model=OPENAI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量時出錯: {e}")
            return []
    elif API_TYPE == 'huggingface':
        try:
            # 初始化Hugging Face的嵌入模型
            model_name = HUGGINGFACE_EMBEDDING_MODEL  # 替換為您的模型名稱
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # 將文本轉換為嵌入
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # 取平均池化的嵌入向量
                
            return embeddings.numpy().tolist()
        except Exception as e:
            logger.info(f"生成向量時出錯: {e}")
            return []
    elif API_TYPE == 'openai':
        try:
            # 初始化ollama的Embedding模型
            client = OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_CHAT_API_KEY
            )
            data = client.embeddings.create(input=texts, model=OPENAI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量時出錯: {e}")
            return []


# 對文本按批次進行向量計算
def generate_vectors(data, max_batch_size=25):
    """
    將文本分批進行向量生成
    """
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        response = get_embeddings(batch)
        results.extend(response)
    return results

# 封裝向量數據庫chromadb類，提供兩種方法
class MyVectorDBConnector:
    """
    向量數據庫接口封裝
    """
    def __init__(self, collection_name, embedding_fn):
        """
        初始化向量數據庫
        """
        global CHROMADB_DIRECTORY
        chroma_client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        """
        向數據庫集合中添加文檔
        """
        self.collection.add(
            embeddings=self.embedding_fn(documents),
            documents=documents,
            ids=[str(uuid.uuid4()) for i in range(len(documents))]
        )

    def search(self, query, top_n):
        """
        在數據庫中進行相似度檢索
        """
        try:
            results = self.collection.query(
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n
            )
            return results
        except Exception as e:
            logger.info(f"檢索向量數據庫時出錯: {e}")
            return []

# 封裝文本預處理及灌庫方法，提供外部調用
def vectorStoreSave():
    """
    文本預處理並將向量存儲到數據庫
    """
    global TEXT_LANGUAGE, CHROMADB_COLLECTION_NAME, INPUT_PDF, PAGE_NUMBERS
    if TEXT_LANGUAGE == 'Chinese':
        paragraphs = pdfSplitTest_Ch.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
        vector_db.add_documents(paragraphs)
        user_query = "Cloud SQL的⽤途是什麼?"
        search_results = vector_db.search(user_query, 5)
        logger.info(f"檢索向量數據庫的結果: {search_results}")

    elif TEXT_LANGUAGE == 'English':
        paragraphs = pdfSplitTest_En.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
        vector_db.add_documents(paragraphs)
        user_query = "llama2安全性如何"
        search_results = vector_db.search(user_query, 5)
        logger.info(f"檢索向量數據庫的結果: {search_results}")

if __name__ == "__main__":
    # 測試文本預處理及灌庫
    vectorStoreSave()
