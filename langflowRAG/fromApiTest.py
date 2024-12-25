# 使用 API 接口進行調用
import requests
import json
import logging
# 環境變數相關
import os
from dotenv import load_dotenv  # pip install python-dotenv

# 設置日誌模板
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 載入 .env 檔案
load_dotenv()

# API設置相關，根據自己的實際情況進行調整
LANGFLOW_BASE_URL = os.getenv("LANGFLOW_BASE_URL") 
logger.info(f"LANGFLOW_BASE_URL: {LANGFLOW_BASE_URL}")

# 1、API 請求相關配置，根據實際情況進行調整
url = f"{LANGFLOW_BASE_URL}api/v1/run/ragflow?stream=false"
headers = {"Content-Type": "application/json"}

# 2、進入循環輸入模式
print("輸入 'exit' 可退出程式")
while True:
    # 提示使用者輸入測試文字
    input_text = input("請輸入您的問題：")
    if input_text.lower() == "exit":
        print("程式結束。")
        break
    data = {
        "input_value": input_text,
        "output_type": "chat",
        "input_type": "chat",
        "ChatInput-tRg6H": {
            "background_color": "",
            "chat_icon": "",
            "files": "",
            "input_value": "cloudsql的用途?",
            "sender": "User",
            "sender_name": "User",
            "session_id": "",
            "should_store_message": True,
            "text_color": ""
        },
        "ParseData-2btL1": {
            "sep": "\n",
            "template": "{text}"
        },
        "Prompt-YKbmZ": {
            "context": "",
            "question": "",
            "template": "{context}\n\n---\n\nGiven the context above, answer the question as best as possible.\n\nQuestion: {question}\n\nAnswer: "
        },
        "SplitText-Uln9u": {
            "chunk_overlap": 200,
            "chunk_size": 4096,
            "separator": "\n"
        },
        "ChatOutput-2PqB3": {
            "background_color": "",
            "chat_icon": "",
            "data_template": "{text}",
            "input_value": "",
            "sender": "Machine",
            "sender_name": "AI",
            "session_id": "",
            "should_store_message": True,
            "text_color": ""
        },
        "Chroma-aUzCJ": {
            "allow_duplicates": False,
            "chroma_server_cors_allow_origins": "",
            "chroma_server_grpc_port": None,
            "chroma_server_host": "",
            "chroma_server_http_port": 0,
            "chroma_server_ssl_enabled": False,
            "collection_name": "langflow20241222_4",
            "limit": None,
            "number_of_results": 10,
            "persist_directory": "langflow_20241222_4",
            "search_query": "",
            "search_type": "Similarity"
        },
        "Chroma-OhEZe": {
            "allow_duplicates": False,
            "chroma_server_cors_allow_origins": "",
            "chroma_server_grpc_port": None,
            "chroma_server_host": "",
            "chroma_server_http_port": 0,
            "chroma_server_ssl_enabled": False,
            "collection_name": "langflow20241222_4",
            "limit": None,
            "number_of_results": 10,
            "persist_directory": "langflow_20241222_4",
            "search_query": "",
            "search_type": "Similarity"
        },
        "Directory-UyRuk": {
            "depth": 0,
            "load_hidden": False,
            "max_concurrency": 2,
            "path": "/app/langflow/pdf_test",
            "recursive": False,
            "silent_errors": False,
            "types": [],
            "use_multithreading": False
        },
        "LMStudioEmbeddingsComponent-icig0": {
            "api_key": "",
            "base_url": "http://163.14.137.66:1234/v1",
            "model": "text-embedding-nomic-embed-text-v1.5",
            "temperature": 0.1
        },
        "LMStudioEmbeddingsComponent-6wXeR": {
            "api_key": "",
            "base_url": "http://163.14.137.66:1234/v1",
            "model": "text-embedding-nomic-embed-text-v1.5-embedding",
            "temperature": 0.1
        },
        "LMStudioModel-yJ5qn": {
            "api_key": "",
            "base_url": "http://163.14.137.66:1234/v1",
            "input_value": "",
            "max_tokens": 8192,
            "model_kwargs": {},
            "model_name": "llama-3.2-3b-instruct",
            "seed": 1,
            "stream": False,
            "system_message": "",
            "temperature": 2
        },
        "APIRequest-q0USK": {
            "body": "{}",
            "curl": "",
            "headers": "{}",
            "method": "GET",
            "timeout": 5,
            "urls": ""
        }
    }

    # 3、發送 POST 請求進行測試
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 檢查響應狀態碼
    if response.status_code == 200:
        try:
            logger.info(f"輸出響應內容是: {response.status_code}\n")
            logger.info(f"輸出響應內容是: {response.json()}\n")
            # 解析具體回覆的內容
            content = response.json()['outputs'][0]['outputs'][0]['results']['message']['data']['text']
            logger.info(f"輸出響應內容是: {content}\n")
        except requests.exceptions.JSONDecodeError:
            # 響應不是 JSON 格式
            logger.info("響應內容不是有效的 JSON")
    else:
        logger.info(f"請求失敗，狀態碼為 {response.status_code}")
