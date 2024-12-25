# LangChain RAG實作

## 1. 簡介

實作LangChain RAG，實現與大語言模型結合的聊天機器人

## 2. 目錄
- [LangChain RAG實作](#langchain-rag實作)
  - [1. 簡介](#1-簡介)
  - [2. 目錄](#2-目錄)
  - [3. 畫面最終呈現](#3-畫面最終呈現)
  - [4. 建立python虛擬環境](#4-建立python虛擬環境)
  - [5. 多檔上傳RAG操作步驟](#5-多檔上傳rag操作步驟)
    - [方法一：python本地執行LangChain主程式](#方法一python本地執行langchain主程式)
      - [STEP1：請先將我的專案整包下載下來](#step1請先將我的專案整包下載下來)
      - [STEP2：建立並修改env檔](#step2建立並修改env檔)
      - [STEP3：安裝python套件](#step3安裝python套件)
      - [STEP4：執行python主程式](#step4執行python主程式)
    - [方法二：docker部署LangChain主程式](#方法二docker部署langchain主程式)
    - [API主程式(此步驟用於測試API時才要做)](#api主程式此步驟用於測試api時才要做)
      - [STEP1：API節點說明](#step1api節點說明)
      - [STEP2：執行api`建立Server`](#step2執行api建立server)
      - [STEP3：使用curl指令測試](#step3使用curl指令測試)
  - [6. PDF檔案上傳優化版RAG操作步驟](#6-pdf檔案上傳優化版rag操作步驟)
    - [STEP1：建立並修改env檔](#step1建立並修改env檔)
    - [STEP2：安裝python套件](#step2安裝python套件)
    - [STEP3：執行文本切割轉向量主程式](#step3執行文本切割轉向量主程式)
    - [STEP4：執行RAG主程式](#step4執行rag主程式)
    - [STEP5：執行api測試介面(結合Gradio與聊天機器人對答)](#step5執行api測試介面結合gradio與聊天機器人對答)

## 3. 畫面最終呈現
![alt text](image.png)

## 4. 建立python虛擬環境

 > [!note] 
 > 要安裝的python套件很多，建議使用虛擬環境

```python
# 建立虛擬環境
python -m venv lanchainRAG_env

# 進入虛擬環境
cd lanchainRAG_env\Scripts
activate

# 離開虛擬環境
deactivate
```


## 5. 多檔上傳RAG操作步驟

### 方法一：python本地執行LangChain主程式  

#### STEP1：請先將我的專案整包下載下來

git指令

```bash
git clone https://github.com/Mark850409/20241224_LangChainWithGradioBot.git
```

沒有git，進入此連結，點擊code → DownloadZIP

```
https://github.com/Mark850409/20241224_LangChainWithGradioBot.git
```

#### STEP2：建立並修改env檔

請自行在專案下建立`.env`，按照以下範例進行調整

請先在地端建立語言模型，若沒有請以下擇一下載

LMstudio

https://lmstudio.ai/

Ollama

https://ollama.com/


 > [!note] 
 > 1. EMBEDDINGS_MODEL_NAME→可以使用HuggingFace、Ollama、LMstudio的模型，請自行更改模型名稱
 > 
 > 2. LLM_MODEL_NAME →可以使用HuggingFace、Ollama、LMStudio的模型，請自行更改模型名稱
 >
 > 3. API_KEY → 不重要，隨便填
 >
 > 4. BASE_URL→
 > 
 > ★ LMstudio：http://localhost:1234/v1/
 >
 > ★ Ollama：http://localhost:11434/v1/

範例：
```
EMBEDDINGS_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL_NAME = "llama-3.2-3b-instruct"
API_KEY = "lm-studio"
BASE_URL ="http://localhost:1234/v1/"
TEMPERATURE = "0.5"
VECTOR_DB_NAME = "faiss_1224_db"
```

#### STEP3：安裝python套件

```python
pip install -r requirements.txt
```

#### STEP4：執行python主程式

```python
python app.py
```

### 方法二：docker部署LangChain主程式  

 > [!note] 
 > Windows請先安裝Docker Desktop再執行此步驟

```docker
docker-compose up -d
```

### API主程式(此步驟用於測試API時才要做)

#### STEP1：API節點說明

 > [!note] 
 > /upload → 上傳檔案
 > 
 > /query → 使用者提問

#### STEP2：執行api`建立Server`

```python
python api.py
```

#### STEP3：使用curl指令測試
```bash
curl -X POST -H "Content-Type: application/json" -d "{\"question\":\"你好\",\"search_mode\":\"cosine\"}" http://localhost:5000/query
```

## 6. PDF檔案上傳優化版RAG操作步驟

### STEP1：建立並修改env檔

請自行在專案下建立`.env`，按照以下範例進行調整

請先在地端建立語言模型，若沒有請以下擇一下載

LMstudio

https://lmstudio.ai/

Ollama

https://ollama.com/

模型相關配置是擇一填寫，例如選擇`lmstuido`，就找到相關配置進行設定，其他部分則`註解`

 > [!note] 
 > API_TYPE → 模型類型
 > 
 > TEXT_LANGUAGE → 文本測試語言(中、英文)
 > 
 > INPUT_PDF → PDF文檔路徑
 > 
 > PAGE_NUMBERS → 要處理的頁碼，全部填None
 > 
 > PROMPT_TEMPLATE_TXT → LLM模板提示詞文件
 > 
 > PORT → API PORT
範例：
```
# 模型設置相關，根據自己的實際情況進行調整
# API_TYPE="openai"
# API_TYPE="huggingface"
API_TYPE="lmstudio"
# API_TYPE="ollama" 

# openai: 調用GPT模型；
# huggingface: 調用huggingface方案支持的模型
# lmstudio: 調用lmstudio方案支持的模型
# ollama: 調用ollama方案支持的模型

# openai模型相關配置，根據自己的實際情況進行調整
# OPENAI_API_BASE="https://api.wlai.vip/v1"
# OPENAI_EMBEDDING_API_KEY=""
# OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# OLLAMA模型相關配置，根據自己的實際情況進行調整
# OPENAI_API_BASE="http://localhost:11434/v1"
# OPENAI_EMBEDDING_API_KEY="ollama"
# OPENAI_CHAT_API_KEY='ollama'
# OPENAI_CHAT_MODEL='llama-3.2-3b-instruct'
# OPENAI_EMBEDDING_MODEL="nomic-embed-text:latest"

# lmstudio模型相關配置，根據自己的實際情況進行調整
OPENAI_API_BASE="http://localhost:1234/v1/"
OPENAI_CHAT_API_KEY='lm-studio'
OPENAI_CHAT_MODEL='llama-3.2-3b-instruct'
OPENAI_EMBEDDING_API_KEY="lm-studio"
OPENAI_EMBEDDING_MODEL="text-embedding-nomic-embed-text-v1.5"

# HUGGINGFACE模型相關配置，根據自己的實際情況進行調整
HUGGINGFACE_CHAT_MODEL="gpt2"
HUGGINGFACE_EMBEDDING_MODEL="sentence-transformers/msmarco-distilbert-base-v3"

# 設置測試文本類型
TEXT_LANGUAGE='Chinese'
#TEXT_LANGUAGE='English'

# 測試的PDF文件路徑
INPUT_PDF="input/健康档案(含表格02).pdf"

# 指定文件中待處理的頁碼，全部頁碼則填None
PAGE_NUMBERS=None
# PAGE_NUMBERS=[2, 3]

# 指定向量數據庫chromaDB的存儲位置和集合，根據自己的實際情況進行調整
CHROMADB_DIRECTORY="chromaDB"
CHROMADB_COLLECTION_NAME="demo006"


# prompt模版設置相關，根據自己的實際情況進行調整
PROMPT_TEMPLATE_TXT="prompt_template.txt"

# API設置相關，根據自己的實際情況進行調整
PORT=8012
```

### STEP2：安裝python套件

```python
pip install -r requirements.txt
```

### STEP3：執行文本切割轉向量主程式

此程式負責將文本切割，並存入ChromaDB`向量資料庫`

```python
python vectorSaveTest.py
```

### STEP4：執行RAG主程式

```python
python main.py
```

### STEP5：執行api測試介面(結合Gradio與聊天機器人對答)

```python
python apiTest.py
```