import gradio as gr
from langchain_openai import ChatOpenAI                                     # pip install langchain-openai
from langchain_community.vectorstores import FAISS                          # pip install langchain-community faiss-cpu
from langchain_core.output_parsers import StrOutputParser                   # pip install langchain
from langchain_core.prompts import ChatPromptTemplate                       # pip install langchain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # pip install langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.schema import Document  
from langchain_core.prompts import PromptTemplate                            
import pdfplumber                                                            # pip install pdfplumber
import os
from dotenv import load_dotenv  # pip install python-dotenv
import logging

# 載入 .env 檔案
load_dotenv()

# 設置日誌模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型設置相關，根據自己的實際情況進行調整
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")
logger.info(f"EMBEDDINGS_MODEL_NAME: {EMBEDDINGS_MODEL_NAME}")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
logger.info(f"LLM_MODEL_NAME: {LLM_MODEL_NAME}")
API_KEY = os.getenv("API_KEY")
logger.info(f"API_KEY: {API_KEY}")
BASE_URL = os.getenv("BASE_URL")
logger.info(f"BASE_URL: {BASE_URL}")
TEMPERATURE = float(os.getenv("TEMPERATURE"))
logger.info(f"TEMPERATURE: {TEMPERATURE}")

# 指定向量數據庫chromaDB的存儲檔名，根據自己的實際情況進行調整
VECTOR_DB_NAME = os.getenv("VECTOR_DB_NAME")
logger.info(f"VECTOR_DB_NAME: {VECTOR_DB_NAME}")

# prompt模版設置相關，根據自己的實際情況進行調整
PROMPT_UPLOAD_TEMPLATE_TXT = "../promt/"+os.getenv("PROMPT_UPLOAD_TEMPLATE_TXT") 
logger.info(f"PROMPT_UPLOAD_TEMPLATE_TXT: {PROMPT_UPLOAD_TEMPLATE_TXT}")

# 使用 HuggingFaceEmbeddings 模型 
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
vector_store = None
retriever = None
current_search_mode = "mmr"  # 預設搜索模式

def get_retriever(store, mode="mmr"):
    """根據選擇的模式返回適當的檢索器"""
    if mode == "mmr":
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 20, 'lambda_mult': 0.5}
        )
    else:  # cosine
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3}
        )

def initialize_vector_store(search_mode="mmr"):
    """初始化向量資料庫和檢索器"""
    global vector_store, retriever, current_search_mode
    if os.path.exists(VECTOR_DB_NAME):
        try:
            vector_store = FAISS.load_local(
                VECTOR_DB_NAME, 
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            current_search_mode = search_mode
            retriever = get_retriever(vector_store, search_mode)
            return True
        except Exception as e:
            print(f"載入向量資料庫時發生錯誤: {str(e)}")
            return False
    return False

# 在程式啟動時初始化
initialize_vector_store()

# 檢查是否已存在本地向量資料庫
if os.path.exists(VECTOR_DB_NAME):
    vector_store = FAISS.load_local(VECTOR_DB_NAME, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = None

def extract_table_from_pdf(file_path):
    """從 PDF 中提取表格"""
    extracted_data = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                        extracted_data.append(cleaned_row)
    except Exception as e:
        print(f"表格提取失敗: {str(e)}")
    return extracted_data

def process_files(file_objs, search_mode):
    documents = []
   
    for file_obj in file_objs:
        try:
            # 取得文件的臨時路徑
            temp_path = file_obj.name
            
            if temp_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                table_data = extract_table_from_pdf(temp_path)
                if table_data:
                    table_documents = [Document(page_content="\n".join(["\t".join(map(str, row)) for row in table_data]), metadata={"source": temp_path})]
                    documents.extend(table_documents)
            elif temp_path.lower().endswith(".csv"):
                loader = CSVLoader(temp_path)
                docs = loader.load()
                for doc in docs:
                    documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))
            elif temp_path.lower().endswith(".txt"):
                loader = TextLoader(temp_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))
            else:
                return f"不支援的檔案格式: {temp_path}"
        except Exception as e:
            return f"處理檔案 {temp_path} 時發生錯誤: {str(e)}"

    try:
        global vector_store, retriever, current_search_mode
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(VECTOR_DB_NAME)
        current_search_mode = search_mode
        retriever = get_retriever(vector_store, search_mode)
    except Exception as e:
        return f"儲存至向量資料庫時發生錯誤: {str(e)}"

    return f"文件已成功處理並存入向量資料庫。使用搜索模式: {search_mode}"

# 並行執行兩個任務
def create_chain():
    if not retriever:
        # 嘗試重新初始化
        if not initialize_vector_store():
            raise ValueError("向量資料庫尚未初始化，請先上傳文件。")
    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

# 初始化 OpenAI 模型
llm = ChatOpenAI(model=LLM_MODEL_NAME,
                 api_key=API_KEY,
                 base_url=BASE_URL,
                 temperature=TEMPERATURE)

# 讀取提示詞模板
with open(PROMPT_UPLOAD_TEMPLATE_TXT, 'r', encoding='utf-8') as f:
    template_content = f.read()
prompt_template = PromptTemplate(input_variables=["query", "context"], template=template_content)
prompt = ChatPromptTemplate.from_messages([("human", str(prompt_template.template))])

# 純文字解析器
output_parser = StrOutputParser()

################################

# Gradio介面測試

################################


def chatbot_interface(question):
    """處理使用者輸入並返回答案"""
    try:
        if not retriever:
            if not initialize_vector_store():
                return "向量資料庫尚未初始化，請先上傳文件。"
        chain = create_chain() | prompt | llm | output_parser
        output = chain.invoke(question)
        return output
    except Exception as e:
        return f"發生錯誤：{str(e)}"

def upload_files(file_paths):
    """上傳檔案並處理"""
    try:
        message = process_files(file_paths)
        return message
    except Exception as e:
        return f"發生錯誤：{str(e)}"

# 整合聊天和文件上傳功能到單一介面
with gr.Blocks() as interface:
    gr.Markdown("# 十萬個為什麼")
    gr.Markdown("這是一個基於LangChain和Gradio的系統，您可以上傳文件以建立向量資料庫，並提出問題以獲得回答。")

    with gr.Row():
        with gr.Column():
            # 新增搜索模式選擇
            search_mode = gr.Dropdown(
                choices=["mmr", "cosine"],
                value="mmr",
                label="選擇向量搜索模式",
                info="MMR: 最大邊際相關算法 | Cosine: 文本相似度算法"
            )
            upload_section = gr.File(
                label="上傳文件 (支援 PDF、CSV、TXT)", 
                file_types=[".pdf", ".csv", ".txt"], 
                file_count="multiple"
            )
            upload_status = gr.Textbox(label="處理狀態", interactive=False)
            upload_button = gr.Button("上傳並處理文件")

        with gr.Column():
            question_input = gr.Textbox(
                label="輸入問題", 
                lines=10, 
                placeholder="請輸入您的問題..."
            )
            response_output = gr.Textbox(label="回答", lines=10, interactive=False)
            ask_button = gr.Button("提交問題")

   # 修改檔案上傳處理函數
    def handle_file_upload(files, mode):
        if not files:
            return "請上傳至少一個文件。"
        return process_files(files, mode)

    upload_button.click(
        fn=handle_file_upload,
        inputs=[upload_section, search_mode],
        outputs=upload_status
    )

    # 問題回答功能
    def handle_question(question):
        if not question.strip():
            return "請先輸入您的問題再提交。"
        if not retriever:
            if not initialize_vector_store(current_search_mode):
                return "向量資料庫尚未初始化，請先上傳文件。"
        return chatbot_interface(question)

    ask_button.click(
        fn=handle_question,
        inputs=question_input,
        outputs=response_output
    )

# 啟動整合介面
if __name__ == "__main__":
    interface.launch(share=True, server_name="0.0.0.0", server_port=7861)