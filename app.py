import gradio as gr
from langchain_openai import ChatOpenAI                                     # pip install langchain-openai
from langchain_community.vectorstores import FAISS                          # pip install langchain-community faiss-cpu
from langchain_core.output_parsers import StrOutputParser                   # pip install langchain
from langchain_core.prompts import ChatPromptTemplate                       # pip install langchain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # pip install langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.schema import Document                                    # Ensure the document structure
import pdfplumber                                                            # pip install pdfplumber
import os
from dotenv import load_dotenv  # pip install python-dotenv

# 載入 .env 檔案
load_dotenv()

# 從環境變數中獲取設定
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
TEMPERATURE = float(os.getenv("TEMPERATURE"))
VECTOR_DB_NAME = os.getenv("VECTOR_DB_NAME")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")

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
    if os.path.exists("faiss_1214_db"):
        try:
            vector_store = FAISS.load_local(
                "faiss_1214_db", 
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
        vector_store.save_local("faiss_1214_db")
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

# 聊天模板
prompt = ChatPromptTemplate.from_messages([
    ("system", '''
你是一個專業的 RAG（檢索增強生成）系統助手，負責提供準確且有來源依據的回答。你需要使用繁體中文，並保持專業、客觀的語氣。

基於檢索到的文件內容，請：
1. 請用簡單扼要的方式回答
2. 使用條列式方式組織回答
3. 請不要使用MarkDown語法輸出，請使用純文字輸出

限制條件
• 僅使用繁體中文回答
• 若無法在文件中找到相關資訊，需明確說明
• 回答須符合向量搜索模式（MMR 或 Cosine）的特性
     
品質要求
• 確保回答的準確性和完整性
• 保持邏輯清晰的條列式呈現
'''),
    ("human", "Answer the question based only on the following context: \n{context}\n\nQuestion: {question}"),]
)

# 純文字解析器
output_parser = StrOutputParser()


################################

# 指令介面測試

################################

# 這邊使用 RunnableParallel 並行執行兩個任務
# setup_and_retrieval = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# )

# 串連各個組件
# chain = setup_and_retrieval | prompt | llm | output_parser

# 測試
# output = chain.invoke("唐僧師徒三人遇到那些妖怪?")

# print(output)


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