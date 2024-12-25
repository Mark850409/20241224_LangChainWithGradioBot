from flask import Flask, request, jsonify
from langChainRAG.UploadFileRAG import initialize_vector_store, create_chain, process_files, prompt, llm,output_parser  # 從 app.py 匯入相關模組
app = Flask(__name__)

# 定義 API 路由
@app.route('/query', methods=['POST'])
def query():
    """
    接收 POST 請求，並使用向量檢索回傳結果。
    請求格式:
    {
        "question": "你的問題",
        "search_mode": "mmr" 或 "cosine"
    }
    """
    data = request.json
    question = data.get("question")
    search_mode = data.get("search_mode", "mmr")

    if not question:
        return jsonify({"error": "請提供問題"}), 400

    # 初始化向量資料庫
    if not initialize_vector_store(search_mode):
        return jsonify({"error": "向量資料庫尚未初始化，請先上傳文件"}), 500

    # 建立 RAG 處理鏈
    chain = create_chain() | prompt | llm | output_parser

    try:
        # 執行檢索和回答
        answer = chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"處理過程中發生錯誤: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload():
    """
    接收檔案上傳請求並處理向量檢索初始化。
    請求格式:
    Multipart form-data，支援多個檔案。
    """
    files = request.files.getlist("files")
    search_mode = request.form.get("search_mode", "mmr")

    if not files:
        return jsonify({"error": "請至少上傳一個檔案"}), 400

    # 處理檔案
    file_objs = [file for file in files]
    result = process_files(file_objs, search_mode)

    if "錯誤" in result:
        return jsonify({"error": result}), 500

    return jsonify({"message": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
