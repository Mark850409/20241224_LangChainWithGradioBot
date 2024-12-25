import requests
import json
import logging
import gradio as gr
import os
from dotenv import load_dotenv  # pip install python-dotenv

# 載入 .env 檔案
load_dotenv()

# 設置日誌模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API服務設置相關，根據自己的實際情況進行調整
PORT = int(os.getenv("PORT"))   # 服務訪問的端口
logger.info(f"PORT: {PORT}")


url = f"http://localhost:{PORT}/v1/chat/completions"
headers = {"Content-Type": "application/json"}

def query_api(user_input, stream_flag=False):
    """與API互動，並返回結果。"""
    data = {
        "messages": [{"role": "user", "content": user_input}],
        "stream": stream_flag,
    }

    try:
        if stream_flag:
            result = ""
            with requests.post(url, stream=True, headers=headers, data=json.dumps(data)) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            data = json.loads(json_str)
                            if data['choices'][0]['finish_reason'] == "stop":
                                break
                            else:
                                result += data['choices'][0]['delta']['content']
            return result
        else:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return content
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return f"Error: {e}"

def test_api(user_input, stream_flag):
    return query_api(user_input, stream_flag)

# 使用Gradio構建GUI
with gr.Blocks() as demo:
    gr.Markdown("""<h1 style='text-align: center;'>十萬個為什麼?PDF優化版聊天機器人</h1>""")
    gr.Markdown("""此界面允許您測試與聊天機器人的互動，請輸入內容並選擇輸出模式。""")

    user_input = gr.Textbox(label="輸入測試內容"
                            ,lines=10, 
                placeholder="請輸入您的問題...")
    stream_flag = gr.Dropdown(label="是否啟用流式輸出", choices=["否", "是"], value="否")
    output = gr.Textbox(label="聊天機器人回應",lines=10, 
                )

    def parse_stream_flag(flag):
        return flag == "是"

    submit_button = gr.Button("提交")
    submit_button.click(lambda inp, flag: test_api(inp, parse_stream_flag(flag)), inputs=[user_input, stream_flag], outputs=output)

# 啟動Gradio
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
