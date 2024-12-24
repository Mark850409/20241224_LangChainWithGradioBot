# 使用 Python 官方映像作為基礎映像
FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 複製本地腳本到容器
COPY . /app

# 安裝必要的 Python 庫
RUN pip install -r requirements.txt --no-cache-dir

# 設置模型啟動腳本
CMD ["python", "app.py"]