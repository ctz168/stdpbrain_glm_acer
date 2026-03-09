# 类人脑双系统全闭环AI架构 - Docker配置
# Human-Like Brain Dual-System Full-Loop AI Architecture - Docker Config

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建模型目录
RUN mkdir -p /app/models

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV TELEGRAM_BOT_TOKEN=""
ENV MODEL_PATH="/app/models/Qwen3-0.8B"

# 暴露端口（如果需要）
EXPOSE 8080

# 启动命令
CMD ["python", "-m", "bot.telegram_bot"]
