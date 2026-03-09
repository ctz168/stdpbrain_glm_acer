#!/bin/bash
# 类人脑双系统全闭环AI架构 - 一键启动脚本
# Human-Like Brain Dual-System Full-Loop AI Architecture - Quick Start Script

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}"
echo "============================================================"
echo "  🧠 类人脑双系统全闭环AI架构"
echo "  Human-Like Brain Dual-System Full-Loop AI Architecture"
echo "============================================================"
echo -e "${NC}"

# 检查Python版本
check_python() {
    echo -e "${YELLOW}检查Python环境...${NC}"
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        echo -e "${RED}❌ 未找到Python，请先安装Python 3.10+${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python版本: $PYTHON_VERSION${NC}"
}

# 创建虚拟环境
create_venv() {
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        $PYTHON_CMD -m venv "$PROJECT_ROOT/venv"
        echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
    else
        echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
    fi
    
    source "$PROJECT_ROOT/venv/bin/activate"
}

# 安装依赖
install_deps() {
    echo -e "${YELLOW}安装依赖包...${NC}"
    
    pip install --upgrade pip
    pip install -r "$PROJECT_ROOT/requirements.txt"
    
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
}

# 下载模型
download_model() {
    echo -e "${YELLOW}检查模型文件...${NC}"
    
    MODEL_DIR="$PROJECT_ROOT/models/Qwen3-0.8B"
    
    if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/model.safetensors" ]; then
        echo -e "${YELLOW}下载Qwen3.5-0.8B模型...${NC}"
        python "$PROJECT_ROOT/scripts/download_qwen3.py"
    else
        echo -e "${GREEN}✓ 模型已存在${NC}"
    fi
}

# 配置环境变量
setup_env() {
    echo -e "${YELLOW}配置环境变量...${NC}"
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo -e "${GREEN}✓ 已创建.env配置文件，请编辑填写Token${NC}"
    else
        echo -e "${GREEN}✓ .env配置文件已存在${NC}"
    fi
    
    # 加载环境变量
    if [ -f "$PROJECT_ROOT/.env" ]; then
        export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
    fi
}

# 启动Bot
start_bot() {
    echo -e "${YELLOW}启动Telegram Bot...${NC}"
    
    cd "$PROJECT_ROOT"
    python -m bot.telegram_bot
}

# 主流程
main() {
    echo -e "${BLUE}开始部署...${NC}"
    echo ""
    
    check_python
    create_venv
    install_deps
    download_model
    setup_env
    
    echo ""
    echo -e "${GREEN}============================================================"
    echo "  ✅ 部署完成！"
    echo "============================================================${NC}"
    echo ""
    
    # 询问是否启动Bot
    read -p "是否立即启动Telegram Bot? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_bot
    else
        echo -e "${YELLOW}手动启动命令: python -m bot.telegram_bot${NC}"
    fi
}

# 运行主流程
main "$@"
