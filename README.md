# 🧠 类人脑双系统全闭环 AI 架构 (Neuromorphic AI)

<div align="center">

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/ctz168/stdpbrain_glm_acer)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Base-Qwen3.5--0.8B-red.svg)](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

**基于 Qwen3.5-0.8B 的端侧「海马体-新皮层」双系统类人脑全栈开发方案**

</div>

---

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ctz168/stdpbrain_glm_acer/blob/main/deploy_colab.ipynb)

</div>

## 🚀 快速开始 (Google Colab)

点击上方 **Open In Colab** 按钮，即可在云端一键部署。
1. 填入您的 `TELEGRAM_BOT_TOKEN`。
2. 依次运行单元格，约 2 分钟后即可在 Telegram 与类脑 AI 互动。

---

## 📖 项目简介

本项目是一套完整的「海马体-新皮层双系统类人脑 AI 架构」，实现与人脑同源的 **"刷新即推理、推理即学习、学习即优化、记忆即锚点"** 全闭环智能能力。

### 核心特性
*   **⚡ 100Hz 刷新即推理**：固定 10ms 认知周期，模拟生物神经元的放电频率。
*   **🧬 STDP 在线学习**：无需反向传播，在推理过程中通过脉冲时序依赖塑性（STDP）实时更新权重。
*   **🧠 海马体记忆系统**：存储-计算分离，模拟 EC-DG-CA3-CA1 生物回路，严格限制 2MB RAM 占用。
*   **🔄 自闭环优化**：通过模型自生成的候选方案进行自博弈与自评判，实现逻辑自我进化。

---

## 🛠️ 安装指南

### 💻 1. 硬件与系统要求
*   **操作系统**: Windows 10/11, macOS (Intel/M1/M2), Linux
*   **内存**: ≥ 4GB RAM (模型运行时占用约 2.4GB)
*   **存储**: ≥ 2GB 可用硬盘空间
*   **Python**: 3.10 或更高版本

### 🔧 2. 环境配置步骤

#### Windows 用户
```powershell
# 1. 克隆项目
git clone https://github.com/ctz168/stdpbrain_glm_acer.git
cd stdpbrain_glm_acer

# 2. 创建并激活虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 3. 安装依赖
pip install --upgrade pip
pip install torch transformers accelerate sentencepiece flask faiss-cpu python-telegram-bot hf_transfer wikipedia-api
```

#### Mac/Linux 用户
```bash
# 1. 克隆项目
git clone https://github.com/ctz168/stdpbrain_glm_acer.git
cd stdpbrain_glm_acer

# 2. 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install --upgrade pip
pip install torch transformers accelerate sentencepiece flask faiss-cpu python-telegram-bot hf_transfer wikipedia-api
```

### 📥 3. 下载模型权重 (高性能引擎)
本项目内置了基于 `hf_transfer` 的快速下载脚本，通常在 1 分钟内完成：
```bash
python download_model.py
```

---

## 🚀 运行服务

### 1. 启动 Telegram 机器人
在 `bot/telegram_bot.py` 中配置您的 Token，然后运行：
```bash
python -u bot/telegram_bot.py
```
*   **特性**: 实时流式响应、100Hz 认知状态显示、海马体记忆动态同步。

### 2. 启动 Web 监控后台
```bash
python web/app.py
```
*   **地址**: `http://localhost:5000`
*   **功能**: 可视化实时神经元评分、海马体存储状态及 STDP 学习曲线。

---

## ✨ 核心模块导航

| 模块 | 关键路径 | 实现功能 |
| :--- | :--- | :--- |
| **刷新引擎** | `core/truly_integrated_engine.py` | 10ms 采样周期、Repetition Penalty、Top-P 优化 |
| **STDP 系统** | `modules/stdp_system.py` | 90% 静态/10% 动态权重切分、LTP/LTD 生物学习规则 |
| **海马体系统** | `modules/hippocampus.py` | EC 编码、DG 模式分离、CA3 长效存储、CA1 门控控制 |
| **优化器** | `modules/self_optimization.py` | 自博弈 (Self-Play)、自评判 (Self-Judgment) |
| **知识检索** | `modules/wikipedia_tool.py` | 多线程 Wikipedia 实时搜索、存储-计算分离支持 |

---

## 📁 项目结构
```text
stdpbrain_glm_acer/
├── bot/                # Telegram 机器人交互层
├── core/               # 核心执行引擎 (Refresh Engine & STDP Integration)
├── modules/            # 生物模块 (海马体、STDP 核、优化器、Wiki工具)
├── training/           # 专项训练与记忆巩固代码
├── web/                # 可视化监控前端 (Flask + HTML/CSS)
├── deploy_mac.sh       # Mac 一键部署脚本
├── download_model.py   # 模型高速下载工具
├── deploy_colab.ipynb  # Colab 一键部署笔记本
└── README.md           # 项目综合文档
```

---

## ⚠️ 开发者红线 (Developer Red Lines)
为了确保类脑架构的纯正性，所有修改必须遵循：
1. **底座不可更換**: 必须使用 Qwen3.5-0.8B-Base，禁止更换为 Chat 版或其他模型。
2. **存算分离**: 知识库（Wiki）必须作为外部工具挂载，不得通过微调改写 Base 模型参数。
3. **实时性优先**: 任何新增逻辑不得破坏 100Hz (10ms) 的主循环实时性。

---

## 📄 许可证
MIT License.
