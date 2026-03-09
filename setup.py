"""
类人脑双系统全闭环AI架构
Human-Like Brain Dual-System Full-Loop AI Architecture

安装配置文件
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="brain-like-ai",
    version="1.0.0",
    author="Brain-Like AI Team",
    author_email="your-email@example.com",
    description="基于Qwen3.5-0.8B的端侧类脑大模型全栈开发方案",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/brain-like-ai",
    license="MIT",
    packages=find_packages(exclude=["tests", "docs", "scripts"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "python-telegram-bot>=20.0",
        "aiohttp>=3.9.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "quantization": [
            "bitsandbytes>=0.41.0",
            "accelerate>=0.24.0",
        ],
        "inference": [
            "onnxruntime>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-like-ai=main:main",
            "brain-like-bot=bot.telegram_bot:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "artificial-intelligence",
        "brain-inspired",
        "hippocampus",
        "stdp",
        "qwen",
        "llm",
        "edge-computing",
        "telegram-bot",
    ],
)
