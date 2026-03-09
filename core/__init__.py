"""
类人脑双系统全闭环AI架构 - 核心模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Core Module
"""

from .config import BrainLikeConfig, ModelMode, OptimizationMode
from .engine import BrainLikeAIEngine, GenerationConfig, get_engine, generate, generate_stream
from .interfaces import ModelInterfaces
from .weight_splitter import WeightSplitter

__all__ = [
    'BrainLikeConfig',
    'ModelMode',
    'OptimizationMode',
    'BrainLikeAIEngine',
    'GenerationConfig',
    'get_engine',
    'generate',
    'generate_stream',
    'ModelInterfaces',
    'WeightSplitter',
]
