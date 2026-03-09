"""
类人脑双系统全闭环AI架构
Human-Like Brain Dual-System Full-Loop AI Architecture

基于Qwen3.5-0.8B的端侧类脑大模型全栈开发方案

核心特性：
- 海马体-新皮层双系统架构
- 100Hz人脑级高刷新推理
- STDP时序可塑性学习机制
- 自生成-自博弈-自评判闭环优化
- 端侧离线运行能力

版本: 1.0.0
"""

from core.config import BrainLikeConfig, ModelMode, OptimizationMode
from core.base_model import BrainLikeQwenModel
from core.weight_splitter import WeightSplitter
from core.interfaces import ModelInterfaces

from modules.refresh_engine import RefreshEngine
from modules.stdp_system import STDPSystem
from modules.self_optimization import SelfClosedLoopOptimization
from modules.hippocampus import HippocampusSystem

from main import BrainLikeAI

__version__ = '1.0.0'
__author__ = 'Brain-Like AI Team'

__all__ = [
    # 核心模块
    'BrainLikeConfig',
    'ModelMode',
    'OptimizationMode',
    'BrainLikeQwenModel',
    'WeightSplitter',
    'ModelInterfaces',
    
    # 功能模块
    'RefreshEngine',
    'STDPSystem',
    'SelfClosedLoopOptimization',
    'HippocampusSystem',
    
    # 主入口
    'BrainLikeAI'
]
