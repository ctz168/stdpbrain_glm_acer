"""
类人脑双系统全闭环AI架构 - 工具模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Utility Module

提供通用工具函数和辅助类
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import time
import json
import os
import logging
from datetime import datetime


def setup_logging(log_dir: str = "./logs", log_level: int = logging.INFO):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir, 
        f"brain_like_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('BrainLikeAI')


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_memory_usage(
    model: nn.Module, 
    quantized: bool = True,
    bits: int = 4
) -> Dict[str, float]:
    """估算模型内存使用"""
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    if quantized:
        bytes_per_param = bits / 8
    else:
        bytes_per_param = 2  # FP16
    
    total_memory = total_params * bytes_per_param / (1024 * 1024)  # MB
    trainable_memory = trainable_params * bytes_per_param / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_memory_mb': total_memory,
        'trainable_memory_mb': trainable_memory,
        'frozen_memory_mb': total_memory - trainable_memory
    }


def measure_inference_time(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """测量推理时间"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
    
    # 测量
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
    }


class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[str] = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict] = None,
        step: int = 0
    ):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.save_dir, 
            f"checkpoint_step_{step}.pt"
        )
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'step': step,
            'timestamp': time.time()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # 清理旧检查点
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict:
        """加载检查点"""
        if checkpoint_path is None:
            if not self.checkpoints:
                return {}
            checkpoint_path = self.checkpoints[-1]
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config: Dict = {}
        
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, path: str):
        """加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.config_path = path
    
    def save(self, path: str = None):
        """保存配置"""
        path = path or self.config_path
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置项"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def move_to_device(
    data: Union[torch.Tensor, Dict, List],
    device: torch.device
) -> Union[torch.Tensor, Dict, List]:
    """将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
