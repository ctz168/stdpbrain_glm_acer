"""
类人脑双系统全闭环AI架构 - 权重拆分模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Weight Splitter Module

实现模型权重的双轨拆分：
- 90%静态基础权重（永久冻结）
- 10%STDP动态增量权重（可更新）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .config import BrainLikeConfig


@dataclass
class WeightSplitResult:
    """权重拆分结果"""
    static_weights: Dict[str, torch.Tensor]
    dynamic_weights: Dict[str, torch.Tensor]
    static_ratio: float
    dynamic_ratio: float
    total_params: int
    static_params: int
    dynamic_params: int


class WeightSplitter:
    """
    权重双轨拆分器
    
    将模型权重拆分为：
    - 90%静态基础权重：继承预训练权重，永久冻结
    - 10%STDP动态增量权重：随机初始化，可更新
    """
    
    def __init__(self, config: BrainLikeConfig):
        self.config = config
        self.static_ratio = config.weight_split.static_ratio
        self.dynamic_ratio = config.weight_split.dynamic_ratio
        
        # 验证比例
        assert abs(self.static_ratio + self.dynamic_ratio - 1.0) < 1e-6, \
            "静态权重比例与动态权重比例之和必须为1"
    
    def split_linear_layer(
        self, 
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        拆分线性层权重
        
        Args:
            weight: 原始权重 [out_features, in_features]
            bias: 原始偏置 [out_features]
            
        Returns:
            static_weight: 静态权重
            dynamic_weight: 动态权重
            static_bias: 静态偏置
            dynamic_bias: 动态偏置
        """
        out_features, in_features = weight.shape
        
        # 计算拆分点
        split_point = int(out_features * self.static_ratio)
        
        # 拆分权重
        static_weight = weight[:split_point, :].clone()
        dynamic_weight = self._init_dynamic_weight(
            (out_features - split_point, in_features)
        )
        
        # 拆分偏置
        static_bias = None
        dynamic_bias = None
        if bias is not None:
            static_bias = bias[:split_point].clone()
            dynamic_bias = self._init_dynamic_weight((out_features - split_point,))
        
        return static_weight, dynamic_weight, static_bias, dynamic_bias
    
    def split_attention_weights(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        拆分注意力层权重
        
        Args:
            q_weight: Query权重
            k_weight: Key权重
            v_weight: Value权重
            o_weight: Output权重
            
        Returns:
            拆分后的权重字典
        """
        result = {}
        
        for name, weight in [('q', q_weight), ('k', k_weight), 
                             ('v', v_weight), ('o', o_weight)]:
            static_w, dynamic_w, _, _ = self.split_linear_layer(weight)
            result[f'{name}_static'] = static_w
            result[f'{name}_dynamic'] = dynamic_w
        
        return result
    
    def split_ffn_weights(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        拆分FFN层权重
        
        Args:
            gate_weight: 门控权重
            up_weight: 上投影权重
            down_weight: 下投影权重
            
        Returns:
            拆分后的权重字典
        """
        result = {}
        
        for name, weight in [('gate', gate_weight), ('up', up_weight), 
                             ('down', down_weight)]:
            static_w, dynamic_w, _, _ = self.split_linear_layer(weight)
            result[f'{name}_static'] = static_w
            result[f'{name}_dynamic'] = dynamic_w
        
        return result
    
    def analyze_model_weights(self, model: nn.Module) -> WeightSplitResult:
        """
        分析模型权重分布
        
        Args:
            model: 待分析的模型
            
        Returns:
            WeightSplitResult: 权重分析结果
        """
        static_weights = {}
        dynamic_weights = {}
        
        total_params = 0
        static_params = 0
        dynamic_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            if param.requires_grad:
                dynamic_weights[name] = param.data.clone()
                dynamic_params += param.numel()
            else:
                static_weights[name] = param.data.clone()
                static_params += param.numel()
        
        return WeightSplitResult(
            static_weights=static_weights,
            dynamic_weights=dynamic_weights,
            static_ratio=static_params / total_params if total_params > 0 else 0,
            dynamic_ratio=dynamic_params / total_params if total_params > 0 else 0,
            total_params=total_params,
            static_params=static_params,
            dynamic_params=dynamic_params
        )
    
    def verify_split_ratio(self, model: nn.Module) -> bool:
        """
        验证权重拆分比例是否符合要求
        
        Args:
            model: 待验证的模型
            
        Returns:
            bool: 是否符合要求
        """
        result = self.analyze_model_weights(model)
        
        # 允许5%的误差
        tolerance = 0.05
        
        static_ok = abs(result.static_ratio - self.static_ratio) < tolerance
        dynamic_ok = abs(result.dynamic_ratio - self.dynamic_ratio) < tolerance
        
        return static_ok and dynamic_ok
    
    def _init_dynamic_weight(
        self, 
        shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        初始化动态权重
        
        使用小权重随机正态分布初始化
        """
        mean = self.config.weight_split.dynamic_init_mean
        std = self.config.weight_split.dynamic_init_std
        
        return torch.randn(shape) * std + mean
    
    def freeze_static_weights(self, model: nn.Module):
        """
        冻结模型的静态权重
        
        Args:
            model: 待冻结的模型
        """
        for name, param in model.named_parameters():
            if 'static' in name or 'base' in name:
                param.requires_grad = False
    
    def get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """
        获取可训练的动态权重参数
        
        Args:
            model: 模型
            
        Returns:
            可训练参数列表
        """
        return [p for p in model.parameters() if p.requires_grad]
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """
        统计模型参数数量
        
        Args:
            model: 模型
            
        Returns:
            参数统计字典
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_ratio': trainable / total if total > 0 else 0,
            'frozen_ratio': frozen / total if total > 0 else 0
        }
    
    def estimate_memory_usage(
        self, 
        model: nn.Module,
        quantized: bool = True
    ) -> Dict[str, float]:
        """
        估算模型内存使用
        
        Args:
            model: 模型
            quantized: 是否量化
            
        Returns:
            内存使用字典（MB）
        """
        params = self.count_parameters(model)
        
        if quantized:
            # INT4量化：每个参数0.5字节
            bytes_per_param = 0.5
        else:
            # FP16：每个参数2字节
            bytes_per_param = 2
        
        total_memory = params['total'] * bytes_per_param / (1024 * 1024)
        trainable_memory = params['trainable'] * bytes_per_param / (1024 * 1024)
        frozen_memory = params['frozen'] * bytes_per_param / (1024 * 1024)
        
        return {
            'total_mb': total_memory,
            'trainable_mb': trainable_memory,
            'frozen_mb': frozen_memory,
            'within_constraint': total_memory <= self.config.weight_split.max_vram_mb
        }


class WeightMerger:
    """权重合并器，用于推理时合并静态和动态权重"""
    
    @staticmethod
    def merge_linear_weights(
        static_weight: torch.Tensor,
        dynamic_weight: torch.Tensor,
        static_bias: Optional[torch.Tensor] = None,
        dynamic_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        合并线性层权重
        
        Args:
            static_weight: 静态权重
            dynamic_weight: 动态权重
            static_bias: 静态偏置
            dynamic_bias: 动态偏置
            
        Returns:
            合并后的权重和偏置
        """
        merged_weight = torch.cat([static_weight, dynamic_weight], dim=0)
        
        merged_bias = None
        if static_bias is not None and dynamic_bias is not None:
            merged_bias = torch.cat([static_bias, dynamic_bias], dim=0)
        
        return merged_weight, merged_bias
    
    @staticmethod
    def apply_stdp_update(
        weight: torch.Tensor,
        update: torch.Tensor,
        learning_rate: float,
        weight_min: float = -1.0,
        weight_max: float = 1.0
    ) -> torch.Tensor:
        """
        应用STDP权重更新
        
        Args:
            weight: 原始权重
            update: 更新量
            learning_rate: 学习率
            weight_min: 权重下限
            weight_max: 权重上限
            
        Returns:
            更新后的权重
        """
        new_weight = weight + learning_rate * update
        new_weight = torch.clamp(new_weight, weight_min, weight_max)
        return new_weight
