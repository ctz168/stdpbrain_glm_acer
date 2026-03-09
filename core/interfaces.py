"""
类人脑双系统全闭环AI架构 - 接口模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Interfaces Module

定义模型与各模块之间的标准接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn

from .config import ModelMode


@dataclass
class AttentionFeatures:
    """注意力特征输出接口数据结构"""
    token_id: int
    layer_idx: int
    query_features: torch.Tensor
    key_features: torch.Tensor
    value_features: torch.Tensor
    attention_weights: torch.Tensor
    timing_ms: float


@dataclass
class MemoryAnchor:
    """记忆锚点数据结构"""
    anchor_id: str
    timestamp_ms: float
    feature_vector: torch.Tensor
    semantic_pointer: str
    causal_links: List[str]
    gate_signal: torch.Tensor
    relevance_score: float


@dataclass
class STDPUpdateSignal:
    """STDP更新信号"""
    pre_activation_time: float
    post_activation_time: float
    weight_name: str
    update_direction: float  # 正为LTP，负为LTD
    update_magnitude: float


class IHippocampusInterface(ABC):
    """海马体模块接口"""
    
    @abstractmethod
    def encode_features(self, features: List[AttentionFeatures]) -> torch.Tensor:
        """编码特征（EC内嗅皮层）"""
        pass
    
    @abstractmethod
    def separate_patterns(self, encoded: torch.Tensor) -> torch.Tensor:
        """模式分离（DG齿状回）"""
        pass
    
    @abstractmethod
    def store_episode(
        self, 
        pattern: torch.Tensor,
        timestamp_ms: float,
        semantic_info: Dict
    ) -> str:
        """存储情景记忆（CA3区）"""
        pass
    
    @abstractmethod
    def recall_memories(
        self, 
        cue: torch.Tensor,
        top_k: int = 2
    ) -> List[MemoryAnchor]:
        """召回记忆（CA3模式补全）"""
        pass
    
    @abstractmethod
    def get_attention_gate(
        self, 
        memories: List[MemoryAnchor]
    ) -> torch.Tensor:
        """获取注意力门控信号（CA1区）"""
        pass


class ISTDPInterface(ABC):
    """STDP更新系统接口"""
    
    @abstractmethod
    def compute_ltp(
        self,
        pre_time: float,
        post_time: float,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """计算LTP（长期增强）更新"""
        pass
    
    @abstractmethod
    def compute_ltd(
        self,
        pre_time: float,
        post_time: float,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """计算LTD（长期减弱）更新"""
        pass
    
    @abstractmethod
    def apply_update(
        self,
        weight: nn.Parameter,
        update: torch.Tensor,
        learning_rate: float
    ) -> None:
        """应用权重更新"""
        pass
    
    @abstractmethod
    def get_update_statistics(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        pass


class IOptimizationInterface(ABC):
    """自闭环优化系统接口"""
    
    @abstractmethod
    def generate_candidates(
        self,
        input_ids: torch.Tensor,
        num_candidates: int = 2
    ) -> List[torch.Tensor]:
        """生成候选输出"""
        pass
    
    @abstractmethod
    def verify_output(
        self,
        output: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[bool, str]:
        """验证输出正确性"""
        pass
    
    @abstractmethod
    def judge_candidates(
        self,
        candidates: List[torch.Tensor],
        dimensions: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """评判候选输出"""
        pass
    
    @abstractmethod
    def select_best(
        self,
        candidates: List[torch.Tensor],
        scores: Dict[str, List[float]]
    ) -> torch.Tensor:
        """选择最优输出"""
        pass


class IRefreshEngineInterface(ABC):
    """刷新引擎接口"""
    
    @abstractmethod
    def start_cycle(self) -> None:
        """开始新周期"""
        pass
    
    @abstractmethod
    def process_token(
        self,
        token_id: int,
        context: torch.Tensor
    ) -> torch.Tensor:
        """处理单个token"""
        pass
    
    @abstractmethod
    def end_cycle(self) -> Dict[str, Any]:
        """结束周期"""
        pass
    
    @abstractmethod
    def get_cycle_statistics(self) -> Dict[str, Any]:
        """获取周期统计"""
        pass


class ITrainingInterface(ABC):
    """训练模块接口"""
    
    @abstractmethod
    def pre_adapt_train(
        self,
        model: nn.Module,
        data_loader: Any,
        config: Dict
    ) -> Dict[str, float]:
        """预适配训练"""
        pass
    
    @abstractmethod
    def online_update(
        self,
        model: nn.Module,
        feedback: Dict
    ) -> None:
        """在线更新"""
        pass
    
    @abstractmethod
    def offline_consolidate(
        self,
        model: nn.Module,
        memory_replay: List
    ) -> Dict[str, float]:
        """离线巩固"""
        pass


class IEvaluationInterface(ABC):
    """测评模块接口"""
    
    @abstractmethod
    def evaluate_memory_recall(
        self,
        model: nn.Module,
        test_cases: List
    ) -> Dict[str, float]:
        """评估记忆召回能力"""
        pass
    
    @abstractmethod
    def evaluate_reasoning(
        self,
        model: nn.Module,
        test_cases: List
    ) -> Dict[str, float]:
        """评估推理能力"""
        pass
    
    @abstractmethod
    def evaluate_edge_performance(
        self,
        model: nn.Module
    ) -> Dict[str, float]:
        """评估端侧性能"""
        pass
    
    @abstractmethod
    def generate_report(self) -> Dict[str, Any]:
        """生成测评报告"""
        pass


class ModelInterfaces:
    """模型接口管理器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._feature_buffer: List[AttentionFeatures] = []
        self._memory_anchors: List[MemoryAnchor] = []
        self._current_mode = ModelMode.GENERATION
    
    # 注意力特征输出接口
    def get_attention_features(self) -> List[AttentionFeatures]:
        """获取注意力层特征输出"""
        return self._feature_buffer
    
    def push_attention_features(self, features: AttentionFeatures) -> None:
        """添加注意力特征到缓存"""
        self._feature_buffer.append(features)
    
    def clear_feature_buffer(self) -> None:
        """清空特征缓存"""
        self._feature_buffer.clear()
    
    # 海马体注意力门控接口
    def set_memory_anchors(self, anchors: List[MemoryAnchor]) -> None:
        """设置海马体记忆锚点"""
        self._memory_anchors = anchors
    
    def get_memory_anchors(self) -> List[MemoryAnchor]:
        """获取当前记忆锚点"""
        return self._memory_anchors
    
    def get_gate_signals(self) -> torch.Tensor:
        """获取门控信号"""
        if not self._memory_anchors:
            return torch.zeros(1)
        
        signals = [anchor.gate_signal for anchor in self._memory_anchors]
        return torch.stack(signals).mean(dim=0)
    
    # 角色适配接口
    def set_mode(self, mode: ModelMode) -> None:
        """设置模型运行模式"""
        self._current_mode = mode
    
    def get_mode(self) -> ModelMode:
        """获取当前模式"""
        return self._current_mode
    
    def get_mode_prompt(self) -> str:
        """获取当前模式的提示词"""
        prompts = {
            ModelMode.GENERATION: (
                "你是一个智能助手，请根据用户输入生成准确、有帮助的回答。"
                "确保回答内容准确、逻辑清晰、语言流畅。"
            ),
            ModelMode.VERIFICATION: (
                "你是一个严谨的验证者，请仔细检查给定内容的逻辑正确性和事实准确性。"
                "指出任何错误、漏洞或不一致之处，并提供修正建议。"
            ),
            ModelMode.JUDGMENT: (
                "你是一个公正的评判者，请从以下四个维度对给定内容进行评分（0-10分）：\n"
                "1. 事实准确性：内容是否符合客观事实\n"
                "2. 逻辑完整性：论证是否完整、逻辑是否自洽\n"
                "3. 语义连贯性：表达是否流畅、语义是否清晰\n"
                "4. 指令遵循度：是否完整遵循用户指令"
            )
        }
        return prompts.get(self._current_mode, prompts[ModelMode.GENERATION])
    
    # STDP权重接口
    def get_dynamic_weights(self) -> Dict[str, nn.Parameter]:
        """获取所有动态权重"""
        weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights[name] = param
        return weights
    
    def get_weight_by_name(self, name: str) -> Optional[nn.Parameter]:
        """按名称获取权重"""
        for n, param in self.model.named_parameters():
            if n == name:
                return param
        return None
    
    # 统计接口
    def get_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'current_mode': self._current_mode.value,
            'feature_buffer_size': len(self._feature_buffer),
            'memory_anchor_count': len(self._memory_anchors)
        }
