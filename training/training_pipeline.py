"""
类人脑双系统全闭环AI架构 - 专项全流程训练模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Training Module

包含三个子模块：
1. 底座预适配微调模块（部署前一次性执行）
2. 在线终身学习训练模块（推理时实时执行）
3. 离线记忆巩固与推理优化模块（空闲时执行）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import os
import math

from core.config import BrainLikeConfig, TrainingConfig


class TrainingPhase(Enum):
    """训练阶段"""
    PRE_ADAPT = "pre_adapt"  # 预适配
    ONLINE = "online"  # 在线学习
    OFFLINE = "offline"  # 离线巩固


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    weight_update_ratio: float = 0.0
    timestamp: float = 0.0


@dataclass
class TrainingResult:
    """训练结果"""
    phase: TrainingPhase
    total_steps: int
    final_loss: float
    metrics_history: List[TrainingMetrics]
    duration_seconds: float
    weight_changes: Dict[str, float]


class AdaptationDataset(Dataset):
    """预适配数据集"""
    
    def __init__(
        self,
        data_path: str = None,
        data_list: List[Dict] = None
    ):
        self.data = []
        
        if data_list:
            self.data = data_list
        elif data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item.get('input_ids', []), dtype=torch.long),
            'labels': torch.tensor(item.get('labels', []), dtype=torch.long),
            'attention_mask': torch.tensor(item.get('attention_mask', [1]), dtype=torch.long)
        }


class PreAdaptTrainer:
    """
    底座预适配微调模块
    
    完成STDP动态分支与海马体模块的初始化适配，
    让模型快速适配高刷新推理模式、STDP更新规则、海马体注意力门控、角色切换逻辑
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: BrainLikeConfig,
        train_config: TrainingConfig = None
    ):
        self.model = model
        self.config = config
        self.train_config = train_config or config.training
        
        # 训练状态
        self._current_epoch = 0
        self._current_step = 0
        self._metrics_history: List[TrainingMetrics] = []
        
        # 优化器
        self._optimizer: Optional[torch.optim.Optimizer] = None
    
    def setup(self):
        """设置训练环境"""
        # 冻结静态权重
        self._freeze_static_weights()
        
        # 创建优化器（仅优化动态权重）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.train_config.pre_adapt_optimizer == "AdamW":
            self._optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.train_config.pre_adapt_lr,
                weight_decay=0.01
            )
        else:
            self._optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.train_config.pre_adapt_lr
            )
    
    def _freeze_static_weights(self):
        """冻结静态基础权重"""
        for name, param in self.model.named_parameters():
            # 冻结不包含'dynamic'的权重
            if 'dynamic' not in name.lower() and 'stdp' not in name.lower():
                param.requires_grad = False
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 前向传播
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch.get('attention_mask', None)
            
            # 移动到设备
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # 计算损失
            with torch.cuda.amp.autocast(enabled=False):
                logits, features, dynamic_weights = self.model(
                    input_ids,
                    attention_mask=attention_mask
                )
                
                # 计算交叉熵损失
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            
            # 反向传播
            self._optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            # 更新权重
            self._optimizer.step()
            
            # 记录指标
            total_loss += loss.item()
            num_batches += 1
            self._current_step += 1
            
            # 记录详细指标
            metrics = TrainingMetrics(
                epoch=epoch,
                step=self._current_step,
                loss=loss.item(),
                learning_rate=self._optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                timestamp=time.time()
            )
            self._metrics_history.append(metrics)
        
        return {
            'epoch': epoch,
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0,
            'total_steps': self._current_step
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> TrainingResult:
        """
        执行完整训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            
        Returns:
            训练结果
        """
        start_time = time.time()
        
        # 设置
        self.setup()
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.pre_adapt_batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 训练循环
        for epoch in range(self.train_config.pre_adapt_epochs):
            self._current_epoch = epoch
            epoch_metrics = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}: Loss = {epoch_metrics['avg_loss']:.4f}")
        
        # 计算权重变化
        weight_changes = self._compute_weight_changes()
        
        duration = time.time() - start_time
        
        return TrainingResult(
            phase=TrainingPhase.PRE_ADAPT,
            total_steps=self._current_step,
            final_loss=self._metrics_history[-1].loss if self._metrics_history else 0,
            metrics_history=self._metrics_history,
            duration_seconds=duration,
            weight_changes=weight_changes
        )
    
    def _compute_weight_changes(self) -> Dict[str, float]:
        """计算权重变化"""
        changes = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                changes[name] = param.grad.abs().mean().item()
        return changes
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict() if self._optimizer else None,
            'config': self.config.to_dict(),
            'epoch': self._current_epoch,
            'step': self._current_step,
            'metrics_history': [m.__dict__ for m in self._metrics_history]
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self._optimizer and checkpoint['optimizer_state_dict']:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._current_epoch = checkpoint.get('epoch', 0)
        self._current_step = checkpoint.get('step', 0)


class OnlineLearner:
    """
    在线终身学习训练模块
    
    实现推理即学习，模型在端侧运行过程中，
    无需人工干预，即可实时学习新内容、适配用户习惯、优化自身能力
    
    全程基于STDP时序可塑性规则，无反向传播、无批量数据、无全局误差计算
    """
    
    def __init__(
        self,
        model: nn.Module,
        stdp_system,
        config: BrainLikeConfig
    ):
        self.model = model
        self.stdp = stdp_system
        self.config = config
        self.train_config = config.training
        
        # 学习状态
        self._update_count = 0
        self._total_updates: List[Dict] = []
    
    def update(
        self,
        input_features: torch.Tensor,
        output_features: torch.Tensor,
        feedback: Dict[str, Any]
    ):
        """
        在线更新
        
        每个刷新周期自动执行，算力开销不超过模型总算力的2%
        
        Args:
            input_features: 输入特征
            output_features: 输出特征
            feedback: 反馈信息（正确性、质量等）
        """
        if not self.train_config.online_learning_enabled:
            return
        
        current_time = time.time() * 1000
        
        # 计算STDP更新
        model_output = {
            'hidden_states': input_features,
            'output_states': output_features,
            'contribution_scores': feedback.get('contribution_scores', torch.ones(1)),
            'activation_times': feedback.get('activation_times', [current_time - 10])
        }
        
        # 获取STDP更新
        updates = self.stdp.compute_all_updates(model_output, current_time)
        
        # 应用更新
        self.stdp.apply_all_updates(updates, self.model)
        
        # 记录
        self._update_count += 1
        self._total_updates.append({
            'timestamp': current_time,
            'updates': {k: v.mean().item() for k, v in updates.items()} if updates else {}
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计"""
        return {
            'total_updates': self._update_count,
            'recent_updates': self._total_updates[-10:] if self._total_updates else []
        }


class OfflineConsolidator:
    """
    离线记忆巩固与推理优化模块
    
    在端侧空闲时，通过海马体记忆回放，
    把短期情景记忆转化为长期语义记忆，
    同时优化模型的推理路径
    """
    
    def __init__(
        self,
        model: nn.Module,
        hippocampus_system,
        stdp_system,
        config: BrainLikeConfig
    ):
        self.model = model
        self.hippocampus = hippocampus_system
        self.stdp = stdp_system
        self.config = config
        self.train_config = config.training
        
        # 巩固状态
        self._consolidation_count = 0
        self._last_consolidation_time = 0.0
        self._is_consolidating = False
    
    def check_should_consolidate(self) -> bool:
        """检查是否应该执行离线巩固"""
        if not self.train_config.offline_consolidation_enabled:
            return False
        
        idle_time = time.time() - self._last_consolidation_time
        return idle_time >= self.train_config.offline_idle_threshold_minutes * 60
    
    def consolidate(self) -> Dict[str, Any]:
        """
        执行离线巩固
        
        Returns:
            巩固结果
        """
        if self._is_consolidating:
            return {'status': 'already_consolidating'}
        
        self._is_consolidating = True
        start_time = time.time()
        
        # 执行海马体离线巩固
        hippocampus_result = self.hippocampus.offline_consolidation()
        
        # 获取回放的记忆序列
        if hippocampus_result.get('status') == 'completed':
            # 对回放的记忆进行STDP强化
            self._consolidate_memories(hippocampus_result)
        
        self._consolidation_count += 1
        self._last_consolidation_time = time.time()
        self._is_consolidating = False
        
        duration = time.time() - start_time
        
        return {
            'status': 'completed',
            'consolidation_count': self._consolidation_count,
            'duration_seconds': duration,
            'hippocampus_result': hippocampus_result
        }
    
    def _consolidate_memories(self, hippocampus_result: Dict):
        """巩固记忆"""
        # 获取STDP统计
        stdp_stats = self.stdp.get_statistics()
        
        # 根据统计调整学习率
        if stdp_stats.total_updates > 0:
            avg_update = stdp_stats.average_update
            # 动态调整学习率
            if avg_update > 0.1:
                # 更新幅度过大，降低学习率
                self.stdp.set_learning_rates(
                    self.config.stdp.alpha * 0.9,
                    self.config.stdp.beta * 0.9
                )
            elif avg_update < 0.001:
                # 更新幅度过小，提高学习率
                self.stdp.set_learning_rates(
                    self.config.stdp.alpha * 1.1,
                    self.config.stdp.beta * 1.1
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取巩固统计"""
        return {
            'consolidation_count': self._consolidation_count,
            'last_consolidation_time': self._last_consolidation_time,
            'is_consolidating': self._is_consolidating
        }


class TrainingPipeline:
    """
    训练流水线
    
    整合预适配、在线学习、离线巩固三个阶段
    """
    
    def __init__(
        self,
        model: nn.Module,
        hippocampus_system,
        stdp_system,
        config: BrainLikeConfig
    ):
        self.model = model
        self.config = config
        
        # 初始化各训练模块
        self.pre_adapt_trainer = PreAdaptTrainer(model, config)
        self.online_learner = OnlineLearner(model, stdp_system, config)
        self.offline_consolidator = OfflineConsolidator(
            model, hippocampus_system, stdp_system, config
        )
    
    def run_pre_adapt(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> TrainingResult:
        """运行预适配训练"""
        return self.pre_adapt_trainer.train(train_dataset, eval_dataset)
    
    def online_update(
        self,
        input_features: torch.Tensor,
        output_features: torch.Tensor,
        feedback: Dict[str, Any]
    ):
        """执行在线更新"""
        self.online_learner.update(input_features, output_features, feedback)
    
    def check_and_consolidate(self) -> Optional[Dict[str, Any]]:
        """检查并执行离线巩固"""
        if self.offline_consolidator.check_should_consolidate():
            return self.offline_consolidator.consolidate()
        return None
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """获取所有训练统计"""
        return {
            'pre_adapt': {
                'current_epoch': self.pre_adapt_trainer._current_epoch,
                'current_step': self.pre_adapt_trainer._current_step,
                'metrics_count': len(self.pre_adapt_trainer._metrics_history)
            },
            'online': self.online_learner.get_statistics(),
            'offline': self.offline_consolidator.get_statistics()
        }
    
    def save_all(self, directory: str):
        """保存所有状态"""
        os.makedirs(directory, exist_ok=True)
        
        # 保存预适配检查点
        self.pre_adapt_trainer.save_checkpoint(
            os.path.join(directory, 'pre_adapt_checkpoint.pt')
        )
        
        # 保存配置
        self.config.save(os.path.join(directory, 'config.json'))
    
    def load_all(self, directory: str):
        """加载所有状态"""
        # 加载预适配检查点
        checkpoint_path = os.path.join(directory, 'pre_adapt_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            self.pre_adapt_trainer.load_checkpoint(checkpoint_path)
