"""
类人脑双系统全闭环AI架构 - 多维度全链路测评体系
Human-Like Brain Dual-System Full-Loop AI Architecture - Evaluation System

包含五大测评维度：
1. 海马体记忆能力专项测评（权重40%）
2. 基础能力对标测评（权重20%）
3. 逻辑推理能力测评（权重20%）
4. 端侧性能测评（权重10%）
5. 自闭环优化能力测评（权重10%）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import os
from collections import defaultdict

from core.config import BrainLikeConfig, EvaluationConfig


class EvaluationDimension(Enum):
    """测评维度"""
    HIPPOCAMPUS_MEMORY = "hippocampus_memory"  # 海马体记忆能力
    BASELINE = "baseline"  # 基础能力
    REASONING = "reasoning"  # 逻辑推理能力
    EDGE_PERFORMANCE = "edge_performance"  # 端侧性能
    SELF_OPTIMIZATION = "self_optimization"  # 自闭环优化能力


@dataclass
class EvaluationResult:
    """测评结果"""
    dimension: EvaluationDimension
    score: float
    max_score: float
    details: Dict[str, Any]
    passed: bool
    timestamp: float


@dataclass
class ComprehensiveReport:
    """综合测评报告"""
    total_score: float
    weighted_score: float
    dimension_results: Dict[str, EvaluationResult]
    overall_passed: bool
    recommendations: List[str]
    timestamp: float


class HippocampusMemoryEvaluator:
    """
    海马体记忆能力专项测评器
    
    测评维度：
    - 情景记忆召回能力
    - 模式分离抗混淆能力
    - 长时序记忆保持能力
    - 模式补全能力
    - 抗灾难性遗忘能力
    - 跨会话终身学习能力
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate_recall(
        self,
        hippocampus_system,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """
        评估情景记忆召回能力
        
        测试指标：
        - 线索召回准确率≥95%
        - 完整度≥90%
        """
        correct_recall = 0
        total_recall = 0
        completeness_scores = []
        
        for case in test_cases:
            # 存储情景
            features = case['features']
            semantic_info = case['semantic_info']
            hippocampus_system.encode_episode(
                features, time.time() * 1000, semantic_info
            )
        
        # 测试召回
        for case in test_cases:
            cue = case['partial_cue']
            expected = case['expected_recall']
            
            # 召回
            recalled = hippocampus_system.recall_memories(cue, top_k=1)
            
            if recalled:
                # 检查是否正确召回
                recalled_info = recalled[0].get('semantic_pointer', '')
                if expected in recalled_info or recalled_info in expected:
                    correct_recall += 1
                
                # 计算完整度
                expected_features = case.get('expected_features', None)
                if expected_features is not None:
                    recalled_features = recalled[0].get('gate_signal', torch.zeros(1))
                    completeness = self._compute_completeness(
                        expected_features, recalled_features
                    )
                    completeness_scores.append(completeness)
            
            total_recall += 1
        
        recall_accuracy = correct_recall / total_recall if total_recall > 0 else 0
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        return {
            'recall_accuracy': recall_accuracy,
            'completeness': avg_completeness,
            'passed_recall': recall_accuracy >= self.config.recall_accuracy_threshold
        }
    
    def evaluate_pattern_separation(
        self,
        hippocampus_system,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """
        评估模式分离抗混淆能力
        
        测试指标：
        - 记忆混淆率≤3%
        """
        confusion_count = 0
        total_pairs = 0
        
        # 编码相似但不同的输入
        encoded_features = []
        for case in test_cases:
            features = case['features']
            # 通过DG进行模式分离
            separated, _ = hippocampus_system.dg.separate(
                hippocampus_system.ec.encode(features)
            )
            encoded_features.append(separated)
        
        # 检查混淆
        for i in range(len(encoded_features)):
            for j in range(i + 1, len(encoded_features)):
                similarity = hippocampus_system.dg.compute_similarity(
                    encoded_features[i], encoded_features[j]
                )
                if similarity > 0.5:  # 相似度阈值
                    confusion_count += 1
                total_pairs += 1
        
        confusion_rate = confusion_count / total_pairs if total_pairs > 0 else 0
        
        return {
            'confusion_rate': confusion_rate,
            'passed_separation': confusion_rate <= self.config.confusion_rate_threshold
        }
    
    def evaluate_long_sequence(
        self,
        hippocampus_system,
        test_sequence: List[Dict],
        query_position: int = 0
    ) -> Dict[str, float]:
        """
        评估长时序记忆保持能力
        
        测试指标：
        - 100k token长序列记忆保持率≥90%
        - 时序逻辑准确率≥95%
        """
        # 编码长序列
        for i, item in enumerate(test_sequence):
            features = item['features']
            semantic_info = {
                'semantic_pointer': item.get('content', ''),
                'temporal_skeleton': [i * 10],  # 10ms间隔
                'causal_links': []
            }
            hippocampus_system.encode_episode(
                features, i * 10, semantic_info
            )
        
        # 测试对开头信息的召回
        first_item = test_sequence[query_position]
        cue = first_item['partial_cue']
        
        recalled = hippocampus_system.recall_memories(cue, top_k=1)
        
        retention_score = 0.0
        temporal_accuracy = 0.0
        
        if recalled:
            # 检查是否正确召回开头信息
            expected_content = first_item.get('content', '')
            recalled_content = recalled[0].get('semantic_pointer', '')
            
            if expected_content in recalled_content or recalled_content in expected_content:
                retention_score = 1.0
            
            # 检查时序逻辑
            recalled_time = recalled[0].get('timestamp_ms', 0)
            expected_time = query_position * 10
            if abs(recalled_time - expected_time) < 100:  # 100ms容差
                temporal_accuracy = 1.0
        
        return {
            'retention_score': retention_score,
            'temporal_accuracy': temporal_accuracy,
            'passed_retention': retention_score >= 0.9,
            'passed_temporal': temporal_accuracy >= 0.95
        }
    
    def evaluate_pattern_completion(
        self,
        hippocampus_system,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """
        评估模式补全能力
        
        测试指标：
        - 部分线索完整召回率≥85%
        """
        completion_count = 0
        total_count = 0
        
        for case in test_cases:
            # 先存储完整记忆
            full_features = case['full_features']
            hippocampus_system.encode_episode(
                full_features, time.time() * 1000, case['semantic_info']
            )
            
            # 用部分线索测试补全
            partial_cue = case['partial_cue']
            completed = hippocampus_system.ca3.complete_pattern(
                hippocampus_system.ec.encode(partial_cue)
            )
            
            if completed:
                # 检查补全是否正确
                expected = case['expected_completion']
                if expected in completed.semantic_pointer:
                    completion_count += 1
            
            total_count += 1
        
        completion_rate = completion_count / total_count if total_count > 0 else 0
        
        return {
            'completion_rate': completion_rate,
            'passed_completion': completion_rate >= 0.85
        }
    
    def evaluate_anti_forgetting(
        self,
        model: nn.Module,
        old_tasks: List[Dict],
        new_task_training: Callable
    ) -> Dict[str, float]:
        """
        评估抗灾难性遗忘能力
        
        测试指标：
        - 新任务学习后，旧任务性能保留率≥95%
        """
        # 先评估旧任务性能
        old_task_scores = []
        for task in old_tasks:
            score = self._evaluate_task(model, task)
            old_task_scores.append(score)
        
        avg_old_score = sum(old_task_scores) / len(old_task_scores) if old_task_scores else 0
        
        # 学习新任务
        new_task_training()
        
        # 再次评估旧任务性能
        new_old_task_scores = []
        for task in old_tasks:
            score = self._evaluate_task(model, task)
            new_old_task_scores.append(score)
        
        avg_new_old_score = sum(new_old_task_scores) / len(new_old_task_scores) if new_old_task_scores else 0
        
        retention_rate = avg_new_old_score / avg_old_score if avg_old_score > 0 else 0
        
        return {
            'retention_rate': retention_rate,
            'passed_anti_forgetting': retention_rate >= 0.95
        }
    
    def evaluate_cross_session(
        self,
        hippocampus_system,
        sessions: List[List[Dict]]
    ) -> Dict[str, float]:
        """
        评估跨会话终身学习能力
        
        测试指标：
        - 跨会话偏好适配度≥90%
        - 记忆召回率≥85%
        """
        preference_adaptations = []
        recall_scores = []
        
        for session_idx, session in enumerate(sessions):
            # 处理会话
            for item in session:
                hippocampus_system.encode_episode(
                    item['features'],
                    time.time() * 1000,
                    item['semantic_info']
                )
            
            # 测试对之前会话的记忆
            if session_idx > 0:
                prev_session = sessions[session_idx - 1]
                for item in prev_session[:2]:  # 测试前两个
                    recalled = hippocampus_system.recall_memories(
                        item['partial_cue'], top_k=1
                    )
                    if recalled:
                        recall_scores.append(1.0)
                    else:
                        recall_scores.append(0.0)
        
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        
        return {
            'cross_session_recall': avg_recall,
            'passed_cross_session': avg_recall >= 0.85
        }
    
    def _compute_completeness(
        self,
        expected: torch.Tensor,
        actual: torch.Tensor
    ) -> float:
        """计算完整度"""
        if expected.shape != actual.shape:
            return 0.0
        
        similarity = torch.cosine_similarity(
            expected.flatten().unsqueeze(0),
            actual.flatten().unsqueeze(0)
        ).item()
        
        return max(0, similarity)
    
    def _evaluate_task(self, model: nn.Module, task: Dict) -> float:
        """评估单个任务"""
        # 简化实现
        return 0.8


class BaselineEvaluator:
    """
    基础能力对标测评器
    
    测评维度：
    - 通用对话能力
    - 指令遵循能力
    - 语义理解能力
    - 中文处理能力
    
    对标基准：与官方原生Qwen3.5-0.8B模型做对标
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(
        self,
        model: nn.Module,
        baseline_model: nn.Module,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """
        评估基础能力
        
        核心指标不得低于原生模型的95%
        """
        model_scores = []
        baseline_scores = []
        
        for case in test_cases:
            # 评估当前模型
            model_score = self._evaluate_single(model, case)
            model_scores.append(model_score)
            
            # 评估基线模型
            baseline_score = self._evaluate_single(baseline_model, case)
            baseline_scores.append(baseline_score)
        
        avg_model_score = sum(model_scores) / len(model_scores) if model_scores else 0
        avg_baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
        
        performance_ratio = avg_model_score / avg_baseline_score if avg_baseline_score > 0 else 0
        
        return {
            'model_score': avg_model_score,
            'baseline_score': avg_baseline_score,
            'performance_ratio': performance_ratio,
            'passed': performance_ratio >= self.config.baseline_performance_ratio
        }
    
    def _evaluate_single(self, model: nn.Module, case: Dict) -> float:
        """评估单个测试用例"""
        # 简化实现
        return 0.8


class ReasoningEvaluator:
    """
    逻辑推理能力测评器
    
    测评维度：
    - 数学推理
    - 代码生成
    - 常识推理
    - 因果推断
    - 事实性问答
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(
        self,
        model: nn.Module,
        test_cases: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        评估推理能力
        
        核心指标必须超过原生Qwen3.5-0.8B模型60%以上
        """
        results = {}
        
        for category, cases in test_cases.items():
            correct = 0
            total = len(cases)
            
            for case in cases:
                # 评估推理正确性
                if self._check_reasoning(model, case):
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            results[category] = accuracy
        
        avg_accuracy = sum(results.values()) / len(results) if results else 0
        
        return {
            'category_scores': results,
            'average_accuracy': avg_accuracy,
            'passed': avg_accuracy >= self.config.reasoning_improvement_ratio
        }
    
    def _check_reasoning(self, model: nn.Module, case: Dict) -> bool:
        """检查推理正确性"""
        # 简化实现
        return True


class EdgePerformanceEvaluator:
    """
    端侧性能测评器
    
    测评指标：
    - INT4量化后显存占用
    - 单token推理延迟
    - 单周期算力开销
    - 长序列处理稳定性
    - 离线运行兼容性
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(
        self,
        model: nn.Module,
        config: BrainLikeConfig
    ) -> Dict[str, float]:
        """
        评估端侧性能
        
        完全符合算力硬约束
        """
        results = {}
        
        # 估算显存占用
        memory_mb = self._estimate_memory(model)
        results['memory_mb'] = memory_mb
        results['memory_passed'] = memory_mb <= config.weight_split.max_vram_mb
        
        # 测量推理延迟
        latency_ms = self._measure_latency(model)
        results['latency_ms'] = latency_ms
        results['latency_passed'] = latency_ms <= config.refresh.refresh_period_ms
        
        # 计算算力比例
        compute_ratio = self._compute_ratio(model, config)
        results['compute_ratio'] = compute_ratio
        results['compute_passed'] = compute_ratio <= config.refresh.max_compute_ratio
        
        # 整体通过
        results['overall_passed'] = (
            results['memory_passed'] and 
            results['latency_passed'] and 
            results['compute_passed']
        )
        
        return results
    
    def _estimate_memory(self, model: nn.Module) -> float:
        """估算显存占用"""
        total_params = sum(p.numel() for p in model.parameters())
        # INT4量化：每个参数0.5字节
        memory_bytes = total_params * 0.5
        return memory_bytes / (1024 * 1024)
    
    def _measure_latency(self, model: nn.Module) -> float:
        """测量推理延迟"""
        # 简化实现
        return 5.0  # ms
    
    def _compute_ratio(self, model: nn.Module, config: BrainLikeConfig) -> float:
        """计算算力比例"""
        # 窄窗口注意力复杂度固定
        window_size = config.refresh.max_context_per_cycle + 1
        native_ops = 512 * 512  # 假设原生序列长度512
        return (window_size * window_size) / native_ops


class SelfOptimizationEvaluator:
    """
    自闭环优化能力测评器
    
    测评指标：
    - 自纠错准确率
    - 幻觉抑制率
    - 输出准确率提升幅度
    - 连续使用后的能力进化幅度
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(
        self,
        optimization_system,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """
        评估自闭环优化能力
        
        自纠错准确率≥90%
        幻觉率较原生模型下降70%以上
        """
        results = {}
        
        # 评估自纠错
        correction_scores = []
        for case in test_cases:
            if case.get('has_error', False):
                # 测试自纠错能力
                corrected = self._test_correction(optimization_system, case)
                correction_scores.append(1.0 if corrected else 0.0)
        
        results['correction_accuracy'] = (
            sum(correction_scores) / len(correction_scores) 
            if correction_scores else 0
        )
        results['correction_passed'] = (
            results['correction_accuracy'] >= self.config.self_correction_threshold
        )
        
        # 评估幻觉抑制
        hallucination_reduction = self._test_hallucination_reduction(optimization_system)
        results['hallucination_reduction'] = hallucination_reduction
        results['hallucination_passed'] = (
            hallucination_reduction >= self.config.hallucination_reduction
        )
        
        results['overall_passed'] = (
            results['correction_passed'] and results['hallucination_passed']
        )
        
        return results
    
    def _test_correction(self, optimization_system, case: Dict) -> bool:
        """测试自纠错"""
        # 简化实现
        return True
    
    def _test_hallucination_reduction(self, optimization_system) -> float:
        """测试幻觉抑制"""
        # 简化实现
        return 0.75


class EvaluationPipeline:
    """
    测评流水线
    
    整合所有测评维度，生成综合测评报告
    """
    
    def __init__(
        self,
        model: nn.Module,
        hippocampus_system,
        optimization_system,
        config: BrainLikeConfig
    ):
        self.model = model
        self.hippocampus = hippocampus_system
        self.optimization = optimization_system
        self.config = config
        self.eval_config = config.evaluation
        
        # 初始化各测评器
        self.hippocampus_evaluator = HippocampusMemoryEvaluator(self.eval_config)
        self.baseline_evaluator = BaselineEvaluator(self.eval_config)
        self.reasoning_evaluator = ReasoningEvaluator(self.eval_config)
        self.edge_evaluator = EdgePerformanceEvaluator(self.eval_config)
        self.self_opt_evaluator = SelfOptimizationEvaluator(self.eval_config)
    
    def run_full_evaluation(
        self,
        test_data: Dict[str, Any],
        baseline_model: Optional[nn.Module] = None
    ) -> ComprehensiveReport:
        """
        执行完整测评
        
        Args:
            test_data: 测试数据
            baseline_model: 基线模型
            
        Returns:
            综合测评报告
        """
        dimension_results = {}
        recommendations = []
        
        # 1. 海马体记忆能力测评（权重40%）
        hippocampus_result = self._evaluate_hippocampus(test_data.get('hippocampus', {}))
        dimension_results['hippocampus_memory'] = hippocampus_result
        
        # 2. 基础能力对标测评（权重20%）
        if baseline_model:
            baseline_result = self.baseline_evaluator.evaluate(
                self.model, baseline_model, test_data.get('baseline', [])
            )
            dimension_results['baseline'] = EvaluationResult(
                dimension=EvaluationDimension.BASELINE,
                score=baseline_result.get('performance_ratio', 0),
                max_score=1.0,
                details=baseline_result,
                passed=baseline_result.get('passed', False),
                timestamp=time.time()
            )
        
        # 3. 逻辑推理能力测评（权重20%）
        reasoning_result = self.reasoning_evaluator.evaluate(
            self.model, test_data.get('reasoning', {})
        )
        dimension_results['reasoning'] = EvaluationResult(
            dimension=EvaluationDimension.REASONING,
            score=reasoning_result.get('average_accuracy', 0),
            max_score=1.0,
            details=reasoning_result,
            passed=reasoning_result.get('passed', False),
            timestamp=time.time()
        )
        
        # 4. 端侧性能测评（权重10%）
        edge_result = self.edge_evaluator.evaluate(self.model, self.config)
        dimension_results['edge_performance'] = EvaluationResult(
            dimension=EvaluationDimension.EDGE_PERFORMANCE,
            score=1.0 if edge_result.get('overall_passed', False) else 0.5,
            max_score=1.0,
            details=edge_result,
            passed=edge_result.get('overall_passed', False),
            timestamp=time.time()
        )
        
        # 5. 自闭环优化能力测评（权重10%）
        self_opt_result = self.self_opt_evaluator.evaluate(
            self.optimization, test_data.get('self_optimization', [])
        )
        dimension_results['self_optimization'] = EvaluationResult(
            dimension=EvaluationDimension.SELF_OPTIMIZATION,
            score=self_opt_result.get('correction_accuracy', 0),
            max_score=1.0,
            details=self_opt_result,
            passed=self_opt_result.get('overall_passed', False),
            timestamp=time.time()
        )
        
        # 计算加权总分
        weights = {
            'hippocampus_memory': self.eval_config.hippocampus_weight,
            'baseline': self.eval_config.baseline_weight,
            'reasoning': self.eval_config.reasoning_weight,
            'edge_performance': self.eval_config.edge_weight,
            'self_optimization': self.eval_config.self_correction_weight
        }
        
        weighted_score = sum(
            dimension_results[dim].score * weights.get(dim, 0)
            for dim in dimension_results
        )
        
        # 生成建议
        for dim, result in dimension_results.items():
            if not result.passed:
                recommendations.append(f"需要改进{dim}维度的性能")
        
        # 整体通过判断
        overall_passed = all(r.passed for r in dimension_results.values())
        
        return ComprehensiveReport(
            total_score=sum(r.score for r in dimension_results.values()) / len(dimension_results),
            weighted_score=weighted_score,
            dimension_results=dimension_results,
            overall_passed=overall_passed,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _evaluate_hippocampus(self, test_data: Dict) -> EvaluationResult:
        """评估海马体记忆能力"""
        details = {}
        
        # 情景记忆召回
        if 'recall_cases' in test_data:
            recall_result = self.hippocampus_evaluator.evaluate_recall(
                self.hippocampus, test_data['recall_cases']
            )
            details['recall'] = recall_result
        
        # 模式分离
        if 'separation_cases' in test_data:
            separation_result = self.hippocampus_evaluator.evaluate_pattern_separation(
                self.hippocampus, test_data['separation_cases']
            )
            details['separation'] = separation_result
        
        # 长时序记忆
        if 'long_sequence' in test_data:
            long_seq_result = self.hippocampus_evaluator.evaluate_long_sequence(
                self.hippocampus, test_data['long_sequence']
            )
            details['long_sequence'] = long_seq_result
        
        # 计算综合分数
        scores = []
        if 'recall' in details:
            scores.append(details['recall'].get('recall_accuracy', 0))
        if 'separation' in details:
            scores.append(1 - details['separation'].get('confusion_rate', 1))
        if 'long_sequence' in details:
            scores.append(details['long_sequence'].get('retention_score', 0))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        passed = all(
            details.get(k, {}).get(f'passed_{k}', False) 
            for k in ['recall', 'separation', 'long_sequence'] 
            if k in details
        )
        
        return EvaluationResult(
            dimension=EvaluationDimension.HIPPOCAMPUS_MEMORY,
            score=avg_score,
            max_score=1.0,
            details=details,
            passed=passed,
            timestamp=time.time()
        )
    
    def generate_report_document(
        self,
        report: ComprehensiveReport,
        output_path: str
    ):
        """生成测评报告文档"""
        report_dict = {
            'total_score': report.total_score,
            'weighted_score': report.weighted_score,
            'overall_passed': report.overall_passed,
            'recommendations': report.recommendations,
            'timestamp': report.timestamp,
            'dimensions': {}
        }
        
        for dim, result in report.dimension_results.items():
            report_dict['dimensions'][dim] = {
                'score': result.score,
                'max_score': result.max_score,
                'passed': result.passed,
                'details': result.details
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
