"""
类人脑双系统全闭环AI架构 - 自闭环优化系统
Human-Like Brain Dual-System Full-Loop AI Architecture - Self-Closed Loop Optimization System

生产级实现 - 单模型内的组合输出、竞争优化、自双输出+自评判全能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import math
import random
import time
from collections import defaultdict

from core.config import BrainLikeConfig, OptimizationConfig, OptimizationMode


class OptimizationPhase(Enum):
    """优化阶段"""
    GENERATION = "generation"       # 生成阶段
    VERIFICATION = "verification"    # 验证阶段
    JUDGMENT = "judgment"           # 评判阶段
    REFINEMENT = "refinement"       # 修正阶段


@dataclass
class CandidateOutput:
    """候选输出"""
    token_ids: torch.Tensor
    text: str
    score: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0
    generation_seed: int = 0
    temperature: float = 1.0
    log_probs: List[float] = field(default_factory=list)


@dataclass
class JudgmentResult:
    """评判结果"""
    candidate_idx: int
    total_score: float
    dimensions: Dict[str, float]
    feedback: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class OptimizationTrace:
    """优化轨迹"""
    input_text: str
    mode: OptimizationMode
    candidates: List[CandidateOutput]
    judgments: List[JudgmentResult]
    final_output: str
    iterations: int
    total_time_ms: float
    quality_improvement: float = 0.0


class TextQualityAnalyzer:
    """
    文本质量分析器
    
    实现多维度的文本质量评估
    """
    
    def __init__(self):
        # 逻辑连接词
        self.logical_connectors = {
            'cause': ['因为', '由于', '所以', '因此', '导致', '使得'],
            'contrast': ['但是', '然而', '不过', '相反', '虽然', '尽管'],
            'sequence': ['首先', '然后', '接着', '最后', '之后', '之前'],
            'addition': ['而且', '并且', '另外', '此外', '同时'],
            'conclusion': ['总之', '综上所述', '因此', '所以']
        }
        
        # 常见错误模式
        self.error_patterns = [
            r'(.{1,5})\1{2,}',  # 重复词
            r'[，。！？]{2,}',   # 重复标点
            r'(.{10,})\1{1,}',  # 重复句子片段
        ]
        
        # 事实性知识库（简化版）
        self.fact_patterns = {
            'numbers': r'\d+(?:\.\d+)?(?:万|亿|千|百)?(?:[人|个|次|件|条])?',
            'dates': r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日',
            'locations': r'[北上广深重武成杭南西][京海津庆汉都州安安]',
        }
    
    def analyze_logical_structure(self, text: str) -> Dict[str, Any]:
        """
        分析逻辑结构
        
        Returns:
            逻辑结构分析结果
        """
        result = {
            'has_cause_effect': False,
            'has_contrast': False,
            'has_sequence': False,
            'has_conclusion': False,
            'connector_count': 0,
            'structure_score': 0.0
        }
        
        # 检查各类逻辑连接词
        for connector_type, connectors in self.logical_connectors.items():
            for connector in connectors:
                if connector in text:
                    if connector_type == 'cause':
                        result['has_cause_effect'] = True
                    elif connector_type == 'contrast':
                        result['has_contrast'] = True
                    elif connector_type == 'sequence':
                        result['has_sequence'] = True
                    elif connector_type == 'conclusion':
                        result['has_conclusion'] = True
                    result['connector_count'] += 1
        
        # 计算结构分数
        structure_elements = sum([
            result['has_cause_effect'],
            result['has_contrast'],
            result['has_sequence'],
            result['has_conclusion']
        ])
        result['structure_score'] = min(1.0, structure_elements * 0.25 + result['connector_count'] * 0.1)
        
        return result
    
    def detect_errors(self, text: str) -> List[Dict[str, Any]]:
        """
        检测文本错误
        
        Returns:
            错误列表
        """
        errors = []
        
        for pattern in self.error_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                errors.append({
                    'type': 'repetition',
                    'content': match.group(),
                    'position': match.start(),
                    'severity': 'medium'
                })
        
        # 检查句子完整性
        sentences = re.split(r'[。！？]', text)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查是否缺少主语（简化检查）
            if len(sentence) > 10 and not any(
                pronoun in sentence[:5] 
                for pronoun in ['我', '你', '他', '她', '它', '这', '那', '我们', '你们', '他们']
            ):
                # 检查是否有其他主语形式
                if not re.search(r'[\u4e00-\u9fa5]{2,}(?:是|有|在|做|说)', sentence[:10]):
                    pass  # 可能缺少主语，但不一定错误
        
        return errors
    
    def calculate_coherence(self, text: str) -> float:
        """
        计算语义连贯性
        
        Returns:
            连贯性分数 (0-1)
        """
        if not text or len(text) < 10:
            return 0.0
        
        score = 1.0
        
        # 1. 检查句子长度分布
        sentences = [s.strip() for s in re.split(r'[。！？\n]', text) if s.strip()]
        if len(sentences) > 1:
            lengths = [len(s) for s in sentences]
            avg_length = sum(lengths) / len(lengths)
            length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            
            # 长度方差过大，可能不连贯
            if length_variance > 1000:
                score -= 0.2
        
        # 2. 检查代词使用
        pronouns = ['这', '那', '它', '其', '此']
        pronoun_count = sum(text.count(p) for p in pronouns)
        if pronoun_count > len(text) / 20:
            score -= 0.1  # 代词过多可能导致指代不清
        
        # 3. 检查逻辑连接词密度
        connector_count = 0
        for connectors in self.logical_connectors.values():
            connector_count += sum(text.count(c) for c in connectors)
        
        if len(sentences) > 2 and connector_count < len(sentences) * 0.3:
            score -= 0.1  # 连接词过少
        
        return max(0.0, min(1.0, score))
    
    def check_instruction_following(
        self,
        response: str,
        instruction: str
    ) -> Tuple[float, List[str]]:
        """
        检查指令遵循度
        
        Args:
            response: 响应文本
            instruction: 原始指令
            
        Returns:
            遵循度分数和问题列表
        """
        issues = []
        score = 1.0
        
        # 提取指令中的关键要求
        requirements = []
        
        # 检查字数要求
        length_match = re.search(r'(\d+)[字个]', instruction)
        if length_match:
            required_length = int(length_match.group(1))
            actual_length = len(response)
            if actual_length < required_length * 0.8:
                issues.append(f"回复过短：要求约{required_length}字，实际{actual_length}字")
                score -= 0.3
            elif actual_length > required_length * 1.5:
                issues.append(f"回复过长：要求约{required_length}字，实际{actual_length}字")
                score -= 0.1
        
        # 检查格式要求
        if '列表' in instruction or '列出' in instruction:
            if '1.' not in response and '一、' not in response and '-' not in response:
                issues.append("未按要求使用列表格式")
                score -= 0.2
        
        if '代码' in instruction:
            if '```' not in response and 'def ' not in response and 'function' not in response:
                issues.append("未提供代码")
                score -= 0.3
        
        # 检查是否回答了问题
        question_words = ['什么', '怎么', '如何', '为什么', '哪', '谁', '多少', '是否']
        has_question = any(q in instruction for q in question_words)
        if has_question:
            # 检查是否有实质性回答
            if len(response) < 20:
                issues.append("回答过于简短")
                score -= 0.3
        
        return max(0.0, min(1.0, score)), issues
    
    def calculate_perplexity_proxy(self, text: str) -> float:
        """
        计算困惑度代理值（简化版）
        
        使用字符频率和n-gram频率作为代理
        """
        if len(text) < 10:
            return 100.0
        
        # 计算字符熵
        char_freq = defaultdict(int)
        for char in text:
            char_freq[char] += 1
        
        total_chars = len(text)
        entropy = 0.0
        for count in char_freq.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * math.log2(p)
        
        # 熵值越高，困惑度越低（更自然）
        # 返回归一化分数
        return max(0.0, min(1.0, entropy / 5.0))  # 5.0 是经验最大熵


class ModeSelector:
    """
    模式自动选择器
    
    根据输入特征自动选择最优优化模式
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._self_play_keywords = set(config.self_play_keywords)
        self._self_judgment_keywords = set(config.self_judgment_keywords)
        
        # 模式选择历史
        self._selection_history: List[Tuple[str, OptimizationMode]] = []
    
    def select_mode(self, input_text: str) -> OptimizationMode:
        """
        选择优化模式
        
        Args:
            input_text: 输入文本
            
        Returns:
            选择的优化模式
        """
        # 分析输入特征
        features = self._analyze_input(input_text)
        
        # 规则匹配
        # 1. 检查自博弈关键词
        for keyword in self._self_play_keywords:
            if keyword in input_text:
                return OptimizationMode.SELF_PLAY
        
        # 2. 检查自评判关键词
        for keyword in self._self_judgment_keywords:
            if keyword in input_text:
                return OptimizationMode.SELF_JUDGMENT
        
        # 3. 基于特征决策
        if features['complexity'] > 0.7:
            # 复杂任务使用自博弈
            return OptimizationMode.SELF_PLAY
        elif features['requires_accuracy'] > 0.6:
            # 需要准确性的任务使用自评判
            return OptimizationMode.SELF_JUDGMENT
        else:
            # 默认使用自生成
            return OptimizationMode.SELF_GENERATION
    
    def _analyze_input(self, text: str) -> Dict[str, float]:
        """分析输入特征"""
        features = {}
        
        # 复杂度：基于长度和结构
        features['complexity'] = min(1.0, len(text) / 200.0)
        
        # 是否需要准确性
        accuracy_keywords = ['准确', '精确', '正确', '计算', '分析', '比较']
        features['requires_accuracy'] = sum(1 for k in accuracy_keywords if k in text) / len(accuracy_keywords)
        
        # 是否需要创造性
        creative_keywords = ['创意', '想象', '创造', '设计', '构思']
        features['requires_creativity'] = sum(1 for k in creative_keywords if k in text) / len(creative_keywords)
        
        return features
    
    def add_self_play_keyword(self, keyword: str):
        """添加自博弈关键词"""
        self._self_play_keywords.add(keyword)
    
    def add_self_judgment_keyword(self, keyword: str):
        """添加自评判关键词"""
        self._self_judgment_keywords.add(keyword)


class SelfGenerationModule:
    """
    自生成组合输出模块
    
    生产级实现：
    - 多候选并行生成
    - STDP加权一致性投票
    - 动态权重调整
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.quality_analyzer = TextQualityAnalyzer()
        
        # 权重历史
        self._accuracy_history: List[float] = []
        self._current_weights: List[float] = [0.5, 0.5]
        
        # 生成统计
        self._generation_count = 0
        self._consensus_rate = 0.0
    
    def generate_candidates(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        tokenizer: Any,
        num_candidates: int = 2
    ) -> List[CandidateOutput]:
        """
        生成候选输出
        
        Args:
            model: 模型实例
            input_ids: 输入token IDs
            tokenizer: tokenizer
            num_candidates: 候选数量
            
        Returns:
            候选输出列表
        """
        candidates = []
        temp_low, temp_high = self.config.self_gen_temperature_range
        
        for i in range(num_candidates):
            # 设置不同的温度和种子
            temperature = temp_low + (temp_high - temp_low) * i / max(1, num_candidates - 1)
            seed = random.randint(0, 100000)
            torch.manual_seed(seed)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # 解码
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 计算log probs
            log_probs = []
            if hasattr(outputs, 'scores'):
                for score in outputs.scores:
                    probs = F.softmax(score[0], dim=-1)
                    log_prob = torch.log(probs.max()).item()
                    log_probs.append(log_prob)
            
            # 归一化权重
            weight = self._current_weights[i] if i < len(self._current_weights) else 1.0 / num_candidates
            
            candidates.append(CandidateOutput(
                token_ids=generated_ids,
                text=text,
                weight=weight,
                generation_seed=seed,
                temperature=temperature,
                log_probs=log_probs
            ))
        
        self._generation_count += 1
        return candidates
    
    def weighted_voting(
        self,
        candidates: List[CandidateOutput]
    ) -> CandidateOutput:
        """
        STDP加权一致性投票
        
        Args:
            candidates: 候选输出列表
            
        Returns:
            最终选择的候选
        """
        if len(candidates) == 1:
            return candidates[0]
        
        # 计算每个候选的综合分数
        scored_candidates = []
        
        for candidate in candidates:
            # 基础分数：权重
            score = candidate.weight
            
            # 质量分数
            quality_score = self._calculate_quality_score(candidate.text)
            score += quality_score * 0.3
            
            # 一致性分数：与其他候选的相似度
            consistency_score = self._calculate_consistency(candidate, candidates)
            score += consistency_score * 0.2
            
            # log prob分数
            if candidate.log_probs:
                avg_log_prob = sum(candidate.log_probs) / len(candidate.log_probs)
                prob_score = min(1.0, max(0.0, (avg_log_prob + 5) / 5))  # 归一化
                score += prob_score * 0.2
            
            candidate.score = score
            scored_candidates.append((candidate, score))
        
        # 排序选择
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 更新一致性率
        best = scored_candidates[0][0]
        self._consensus_rate = self._consensus_rate * 0.9 + 0.1 * (
            1.0 if sum(1 for c in candidates if c.text == best.text) > len(candidates) / 2 else 0.0
        )
        
        return best
    
    def _calculate_quality_score(self, text: str) -> float:
        """计算质量分数"""
        score = 0.0
        
        # 连贯性
        coherence = self.quality_analyzer.calculate_coherence(text)
        score += coherence * 0.4
        
        # 逻辑结构
        structure = self.quality_analyzer.analyze_logical_structure(text)
        score += structure['structure_score'] * 0.3
        
        # 错误检测
        errors = self.quality_analyzer.detect_errors(text)
        error_penalty = len(errors) * 0.1
        score = max(0, score - error_penalty)
        
        return score
    
    def _calculate_consistency(
        self,
        candidate: CandidateOutput,
        all_candidates: List[CandidateOutput]
    ) -> float:
        """计算一致性分数"""
        if len(all_candidates) <= 1:
            return 1.0
        
        # 简化：基于文本相似度
        similarities = []
        for other in all_candidates:
            if other is candidate:
                continue
            
            # Jaccard相似度
            words1 = set(candidate.text)
            words2 = set(other.text)
            if words1 or words2:
                similarity = len(words1 & words2) / len(words1 | words2)
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def update_weights(self, feedback_score: float):
        """
        更新候选权重
        
        Args:
            feedback_score: 反馈分数 (0-1)
        """
        self._accuracy_history.append(feedback_score)
        if len(self._accuracy_history) > 10:
            self._accuracy_history.pop(0)
        
        # 根据历史调整权重
        if len(self._accuracy_history) >= 5:
            recent_avg = sum(self._accuracy_history[-5:]) / 5
            
            # 如果准确率高，增加低温候选权重
            if recent_avg > 0.7:
                self._current_weights = [0.6, 0.4]  # 更倾向于确定性输出
            elif recent_avg < 0.4:
                self._current_weights = [0.4, 0.6]  # 更倾向于多样性
            else:
                self._current_weights = [0.5, 0.5]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'generation_count': self._generation_count,
            'consensus_rate': self._consensus_rate,
            'current_weights': self._current_weights,
            'average_accuracy': sum(self._accuracy_history) / len(self._accuracy_history) if self._accuracy_history else 0.0
        }


class SelfPlayModule:
    """
    自博弈竞争优化模块
    
    生产级实现：
    - 提案-验证对抗
    - 迭代优化
    - 收敛检测
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.quality_analyzer = TextQualityAnalyzer()
        
        # 迭代状态
        self._iteration_count = 0
        self._proposal_history: List[str] = []
        self._verification_history: List[str] = []
        self._convergence_threshold = 0.95
    
    def propose(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        tokenizer: Any,
        context: str,
        previous_feedback: str = ""
    ) -> CandidateOutput:
        """
        提案角色：生成推理结果
        
        Args:
            model: 模型实例
            input_ids: 输入token IDs
            tokenizer: tokenizer
            context: 上下文
            previous_feedback: 之前的反馈
            
        Returns:
            提案输出
        """
        # 构建提案提示
        if previous_feedback:
            proposal_prompt = f"{context}\n\n之前的反馈：{previous_feedback}\n请根据反馈改进你的回答："
        else:
            proposal_prompt = context
        
        # 编码
        proposal_input = tokenizer(
            proposal_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).input_ids.to(input_ids.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                proposal_input,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        text = tokenizer.decode(outputs[0][proposal_input.shape[1]:], skip_special_tokens=True)
        
        return CandidateOutput(
            token_ids=outputs[0],
            text=text,
            score=1.0
        )
    
    def verify(
        self,
        model: nn.Module,
        proposal: CandidateOutput,
        input_ids: torch.Tensor,
        tokenizer: Any,
        context: str
    ) -> Tuple[bool, str, List[str]]:
        """
        验证角色：校验提案结果
        
        Args:
            model: 模型实例
            proposal: 提案输出
            input_ids: 输入token IDs
            tokenizer: tokenizer
            context: 上下文
            
        Returns:
            is_correct: 是否正确
            feedback: 反馈信息
            issues: 问题列表
        """
        issues = []
        feedback_parts = []
        is_correct = True
        
        # 1. 逻辑一致性检查
        logical_result = self._check_logical_consistency(proposal.text, context)
        if not logical_result['consistent']:
            issues.extend(logical_result['issues'])
            feedback_parts.append(f"逻辑问题：{', '.join(logical_result['issues'])}")
            is_correct = False
        
        # 2. 事实准确性检查
        factual_result = self._check_factual_accuracy(proposal.text, context)
        if not factual_result['accurate']:
            issues.extend(factual_result['issues'])
            feedback_parts.append(f"事实问题：{', '.join(factual_result['issues'])}")
            is_correct = False
        
        # 3. 完整性检查
        completeness_result = self._check_completeness(proposal.text, context)
        if not completeness_result['complete']:
            issues.extend(completeness_result['issues'])
            feedback_parts.append(f"完整性问题：{', '.join(completeness_result['issues'])}")
            is_correct = False
        
        # 4. 质量检查
        quality_score = self._calculate_quality_score(proposal.text)
        if quality_score < 0.5:
            issues.append(f"质量分数较低：{quality_score:.2f}")
            feedback_parts.append("回答质量需要提升")
            is_correct = False
        
        # 生成反馈
        if is_correct:
            feedback = "验证通过，回答质量良好。"
        else:
            feedback = " | ".join(feedback_parts)
        
        return is_correct, feedback, issues
    
    def _check_logical_consistency(self, text: str, context: str) -> Dict[str, Any]:
        """检查逻辑一致性"""
        result = {
            'consistent': True,
            'issues': []
        }
        
        # 检查自相矛盾
        contradiction_patterns = [
            (r'不是(.{1,10})，而是\1', '自相矛盾'),
            (r'不可能.*可能', '前后矛盾'),
            (r'一定.*不一定', '确定性矛盾'),
        ]
        
        for pattern, issue in contradiction_patterns:
            if re.search(pattern, text):
                result['issues'].append(issue)
                result['consistent'] = False
        
        # 检查逻辑结构
        structure = self.quality_analyzer.analyze_logical_structure(text)
        if structure['structure_score'] < 0.3:
            result['issues'].append('逻辑结构不清晰')
            result['consistent'] = False
        
        return result
    
    def _check_factual_accuracy(self, text: str, context: str) -> Dict[str, Any]:
        """检查事实准确性"""
        result = {
            'accurate': True,
            'issues': []
        }
        
        # 检查数字一致性
        numbers_in_text = re.findall(r'\d+(?:\.\d+)?', text)
        numbers_in_context = re.findall(r'\d+(?:\.\d+)?', context)
        
        # 如果上下文有数字，检查是否被正确引用
        if numbers_in_context:
            for num in numbers_in_context[:3]:  # 只检查前3个
                if num not in text and len(text) > 50:
                    # 可能遗漏了重要数字
                    pass  # 不一定错误，只是可能遗漏
        
        # 检查明显的错误
        error_patterns = [
            (r'(\d{4})年(\d{2})月(\d{2})日', lambda m: int(m.group(2)) > 12 or int(m.group(3)) > 31, '日期错误'),
        ]
        
        for pattern, check, issue in error_patterns:
            for match in re.finditer(pattern, text):
                if check(match):
                    result['issues'].append(issue)
                    result['accurate'] = False
        
        return result
    
    def _check_completeness(self, text: str, context: str) -> Dict[str, Any]:
        """检查完整性"""
        result = {
            'complete': True,
            'issues': []
        }
        
        # 检查是否太短
        if len(text) < 20:
            result['issues'].append('回答过于简短')
            result['complete'] = False
        
        # 检查是否截断
        truncation_markers = ['...', '等等', '例如', '如']
        for marker in truncation_markers:
            if text.rstrip().endswith(marker) and len(text) < 100:
                result['issues'].append('回答可能被截断')
                result['complete'] = False
                break
        
        return result
    
    def _calculate_quality_score(self, text: str) -> float:
        """计算质量分数"""
        score = 0.0
        
        # 连贯性
        coherence = self.quality_analyzer.calculate_coherence(text)
        score += coherence * 0.4
        
        # 困惑度代理
        perplexity_score = self.quality_analyzer.calculate_perplexity_proxy(text)
        score += perplexity_score * 0.3
        
        # 结构
        structure = self.quality_analyzer.analyze_logical_structure(text)
        score += structure['structure_score'] * 0.3
        
        return score
    
    def iterate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        tokenizer: Any,
        context: str,
        max_iterations: int = 5
    ) -> Tuple[CandidateOutput, List[Dict]]:
        """
        迭代优化
        
        Args:
            model: 模型实例
            input_ids: 输入token IDs
            tokenizer: tokenizer
            context: 上下文
            max_iterations: 最大迭代次数
            
        Returns:
            最终优化结果和迭代历史
        """
        self._iteration_count = 0
        current_proposal = None
        iteration_history = []
        previous_feedback = ""
        
        for i in range(max_iterations):
            self._iteration_count = i + 1
            
            # 提案
            current_proposal = self.propose(
                model, input_ids, tokenizer, context, previous_feedback
            )
            self._proposal_history.append(current_proposal.text)
            
            # 验证
            is_correct, feedback, issues = self.verify(
                model, current_proposal, input_ids, tokenizer, context
            )
            self._verification_history.append(feedback)
            
            # 记录迭代
            iteration_history.append({
                'iteration': i + 1,
                'proposal': current_proposal.text[:100],
                'is_correct': is_correct,
                'feedback': feedback,
                'issues': issues
            })
            
            # 如果正确或收敛，结束迭代
            if is_correct or self._check_convergence():
                break
            
            previous_feedback = feedback
        
        return current_proposal, iteration_history
    
    def _check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self._proposal_history) < 2:
            return False
        
        # 计算最近两次提案的相似度
        recent = self._proposal_history[-2:]
        
        # Jaccard相似度
        words1 = set(recent[0])
        words2 = set(recent[1])
        
        if not (words1 or words2):
            return True
        
        similarity = len(words1 & words2) / len(words1 | words2)
        
        return similarity >= self._convergence_threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'iteration_count': self._iteration_count,
            'proposal_count': len(self._proposal_history),
            'verification_count': len(self._verification_history),
            'convergence_rate': sum(1 for i in range(1, len(self._proposal_history)) 
                                   if self._proposal_history[i] == self._proposal_history[i-1]) / 
                               max(1, len(self._proposal_history) - 1)
        }


class SelfJudgmentModule:
    """
    自双输出+自评判选优模块
    
    生产级实现：
    - 多维度评判
    - 智能选优
    - 反馈生成
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.quality_analyzer = TextQualityAnalyzer()
        
        self.dimensions = config.self_judgment_dimensions
        self.max_score = config.self_judgment_max_score
        self.total_score = config.self_judgment_total_score
        
        # 评判历史
        self._judgment_history: List[JudgmentResult] = []
        self._dimension_weights: Dict[str, float] = {
            'factual_accuracy': 0.3,
            'logical_completeness': 0.25,
            'semantic_coherence': 0.25,
            'instruction_following': 0.2
        }
    
    def generate_dual_candidates(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        tokenizer: Any
    ) -> List[CandidateOutput]:
        """
        生成双候选输出
        
        Args:
            model: 模型实例
            input_ids: 输入token IDs
            tokenizer: tokenizer
            
        Returns:
            两个候选输出
        """
        candidates = []
        
        for i in range(2):
            # 使用不同的随机种子和温度
            seed = random.randint(0, 100000)
            torch.manual_seed(seed)
            temperature = 0.7 + i * 0.2  # 第二个候选温度更高
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            candidates.append(CandidateOutput(
                token_ids=outputs[0],
                text=text,
                generation_seed=seed,
                temperature=temperature
            ))
        
        return candidates
    
    def judge(
        self,
        model: nn.Module,
        candidates: List[CandidateOutput],
        context: str
    ) -> List[JudgmentResult]:
        """
        评判候选输出
        
        Args:
            model: 模型实例
            candidates: 候选输出列表
            context: 上下文
            
        Returns:
            评判结果列表
        """
        results = []
        
        for idx, candidate in enumerate(candidates):
            # 对每个维度进行评分
            dimension_scores = {}
            all_issues = []
            
            for dim in self.dimensions:
                score, issues = self._evaluate_dimension(
                    candidate.text, context, dim
                )
                dimension_scores[dim] = score
                all_issues.extend(issues)
            
            # 计算加权总分
            total = sum(
                dimension_scores.get(dim, 0) * weight 
                for dim, weight in self._dimension_weights.items()
            )
            
            # 生成反馈
            feedback = self._generate_feedback(dimension_scores, all_issues)
            
            result = JudgmentResult(
                candidate_idx=idx,
                total_score=total,
                dimensions=dimension_scores,
                feedback=feedback,
                issues=all_issues
            )
            results.append(result)
        
        self._judgment_history.extend(results)
        return results
    
    def _evaluate_dimension(
        self,
        text: str,
        context: str,
        dimension: str
    ) -> Tuple[float, List[str]]:
        """
        评估单个维度
        
        Args:
            text: 候选文本
            context: 上下文
            dimension: 评估维度
            
        Returns:
            维度分数和问题列表
        """
        issues = []
        score = self.max_score / 2  # 基础分
        
        if dimension == "factual_accuracy":
            # 事实准确性
            result = self._evaluate_factual_accuracy(text, context)
            score = result['score'] * self.max_score
            issues = result.get('issues', [])
        
        elif dimension == "logical_completeness":
            # 逻辑完整性
            result = self._evaluate_logical_completeness(text)
            score = result['score'] * self.max_score
            issues = result.get('issues', [])
        
        elif dimension == "semantic_coherence":
            # 语义连贯性
            result = self._evaluate_semantic_coherence(text)
            score = result['score'] * self.max_score
            issues = result.get('issues', [])
        
        elif dimension == "instruction_following":
            # 指令遵循度
            result = self._evaluate_instruction_following(text, context)
            score = result['score'] * self.max_score
            issues = result.get('issues', [])
        
        return min(self.max_score, max(0, score)), issues
    
    def _evaluate_factual_accuracy(self, text: str, context: str) -> Dict[str, Any]:
        """评估事实准确性"""
        result = {'score': 0.7, 'issues': []}
        
        # 检查数字
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            # 有数字，检查是否合理
            for num in numbers:
                try:
                    n = float(num)
                    if n < 0 and '温度' not in context and '度' not in context:
                        result['issues'].append(f'负数可能不合理: {num}')
                        result['score'] -= 0.1
                except:
                    pass
        
        # 检查日期
        dates = re.findall(r'\d{4}年|\d{1,2}月\d{1,2}日', text)
        for date in dates:
            if '年' in date:
                year = int(re.search(r'\d{4}', date).group())
                if year > 2030 or year < 1900:
                    result['issues'].append(f'年份可能不合理: {date}')
                    result['score'] -= 0.1
        
        return result
    
    def _evaluate_logical_completeness(self, text: str) -> Dict[str, Any]:
        """评估逻辑完整性"""
        result = {'score': 0.5, 'issues': []}
        
        # 分析逻辑结构
        structure = self.quality_analyzer.analyze_logical_structure(text)
        
        # 有因果结构加分
        if structure['has_cause_effect']:
            result['score'] += 0.15
        
        # 有对比结构加分
        if structure['has_contrast']:
            result['score'] += 0.1
        
        # 有结论加分
        if structure['has_conclusion']:
            result['score'] += 0.15
        
        # 有序列加分
        if structure['has_sequence']:
            result['score'] += 0.1
        
        # 检查逻辑错误
        errors = self.quality_analyzer.detect_errors(text)
        if errors:
            result['score'] -= len(errors) * 0.1
            result['issues'].extend([e['type'] for e in errors])
        
        return result
    
    def _evaluate_semantic_coherence(self, text: str) -> Dict[str, Any]:
        """评估语义连贯性"""
        result = {'score': 0.5, 'issues': []}
        
        # 计算连贯性
        coherence = self.quality_analyzer.calculate_coherence(text)
        result['score'] = coherence
        
        # 计算困惑度代理
        perplexity_score = self.quality_analyzer.calculate_perplexity_proxy(text)
        result['score'] = (coherence + perplexity_score) / 2
        
        return result
    
    def _evaluate_instruction_following(self, text: str, context: str) -> Dict[str, Any]:
        """评估指令遵循度"""
        score, issues = self.quality_analyzer.check_instruction_following(text, context)
        return {'score': score, 'issues': issues}
    
    def _generate_feedback(
        self,
        dimensions: Dict[str, float],
        issues: List[str]
    ) -> str:
        """生成反馈"""
        feedback_parts = []
        
        # 维度分数
        dim_names = {
            "factual_accuracy": "事实准确性",
            "logical_completeness": "逻辑完整性",
            "semantic_coherence": "语义连贯性",
            "instruction_following": "指令遵循度"
        }
        
        for dim, score in dimensions.items():
            name = dim_names.get(dim, dim)
            level = "优秀" if score >= 8 else "良好" if score >= 6 else "需改进"
            feedback_parts.append(f"{name}: {score:.1f}分({level})")
        
        # 问题汇总
        if issues:
            unique_issues = list(set(issues))[:3]  # 最多显示3个
            feedback_parts.append(f"主要问题: {', '.join(unique_issues)}")
        
        return " | ".join(feedback_parts)
    
    def select_best(
        self,
        candidates: List[CandidateOutput],
        judgments: List[JudgmentResult]
    ) -> CandidateOutput:
        """
        选择最优候选
        
        Args:
            candidates: 候选输出列表
            judgments: 评判结果列表
            
        Returns:
            最优候选
        """
        if not judgments:
            return candidates[0] if candidates else None
        
        # 选择总分最高的
        best_idx = max(range(len(judgments)), key=lambda i: judgments[i].total_score)
        best_candidate = candidates[best_idx]
        best_candidate.score = judgments[best_idx].total_score
        best_candidate.dimensions = judgments[best_idx].dimensions
        
        return best_candidate
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._judgment_history:
            return {'judgment_count': 0}
        
        avg_scores = defaultdict(float)
        for result in self._judgment_history:
            for dim, score in result.dimensions.items():
                avg_scores[dim] += score
        
        count = len(self._judgment_history)
        for dim in avg_scores:
            avg_scores[dim] /= count
        
        return {
            'judgment_count': count,
            'average_scores': dict(avg_scores),
            'dimension_weights': self._dimension_weights
        }


class SelfClosedLoopOptimization:
    """
    自闭环优化系统
    
    整合三种优化模式，实现自动模式切换和全链路优化
    """
    
    def __init__(self, config: BrainLikeConfig):
        self.config = config
        self.opt_config = config.optimization
        
        # 初始化各模块
        self.mode_selector = ModeSelector(self.opt_config)
        self.self_generation = SelfGenerationModule(self.opt_config)
        self.self_play = SelfPlayModule(self.opt_config)
        self.self_judgment = SelfJudgmentModule(self.opt_config)
        
        # 当前模式
        self._current_mode = OptimizationMode.SELF_GENERATION
        
        # 优化历史
        self._optimization_history: List[OptimizationTrace] = []
    
    def optimize(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        tokenizer: Any,
        context: str = ""
    ) -> Tuple[CandidateOutput, Dict[str, Any]]:
        """
        执行优化
        
        Args:
            model: 模型实例
            input_ids: 输入token IDs
            tokenizer: tokenizer
            context: 上下文
            
        Returns:
            最优输出和优化信息
        """
        start_time = time.time()
        
        # 自动选择模式
        self._current_mode = self.mode_selector.select_mode(context)
        
        optimization_info = {
            'mode': self._current_mode.value,
            'iterations': 0,
            'judgments': [],
            'phase': OptimizationPhase.GENERATION.value
        }
        
        if self._current_mode == OptimizationMode.SELF_GENERATION:
            # 模式1：自生成组合输出
            candidates = self.self_generation.generate_candidates(
                model, input_ids, tokenizer
            )
            result = self.self_generation.weighted_voting(candidates)
            optimization_info['candidates_count'] = len(candidates)
            optimization_info['consensus_rate'] = self.self_generation._consensus_rate
        
        elif self._current_mode == OptimizationMode.SELF_PLAY:
            # 模式2：自博弈竞争优化
            result, iteration_history = self.self_play.iterate(
                model, input_ids, tokenizer, context
            )
            optimization_info['iterations'] = self.self_play._iteration_count
            optimization_info['iteration_history'] = iteration_history[-3:]  # 只保留最近3次
            optimization_info['phase'] = OptimizationPhase.REFINEMENT.value
        
        elif self._current_mode == OptimizationMode.SELF_JUDGMENT:
            # 模式3：自双输出+自评判选优
            candidates = self.self_judgment.generate_dual_candidates(
                model, input_ids, tokenizer
            )
            judgments = self.self_judgment.judge(model, candidates, context)
            result = self.self_judgment.select_best(candidates, judgments)
            optimization_info['judgments'] = [
                {
                    'idx': j.candidate_idx,
                    'score': j.total_score,
                    'feedback': j.feedback
                }
                for j in judgments
            ]
            optimization_info['phase'] = OptimizationPhase.JUDGMENT.value
        
        else:
            # 默认直接生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=self.opt_config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            result = CandidateOutput(token_ids=outputs[0], text=text)
        
        # 记录优化时间
        optimization_info['optimization_time_ms'] = (time.time() - start_time) * 1000
        
        return result, optimization_info
    
    def get_mode(self) -> OptimizationMode:
        """获取当前模式"""
        return self._current_mode
    
    def set_mode(self, mode: OptimizationMode):
        """手动设置模式"""
        self._current_mode = mode
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'current_mode': self._current_mode.value,
            'self_generation': self.self_generation.get_statistics(),
            'self_play': self.self_play.get_statistics(),
            'self_judgment': self.self_judgment.get_statistics(),
            'optimization_count': len(self._optimization_history)
        }
    
    def update_weights_from_feedback(self, feedback_score: float):
        """
        根据反馈更新权重
        
        Args:
            feedback_score: 反馈分数 (0-1)
        """
        self.self_generation.update_weights(feedback_score)
