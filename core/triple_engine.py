#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 三大增强引擎
Three-Enhancement Engine: Memory + Logic + Joint

1. 记忆增强 (Memory Enhancement)
   - 纠正检测和存储
   - 推理规则知识库
   - 自动召回机制

2. 逻辑增强 (Logic Enhancement)  
   - 语义解析器
   - 数值推理器
   - 答案验证器

3. 联合增强 (Joint Enhancement)
   - 反馈学习
   - 一致性检查
   - 规则更新
"""

import os
import sys
import logging
import time
import re
import math
import json
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import torch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem
from modules.refresh_engine import RefreshEngine as ModuleRefreshEngine
from modules.stdp_system import STDPKernel as ModuleSTDPKernel


# ============================================================================
# 第一部分：记忆增强系统
# ============================================================================

@dataclass
class InferenceRule:
    """推理规则"""
    rule_id: str
    pattern: str  # 匹配模式
    template: str  # 推理模板
    example: str  # 示例
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    corrected: bool = False


class MemoryEnhancer:
    """
    记忆增强系统
    
    功能：
    1. 存储用户纠正的推理规则
    2. 建立知识库
    3. 自动召回相关记忆
    """
    
    def __init__(self):
        # 推理规则库
        self.rules: Dict[str, InferenceRule] = {}
        
        # 纠正历史
        self.corrections: List[Dict] = []
        
        # 会话知识
        self.session_knowledge: Dict[str, Any] = {}
        
        # 预定义规则
        self._init_default_rules()
        
        logger.info("记忆增强系统初始化完成")
    
    def _init_default_rules(self):
        """初始化默认推理规则"""
        default_rules = [
            InferenceRule(
                rule_id="rent_calc",
                pattern=r"(\d+)\s*天\s*房租\s*(\d+)\s*元",
                template="日租金=总租金÷天数，月租金=日租金×30",
                example="20天房租1600元 → 日租金=1600÷20=80元，月租金=80×30=2400元"
            ),
            InferenceRule(
                rule_id="deposit_calc",
                pattern=r"押金[：:]*\s*(\d+)",
                template="押金是固定金额，退房时退还",
                example="押金2400元 → 退房时退还2400元"
            ),
            InferenceRule(
                rule_id="cleaning_fee",
                pattern=r"卫生费[：:]*\s*(\d+)",
                template="卫生费在卫生干净时退还",
                example="卫生费200元 → 卫生干净退200元"
            ),
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def detect_correction(self, user_input: str, prev_answer: str) -> Optional[Dict]:
        """检测用户纠正"""
        correction_keywords = ['不对', '错误', '错了', '不是', '应该是', '正确的是']
        
        is_correction = any(kw in user_input for kw in correction_keywords)
        
        if is_correction:
            # 提取正确答案
            numbers = re.findall(r'\d+', user_input)
            
            correction = {
                'user_input': user_input,
                'prev_answer': prev_answer,
                'numbers': numbers,
                'timestamp': time.time()
            }
            
            self.corrections.append(correction)
            logger.info(f"检测到纠正: {user_input[:50]}")
            
            return correction
        
        return None
    
    def learn_from_correction(self, correction: Dict, correct_answer: str):
        """从纠正中学习"""
        # 创建新规则
        rule_id = f"learned_{int(time.time())}"
        
        rule = InferenceRule(
            rule_id=rule_id,
            pattern=correction['user_input'][:50],
            template=correct_answer,
            example=f"用户纠正: {correction['user_input']} → {correct_answer}",
            corrected=True
        )
        
        self.rules[rule_id] = rule
        logger.info(f"学习新规则: {rule_id}")
    
    def recall_relevant_rules(self, query: str) -> List[InferenceRule]:
        """召回相关规则"""
        relevant = []
        
        for rule in self.rules.values():
            if re.search(rule.pattern, query, re.IGNORECASE):
                relevant.append(rule)
        
        # 优先返回被纠正过的规则
        relevant.sort(key=lambda r: (r.corrected, r.confidence), reverse=True)
        
        return relevant
    
    def store_session_knowledge(self, key: str, value: Any):
        """存储会话知识"""
        self.session_knowledge[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def get_session_knowledge(self, key: str) -> Optional[Any]:
        """获取会话知识"""
        if key in self.session_knowledge:
            return self.session_knowledge[key]['value']
        return None


# ============================================================================
# 第二部分：逻辑增强系统
# ============================================================================

class LogicEnhancer:
    """
    逻辑增强系统
    
    功能：
    1. 语义解析 - 正确理解问题含义
    2. 数值推理 - 自动提取和计算
    3. 答案验证 - 检查答案合理性
    """
    
    def __init__(self):
        # 语义模式
        self.semantic_patterns = {
            # 租金计算模式
            'rent_total': {
                'pattern': r'(\d+)\s*天\s*房租\s*(\d+)\s*元',
                'parse': lambda m: {'days': int(m.group(1)), 'total_rent': int(m.group(2))},
                'meaning': 'X天的总租金是Y元'
            },
            # 押金模式
            'deposit': {
                'pattern': r'押金[：:]*\s*(\d+)',
                'parse': lambda m: {'deposit': int(m.group(1))},
                'meaning': '押金金额'
            },
            # 卫生费模式
            'cleaning': {
                'pattern': r'卫生费[：:]*\s*(\d+)',
                'parse': lambda m: {'cleaning_fee': int(m.group(1))},
                'meaning': '卫生费金额'
            },
            # 范围求偶数模式
            'range_even': {
                'pattern': r'(\d+).*?(\d+).*?最大.*?偶数',
                'parse': lambda m: {'start': int(m.group(1)), 'end': int(m.group(2))},
                'meaning': '求范围内的最大偶数'
            },
        }
        
        logger.info("逻辑增强系统初始化完成")
    
    def parse_semantics(self, text: str) -> Dict[str, Any]:
        """语义解析"""
        result = {
            'original': text,
            'parsed': {},
            'patterns_matched': []
        }
        
        for pattern_name, pattern_info in self.semantic_patterns.items():
            match = re.search(pattern_info['pattern'], text)
            if match:
                parsed_data = pattern_info['parse'](match)
                result['parsed'].update(parsed_data)
                result['patterns_matched'].append({
                    'name': pattern_name,
                    'meaning': pattern_info['meaning'],
                    'data': parsed_data
                })
        
        return result
    
    def numerical_reasoning(self, parsed: Dict) -> Dict[str, Any]:
        """数值推理"""
        results = {}
        
        # 租金计算
        if 'days' in parsed and 'total_rent' in parsed:
            days = parsed['days']
            total = parsed['total_rent']
            daily = total / days
            monthly = daily * 30
            
            results['rent_calc'] = {
                'daily_rent': daily,
                'monthly_rent': monthly,
                'formula': f'日租金={total}÷{days}={daily:.0f}元，月租金={daily:.0f}×30={monthly:.0f}元'
            }
        
        # 范围最大偶数
        if 'start' in parsed and 'end' in parsed:
            start = parsed['start']
            end = parsed['end']
            
            # 找最大偶数
            if end % 2 == 0:
                max_even = end
            else:
                max_even = end - 1
            
            if max_even >= start:
                results['max_even'] = {
                    'value': max_even,
                    'formula': f'{start}和{end}之间的最大偶数是{max_even}'
                }
        
        return results
    
    def validate_answer(self, question: str, answer: str, reasoning: Dict) -> Tuple[bool, str]:
        """验证答案"""
        # 检查答案中是否包含推理结果
        for key, result in reasoning.items():
            if isinstance(result, dict) and 'formula' in result:
                # 检查答案是否包含关键数字
                expected_values = re.findall(r'\d+', result['formula'])
                answer_values = re.findall(r'\d+', answer)
                
                # 如果答案中的数字与预期不符
                if expected_values and answer_values:
                    # 检查关键数字是否匹配
                    key_numbers = [int(v) for v in expected_values[-2:]]  # 最后两个数字是关键
                    answer_numbers = [int(v) for v in answer_values]
                    
                    for kn in key_numbers:
                        if kn not in answer_numbers:
                            return False, f"答案可能缺少关键数字: {kn}"
        
        return True, "答案验证通过"


# ============================================================================
# 第三部分：联合增强系统
# ============================================================================

class JointEnhancer:
    """
    联合增强系统
    
    功能：
    1. 反馈学习 - 从用户纠正中学习
    2. 一致性检查 - 确保回答一致
    3. 规则更新 - 动态更新推理规则
    """
    
    def __init__(self, memory: MemoryEnhancer, logic: LogicEnhancer):
        self.memory = memory
        self.logic = logic
        
        # 回答历史
        self.answer_history: List[Dict] = []
        
        # 学习到的规则
        self.learned_rules: Dict[str, str] = {}
        
        logger.info("联合增强系统初始化完成")
    
    def process_with_feedback(self, user_input: str, prev_answer: str = None) -> Dict:
        """带反馈的处理"""
        result = {
            'is_correction': False,
            'learned': False,
            'semantic': None,
            'reasoning': None,
            'validation': None
        }
        
        # 1. 检测纠正
        if prev_answer:
            correction = self.memory.detect_correction(user_input, prev_answer)
            if correction:
                result['is_correction'] = True
                
                # 从纠正中提取正确答案
                correct_answer = self._extract_correct_answer(user_input)
                if correct_answer:
                    self.memory.learn_from_correction(correction, correct_answer)
                    result['learned'] = True
        
        # 2. 语义解析
        semantic = self.logic.parse_semantics(user_input)
        result['semantic'] = semantic
        
        # 3. 数值推理
        if semantic['parsed']:
            reasoning = self.logic.numerical_reasoning(semantic['parsed'])
            result['reasoning'] = reasoning
        
        # 4. 召回相关规则
        relevant_rules = self.memory.recall_relevant_rules(user_input)
        result['relevant_rules'] = relevant_rules
        
        return result
    
    def _extract_correct_answer(self, text: str) -> Optional[str]:
        """从纠正中提取正确答案"""
        # 查找"应该是X"或"正确的是X"模式
        patterns = [
            r'应该是\s*(.+?)(?:\s|$)',
            r'正确的是\s*(.+?)(?:\s|$)',
            r'是\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def check_consistency(self, question: str, new_answer: str) -> Tuple[bool, Optional[str]]:
        """一致性检查"""
        # 查找历史中的相同问题
        for hist in self.answer_history:
            if hist['question'] == question:
                if hist['answer'] != new_answer:
                    return False, hist['answer']
        
        return True, None
    
    def record_answer(self, question: str, answer: str, reasoning: Dict = None):
        """记录回答"""
        self.answer_history.append({
            'question': question,
            'answer': answer,
            'reasoning': reasoning,
            'timestamp': time.time()
        })
        
        # 保持历史在合理范围
        if len(self.answer_history) > 100:
            self.answer_history = self.answer_history[-100:]


# ============================================================================
# 第四部分：数学增强器（保留原有功能）
# ============================================================================

class MathEnhancer:
    """数学增强器 - 康托集+斐波那契+素数+黎曼zeta"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.cantor = self._compute_cantor()
        self.fib = self._compute_fibonacci()
        self.primes = self._compute_primes()
        self.zeta = self._compute_zeta()
        
        logger.info(f"数学增强器: 康托集{len(self.cantor)}, 斐波那契{len(self.fib)}, 素数{len(self.primes)}")
    
    def _compute_cantor(self) -> List[int]:
        indices = []
        for i in range(min(self.vocab_size, 100000)):
            n = i
            is_cantor = True
            while n > 0:
                if n % 3 == 1:
                    is_cantor = False
                    break
                n //= 3
            if is_cantor:
                indices.append(i)
        return indices
    
    def _compute_fibonacci(self) -> List[int]:
        indices = []
        a, b = 1, 1
        while b < self.vocab_size:
            indices.append(b)
            a, b = b, a + b
        return indices
    
    def _compute_primes(self) -> List[int]:
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True
        return [i for i in range(min(self.vocab_size, 20000)) if is_prime(i)]
    
    def _compute_zeta(self) -> torch.Tensor:
        weights = torch.zeros(self.vocab_size)
        for i in range(1, min(self.vocab_size, 1000)):
            weights[i] = 1.0 / (i ** 1.5)
        return weights / (weights.sum() + 1e-8)
    
    def enhance(self, logits: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
        logits = logits.clone()
        
        for idx in self.cantor:
            if idx < logits.shape[-1]:
                logits[0, idx] += strength
        
        for idx in self.fib:
            if idx < logits.shape[-1]:
                logits[0, idx] += strength * 0.5
        
        for idx in self.primes:
            if idx < logits.shape[-1] and idx < len(self.zeta):
                w = self.zeta[idx].item()
                logits[0, idx] += strength * w * 2
        
        return logits


# ============================================================================
# 第五部分：三大增强引擎
# ============================================================================

class TripleEnhancedEngine:
    """
    三大增强引擎
    
    整合：
    1. 记忆增强
    2. 逻辑增强
    3. 联合增强
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        
        # 三大增强系统
        self.memory_enhancer: Optional[MemoryEnhancer] = None
        self.logic_enhancer: Optional[LogicEnhancer] = None
        self.joint_enhancer: Optional[JointEnhancer] = None
        self.math_enhancer: Optional[MathEnhancer] = None
        
        self.refresh_engine = None
        
        # 会话状态
        self.session: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self._prev_answer: Optional[str] = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"三大增强引擎运行设备: {self.device}")
        self._initialized = False
    
    def initialize(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化三大增强引擎: {self.model_path}")
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 初始化三大增强系统
        self.memory_enhancer = MemoryEnhancer()
        self.logic_enhancer = LogicEnhancer()
        self.joint_enhancer = JointEnhancer(self.memory_enhancer, self.logic_enhancer)
        self.math_enhancer = MathEnhancer(len(self.tokenizer))
        
        # 冻结权重
        self._freeze_weights()
        
        # 初始化子模块
        hippo = HippocampusSystem(self.config)
        stdp = ModuleSTDPKernel(self.config.stdp)
        self.refresh_engine = ModuleRefreshEngine(
            self.model, self.config, hippocampus_module=hippo, stdp_module=stdp
        )
        
        self._initialized = True
        logger.info("三大增强引擎初始化完成！")
        return True
    
    def _freeze_weights(self):
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * self.config.weight_split.static_ratio)
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"冻结权重: {freeze_count}/{len(all_params)} 层")
        logger.info(f"可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def _build_enhanced_prompt(self, user_input: str, joint_result: Dict) -> str:
        """构建增强提示"""
        
        # 基础系统提示
        system = "你是AI助手，请准确计算并简洁回答。"
        
        # 添加推理规则
        if joint_result.get('relevant_rules'):
            rules_text = "\n".join([
                f"- {rule.example}" 
                for rule in joint_result['relevant_rules'][:3]
            ])
            system += f"\n\n已知规则:\n{rules_text}"
        
        # 添加数值推理结果
        if joint_result.get('reasoning'):
            reasoning_text = "\n".join([
                f"- {v['formula']}" 
                for v in joint_result['reasoning'].values() 
                if isinstance(v, dict) and 'formula' in v
            ])
            if reasoning_text:
                system += f"\n\n计算结果:\n{reasoning_text}"
        
        # 添加学习到的规则
        if self.joint_enhancer.learned_rules:
            learned_text = "\n".join([
                f"- {k}: {v}" 
                for k, v in list(self.joint_enhancer.learned_rules.items())[:3]
            ])
            system += f"\n\n用户纠正过的规则:\n{learned_text}"
        
        return system
    
    def generate(self, prompt: str, max_new_tokens: int = 600) -> str:
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 600
    ) -> Generator[str, None, None]:
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        # 联合处理
        joint_result = self.joint_enhancer.process_with_feedback(
            prompt, self._prev_answer
        )
        
        # 构建增强提示
        system_prompt = self._build_enhanced_prompt(prompt, joint_result)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for h in self.history[-5:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a']})
        
        messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        generated = []
        past = None
        response = ""
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                out = self.model(
                    input_ids=input_ids if past is None else input_ids[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                logits = out.logits[:, -1, :].clone()
                hidden = out.hidden_states[-1][:, -1, :]
                past = out.past_key_values
                
                # 数学增强
                logits = self.math_enhancer.enhance(logits, strength=0.3)
                
                # 温度采样
                logits = logits / 0.6
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                
                tid = next_tok.item()
                
                if tid == self.tokenizer.eos_token_id:
                    break
                
                tok_text = self.tokenizer.decode([tid], skip_special_tokens=False)
                if "<|im_" in tok_text:
                    break
                
                generated.append(tid)
                text = self.tokenizer.decode([tid], skip_special_tokens=True)
                response += text
                yield text
                
                input_ids = torch.cat([input_ids, next_tok], dim=-1)
        
        # 记录回答
        self.joint_enhancer.record_answer(prompt, response, joint_result.get('reasoning'))
        
        # 保存历史
        self.history.append({'q': prompt, 'a': response})
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        self._prev_answer = response
    
    def get_statistics(self) -> Dict:
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'session': self.session,
            'history_count': len(self.history),
            'rules_count': len(self.memory_enhancer.rules) if self.memory_enhancer else 0,
            'corrections_count': len(self.memory_enhancer.corrections) if self.memory_enhancer else 0,
        }
    
    def clear_memory(self):
        self.session = {}
        self.history = []
        self._prev_answer = None
        if self.memory_enhancer:
            self.memory_enhancer.corrections = []
            self.memory_enhancer.session_knowledge = {}
        if self.joint_enhancer:
            self.joint_enhancer.answer_history = []
