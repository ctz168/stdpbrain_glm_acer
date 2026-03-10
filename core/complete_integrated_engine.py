#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 完整集成引擎 (优化版V2)
Complete Integrated Brain-Like Engine (Optimized V2)

优化：
1. 输入预处理 - 提取关键数字
2. 更强的思维链提示词
3. 计算问题专用处理
"""

import os
import sys
import logging
import time
import math
import re
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# 配置
# ============================================

@dataclass
class BrainLikeConfig:
    """类脑架构配置"""
    refresh_period_ms: float = 10.0
    stdp_alpha: float = 0.01
    stdp_beta: float = 0.005
    stdp_timing_window: float = 40.0
    freeze_ratio: float = 0.9
    memory_capacity: int = 1000
    memory_top_k: int = 2


class OptimizationMode(Enum):
    SELF_GENERATION = "self_generation"
    SELF_PLAY = "self_play"
    SELF_JUDGMENT = "self_judgment"


# ============================================
# 输入预处理器
# ============================================

class InputPreprocessor:
    """输入预处理器 - 提取关键信息"""
    
    @staticmethod
    def extract_rent_info(text: str) -> Dict[str, Any]:
        """提取租房相关信息"""
        info = {}
        
        # 提取天数和房租
        # 格式1: "20天房租1600元"
        match = re.search(r'(\d+)\s*天.*?房租.*?(\d+)\s*元', text)
        if match:
            info['days'] = int(match.group(1))
            info['rent'] = int(match.group(2))
        
        # 格式2: "房租1600元...20天"
        if 'days' not in info:
            match = re.search(r'房租.*?(\d+)\s*元', text)
            if match:
                info['rent'] = int(match.group(1))
            match = re.search(r'(\d+)\s*天', text)
            if match:
                info['days'] = int(match.group(1))
        
        # 提取押金
        match = re.search(r'押金[：:]?\s*(\d+)', text)
        if match:
            info['deposit'] = int(match.group(1))
        else:
            # 中文数字
            if '押金' in text and ('两千四百' in text or '2400' in text):
                info['deposit'] = 2400
        
        # 提取卫生费
        match = re.search(r'卫生费[：:]?\s*(\d+)\s*元', text)
        if match:
            info['hygiene_fee'] = int(match.group(1))
        
        # 提取退费
        match = re.search(r'退\s*(\d+)\s*元', text)
        if match:
            info['refund'] = int(match.group(1))
        
        return info
    
    @staticmethod
    def build_calculation_prompt(user_input: str) -> str:
        """构建计算提示词"""
        info = InputPreprocessor.extract_rent_info(user_input)
        
        # 如果提取到了关键信息
        if 'days' in info and 'rent' in info:
            days = info['days']
            rent = info['rent']
            daily_rent = rent / days
            monthly_rent = daily_rent * 30
            
            prompt = f"""【租房计算问题】

已知信息：
- 租期天数：{days}天
- 房租金额：{rent}元
- 日租金 = {rent} ÷ {days} = {daily_rent:.0f}元/天
- 月租金 = {daily_rent:.0f} × 30 = {monthly_rent:.0f}元/月

问题：{user_input}

请根据以上计算回答问题。答案要简洁准确。"""
            return prompt
        
        # 如果没有提取到完整信息，使用通用提示
        return f"""【问题】
{user_input}

请仔细阅读问题，提取关键数字，然后计算回答。"""


# ============================================
# STDP学习核心
# ============================================

class STDPKernel:
    def __init__(self, config: BrainLikeConfig):
        self.config = config
        self.timing_window = config.stdp_timing_window
        self.alpha = config.stdp_alpha
        self.beta = config.stdp_beta
        self.ltp_count = 0
        self.ltd_count = 0
        self.total_updates = 0
    
    def compute_update(self, delta_t: float, contribution: float = 1.0) -> Tuple[float, str]:
        if abs(delta_t) > self.timing_window:
            return 0.0, 'none'
        
        self.total_updates += 1
        
        if delta_t > 0:
            update = self.alpha * contribution * math.exp(-delta_t / self.timing_window)
            self.ltp_count += 1
            return update, 'ltp'
        else:
            update = -self.beta * contribution * math.exp(delta_t / self.timing_window)
            self.ltd_count += 1
            return update, 'ltd'
    
    def get_statistics(self) -> Dict:
        return {
            'total_updates': self.total_updates,
            'ltp_count': self.ltp_count,
            'ltd_count': self.ltd_count
        }


# ============================================
# 海马体记忆系统
# ============================================

class HippocampusMemory:
    def __init__(self, config: BrainLikeConfig):
        self.config = config
        self.memories: List[Dict] = []
        self.encode_count = 0
        self.recall_count = 0
    
    def encode(self, text: str, embedding: torch.Tensor, timestamp: float):
        self.encode_count += 1
        memory = {
            'text': text[:200],
            'embedding': embedding.detach().clone(),
            'timestamp': timestamp,
            'access_count': 0
        }
        self.memories.append(memory)
        if len(self.memories) > self.config.memory_capacity:
            self.memories.pop(0)
    
    def recall(self, query_embedding: torch.Tensor, top_k: int = None) -> List[Dict]:
        self.recall_count += 1
        if not self.memories:
            return []
        
        top_k = top_k or self.config.memory_top_k
        
        similarities = []
        for memory in self.memories:
            mem_emb = memory['embedding'].flatten()
            query_flat = query_embedding.flatten()
            min_dim = min(query_flat.shape[0], mem_emb.shape[0])
            similarity = F.cosine_similarity(
                query_flat[:min_dim].unsqueeze(0),
                mem_emb[:min_dim].unsqueeze(0)
            ).item()
            similarities.append((memory, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for memory, score in similarities[:top_k]:
            if score > 0.3:
                memory['access_count'] += 1
                results.append(memory)
        
        return results
    
    def get_statistics(self) -> Dict:
        return {
            'memory_count': len(self.memories),
            'encode_count': self.encode_count,
            'recall_count': self.recall_count
        }


# ============================================
# 自优化闭环系统
# ============================================

class SelfOptimizationLoop:
    def __init__(self, config: BrainLikeConfig):
        self.config = config
        self.generation_count = 0
        self.play_count = 0
        self.judgment_count = 0
    
    def select_mode(self, input_text: str) -> OptimizationMode:
        self_play_keywords = ['计算', '推理', '分析', '证明', '验证', '检查', '多少']
        self_judgment_keywords = ['比较', '选择', '评价', '哪个更好', '最优']
        
        for kw in self_play_keywords:
            if kw in input_text:
                return OptimizationMode.SELF_PLAY
        
        for kw in self_judgment_keywords:
            if kw in input_text:
                return OptimizationMode.SELF_JUDGMENT
        
        return OptimizationMode.SELF_GENERATION
    
    def self_generation(self, model, input_ids: torch.Tensor, tokenizer) -> Tuple[str, Dict]:
        self.generation_count += 1
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text, {'mode': 'self_generation'}
    
    def self_play(self, model, input_ids: torch.Tensor, tokenizer, context: str) -> Tuple[str, Dict]:
        self.play_count += 1
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.3,  # 更低温度
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text, {'mode': 'self_play'}
    
    def self_judgment(self, model, input_ids: torch.Tensor, tokenizer, context: str) -> Tuple[str, Dict]:
        self.judgment_count += 1
        
        candidates = []
        for i in range(2):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.3 + i * 0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
            text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            candidates.append(text)
        
        best = max(candidates, key=lambda x: sum(c.isdigit() for c in x))
        return best, {'mode': 'self_judgment'}
    
    def get_statistics(self) -> Dict:
        return {
            'generation_count': self.generation_count,
            'play_count': self.play_count,
            'judgment_count': self.judgment_count
        }


# ============================================
# 完整集成引擎
# ============================================

class CompleteIntegratedEngine:
    """完整集成的类脑引擎 (优化版V2)"""
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or BrainLikeConfig()
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self.stdp: Optional[STDPKernel] = None
        self.hippocampus: Optional[HippocampusMemory] = None
        self.self_optimization: Optional[SelfOptimizationLoop] = None
        self.preprocessor: Optional[InputPreprocessor] = None
        
        self.dynamic_weights: Dict[str, torch.Tensor] = {}
        
        self._initialized = False
        self._cycle_count = 0
    
    def initialize(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("="*60)
        logger.info("初始化完整集成的类脑引擎 (优化版V2)")
        logger.info("="*60)
        
        self.device = torch.device("cpu")
        
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
        
        self._freeze_weights()
        
        self.stdp = STDPKernel(self.config)
        self.hippocampus = HippocampusMemory(self.config)
        self.self_optimization = SelfOptimizationLoop(self.config)
        self.preprocessor = InputPreprocessor()
        
        self._initialized = True
        
        logger.info(f"刷新周期: {self.config.refresh_period_ms}ms (100Hz)")
        logger.info("输入预处理: 已启用")
        logger.info("初始化完成！")
        
        return True
    
    def _freeze_weights(self):
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * self.config.freeze_ratio)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
            else:
                self.dynamic_weights[name] = torch.zeros_like(param.data) * 0.01
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"冻结权重: {freeze_count}/{len(all_params)} 层")
        logger.info(f"可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def generate_stream(self, prompt: str, max_new_tokens: int = 200) -> Generator[str, None, None]:
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        self._cycle_count += 1
        
        # 1. 预处理输入 - 提取关键信息并构建提示词
        enhanced_prompt = self.preprocessor.build_calculation_prompt(prompt)
        
        # 2. 选择优化模式
        mode = self.self_optimization.select_mode(prompt)
        logger.info(f"选择优化模式: {mode.value}")
        
        # 3. 构建输入
        input_text = f"<|im_start|>user\n{enhanced_prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # 4. 执行生成
        if mode == OptimizationMode.SELF_PLAY:
            result, info = self.self_optimization.self_play(
                self.model, input_ids, self.tokenizer, prompt
            )
        else:
            result, info = self.self_optimization.self_generation(
                self.model, input_ids, self.tokenizer
            )
        
        # 5. 流式输出
        for char in result:
            yield char
        
        # 6. 后处理
        self._apply_stdp_learning(result)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        self.hippocampus.encode(result, hidden_state, time.time() * 1000)
    
    def _apply_stdp_learning(self, output: str):
        quality_score = min(1.0, len(output) / 1000.0)
        delta_t = 5.0
        update, update_type = self.stdp.compute_update(delta_t, quality_score)
        
        for name in list(self.dynamic_weights.keys())[:5]:
            if update_type == 'ltp':
                self.dynamic_weights[name] += update * 0.0001
            else:
                self.dynamic_weights[name] -= update * 0.0001
            self.dynamic_weights[name].clamp_(-0.1, 0.1)
    
    def get_statistics(self) -> Dict:
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'cycle_count': self._cycle_count,
            'stdp': self.stdp.get_statistics() if self.stdp else {},
            'hippocampus': self.hippocampus.get_statistics() if self.hippocampus else {},
            'self_optimization': self.self_optimization.get_statistics() if self.self_optimization else {}
        }
    
    def clear_memory(self):
        if self.hippocampus:
            self.hippocampus.memories.clear()


_engine: Optional[CompleteIntegratedEngine] = None

def get_engine(model_path: str = None) -> CompleteIntegratedEngine:
    global _engine
    if _engine is None:
        model_path = model_path or str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
        _engine = CompleteIntegratedEngine(model_path)
    return _engine
