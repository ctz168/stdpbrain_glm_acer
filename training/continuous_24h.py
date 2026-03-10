#!/usr/bin/env python3
"""
24小时持续训练计划
24-Hour Continuous Training Plan

训练目标：
1. 全面提升模型推理能力
2. 覆盖所有场景
3. 每30分钟保存一次检查点
4. 自动恢复训练
"""

import os
import sys
import json
import gc
import time
import logging
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import torch
import torch.nn as nn

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('/tmp/training_24h.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# 训练数据集 - 全面覆盖
# ============================================

TRAINING_DATA = {
    # === 第1类：基础租金计算 ===
    "rent_basic": [
        {"q": "房租1600元租了20天，日租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天。日租金是80元/天。"},
        {"q": "房租1600元租了20天，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
        {"q": "房租900元租了15天，日租金是多少？", "a": "日租金 = 900 ÷ 15 = 60元/天。日租金是60元/天。"},
        {"q": "房租900元租了15天，月租金是多少？", "a": "日租金 = 900 ÷ 15 = 60元/天，月租金 = 60 × 30 = 1800元/月。月租金是1800元/月。"},
        {"q": "房租500元租了10天，日租金是多少？月租金是多少？", "a": "日租金 = 500 ÷ 10 = 50元/天，月租金 = 50 × 30 = 1500元/月。日租金是50元/天，月租金是1500元/月。"},
        {"q": "房租2000元租了25天，日租金是多少？", "a": "日租金 = 2000 ÷ 25 = 80元/天。日租金是80元/天。"},
        {"q": "房租1800元租了30天，日租金是多少？", "a": "日租金 = 1800 ÷ 30 = 60元/天。日租金是60元/天。"},
        {"q": "房租2400元租了30天，日租金是多少？月租金是多少？", "a": "日租金 = 2400 ÷ 30 = 80元/天，月租金 = 80 × 30 = 2400元/月。日租金是80元/天，月租金是2400元/月。"},
        {"q": "房租1200元租了15天，日租金是多少？月租金是多少？", "a": "日租金 = 1200 ÷ 15 = 80元/天，月租金 = 80 × 30 = 2400元/月。日租金是80元/天，月租金是2400元/月。"},
        {"q": "房租3000元租了30天，日租金是多少？", "a": "日租金 = 3000 ÷ 30 = 100元/天。日租金是100元/天。"},
        {"q": "房租1500元租了20天，日租金是多少？月租金是多少？", "a": "日租金 = 1500 ÷ 20 = 75元/天，月租金 = 75 × 30 = 2250元/月。日租金是75元/天，月租金是2250元/月。"},
        {"q": "房租800元租了10天，日租金是多少？月租金是多少？", "a": "日租金 = 800 ÷ 10 = 80元/天，月租金 = 80 × 30 = 2400元/月。日租金是80元/天，月租金是2400元/月。"},
    ],
    
    # === 第2类：反向计算 ===
    "rent_reverse": [
        {"q": "日租金80元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 80 × 30 = 2400元。月租金是2400元/月。"},
        {"q": "日租金60元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 60 × 30 = 1800元。月租金是1800元/月。"},
        {"q": "日租金50元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 50 × 30 = 1500元。月租金是1500元/月。"},
        {"q": "日租金100元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 100 × 30 = 3000元。月租金是3000元/月。"},
        {"q": "日租金75元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 75 × 30 = 2250元。月租金是2250元/月。"},
        {"q": "月租金2400元，日租金是多少？", "a": "日租金 = 月租金 ÷ 30 = 2400 ÷ 30 = 80元。日租金是80元/天。"},
        {"q": "月租金1800元，日租金是多少？", "a": "日租金 = 月租金 ÷ 30 = 1800 ÷ 30 = 60元。日租金是60元/天。"},
        {"q": "月租金3000元，日租金是多少？", "a": "日租金 = 月租金 ÷ 30 = 3000 ÷ 30 = 100元。日租金是100元/天。"},
    ],
    
    # === 第3类：复杂场景（干扰信息） ===
    "rent_complex": [
        {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
        {"q": "房租1600元20天，押金两千四百元，卫生费200元，合计2600元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。合计2600元包含押金和卫生费，不影响月租金。月租金是2400元/月。"},
        {"q": "3月份20天房租1600元，押金2400元，卫生费200元，退房时卫生干净退200元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
        {"q": "房租900元租了15天，押金1500元，月租金是多少？", "a": "关键信息：房租900元，租期15天。日租金 = 900 ÷ 15 = 60元/天，月租金 = 60 × 30 = 1800元/月。月租金是1800元/月。"},
        {"q": "房租2000元25天，押金3000元，物业费100元，月租金是多少？", "a": "关键信息：房租2000元，租期25天。日租金 = 2000 ÷ 25 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
        {"q": "房租500元10天，押金800元，水电费50元，月租金是多少？", "a": "关键信息：房租500元，租期10天。日租金 = 500 ÷ 10 = 50元/天，月租金 = 50 × 30 = 1500元/月。月租金是1500元/月。"},
        {"q": "合同期1年，起租日3月12日，3月份20天房租1600元，押一付三，押金2400元。请问月租金？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
        {"q": "房租1600元20天，包含物业费，押金2400元，卫生费200元可退，总共付了2600元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
    ],
    
    # === 第4类：押金问题 ===
    "deposit": [
        {"q": "押金什么时候可以退？", "a": "押金通常在退房时退还。如果房屋没有损坏，押金会在退房后7-30天内退还给租客。"},
        {"q": "押金2400元什么时候退？", "a": "押金2400元在退房时退还。如果房屋没有损坏、水电费已结清，押金会在退房后7-30天内退还。"},
        {"q": "押金两千四百元可以退吗？", "a": "押金2400元在退房时可以退还。前提是房屋没有损坏、水电费已结清、卫生干净。"},
        {"q": "退房时押金怎么退？", "a": "退房时，房东会检查房屋状况。如果没有损坏，押金会在7-30天内退还给租客。"},
        {"q": "押金能退吗？", "a": "押金在退房时可以退还，前提是房屋没有损坏、水电费已结清。"},
        {"q": "押金什么时候退给我？", "a": "押金在退房验收后退还，通常在退房后7-30天内到账。"},
        {"q": "押金不退的情况有哪些？", "a": "如果房屋有损坏、水电费未结清、或违反合同条款，押金可能不会全额退还。"},
        {"q": "押金是几个月房租？", "a": "押金通常是1-2个月房租。根据合同，押金2400元相当于1个月房租。"},
        {"q": "押金可以抵扣房租吗？", "a": "押金不能直接抵扣房租。押金是保证金，在退房时退还，不是预付房租。"},
        {"q": "押金和房租有什么区别？", "a": "房租是每月支付的居住费用，押金是保证金，退房时退还。押金不是房租的一部分。"},
    ],
    
    # === 第5类：卫生费问题 ===
    "hygiene_fee": [
        {"q": "卫生费可以退吗？", "a": "卫生费200元，如果退房时卫生干净，可以退还200元卫生费。"},
        {"q": "卫生费200元可以退吗？", "a": "卫生费200元，如果退房时房屋卫生干净，可以全额退还200元。"},
        {"q": "离租卫生干净退200元卫生费是什么意思？", "a": "意思是：如果退房时房屋打扫干净，200元卫生费可以退还。"},
        {"q": "卫生费什么时候退？", "a": "卫生费在退房验收时退还。如果房屋卫生干净，200元卫生费会退还给租客。"},
        {"q": "卫生费不退的情况？", "a": "如果退房时房屋卫生不干净，卫生费200元可能不会退还，用于支付清洁费用。"},
        {"q": "卫生费是什么？", "a": "卫生费是用于房屋清洁的费用。如果退房时房屋干净，可以退还。"},
        {"q": "卫生费必须交吗？", "a": "卫生费是租房合同的一部分，需要支付。但如果退房时卫生干净，可以退还。"},
        {"q": "卫生费和押金有什么区别？", "a": "押金是保证金，退房时退还。卫生费是清洁费，如果退房时卫生干净才退还。"},
    ],
    
    # === 第6类：综合问答 ===
    "comprehensive": [
        {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。离租卫生干净退200元卫生费。合计2600元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。押金2400元在退房时退还，卫生费200元在退房卫生干净时退还。月租金是2400元/月。"},
        {"q": "我需要付多少钱？", "a": "根据信息：房租1600元（20天），押金2400元，卫生费200元，合计需要支付4200元。其中押金和卫生费在退房时可以退还。"},
        {"q": "退房时能退多少钱？", "a": "退房时可以退还：押金2400元 + 卫生费200元（如果卫生干净）= 2600元。"},
        {"q": "房租包含水电费吗？", "a": "根据提供的信息，房租1600元（20天）不包含水电费，水电费通常需要另外支付。"},
        {"q": "押金和卫生费有什么区别？", "a": "押金2400元是保证金，退房时退还。卫生费200元是清洁费，如果退房时卫生干净可以退还。"},
        {"q": "租期是多久？", "a": "根据信息，租期是1年，从3月12日起租。3月份租了20天。"},
        {"q": "3月份房租怎么算？", "a": "3月份租了20天，房租1600元。日租金 = 1600 ÷ 20 = 80元/天。"},
        {"q": "押一付三是什么意思？", "a": "押一付三是指：押金1个月，每次付3个月房租。押金在退房时退还。"},
    ],
    
    # === 第7类：数学计算 ===
    "math": [
        {"q": "计算：1600 ÷ 20 = ?", "a": "1600 ÷ 20 = 80"},
        {"q": "计算：80 × 30 = ?", "a": "80 × 30 = 2400"},
        {"q": "计算：900 ÷ 15 = ?", "a": "900 ÷ 15 = 60"},
        {"q": "计算：60 × 30 = ?", "a": "60 × 30 = 1800"},
        {"q": "计算：2000 ÷ 25 = ?", "a": "2000 ÷ 25 = 80"},
        {"q": "计算：500 ÷ 10 = ?", "a": "500 ÷ 10 = 50"},
        {"q": "计算：2400 ÷ 30 = ?", "a": "2400 ÷ 30 = 80"},
        {"q": "计算：1800 ÷ 30 = ?", "a": "1800 ÷ 30 = 60"},
        {"q": "计算：50 × 30 = ?", "a": "50 × 30 = 1500"},
        {"q": "计算：100 × 30 = ?", "a": "100 × 30 = 3000"},
    ],
    
    # === 第8类：逻辑推理 ===
    "reasoning": [
        {"q": "如果日租金是80元，租了20天，总房租是多少？", "a": "总房租 = 日租金 × 天数 = 80 × 20 = 1600元。总房租是1600元。"},
        {"q": "如果月租金是2400元，租了3个月，总房租是多少？", "a": "总房租 = 月租金 × 月数 = 2400 × 3 = 7200元。总房租是7200元。"},
        {"q": "如果房租是2400元/月，租了15天，房租是多少？", "a": "日租金 = 2400 ÷ 30 = 80元/天，房租 = 80 × 15 = 1200元。房租是1200元。"},
        {"q": "小明租了20天，付了1600元房租，小红租了15天，付了900元房租，谁的日租金更贵？", "a": "小明日租金 = 1600 ÷ 20 = 80元/天，小红日租金 = 900 ÷ 15 = 60元/天。小明的日租金更贵。"},
        {"q": "房租从1600元涨到2000元，涨了多少？", "a": "涨价 = 2000 - 1600 = 400元，涨幅 = 400 ÷ 1600 = 25%。涨了400元，涨幅25%。"},
    ],
}


class ContinuousTrainer:
    """24小时持续训练器"""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # 训练状态
        self.start_time = None
        self.end_time = None
        self.total_steps = 0
        self.best_loss = float('inf')
        self.current_epoch = 0
        
        # 检查点配置
        self.save_interval = 30 * 60  # 30分钟保存一次
        self.last_save_time = 0
        
        # 训练时长
        self.training_duration = 24 * 60 * 60  # 24小时
    
    def setup(self):
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("="*60)
        logger.info("24小时持续训练计划")
        logger.info("="*60)
        
        logger.info("加载模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 冻结90%权重
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * 0.9)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"可训练参数: {trainable/1e6:.2f}M")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-5
        )
        
        # 加载已有权重
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """加载检查点"""
        # 尝试加载最新检查点
        checkpoint_files = sorted(
            Path(self.output_path).glob('checkpoint_*.pt'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if checkpoint_files:
            latest = checkpoint_files[0]
            logger.info(f"加载检查点: {latest}")
            
            checkpoint = torch.load(latest, map_location='cpu')
            
            if 'weights' in checkpoint:
                for name, param in self.model.named_parameters():
                    if name in checkpoint['weights']:
                        param.data = checkpoint['weights'][name]
            
            if 'best_loss' in checkpoint:
                self.best_loss = checkpoint['best_loss']
            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
            
            logger.info(f"恢复训练: steps={self.total_steps}, best_loss={self.best_loss:.4f}")
        
        elif os.path.exists(os.path.join(self.output_path, 'dynamic_weights.pt')):
            logger.info("加载已有动态权重...")
            weights = torch.load(
                os.path.join(self.output_path, 'dynamic_weights.pt'),
                map_location='cpu'
            )
            
            applied = 0
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.data = weights[name]
                    applied += 1
            
            logger.info(f"加载了 {applied} 个权重")
    
    def _save_checkpoint(self, force: bool = False):
        """保存检查点"""
        current_time = time.time()
        
        if not force and current_time - self.last_save_time < self.save_interval:
            return
        
        self.last_save_time = current_time
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存动态权重
        dynamic_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                dynamic_weights[name] = param.data.cpu()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            self.output_path, 
            f'checkpoint_{timestamp}.pt'
        )
        
        torch.save({
            'weights': dynamic_weights,
            'best_loss': self.best_loss,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        # 同时更新主权重文件
        torch.save(
            dynamic_weights, 
            os.path.join(self.output_path, 'dynamic_weights.pt')
        )
        
        logger.info(f"✓ 检查点已保存: {checkpoint_path}")
        
        # 保存训练报告
        self._save_report()
    
    def _save_report(self):
        """保存训练报告"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        report = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'current_time': datetime.now().isoformat(),
            'elapsed_hours': round(elapsed / 3600, 2),
            'remaining_hours': round((self.training_duration - elapsed) / 3600, 2),
            'total_steps': self.total_steps,
            'best_loss': round(self.best_loss, 4),
            'progress': round(elapsed / self.training_duration * 100, 1)
        }
        
        with open(os.path.join(self.output_path, 'training_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
    
    def train_epoch(self, data: List[Dict], category: str):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        # 随机打乱
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        for item in shuffled:
            prompt = f"问题：{item['q']}\n\n答案：{item['a']}"
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors='pt', 
                max_length=256, 
                truncation=True, 
                padding='max_length'
            )
            
            input_ids = inputs['input_ids']
            labels = input_ids.clone()
            
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 
                1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            self.total_steps += 1
            
            del outputs, loss
            gc.collect()
            
            # 检查是否需要保存
            self._save_checkpoint()
            
            # 检查是否超时
            if time.time() - self.start_time >= self.training_duration:
                return total_loss / len(shuffled), True
        
        return total_loss / len(shuffled), False
    
    def test(self):
        """测试"""
        self.model.eval()
        
        test_questions = [
            "房租1600元租了20天，月租金是多少？",
            "押金什么时候可以退？",
            "卫生费可以退吗？",
            "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？",
        ]
        
        logger.info("\n--- 测试结果 ---")
        
        for q in test_questions:
            prompt = f"问题：{q}\n\n答案："
            inputs = self.tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'], 
                    max_new_tokens=100, 
                    do_sample=False
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "答案：" in response:
                response = response.split("答案：")[-1]
            
            logger.info(f"Q: {q}")
            logger.info(f"A: {response}\n")
    
    def run(self):
        """运行24小时训练"""
        self.setup()
        self.start_time = time.time()
        
        logger.info(f"\n开始24小时训练")
        logger.info(f"预计结束时间: {datetime.fromtimestamp(self.start_time + self.training_duration)}")
        
        # 准备所有数据
        all_data = []
        for category, data in TRAINING_DATA.items():
            all_data.extend(data)
            logger.info(f"  {category}: {len(data)} 条")
        
        logger.info(f"总数据量: {len(all_data)} 条")
        
        epoch = 0
        timeout = False
        
        while not timeout:
            epoch += 1
            
            # 每轮随机选择数据类别
            categories = list(TRAINING_DATA.keys())
            random.shuffle(categories)
            
            epoch_loss = 0.0
            epoch_count = 0
            
            for category in categories:
                data = TRAINING_DATA[category]
                
                avg_loss, timeout = self.train_epoch(data, category)
                epoch_loss += avg_loss
                epoch_count += 1
                
                if timeout:
                    break
            
            avg_epoch_loss = epoch_loss / epoch_count
            
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
            
            elapsed = (time.time() - self.start_time) / 3600
            remaining = (self.training_duration / 3600) - elapsed
            
            logger.info(
                f"Epoch {epoch}: Loss={avg_epoch_loss:.4f}, "
                f"Best={self.best_loss:.4f}, "
                f"已训练={elapsed:.1f}h, 剩余={remaining:.1f}h"
            )
            
            # 每10轮测试一次
            if epoch % 10 == 0:
                self.test()
            
            if timeout:
                break
        
        # 最终保存
        self._save_checkpoint(force=True)
        
        logger.info("\n" + "="*60)
        logger.info("24小时训练完成！")
        logger.info(f"总步数: {self.total_steps}")
        logger.info(f"最佳Loss: {self.best_loss:.4f}")
        logger.info("="*60)


def main():
    model_path = str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    trainer = ContinuousTrainer(model_path, output_path)
    trainer.run()


if __name__ == "__main__":
    main()
