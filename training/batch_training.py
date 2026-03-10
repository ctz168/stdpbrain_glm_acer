#!/usr/bin/env python3
"""
分批次优化训练 - 每批次保存权重
Batch Training with Checkpoint Saving
"""

import os
import sys
import json
import gc
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn as nn

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# 分批次训练数据
# ============================================

# 批次1：基础租金计算
BATCH_1_DATA = [
    {"q": "房租1600元租了20天，日租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天。日租金是80元/天。"},
    {"q": "房租1600元租了20天，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
    {"q": "房租900元租了15天，日租金和月租金各是多少？", "a": "日租金 = 900 ÷ 15 = 60元/天，月租金 = 60 × 30 = 1800元/月。日租金是60元/天，月租金是1800元/月。"},
    {"q": "房租500元租了10天，日租金是多少？月租金是多少？", "a": "日租金 = 500 ÷ 10 = 50元/天，月租金 = 50 × 30 = 1500元/月。日租金是50元/天，月租金是1500元/月。"},
    {"q": "房租2000元租了25天，日租金是多少？", "a": "日租金 = 2000 ÷ 25 = 80元/天。日租金是80元/天。"},
    {"q": "日租金80元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 80 × 30 = 2400元。月租金是2400元/月。"},
    {"q": "日租金60元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 60 × 30 = 1800元。月租金是1800元/月。"},
]

# 批次2：复杂场景（包含干扰信息）
BATCH_2_DATA = [
    {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？", 
     "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
    {"q": "房租1600元20天，押金两千四百元，卫生费200元，合计2600元。月租金是多少？", 
     "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。合计2600元包含押金和卫生费，不影响月租金计算。月租金是2400元/月。"},
    {"q": "3月份20天房租1600元，押金2400元，卫生费200元，退房时卫生干净退200元。月租金是多少？", 
     "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
    {"q": "房租900元租了15天，押金1500元，月租金是多少？", 
     "a": "关键信息：房租900元，租期15天。日租金 = 900 ÷ 15 = 60元/天，月租金 = 60 × 30 = 1800元/月。月租金是1800元/月。"},
    {"q": "房租2000元25天，押金3000元，物业费100元，月租金是多少？", 
     "a": "关键信息：房租2000元，租期25天。日租金 = 2000 ÷ 25 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
]

# 批次3：押金相关问题
BATCH_3_DATA = [
    {"q": "押金什么时候可以退？", "a": "押金通常在退房时退还。如果房屋没有损坏，押金会在退房后7-30天内退还给租客。"},
    {"q": "押金2400元什么时候退？", "a": "押金2400元在退房时退还。如果房屋没有损坏、水电费已结清，押金会在退房后7-30天内退还。"},
    {"q": "押金两千四百元可以退吗？", "a": "押金2400元在退房时可以退还。前提是房屋没有损坏、水电费已结清、卫生干净。"},
    {"q": "退房时押金怎么退？", "a": "退房时，房东会检查房屋状况。如果没有损坏，押金会在7-30天内退还给租客。"},
    {"q": "押金能退吗？", "a": "押金在退房时可以退还，前提是房屋没有损坏、水电费已结清。"},
    {"q": "押金什么时候退给我？", "a": "押金在退房验收后退还，通常在退房后7-30天内到账。"},
]

# 批次4：卫生费相关问题
BATCH_4_DATA = [
    {"q": "卫生费可以退吗？", "a": "卫生费200元，如果退房时卫生干净，可以退还200元卫生费。"},
    {"q": "卫生费200元可以退吗？", "a": "卫生费200元，如果退房时房屋卫生干净，可以全额退还200元。"},
    {"q": "离租卫生干净退200元卫生费是什么意思？", "a": "意思是：如果退房时房屋打扫干净，200元卫生费可以退还。"},
    {"q": "卫生费什么时候退？", "a": "卫生费在退房验收时退还。如果房屋卫生干净，200元卫生费会退还给租客。"},
    {"q": "卫生费不退的情况？", "a": "如果退房时房屋卫生不干净，卫生费200元可能不会退还，用于支付清洁费用。"},
]

# 批次5：综合问答
BATCH_5_DATA = [
    {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。离租卫生干净退200元卫生费。合计2600元。月租金是多少？", 
     "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。押金2400元在退房时退还，卫生费200元在退房卫生干净时退还。月租金是2400元/月。"},
    {"q": "我需要付多少钱？", "a": "根据信息：房租1600元（20天），押金2400元，卫生费200元，合计需要支付4200元。其中押金和卫生费在退房时可以退还。"},
    {"q": "退房时能退多少钱？", "a": "退房时可以退还：押金2400元 + 卫生费200元（如果卫生干净）= 2600元。"},
    {"q": "房租包含水电费吗？", "a": "根据提供的信息，房租1600元（20天）不包含水电费，水电费通常需要另外支付。"},
    {"q": "押金和卫生费有什么区别？", "a": "押金2400元是保证金，退房时退还。卫生费200元是清洁费，如果退房时卫生干净可以退还。"},
]

# 所有批次
ALL_BATCHES = [
    ("batch_1_basic", BATCH_1_DATA),
    ("batch_2_complex", BATCH_2_DATA),
    ("batch_3_deposit", BATCH_3_DATA),
    ("batch_4_hygiene", BATCH_4_DATA),
    ("batch_5_comprehensive", BATCH_5_DATA),
]


class BatchTrainer:
    """分批次训练器"""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        self.current_weights = None
        self.best_loss = float('inf')
        self.total_steps = 0
    
    def setup(self):
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
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
        
        # 尝试加载已有权重
        self._load_existing_weights()
    
    def _load_existing_weights(self):
        """加载已有权重"""
        weights_path = os.path.join(self.output_path, 'dynamic_weights.pt')
        if os.path.exists(weights_path):
            logger.info("加载已有动态权重...")
            weights = torch.load(weights_path, map_location='cpu')
            
            applied = 0
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.data = weights[name]
                    applied += 1
            
            logger.info(f"加载了 {applied} 个权重")
    
    def train_batch(self, batch_name: str, data: List[Dict], epochs: int = 3):
        """训练单个批次"""
        logger.info(f"\n{'='*60}")
        logger.info(f"训练批次: {batch_name}")
        logger.info(f"数据量: {len(data)} 条")
        logger.info(f"{'='*60}")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for item in data:
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
            
            avg_loss = total_loss / len(data)
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
            
            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        # 每批次训练后保存权重
        self._save_checkpoint(batch_name)
        
        return avg_loss
    
    def _save_checkpoint(self, batch_name: str):
        """保存检查点"""
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存动态权重
        dynamic_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                dynamic_weights[name] = param.data.cpu()
        
        weights_path = os.path.join(self.output_path, 'dynamic_weights.pt')
        torch.save(dynamic_weights, weights_path)
        
        # 保存检查点（带批次名）
        checkpoint_path = os.path.join(self.output_path, f'checkpoint_{batch_name}.pt')
        torch.save({
            'weights': dynamic_weights,
            'best_loss': self.best_loss,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"  ✓ 检查点已保存: {checkpoint_path}")
        
        # 保存训练报告
        self._save_report()
    
    def _save_report(self):
        """保存训练报告"""
        report = {
            'model_path': self.model_path,
            'output_path': self.output_path,
            'best_loss': self.best_loss,
            'total_steps': self.total_steps,
            'last_update': datetime.now().isoformat()
        }
        
        report_path = os.path.join(self.output_path, 'training_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def test(self, questions: List[str]):
        """测试"""
        logger.info("\n测试模型...")
        self.model.eval()
        
        for q in questions:
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
            
            logger.info(f"\nQ: {q}")
            logger.info(f"A: {response}")
    
    def run_all_batches(self):
        """运行所有批次"""
        self.setup()
        
        for batch_name, data in ALL_BATCHES:
            self.train_batch(batch_name, data, epochs=3)
        
        # 最终测试
        test_questions = [
            "房租1600元租了20天，月租金是多少？",
            "押金什么时候可以退？",
            "卫生费可以退吗？",
            "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？",
        ]
        self.test(test_questions)
        
        logger.info("\n所有批次训练完成！")


def main():
    model_path = str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    trainer = BatchTrainer(model_path, output_path)
    trainer.run_all_batches()


if __name__ == "__main__":
    main()
