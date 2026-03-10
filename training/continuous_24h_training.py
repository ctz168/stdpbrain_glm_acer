#!/usr/bin/env python3
"""
24小时持续训练计划
24-Hour Continuous Training Plan

特点：
1. 分批次训练，每批次保存权重
2. 每批次完成后发送Telegram消息报告进展
3. 持续运行24小时
4. 自动恢复训练
"""

import os
import sys
import json
import gc
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import torch
import torch.nn as nn

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Telegram Bot配置
TELEGRAM_BOT_TOKEN = "8534413276:AAHzqgxVTOL2fapd8NV7UjppF4NXr1zSUek"
TELEGRAM_CHAT_ID = "7234749146"  # FirstLandingMarsFederation


def send_telegram_message(message: str):
    """发送Telegram消息"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            logger.info("Telegram消息发送成功")
        else:
            logger.warning(f"Telegram消息发送失败: {response.text}")
    except Exception as e:
        logger.error(f"发送Telegram消息异常: {e}")


# ============================================
# 24小时训练数据计划
# ============================================

TRAINING_PLAN = {
    "hour_0-4": {
        "name": "阶段1：基础能力",
        "batches": [
            {
                "name": "batch_01_basic_calc",
                "data": [
                    {"q": "房租1600元租了20天，日租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天。日租金是80元/天。"},
                    {"q": "房租1600元租了20天，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "房租900元租了15天，日租金和月租金各是多少？", "a": "日租金 = 900 ÷ 15 = 60元/天，月租金 = 60 × 30 = 1800元/月。日租金是60元/天，月租金是1800元/月。"},
                    {"q": "房租500元租了10天，日租金是多少？月租金是多少？", "a": "日租金 = 500 ÷ 10 = 50元/天，月租金 = 50 × 30 = 1500元/月。日租金是50元/天，月租金是1500元/月。"},
                    {"q": "房租2000元租了25天，日租金是多少？", "a": "日租金 = 2000 ÷ 25 = 80元/天。日租金是80元/天。"},
                    {"q": "房租1800元租了30天，日租金是多少？", "a": "日租金 = 1800 ÷ 30 = 60元/天。日租金是60元/天。"},
                    {"q": "房租2400元租了30天，日租金是多少？月租金是多少？", "a": "日租金 = 2400 ÷ 30 = 80元/天，月租金 = 80 × 30 = 2400元/月。日租金是80元/天，月租金是2400元/月。"},
                ],
                "epochs": 5
            },
            {
                "name": "batch_02_daily_monthly",
                "data": [
                    {"q": "日租金80元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 80 × 30 = 2400元。月租金是2400元/月。"},
                    {"q": "日租金60元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 60 × 30 = 1800元。月租金是1800元/月。"},
                    {"q": "日租金50元，月租金是多少？", "a": "月租金 = 日租金 × 30 = 50 × 30 = 1500元。月租金是1500元/月。"},
                    {"q": "月租金2400元，日租金是多少？", "a": "日租金 = 月租金 ÷ 30 = 2400 ÷ 30 = 80元/天。日租金是80元/天。"},
                    {"q": "月租金1800元，日租金是多少？", "a": "日租金 = 月租金 ÷ 30 = 1800 ÷ 30 = 60元/天。日租金是60元/天。"},
                    {"q": "月租金1500元，日租金是多少？", "a": "日租金 = 月租金 ÷ 30 = 1500 ÷ 30 = 50元/天。日租金是50元/天。"},
                ],
                "epochs": 5
            },
        ]
    },
    "hour_4-8": {
        "name": "阶段2：复杂场景",
        "batches": [
            {
                "name": "batch_03_complex_1",
                "data": [
                    {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "房租1600元20天，押金两千四百元，卫生费200元，合计2600元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。合计2600元包含押金和卫生费，不影响月租金。月租金是2400元/月。"},
                    {"q": "3月份20天房租1600元，押金2400元，卫生费200元，退房时卫生干净退200元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "房租900元租了15天，押金1500元，月租金是多少？", "a": "关键信息：房租900元，租期15天。日租金 = 900 ÷ 15 = 60元/天，月租金 = 60 × 30 = 1800元/月。月租金是1800元/月。"},
                    {"q": "房租2000元25天，押金3000元，物业费100元，月租金是多少？", "a": "关键信息：房租2000元，租期25天。日租金 = 2000 ÷ 25 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                ],
                "epochs": 5
            },
            {
                "name": "batch_04_complex_2",
                "data": [
                    {"q": "合同期1年，起租日3月12日，3月份20天房租1600元，押一付三，押金2400元。请问月租金？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "房租1600元20天，包含物业费，押金2400元，卫生费200元可退，总共付了2600元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "租房押一付三，房租1600元20天，押金2400元，首付多少？", "a": "首付 = 押金 + 3个月房租 = 2400 + 2400×3 = 9600元。首付是9600元。"},
                    {"q": "房租1200元租了20天，押金2000元，物业费150元，月租金是多少？", "a": "关键信息：房租1200元，租期20天。日租金 = 1200 ÷ 20 = 60元/天，月租金 = 60 × 30 = 1800元/月。月租金是1800元/月。"},
                ],
                "epochs": 5
            },
        ]
    },
    "hour_8-12": {
        "name": "阶段3：押金问题",
        "batches": [
            {
                "name": "batch_05_deposit_1",
                "data": [
                    {"q": "押金什么时候可以退？", "a": "押金通常在退房时退还。如果房屋没有损坏，押金会在退房后7-30天内退还给租客。"},
                    {"q": "押金2400元什么时候退？", "a": "押金2400元在退房时退还。如果房屋没有损坏、水电费已结清，押金会在退房后7-30天内退还。"},
                    {"q": "押金两千四百元可以退吗？", "a": "押金2400元在退房时可以退还。前提是房屋没有损坏、水电费已结清、卫生干净。"},
                    {"q": "退房时押金怎么退？", "a": "退房时，房东会检查房屋状况。如果没有损坏，押金会在7-30天内退还给租客。"},
                    {"q": "押金能退吗？", "a": "押金在退房时可以退还，前提是房屋没有损坏、水电费已结清。"},
                    {"q": "押金什么时候退给我？", "a": "押金在退房验收后退还，通常在退房后7-30天内到账。"},
                ],
                "epochs": 5
            },
            {
                "name": "batch_06_deposit_2",
                "data": [
                    {"q": "押金不退的情况有哪些？", "a": "押金可能不退的情况：1.房屋有损坏；2.水电费未结清；3.提前退租违约；4.卫生太差需要清洁。"},
                    {"q": "押金可以抵扣房租吗？", "a": "押金通常不能抵扣房租。押金是保证金，在退房时退还，不是预付房租。"},
                    {"q": "押金和房租有什么区别？", "a": "房租是每月的租金费用，需要按时支付。押金是保证金，退房时退还，通常为1-2个月房租。"},
                    {"q": "押金一般是多少？", "a": "押金通常是1-2个月的房租。比如月租2400元，押金可能是2400-4800元。"},
                    {"q": "押金条丢了怎么办？", "a": "押金条丢失可以：1.找房东补开收据；2.提供转账记录证明；3.与房东协商解决。"},
                ],
                "epochs": 5
            },
        ]
    },
    "hour_12-16": {
        "name": "阶段4：卫生费问题",
        "batches": [
            {
                "name": "batch_07_hygiene_1",
                "data": [
                    {"q": "卫生费可以退吗？", "a": "卫生费200元，如果退房时卫生干净，可以退还200元卫生费。"},
                    {"q": "卫生费200元可以退吗？", "a": "卫生费200元，如果退房时房屋卫生干净，可以全额退还200元。"},
                    {"q": "离租卫生干净退200元卫生费是什么意思？", "a": "意思是：如果退房时房屋打扫干净，200元卫生费可以退还。"},
                    {"q": "卫生费什么时候退？", "a": "卫生费在退房验收时退还。如果房屋卫生干净，200元卫生费会退还给租客。"},
                    {"q": "卫生费不退的情况？", "a": "如果退房时房屋卫生不干净，卫生费200元可能不会退还，用于支付清洁费用。"},
                ],
                "epochs": 5
            },
            {
                "name": "batch_08_hygiene_2",
                "data": [
                    {"q": "卫生费和押金有什么区别？", "a": "押金是保证金，退房时退还。卫生费是清洁费，如果退房时卫生干净可以退还，否则用于清洁。"},
                    {"q": "卫生费一般是多少？", "a": "卫生费通常是100-300元，用于退房时的清洁费用。如果自己打扫干净可以退还。"},
                    {"q": "退房时卫生标准是什么？", "a": "退房卫生标准：地面干净、家具无灰尘、厨房无油污、卫生间无污渍、垃圾已清理。"},
                    {"q": "卫生费必须交吗？", "a": "卫生费是租房合同的一部分，通常需要缴纳。如果退房时自己打扫干净，可以退还。"},
                ],
                "epochs": 5
            },
        ]
    },
    "hour_16-20": {
        "name": "阶段5：综合问答",
        "batches": [
            {
                "name": "batch_09_comprehensive_1",
                "data": [
                    {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。离租卫生干净退200元卫生费。合计2600元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。押金2400元在退房时退还，卫生费200元在退房卫生干净时退还。月租金是2400元/月。"},
                    {"q": "我需要付多少钱？", "a": "根据信息：房租1600元（20天），押金2400元，卫生费200元，合计需要支付4200元。其中押金和卫生费在退房时可以退还。"},
                    {"q": "退房时能退多少钱？", "a": "退房时可以退还：押金2400元 + 卫生费200元（如果卫生干净）= 2600元。"},
                    {"q": "房租包含水电费吗？", "a": "根据提供的信息，房租1600元（20天）不包含水电费，水电费通常需要另外支付。"},
                    {"q": "押金和卫生费有什么区别？", "a": "押金2400元是保证金，退房时退还。卫生费200元是清洁费，如果退房时卫生干净可以退还。"},
                ],
                "epochs": 5
            },
            {
                "name": "batch_10_comprehensive_2",
                "data": [
                    {"q": "租房需要注意什么？", "a": "租房注意事项：1.查看房屋状况；2.确认租金和押金；3.签订正规合同；4.检查水电设施；5.了解退租规则。"},
                    {"q": "提前退租怎么办？", "a": "提前退租通常需要：1.提前通知房东（通常30天）；2.可能扣除押金；3.结清水电费；4.办理退房手续。"},
                    {"q": "租房合同包括哪些内容？", "a": "租房合同包括：租期、租金、押金、水电费、维修责任、退租条件、违约责任等。"},
                    {"q": "房东不退押金怎么办？", "a": "房东不退押金可以：1.协商解决；2.查看合同条款；3.保留证据；4.向相关部门投诉；5.法律途径解决。"},
                ],
                "epochs": 5
            },
        ]
    },
    "hour_20-24": {
        "name": "阶段6：强化训练",
        "batches": [
            {
                "name": "batch_11_reinforce_1",
                "data": [
                    {"q": "房租1600元租了20天，日租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天。日租金是80元/天。"},
                    {"q": "房租1600元租了20天，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "押金什么时候可以退？", "a": "押金在退房时退还。如果房屋没有损坏，押金会在退房后7-30天内退还。"},
                    {"q": "卫生费可以退吗？", "a": "卫生费200元，如果退房时卫生干净，可以退还200元卫生费。"},
                    {"q": "退房时能退多少钱？", "a": "退房时可以退还：押金2400元 + 卫生费200元（如果卫生干净）= 2600元。"},
                ],
                "epochs": 10
            },
            {
                "name": "batch_12_reinforce_2",
                "data": [
                    {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "房租1600元20天，押金2400元，卫生费200元，合计2600元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。月租金是2400元/月。"},
                    {"q": "押金2400元什么时候退？", "a": "押金2400元在退房时退还。如果房屋没有损坏、水电费已结清，押金会在退房后7-30天内退还。"},
                    {"q": "卫生费200元可以退吗？", "a": "卫生费200元，如果退房时房屋卫生干净，可以全额退还200元。"},
                ],
                "epochs": 10
            },
        ]
    },
}


class ContinuousTrainer:
    """24小时持续训练器"""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        self.best_loss = float('inf')
        self.total_steps = 0
        self.completed_batches = []
        self.start_time = None
        
        # 加载训练状态
        self._load_state()
    
    def _load_state(self):
        """加载训练状态"""
        state_path = os.path.join(self.output_path, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.completed_batches = state.get('completed_batches', [])
            self.best_loss = state.get('best_loss', float('inf'))
            self.total_steps = state.get('total_steps', 0)
            logger.info(f"加载训练状态: 已完成 {len(self.completed_batches)} 个批次")
    
    def _save_state(self):
        """保存训练状态"""
        os.makedirs(self.output_path, exist_ok=True)
        state = {
            'completed_batches': self.completed_batches,
            'best_loss': self.best_loss,
            'total_steps': self.total_steps,
            'last_update': datetime.now().isoformat()
        }
        state_path = os.path.join(self.output_path, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
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
        
        # 加载已有权重
        self._load_weights()
    
    def _load_weights(self):
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
    
    def train_batch(self, batch_name: str, data: List[Dict], epochs: int = 5) -> float:
        """训练单个批次"""
        # 检查是否已完成
        if batch_name in self.completed_batches:
            logger.info(f"批次 {batch_name} 已完成，跳过")
            return 0.0
        
        logger.info(f"\n训练批次: {batch_name}")
        logger.info(f"数据量: {len(data)} 条, 轮数: {epochs}")
        
        batch_start = time.time()
        
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
        
        batch_time = time.time() - batch_start
        
        # 保存权重
        self._save_weights(batch_name)
        
        # 记录完成
        self.completed_batches.append(batch_name)
        self._save_state()
        
        return avg_loss
    
    def _save_weights(self, batch_name: str):
        """保存权重"""
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存动态权重
        dynamic_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                dynamic_weights[name] = param.data.cpu()
        
        weights_path = os.path.join(self.output_path, 'dynamic_weights.pt')
        torch.save(dynamic_weights, weights_path)
        
        # 保存检查点
        checkpoint_path = os.path.join(self.output_path, f'checkpoint_{batch_name}.pt')
        torch.save({
            'weights': dynamic_weights,
            'best_loss': self.best_loss,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"  ✓ 权重已保存")
    
    def send_progress_report(self, batch_name: str, loss: float, stage_name: str):
        """发送进度报告"""
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        message = f"""🧠 *训练进度报告*

📊 *当前批次*: `{batch_name}`
📈 *Loss*: {loss:.4f}
⏱ *已训练*: {elapsed.seconds // 3600}小时{(elapsed.seconds // 60) % 60}分钟
✅ *已完成批次*: {len(self.completed_batches)}
📉 *最佳Loss*: {self.best_loss:.4f}
🔢 *总步数*: {self.total_steps}

📝 *当前阶段*: {stage_name}
"""
        send_telegram_message(message)
    
    def run_24h_training(self):
        """运行24小时训练"""
        self.start_time = datetime.now()
        
        # 发送开始消息
        send_telegram_message(f"""🚀 *24小时训练计划启动*

⏰ 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
📊 总批次: {sum(len(stage['batches']) for stage in TRAINING_PLAN.values())}
✅ 已完成: {len(self.completed_batches)}

训练将分6个阶段进行，每个阶段完成后发送进度报告。
""")
        
        self.setup()
        
        total_batches = sum(len(stage['batches']) for stage in TRAINING_PLAN.values())
        
        for stage_key, stage in TRAINING_PLAN.items():
            stage_name = stage['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"开始阶段: {stage_name}")
            logger.info(f"{'='*60}")
            
            for batch in stage['batches']:
                batch_name = batch['name']
                data = batch['data']
                epochs = batch.get('epochs', 5)
                
                loss = self.train_batch(batch_name, data, epochs)
                
                if loss > 0:
                    # 发送进度报告
                    self.send_progress_report(batch_name, loss, stage_name)
                
                # 检查是否超过24小时
                elapsed = datetime.now() - self.start_time
                if elapsed.seconds >= 24 * 3600:
                    logger.info("24小时训练完成！")
                    break
            
            # 阶段完成报告
            elapsed = datetime.now() - self.start_time
            send_telegram_message(f"""✅ *阶段完成*

📝 {stage_name}
⏱ 已训练: {elapsed.seconds // 3600}小时{(elapsed.seconds // 60) % 60}分钟
✅ 已完成批次: {len(self.completed_batches)}/{total_batches}
""")
        
        # 发送完成消息
        elapsed = datetime.now() - self.start_time
        send_telegram_message(f"""🎉 *24小时训练完成！*

⏱ 总训练时间: {elapsed.seconds // 3600}小时{(elapsed.seconds // 60) % 60}分钟
✅ 完成批次: {len(self.completed_batches)}
📉 最佳Loss: {self.best_loss:.4f}
🔢 总步数: {self.total_steps}

模型权重已保存，可以测试效果了！
""")


def main():
    model_path = str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    trainer = ContinuousTrainer(model_path, output_path)
    trainer.run_24h_training()


if __name__ == "__main__":
    main()
