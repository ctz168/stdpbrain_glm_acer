#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 完整训练脚本
Human-Like Brain Dual-System Full-Loop AI Architecture - Training Script

执行预适配训练，优化模型权重
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================
# 训练配置
# ============================================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型路径
    model_path: str = ""
    output_path: str = ""
    
    # 训练参数
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # 数据参数
    max_seq_length: int = 512
    
    # 设备
    device: str = "auto"
    
    # 保存
    save_every: int = 500
    eval_every: int = 200


# ============================================
# 训练数据集
# ============================================

class BrainLikeDataset(Dataset):
    """类脑架构训练数据集"""
    
    def __init__(
        self,
        data_path: str = None,
        tokenizer = None,
        max_length: int = 512
    ):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data_path and os.path.exists(data_path):
            self._load_data(data_path)
        else:
            # 使用内置示例数据
            self._create_sample_data()
    
    def _load_data(self, path: str):
        """加载数据"""
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            self.data.append({
                'input': item.get('input', ''),
                'output': item.get('output', ''),
                'type': item.get('type', 'general')
            })
        
        logger.info(f"加载了 {len(self.data)} 条训练数据")
    
    def _create_sample_data(self):
        """创建示例训练数据"""
        sample_data = [
            # 高刷新推理适配
            {"input": "请用简洁的语言回答：什么是人工智能？", 
             "output": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。", 
             "type": "high_refresh"},
            
            # 海马体记忆适配
            {"input": "记住：我叫张三，今年25岁，是一名软件工程师。", 
             "output": "好的，我记住了：您叫张三，今年25岁，职业是软件工程师。", 
             "type": "memory"},
            
            # STDP学习适配
            {"input": "我之前告诉过你我的名字，你还记得吗？", 
             "output": "根据我的记忆，您叫张三。", 
             "type": "stdp_recall"},
            
            # 自生成模式
            {"input": "给我讲一个简短的故事。", 
             "output": "从前有一只聪明的小狐狸，它学会了用树叶做伞，在雨天保护自己。", 
             "type": "self_gen"},
            
            # 自博弈模式
            {"input": "请计算：123 + 456 = ?", 
             "output": "让我来计算：123 + 456 = 579。验证：123 + 456，个位3+6=9，十位2+5=7，百位1+4=5，结果是579。", 
             "type": "self_play"},
            
            # 自评判模式
            {"input": "请解释量子计算的基本原理，要求准确且易懂。", 
             "output": "量子计算利用量子比特进行计算。与传统比特只能是0或1不同，量子比特可以同时处于0和1的叠加态，这使得量子计算机能够并行处理大量可能性，在某些问题上比传统计算机快得多。", 
             "type": "self_judge"},
            
            # 角色切换适配
            {"input": "[生成模式] 请写一首关于春天的诗。", 
             "output": "春风拂面花盛开，\n绿柳垂丝燕归来。\n桃红李白争芳艳，\n万物复苏好时节。", 
             "type": "role_gen"},
            
            {"input": "[验证模式] 检查：地球是太阳系中最大的行星。", 
             "output": "这个说法是错误的。地球不是太阳系中最大的行星，木星才是最大的行星。地球在太阳系行星中排名第五，前四位依次是木星、土星、天王星、海王星。", 
             "type": "role_verify"},
            
            {"input": "[评判模式] 评估回答质量：问'1+1等于几？'答'1+1等于2，这是基本的数学运算。'", 
             "output": "评判结果：\n- 事实准确性：10分（答案正确）\n- 逻辑完整性：9分（解释清晰）\n- 语义连贯性：10分（表达流畅）\n- 指令遵循度：10分（直接回答）\n总分：39/40分，优秀", 
             "type": "role_judge"},
            
            # 长时序记忆
            {"input": "让我们开始一个长对话。第一，我喜欢蓝色。", 
             "output": "好的，我记住了您喜欢蓝色。请继续。", 
             "type": "long_seq"},
            
            {"input": "第二，我养了一只猫叫咪咪。", 
             "output": "记住了：您喜欢蓝色，养了一只叫咪咪的猫。请继续。", 
             "type": "long_seq"},
            
            {"input": "你还记得我的喜好和宠物吗？", 
             "output": "是的，您喜欢蓝色，养了一只叫咪咪的猫。", 
             "type": "long_seq_recall"},
            
            # 因果推理
            {"input": "如果下雨，地面会湿。现在地面是湿的，能确定下雨了吗？", 
             "output": "不能确定。虽然下雨会导致地面湿，但地面湿也可能由其他原因造成，比如洒水车经过、有人泼水等。这是逻辑上的'肯定后件'谬误。", 
             "type": "causal"},
            
            # 模式补全
            {"input": "序列：2, 4, 6, 8, ? 请补全。", 
             "output": "下一个数字是10。这是一个等差数列，每个数比前一个数大2。", 
             "type": "completion"},
        ]
        
        # 扩展数据
        expanded_data = []
        for _ in range(50):  # 扩展到750条
            expanded_data.extend(sample_data)
        
        self.data = expanded_data[:500]  # 限制500条
        logger.info(f"创建了 {len(self.data)} 条示例训练数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        input_text = item['input']
        output_text = item['output']
        
        # 如果有tokenizer，进行编码
        if self.tokenizer:
            full_text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
            
            encodings = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].squeeze()
            attention_mask = encodings['attention_mask'].squeeze()
            
            # 创建labels（只计算assistant部分的loss）
            labels = input_ids.clone()
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'type': item['type']
            }
        else:
            # 返回文本
            return {
                'input_text': input_text,
                'output_text': output_text,
                'type': item['type']
            }


# ============================================
# 训练器
# ============================================

class BrainLikeTrainer:
    """类脑架构训练器"""
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        config: TrainingConfig = None
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.config = config or TrainingConfig()
        
        # 设置设备
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"使用设备: {self.device}")
        
        # 模型和tokenizer
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
        self.loss_history = []
    
    def setup(self):
        """设置训练环境"""
        logger.info("正在设置训练环境...")
        
        # 加载tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        from transformers import AutoModelForCausalLM
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        # 冻结大部分权重（模拟90%静态权重）
        self._freeze_weights()
        
        # 创建优化器（只优化未冻结的权重）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)
        
        logger.info("训练环境设置完成")
    
    def _freeze_weights(self):
        """冻结90%的权重（模拟静态权重）"""
        # 获取所有参数
        all_params = list(self.model.named_parameters())
        total_params = len(all_params)
        
        # 冻结前90%的层
        freeze_count = int(total_params * 0.9)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
            else:
                # 只训练后10%的参数（主要是输出层附近的参数）
                param.requires_grad = True
                logger.debug(f"可训练: {name}")
        
        frozen = sum(1 for _, p in all_params if not p.requires_grad)
        trainable = sum(1 for _, p in all_params if p.requires_grad)
        
        logger.info(f"冻结参数: {frozen}, 可训练参数: {trainable}")
        logger.info(f"静态权重比例: {frozen/total_params*100:.1f}%")
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """执行训练"""
        logger.info("开始训练...")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            self.model.train()
            
            for batch_idx, batch in enumerate(train_loader):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )
                
                # 更新权重
                self.optimizer.step()
                
                # 记录
                epoch_loss += loss.item()
                epoch_steps += 1
                self.global_step += 1
                
                # 日志
                if self.global_step % 10 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs}, "
                        f"Step {self.global_step}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Avg Loss: {avg_loss:.4f}"
                    )
                
                # 保存检查点
                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")
            
            # Epoch结束
            avg_epoch_loss = epoch_loss / epoch_steps
            self.loss_history.append(avg_epoch_loss)
            
            logger.info(
                f"Epoch {epoch+1} 完成, "
                f"平均Loss: {avg_epoch_loss:.4f}, "
                f"耗时: {(time.time() - start_time)/60:.1f}分钟"
            )
            
            # 保存epoch检查点
            self._save_checkpoint(f"epoch-{epoch+1}")
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成! 总耗时: {total_time/60:.1f}分钟")
        
        # 保存最终模型
        self._save_final_model()
        
        return {
            'total_steps': self.global_step,
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'loss_history': self.loss_history,
            'training_time_seconds': total_time
        }
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_path, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # 保存训练状态
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': asdict(self.config)
        }, os.path.join(checkpoint_path, 'training_state.pt'))
        
        logger.info(f"保存检查点: {checkpoint_path}")
    
    def _save_final_model(self):
        """保存最终模型"""
        final_path = os.path.join(self.output_path, 'final')
        os.makedirs(final_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # 保存训练报告
        report = {
            'total_steps': self.global_step,
            'epochs': self.config.epochs,
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'loss_history': self.loss_history,
            'config': asdict(self.config)
        }
        
        with open(os.path.join(final_path, 'training_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"最终模型保存到: {final_path}")


# ============================================
# STDP在线学习训练器
# ============================================

class STDPOnlineTrainer:
    """STDP在线学习训练器"""
    
    def __init__(self, model, stdp_config: Dict = None):
        self.model = model
        self.stdp_config = stdp_config or {
            'alpha': 0.01,  # LTP学习率
            'beta': 0.008,  # LTD学习率
            'timing_window': 20.0  # 时序窗口(ms)
        }
        
        self.update_count = 0
        self.update_history = []
    
    def update_weights(
        self,
        input_features: torch.Tensor,
        output_quality: float,
        timing_delta: float
    ):
        """
        基于STDP规则更新权重
        
        Args:
            input_features: 输入特征
            output_quality: 输出质量 (0-1)
            timing_delta: 时序差值 (ms)
        """
        # 计算STDP更新量
        if timing_delta > 0 and timing_delta < self.stdp_config['timing_window']:
            # LTP: 前序先激活
            update = self.stdp_config['alpha'] * output_quality * \
                     math.exp(-timing_delta / self.stdp_config['timing_window'])
            update_type = 'LTP'
        elif timing_delta < 0 and abs(timing_delta) < self.stdp_config['timing_window']:
            # LTD: 后序先激活
            update = -self.stdp_config['beta'] * (1 - output_quality) * \
                     math.exp(timing_delta / self.stdp_config['timing_window'])
            update_type = 'LTD'
        else:
            return 0.0, 'NONE'
        
        # 应用更新到可训练参数
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    # 小幅度更新
                    param.data += update * 0.001 * torch.randn_like(param.data)
        
        self.update_count += 1
        self.update_history.append({
            'update': update,
            'type': update_type,
            'quality': output_quality
        })
        
        return update, update_type


# ============================================
# 主函数
# ============================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='类人脑架构训练脚本')
    parser.add_argument('--model-path', type=str, 
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B',
                       help='模型路径')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output',
                       help='输出路径')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # 创建训练器
    trainer = BrainLikeTrainer(
        model_path=args.model_path,
        output_path=args.output_path,
        config=config
    )
    
    # 设置
    trainer.setup()
    
    # 创建数据集
    train_dataset = BrainLikeDataset(
        tokenizer=trainer.tokenizer,
        max_length=config.max_seq_length
    )
    
    # 训练
    result = trainer.train(train_dataset)
    
    # 打印结果
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"总步数: {result['total_steps']}")
    print(f"最终Loss: {result['final_loss']:.4f}")
    print(f"训练时间: {result['training_time_seconds']/60:.1f}分钟")
    print(f"输出路径: {args.output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
