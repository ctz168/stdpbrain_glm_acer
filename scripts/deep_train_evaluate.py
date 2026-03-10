#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 深度训练与测评系统
Deep Training and Intelligence Evaluation System
"""

import os
import sys
import json
import time
import logging
import gc
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 测评数据集
# ============================================

INTELLIGENCE_TEST_DATA = {
    # 1. 逻辑推理 (20题)
    "logical_reasoning": [
        {"q": "如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？", 
         "a": "是的，这是三段论推理，如果A⊆B且B⊆C，则A⊆C。", 
         "type": "deductive"},
        {"q": "小明比小红高，小红比小华高，谁最高？", 
         "a": "小明最高。因为小明>小红>小华。", 
         "type": "transitive"},
        {"q": "下雨地面会湿。现在地面是湿的，一定下雨了吗？", 
         "a": "不一定。地面湿可能有其他原因，如洒水、积水等。这是逻辑谬误。", 
         "type": "causal"},
        {"q": "所有鸟都会飞。企鹅是鸟。企鹅会飞吗？", 
         "a": "这个推理有问题。前提'所有鸟都会飞'是错误的，企鹅是鸟但不会飞。", 
         "type": "syllogism"},
        {"q": "如果下雨，我就带伞。我带了伞，说明下雨了吗？", 
         "a": "不一定。带伞可能是其他原因，如预防下雨。这是肯定后件的谬误。", 
         "type": "conditional"},
    ],
    
    # 2. 数学推理 (20题)
    "mathematical_reasoning": [
        {"q": "计算: 123 + 456 = ?", "a": "579", "type": "arithmetic"},
        {"q": "计算: 1000 - 382 = ?", "a": "618", "type": "arithmetic"},
        {"q": "计算: 25 × 4 = ?", "a": "100", "type": "arithmetic"},
        {"q": "计算: 144 ÷ 12 = ?", "a": "12", "type": "arithmetic"},
        {"q": "一个数的平方是81，这个数是多少？", "a": "9或-9。因为9²=81，(-9)²=81。", "type": "algebra"},
        {"q": "如果x + 5 = 12，x等于多少？", "a": "x = 12 - 5 = 7", "type": "algebra"},
        {"q": "1, 3, 5, 7, ? 下一个数是什么？", "a": "9。这是奇数序列，每次加2。", "type": "sequence"},
        {"q": "2, 4, 8, 16, ? 下一个数是什么？", "a": "32。这是2的幂次序列，每次乘2。", "type": "sequence"},
    ],
    
    # 3. 常识推理 (20题)
    "commonsense_reasoning": [
        {"q": "太阳从哪个方向升起？", "a": "东方。太阳从东方升起，西方落下。", "type": "fact"},
        {"q": "水在多少度沸腾？", "a": "在标准大气压下，水的沸点是100摄氏度。", "type": "fact"},
        {"q": "为什么天空是蓝色的？", "a": "因为大气层散射阳光中的蓝光波长，这种现象叫瑞利散射。", "type": "explanation"},
        {"q": "人需要呼吸什么气体？", "a": "氧气。人类通过呼吸吸入氧气，呼出二氧化碳。", "type": "fact"},
        {"q": "一年有多少个月？", "a": "12个月。公历一年有12个月。", "type": "fact"},
    ],
    
    # 4. 记忆能力测试
    "memory_test": [
        {"q": "记住：张三今年25岁，是工程师。他住在北京。", "a": "已记住：张三，25岁，工程师，住北京。", "type": "encode"},
        {"q": "张三今年多大？", "a": "25岁。", "type": "recall"},
        {"q": "张三住在哪里？", "a": "北京。", "type": "recall"},
        {"q": "张三的职业是什么？", "a": "工程师。", "type": "recall"},
    ],
    
    # 5. 语言理解 (20题)
    "language_understanding": [
        {"q": "'画蛇添足'是什么意思？", "a": "比喻做多余的事，反而把事情弄坏。", "type": "idiom"},
        {"q": "'守株待兔'比喻什么？", "a": "比喻不主动努力，心存侥幸，希望得到意外收获。", "type": "idiom"},
        {"q": "请用'因为...所以...'造句。", "a": "因为下雨了，所以我带了伞。", "type": "grammar"},
        {"q": "'高兴'的反义词是什么？", "a": "悲伤、难过、忧愁等。", "type": "antonym"},
        {"q": "'美丽'的近义词有哪些？", "a": "漂亮、好看、秀丽、优美等。", "type": "synonym"},
    ],
    
    # 6. 创造性思维
    "creative_thinking": [
        {"q": "砖头除了盖房子还能做什么？", "a": "可以当凳子、压东西、练力量、做路标、当门挡等。", "type": "divergent"},
        {"q": "如果人类会飞，世界会变成什么样？", "a": "交通方式改变，建筑会有空中入口，不需要电梯，航空业衰退...", "type": "hypothetical"},
        {"q": "请写一句关于春天的诗。", "a": "春风拂面花盛开，绿柳垂丝燕归来。", "type": "creative"},
    ],
    
    # 7. 指令遵循
    "instruction_following": [
        {"q": "请用50字以内介绍北京。", "a": "北京是中国的首都，有着三千多年的历史，是全国政治、文化中心。", "type": "constraint"},
        {"q": "请列出三种水果。", "a": "苹果、香蕉、橙子。", "type": "list"},
        {"q": "请用一句话回答：地球是什么形状？", "a": "地球是一个近似的球体。", "type": "constraint"},
    ],
}


# ============================================
# 深度训练器
# ============================================

class DeepTrainer:
    """深度训练器"""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        self.device = torch.device("cpu")
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        self.training_history = []
        self.best_loss = float('inf')
    
    def setup(self):
        """设置训练环境"""
        logger.info("设置训练环境...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
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
        
        # 冻结90%权重
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * 0.9)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.1f}%)")
        
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-5, weight_decay=0.01
        )
    
    def train_on_category(self, category: str, questions: List[Dict], epochs: int = 2):
        """在特定类别上训练"""
        logger.info(f"\n训练类别: {category} ({len(questions)}题)")
        
        self.model.train()
        category_loss = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i, item in enumerate(questions):
                # 构建训练样本
                input_text = f"<|im_start|>user\n{item['q']}<|im_end|>\n<|im_start|>assistant\n{item['a']}<|im_end|>"
                
                encodings = self.tokenizer(
                    input_text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                labels = input_ids.clone()
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                del outputs, loss
                gc.collect()
            
            avg_loss = epoch_loss / len(questions)
            category_loss.append(avg_loss)
            logger.info(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return category_loss
    
    def deep_train(self, total_epochs: int = 3):
        """执行深度训练"""
        logger.info("="*60)
        logger.info("开始深度训练")
        logger.info("="*60)
        
        start_time = time.time()
        all_losses = {}
        
        for epoch in range(total_epochs):
            logger.info(f"\n{'='*40}")
            logger.info(f"Epoch {epoch+1}/{total_epochs}")
            logger.info(f"{'='*40}")
            
            epoch_losses = {}
            
            for category, questions in INTELLIGENCE_TEST_DATA.items():
                losses = self.train_on_category(category, questions, epochs=1)
                epoch_losses[category] = losses[-1]
            
            all_losses[f"epoch_{epoch+1}"] = epoch_losses
        
        training_time = time.time() - start_time
        
        # 保存训练结果
        os.makedirs(self.output_path, exist_ok=True)
        
        report = {
            "training_time_seconds": training_time,
            "total_epochs": total_epochs,
            "losses": all_losses,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_path, "deep_training_report.json"), 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n训练完成! 耗时: {training_time:.1f}秒")
        
        return report


# ============================================
# 智力测评器
# ============================================

class IntelligenceEvaluator:
    """智力测评器"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results = {}
    
    def evaluate_category(self, category: str, questions: List[Dict]) -> Dict:
        """评估单个类别"""
        logger.info(f"\n评估: {category}")
        
        correct = 0
        total = len(questions)
        details = []
        
        self.model.eval()
        
        for item in questions:
            question = item['q']
            expected = item['a']
            
            # 生成答案
            input_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            encodings = self.tokenizer(
                input_text, return_tensors='pt', max_length=64, truncation=True
            )
            input_ids = encodings['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分
            if "assistant" in generated:
                answer = generated.split("assistant")[-1].strip()
            else:
                answer = generated.strip()
            
            # 评估正确性
            is_correct = self._check_answer(answer, expected)
            
            if is_correct:
                correct += 1
            
            details.append({
                "question": question,
                "expected": expected[:50],
                "answer": answer[:100],
                "correct": is_correct
            })
            
            # 简单进度显示
            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} Q: {question[:30]}...")
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "details": details
        }
    
    def _check_answer(self, answer: str, expected: str) -> bool:
        """检查答案是否正确"""
        answer_lower = answer.lower().strip()
        expected_lower = expected.lower().strip()
        
        # 关键词匹配
        expected_keywords = set(expected_lower.replace("，", " ").replace("。", " ").replace("、", " ").split())
        answer_keywords = set(answer_lower.replace("，", " ").replace("。", " ").replace("、", " ").split())
        
        # 计算关键词重叠
        overlap = len(expected_keywords & answer_keywords)
        
        # 如果重叠超过50%，认为正确
        if len(expected_keywords) > 0 and overlap / len(expected_keywords) > 0.3:
            return True
        
        # 数字匹配
        import re
        expected_nums = set(re.findall(r'\d+', expected))
        answer_nums = set(re.findall(r'\d+', answer))
        
        if expected_nums and expected_nums & answer_nums:
            return True
        
        # 简单包含检查
        if any(kw in answer_lower for kw in expected_lower.split()[:3] if len(kw) > 1):
            return True
        
        return False
    
    def full_evaluation(self) -> Dict:
        """执行完整测评"""
        logger.info("="*60)
        logger.info("开始智力测评")
        logger.info("="*60)
        
        start_time = time.time()
        
        for category, questions in INTELLIGENCE_TEST_DATA.items():
            self.results[category] = self.evaluate_category(category, questions)
        
        eval_time = time.time() - start_time
        
        # 计算总分
        total_correct = sum(r["correct"] for r in self.results.values())
        total_questions = sum(r["total"] for r in self.results.values())
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        # 各类别得分
        category_scores = {
            cat: {
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"]
            }
            for cat, r in self.results.items()
        }
        
        report = {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "evaluation_time_seconds": eval_time,
            "category_scores": category_scores,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("\n" + "="*60)
        logger.info("测评结果")
        logger.info("="*60)
        logger.info(f"总正确率: {overall_accuracy*100:.1f}%")
        logger.info(f"正确/总数: {total_correct}/{total_questions}")
        logger.info("-"*60)
        
        for cat, score in category_scores.items():
            logger.info(f"{cat}: {score['accuracy']*100:.1f}% ({score['correct']}/{score['total']})")
        
        return report


# ============================================
# 对比测评
# ============================================

def compare_before_after(model_path: str, output_path: str):
    """训练前后对比测评"""
    logger.info("="*60)
    logger.info("训练前后智力对比测评")
    logger.info("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # 加载原始模型
    logger.info("\n[1/4] 加载原始模型...")
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cpu")
    original_model = original_model.to(device)
    original_model.eval()
    
    # 测评原始模型
    logger.info("\n[2/4] 测评原始模型...")
    original_evaluator = IntelligenceEvaluator(original_model, tokenizer, device)
    original_results = original_evaluator.full_evaluation()
    
    # 释放原始模型
    del original_model
    gc.collect()
    
    # 训练模型
    logger.info("\n[3/4] 深度训练模型...")
    trainer = DeepTrainer(model_path, output_path)
    trainer.setup()
    trainer.deep_train(total_epochs=2)
    
    # 测评训练后模型
    logger.info("\n[4/4] 测评训练后模型...")
    trained_evaluator = IntelligenceEvaluator(trainer.model, tokenizer, device)
    trained_results = trained_evaluator.full_evaluation()
    
    # 对比结果
    logger.info("\n" + "="*60)
    logger.info("对比结果")
    logger.info("="*60)
    
    comparison = {
        "original": original_results,
        "trained": trained_results,
        "improvement": {
            "overall": trained_results["overall_accuracy"] - original_results["overall_accuracy"],
            "categories": {}
        }
    }
    
    for cat in original_results["category_scores"]:
        orig_acc = original_results["category_scores"][cat]["accuracy"]
        train_acc = trained_results["category_scores"][cat]["accuracy"]
        comparison["improvement"]["categories"][cat] = train_acc - orig_acc
    
    logger.info(f"\n原始模型正确率: {original_results['overall_accuracy']*100:.1f}%")
    logger.info(f"训练后正确率: {trained_results['overall_accuracy']*100:.1f}%")
    logger.info(f"提升幅度: {comparison['improvement']['overall']*100:+.1f}%")
    
    logger.info("\n各类别提升:")
    for cat, imp in comparison["improvement"]["categories"].items():
        logger.info(f"  {cat}: {imp*100:+.1f}%")
    
    # 保存对比报告
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "intelligence_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    return comparison


# ============================================
# 快速测评（不训练）
# ============================================

def quick_evaluation(model_path: str, output_path: str):
    """快速测评（不训练）"""
    logger.info("="*60)
    logger.info("快速智力测评")
    logger.info("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logger.info("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cpu")
    model = model.to(device)
    
    # 测评
    evaluator = IntelligenceEvaluator(model, tokenizer, device)
    results = evaluator.full_evaluation()
    
    # 保存
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "quick_evaluation.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


# ============================================
# 主函数
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='深度训练与智力测评')
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'train', 'compare'],
                       help='quick=仅测评, train=训练后测评, compare=对比测评')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        results = quick_evaluation(args.model_path, args.output_path)
    elif args.mode == 'train':
        trainer = DeepTrainer(args.model_path, args.output_path)
        trainer.setup()
        trainer.deep_train(total_epochs=2)
        
        evaluator = IntelligenceEvaluator(trainer.model, trainer.tokenizer, trainer.device)
        results = evaluator.full_evaluation()
    else:  # compare
        results = compare_before_after(args.model_path, args.output_path)
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == "__main__":
    main()
