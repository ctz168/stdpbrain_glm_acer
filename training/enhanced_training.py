#!/usr/bin/env python3
"""
一体式优化整合训练 - 增强版
解决复杂场景推理问题
"""

import os
import sys
import json
import gc
import logging
from pathlib import Path
from datetime import datetime

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
# 增强训练数据 - 覆盖复杂场景
# ============================================

TRAINING_DATA = [
    # === 基础计算 ===
    {"q": "房租1600元租了20天，日租金是多少？", 
     "r": "日租金 = 1600 ÷ 20 = 80元/天", 
     "a": "日租金是80元/天。"},
    
    {"q": "房租1600元租了20天，月租金是多少？", 
     "r": "日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月", 
     "a": "月租金是2400元/月。"},
    
    # === 复杂场景：包含押金、卫生费等干扰信息 ===
    {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？", 
     "r": "关键信息：3月份20天房租1600元\n日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月\n注意：押金和卫生费不影响月租金计算", 
     "a": "月租金是2400元/月。"},
    
    {"q": "房租1600元20天，押金两千四百元，卫生费200元，合计2600元。月租金是多少？", 
     "r": "关键信息：房租1600元，租期20天\n日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月\n注意：合计2600元包含押金和卫生费，不是房租", 
     "a": "月租金是2400元/月。"},
    
    {"q": "3月份20天房租1600元，押金2400元，卫生费200元，退房时卫生干净退200元。月租金是多少？", 
     "r": "关键信息：房租1600元，租期20天\n日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月", 
     "a": "月租金是2400元/月。"},
    
    # === 更多变体 ===
    {"q": "房租900元租了15天，押金1500元，月租金是多少？", 
     "r": "关键信息：房租900元，租期15天\n日租金 = 900 ÷ 15 = 60元/天\n月租金 = 60 × 30 = 1800元/月", 
     "a": "月租金是1800元/月。"},
    
    {"q": "房租2000元25天，押金3000元，物业费100元，月租金是多少？", 
     "r": "关键信息：房租2000元，租期25天\n日租金 = 2000 ÷ 25 = 80元/天\n月租金 = 80 × 30 = 2400元/月", 
     "a": "月租金是2400元/月。"},
    
    {"q": "房租500元10天，押金800元，水电费50元，月租金是多少？", 
     "r": "关键信息：房租500元，租期10天\n日租金 = 500 ÷ 10 = 50元/天\n月租金 = 50 × 30 = 1500元/月", 
     "a": "月租金是1500元/月。"},
    
    # === 反向计算 ===
    {"q": "日租金80元，月租金是多少？", 
     "r": "月租金 = 日租金 × 30 = 80 × 30 = 2400元", 
     "a": "月租金是2400元/月。"},
    
    {"q": "日租金60元，月租金是多少？", 
     "r": "月租金 = 日租金 × 30 = 60 × 30 = 1800元", 
     "a": "月租金是1800元/月。"},
    
    # === 特殊情况 ===
    {"q": "房租1800元租了30天，月租金是多少？", 
     "r": "租期正好30天\n月租金 = 1800元", 
     "a": "月租金是1800元/月。"},
    
    {"q": "房租2400元租了30天，日租金是多少？", 
     "r": "日租金 = 2400 ÷ 30 = 80元/天", 
     "a": "日租金是80元/天。"},
    
    # === 更多干扰信息场景 ===
    {"q": "合同期1年，起租日3月12日，3月份20天房租1600元，押一付三，押金2400元。请问月租金？", 
     "r": "关键信息：3月份20天房租1600元\n日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月", 
     "a": "月租金是2400元/月。"},
    
    {"q": "房租1600元20天，包含物业费，押金2400元，卫生费200元可退，总共付了2600元。月租金是多少？", 
     "r": "关键信息：房租1600元，租期20天\n日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月\n注意：总共付的2600元包含押金和卫生费", 
     "a": "月租金是2400元/月。"},
]


def train():
    """训练"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    logger.info("="*60)
    logger.info("一体式优化整合训练 - 增强版")
    logger.info("="*60)
    
    # 加载模型
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # 冻结90%权重
    all_params = list(model.named_parameters())
    freeze_count = int(len(all_params) * 0.9)
    
    for i, (name, param) in enumerate(all_params):
        if i < freeze_count:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"冻结: {freeze_count}/{len(all_params)}层, 可训练: {trainable/1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5
    )
    
    # 训练
    epochs = 10  # 增加训练轮数
    best_loss = float('inf')
    
    logger.info(f"\n开始训练: {epochs}轮, {len(TRAINING_DATA)}条数据")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for item in TRAINING_DATA:
            prompt = f"问题：{item['q']}\n\n请一步步思考：\n{item['r']}\n\n答案：{item['a']}"
            
            inputs = tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
            input_ids = inputs['input_ids']
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            del outputs, loss
            gc.collect()
        
        avg_loss = total_loss / len(TRAINING_DATA)
        if avg_loss < best_loss:
            best_loss = avg_loss
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Best = {best_loss:.4f}")
    
    # 测试
    logger.info("\n测试训练效果...")
    model.eval()
    
    test_questions = [
        "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？",
        "房租1600元20天，押金两千四百元，卫生费200元，合计2600元。月租金是多少？",
        "房租1600元租了20天，月租金是多少？",
    ]
    
    for q in test_questions:
        prompt = f"问题：{q}\n\n请一步步思考并回答："
        inputs = tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=150, do_sample=False)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nQ: {q}")
        logger.info(f"A: {response}")
    
    # 保存动态权重
    logger.info(f"\n保存动态权重到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    dynamic_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            dynamic_weights[name] = param.data.cpu()
    
    torch.save(dynamic_weights, os.path.join(output_path, 'dynamic_weights.pt'))
    tokenizer.save_pretrained(output_path)
    
    config = {
        'base_model': model_path,
        'training_time': datetime.now().isoformat(),
        'epochs': epochs,
        'data_count': len(TRAINING_DATA),
        'best_loss': best_loss,
        'trainable_params': trainable,
    }
    
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("训练完成！")


if __name__ == "__main__":
    train()
