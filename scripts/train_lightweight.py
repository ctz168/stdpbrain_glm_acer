#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 轻量级训练脚本
Human-Like Brain Dual-System Full-Loop AI Architecture - Lightweight Training

针对低内存环境优化的训练脚本
"""

import os
import sys
import json
import time
import logging
import gc
from pathlib import Path

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


def lightweight_training(
    model_path: str,
    output_path: str,
    num_steps: int = 50
):
    """
    轻量级训练
    
    使用梯度累积和内存优化技术
    """
    logger.info("="*60)
    logger.info("类人脑架构 - 轻量级训练")
    logger.info("="*60)
    
    # 设置设备
    device = torch.device("cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer
    from transformers import AutoTokenizer
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    from transformers import AutoModelForCausalLM
    logger.info("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数: {total_params/1e9:.2f}B")
    
    # 冻结90%权重
    logger.info("冻结90%静态权重...")
    all_params = list(model.named_parameters())
    freeze_count = int(len(all_params) * 0.9)
    
    for i, (name, param) in enumerate(all_params):
        if i < freeze_count:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=0.01
    )
    
    # 训练数据
    training_samples = [
        {"input": "你好", "output": "你好！有什么我可以帮助你的吗？"},
        {"input": "什么是人工智能？", "output": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"},
        {"input": "请记住：我叫小明", "output": "好的，我记住了，您叫小明。"},
        {"input": "我叫什么名字？", "output": "根据之前的对话，您叫小明。"},
        {"input": "1+1等于几？", "output": "1+1等于2。"},
        {"input": "请写一首短诗", "output": "春风拂面花盛开，绿柳垂丝燕归来。"},
        {"input": "谢谢", "output": "不客气！还有什么我可以帮助你的吗？"},
        {"input": "再见", "output": "再见！祝你有美好的一天！"},
    ]
    
    # 训练循环
    logger.info(f"开始训练，共 {num_steps} 步...")
    start_time = time.time()
    loss_history = []
    
    model.train()
    
    for step in range(num_steps):
        # 选择样本
        sample = training_samples[step % len(training_samples)]
        
        # 编码
        input_text = f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"
        
        encodings = tokenizer(
            input_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        labels = input_ids.clone()
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            1.0
        )
        
        # 更新
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # 日志
        if (step + 1) % 5 == 0:
            avg_loss = sum(loss_history[-5:]) / 5
            logger.info(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
        
        # 清理内存
        del outputs, loss
        gc.collect()
    
    # 训练完成
    training_time = time.time() - start_time
    avg_loss = sum(loss_history) / len(loss_history)
    
    logger.info("="*60)
    logger.info("训练完成!")
    logger.info(f"总步数: {num_steps}")
    logger.info(f"平均Loss: {avg_loss:.4f}")
    logger.info(f"训练时间: {training_time:.1f}秒")
    logger.info("="*60)
    
    # 保存模型
    os.makedirs(output_path, exist_ok=True)
    
    # 保存最终模型
    final_path = os.path.join(output_path, 'trained_model')
    os.makedirs(final_path, exist_ok=True)
    
    logger.info(f"保存模型到: {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # 保存训练报告
    report = {
        'total_steps': num_steps,
        'average_loss': avg_loss,
        'loss_history': loss_history,
        'training_time_seconds': training_time,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'static_weight_ratio': 0.9
    }
    
    with open(os.path.join(output_path, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("模型保存完成!")
    
    # 测试生成
    logger.info("\n测试生成...")
    model.eval()
    
    test_input = "你好，请介绍一下自己"
    test_encodings = tokenizer(
        f"<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors='pt'
    )
    test_input_ids = test_encodings['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            test_input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"输入: {test_input}")
    logger.info(f"输出: {generated_text}")
    
    return report


def stdp_online_learning_demo():
    """STDP在线学习演示"""
    logger.info("\n" + "="*60)
    logger.info("STDP在线学习演示")
    logger.info("="*60)
    
    # STDP参数
    alpha = 0.01  # LTP学习率
    beta = 0.008  # LTD学习率
    timing_window = 20.0  # 时序窗口(ms)
    
    # 模拟STDP更新
    updates = []
    
    for i in range(10):
        # 模拟时序差
        timing_delta = (i % 3 - 1) * 10  # -10, 0, 10 ms
        output_quality = 0.7 + (i % 5) * 0.06  # 0.7-0.94
        
        if timing_delta > 0 and timing_delta < timing_window:
            # LTP
            update = alpha * output_quality * (timing_delta / timing_window)
            update_type = 'LTP'
        elif timing_delta < 0 and abs(timing_delta) < timing_window:
            # LTD
            update = -beta * (1 - output_quality) * (abs(timing_delta) / timing_window)
            update_type = 'LTD'
        else:
            update = 0
            update_type = 'NONE'
        
        updates.append({
            'step': i,
            'timing_delta_ms': timing_delta,
            'quality': output_quality,
            'update': update,
            'type': update_type
        })
        
        logger.info(f"Step {i}: Δt={timing_delta}ms, Q={output_quality:.2f}, "
                   f"Update={update:.6f} ({update_type})")
    
    return updates


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='类人脑架构轻量级训练')
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B',
                       help='模型路径')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output',
                       help='输出路径')
    parser.add_argument('--steps', type=int, default=30, help='训练步数')
    
    args = parser.parse_args()
    
    # 执行训练
    report = lightweight_training(
        model_path=args.model_path,
        output_path=args.output_path,
        num_steps=args.steps
    )
    
    # STDP演示
    stdp_updates = stdp_online_learning_demo()
    
    print("\n训练完成!")
    print(f"输出路径: {args.output_path}")


if __name__ == "__main__":
    main()
