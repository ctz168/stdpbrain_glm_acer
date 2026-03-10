#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 内存优化训练脚本
Memory-Optimized Training Script
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def memory_efficient_training(
    model_path: str,
    output_path: str,
    num_steps: int = 30
):
    """内存高效训练"""
    logger.info("="*60)
    logger.info("类人脑架构 - 内存优化训练")
    logger.info("="*60)
    
    device = torch.device("cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数: {total_params/1e9:.2f}B")
    
    # 冻结90%权重
    all_params = list(model.named_parameters())
    freeze_count = int(len(all_params) * 0.9)
    
    for i, (name, param) in enumerate(all_params):
        if i < freeze_count:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5, weight_decay=0.01
    )
    
    # 训练数据
    training_samples = [
        {"input": "你好", "output": "你好！有什么我可以帮助你的吗？"},
        {"input": "什么是人工智能？", "output": "人工智能是计算机科学的一个分支。"},
        {"input": "请记住：我叫小明", "output": "好的，我记住了，您叫小明。"},
        {"input": "我叫什么名字？", "output": "根据之前的对话，您叫小明。"},
        {"input": "1+1等于几？", "output": "1+1等于2。"},
        {"input": "谢谢", "output": "不客气！"},
    ]
    
    # 训练
    logger.info(f"开始训练，共 {num_steps} 步...")
    start_time = time.time()
    loss_history = []
    
    model.train()
    
    for step in range(num_steps):
        sample = training_samples[step % len(training_samples)]
        
        input_text = f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"
        
        encodings = tokenizer(
            input_text, max_length=64, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (step + 1) % 5 == 0:
            avg_loss = sum(loss_history[-5:]) / 5
            logger.info(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")
        
        del outputs, loss
        gc.collect()
    
    training_time = time.time() - start_time
    avg_loss = sum(loss_history) / len(loss_history)
    
    logger.info("="*60)
    logger.info("训练完成!")
    logger.info(f"总步数: {num_steps}")
    logger.info(f"平均Loss: {avg_loss:.4f}")
    logger.info(f"训练时间: {training_time:.1f}秒")
    logger.info("="*60)
    
    # 测试生成
    logger.info("\n测试生成...")
    model.eval()
    
    test_input = "你好"
    test_encodings = tokenizer(
        f"<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors='pt'
    )
    test_input_ids = test_encodings['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            test_input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"输入: {test_input}")
    logger.info(f"输出: {generated}")
    
    # 保存训练报告（不保存完整模型以节省内存）
    os.makedirs(output_path, exist_ok=True)
    
    report = {
        'total_steps': num_steps,
        'average_loss': avg_loss,
        'loss_history': loss_history,
        'training_time_seconds': training_time,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'static_weight_ratio': 0.9,
        'final_test_input': test_input,
        'final_test_output': generated
    }
    
    with open(os.path.join(output_path, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n训练报告已保存: {output_path}/training_report.json")
    
    return report


def stdp_learning_simulation():
    """STDP学习模拟"""
    logger.info("\n" + "="*60)
    logger.info("STDP在线学习模拟")
    logger.info("="*60)
    
    # STDP参数
    alpha = 0.01   # LTP学习率
    beta = 0.008   # LTD学习率
    timing_window = 20.0  # 时序窗口(ms)
    
    logger.info(f"LTP学习率(α): {alpha}")
    logger.info(f"LTD学习率(β): {beta}")
    logger.info(f"时序窗口: {timing_window}ms")
    logger.info("-"*60)
    
    updates = []
    
    scenarios = [
        ("正确推理", 10, 0.95),    # LTP
        ("部分正确", 5, 0.7),      # LTP (弱)
        ("错误推理", -8, 0.3),     # LTD
        ("无关激活", -15, 0.1),    # LTD (强)
        ("快速正确", 3, 0.9),      # LTP (强)
        ("延迟错误", -12, 0.2),    # LTD
    ]
    
    for name, timing_delta, quality in scenarios:
        if timing_delta > 0 and timing_delta < timing_window:
            # LTP: 前序先激活，正确贡献
            update = alpha * quality * math.exp(-timing_delta / timing_window)
            update_type = 'LTP'
            desc = "权重增强"
        elif timing_delta < 0 and abs(timing_delta) < timing_window:
            # LTD: 后序先激活，或错误贡献
            update = -beta * (1 - quality) * math.exp(timing_delta / timing_window)
            update_type = 'LTD'
            desc = "权重减弱"
        else:
            update = 0
            update_type = 'NONE'
            desc = "无更新"
        
        updates.append({
            'scenario': name,
            'timing_delta': timing_delta,
            'quality': quality,
            'update': update,
            'type': update_type,
            'description': desc
        })
        
        logger.info(f"{name}: Δt={timing_delta:+3d}ms, Q={quality:.2f} → "
                   f"{update_type}: {update:+.6f} ({desc})")
    
    # 统计
    ltp_count = sum(1 for u in updates if u['type'] == 'LTP')
    ltd_count = sum(1 for u in updates if u['type'] == 'LTD')
    total_ltp = sum(u['update'] for u in updates if u['type'] == 'LTP')
    total_ltd = sum(u['update'] for u in updates if u['type'] == 'LTD')
    
    logger.info("-"*60)
    logger.info(f"LTP次数: {ltp_count}, 总增强: {total_ltp:.6f}")
    logger.info(f"LTD次数: {ltd_count}, 总减弱: {total_ltd:.6f}")
    logger.info(f"净变化: {total_ltp + total_ltd:.6f}")
    
    return updates


def hippocampus_memory_simulation():
    """海马体记忆系统模拟"""
    logger.info("\n" + "="*60)
    logger.info("海马体记忆系统模拟")
    logger.info("="*60)
    
    # 模拟记忆单元
    memories = []
    
    # 编码阶段
    logger.info("\n[EC编码阶段]")
    episodes = [
        "用户说喜欢蓝色",
        "用户说养了一只猫叫咪咪",
        "用户问天气",
        "用户说喜欢蓝色"  # 重复
    ]
    
    for i, episode in enumerate(episodes):
        memory_id = f"mem_{i:03d}_{hash(episode) % 10000:04d}"
        timestamp = time.time() * 1000 + i * 10  # 10ms间隔
        
        memories.append({
            'id': memory_id,
            'content': episode,
            'timestamp': timestamp,
            'access_count': 0
        })
        
        logger.info(f"  编码: {memory_id} - '{episode}'")
    
    # DG模式分离
    logger.info("\n[DG模式分离]")
    unique_memories = {}
    for mem in memories:
        key = mem['content']
        if key not in unique_memories:
            unique_memories[key] = mem
            logger.info(f"  分离: {mem['id']} - 唯一记忆")
        else:
            logger.info(f"  合并: {mem['id']} - 与已有记忆相同")
    
    # CA3召回
    logger.info("\n[CA3召回阶段]")
    query = "用户喜欢什么颜色"
    logger.info(f"  查询: '{query}'")
    
    # 模拟召回
    recalled = [m for m in unique_memories.values() if '颜色' in m['content'] or '蓝色' in m['content']]
    for mem in recalled:
        mem['access_count'] += 1
        logger.info(f"  召回: {mem['id']} - '{mem['content']}' (访问次数: {mem['access_count']})")
    
    # CA1时序编码
    logger.info("\n[CA1时序编码]")
    for i, mem in enumerate(unique_memories.values()):
        logger.info(f"  时序链: [{i}] {mem['id']} @ {mem['timestamp']:.0f}ms")
    
    # SWR离线巩固
    logger.info("\n[SWR离线巩固]")
    logger.info("  模拟空闲时记忆回放...")
    for mem in unique_memories.values():
        if mem['access_count'] > 0:
            logger.info(f"  巩固: {mem['id']} - 访问{mem['access_count']}次，权重增强")
        else:
            logger.info(f"  修剪: {mem['id']} - 未被访问，可能遗忘")
    
    return memories


import math

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='类人脑架构训练')
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output')
    parser.add_argument('--steps', type=int, default=20)
    
    args = parser.parse_args()
    
    # 训练
    report = memory_efficient_training(
        args.model_path, args.output_path, args.steps
    )
    
    # STDP模拟
    stdp_updates = stdp_learning_simulation()
    
    # 海马体模拟
    memories = hippocampus_memory_simulation()
    
    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)


if __name__ == "__main__":
    main()
