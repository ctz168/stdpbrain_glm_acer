#!/usr/bin/env python3
"""
均衡训练 - 避免过拟合
Balanced Training - Avoid Overfitting

问题：之前训练数据中租金计算问题太多，导致模型过拟合
解决：均衡分配各类数据，增加通用对话
"""

import os
import sys
import json
import gc
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TELEGRAM_BOT_TOKEN = "8534413276:AAHzqgxVTOL2fapd8NV7UjppF4NXr1zSUek"
TELEGRAM_CHAT_ID = "8398597279"


def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except:
        pass


# ============================================
# 均衡训练数据 - 各类数据比例均衡
# ============================================

TRAINING_DATA = {
    # === 第1类：通用对话 (最重要，防止过拟合) ===
    "general": [
        {"q": "你好", "a": "你好！我是类人脑AI助手，有什么可以帮助你的吗？"},
        {"q": "您好", "a": "您好！很高兴为您服务，请问有什么问题？"},
        {"q": "嗨", "a": "嗨！你好，有什么我可以帮你的吗？"},
        {"q": "hello", "a": "Hello! 你好，我是AI助手，有什么可以帮助你的？"},
        {"q": "你好啊", "a": "你好啊！很高兴见到你，有什么问题尽管问我。"},
        {"q": "在吗", "a": "在的，请问有什么可以帮助你？"},
        {"q": "有人吗", "a": "我在的，请问有什么问题？"},
        {"q": "你是谁", "a": "我是类人脑双系统AI架构的智能助手，可以帮你解答各种问题。"},
        {"q": "你叫什么名字", "a": "我是类人脑AI助手，你可以叫我小脑。"},
        {"q": "介绍一下你自己", "a": "我是类人脑双系统AI架构的智能助手，具有100Hz高刷新、STDP在线学习、海马体记忆等特点，可以帮你解答租房计算、逻辑推理等问题。"},
        {"q": "谢谢", "a": "不客气！有其他问题随时问我。"},
        {"q": "感谢", "a": "不用谢，能帮到你就好！"},
        {"q": "再见", "a": "再见！有问题随时来找我。"},
        {"q": "拜拜", "a": "拜拜！祝你一切顺利！"},
        {"q": "晚安", "a": "晚安，祝你有个好梦！"},
        {"q": "早上好", "a": "早上好！新的一天开始了，有什么我可以帮你的吗？"},
    ],
    
    # === 第2类：租金计算 (适量，不要太多) ===
    "rent_calc": [
        {"q": "房租1600元租了20天，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。"},
        {"q": "房租900元租了15天，月租金是多少？", "a": "日租金 = 900 ÷ 15 = 60元/天，月租金 = 60 × 30 = 1800元/月。"},
        {"q": "日租金80元，月租金是多少？", "a": "月租金 = 80 × 30 = 2400元/月。"},
        {"q": "月租金2400元，日租金是多少？", "a": "日租金 = 2400 ÷ 30 = 80元/天。"},
    ],
    
    # === 第3类：复杂场景租金计算 ===
    "rent_complex": [
        {"q": "房租1600元20天，押金2400元，卫生费200元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。押金和卫生费不影响月租金计算。"},
        {"q": "租期1年，3月份20天房租1600元，押金2400元。月租金是多少？", "a": "关键信息：房租1600元，租期20天。日租金 = 1600 ÷ 20 = 80元/天，月租金 = 80 × 30 = 2400元/月。"},
    ],
    
    # === 第4类：押金问题 ===
    "deposit": [
        {"q": "押金什么时候可以退？", "a": "押金在退房时退还。如果房屋没有损坏、水电费已结清，押金会在退房后7-30天内退还。"},
        {"q": "押金可以退吗？", "a": "可以退。押金在退房时退还，前提是房屋没有损坏、水电费已结清。"},
        {"q": "押金2400元什么时候退？", "a": "押金2400元在退房时退还，需要房屋无损坏、水电费已结清。"},
    ],
    
    # === 第5类：卫生费问题 ===
    "hygiene": [
        {"q": "卫生费可以退吗？", "a": "可以退。如果退房时房屋卫生干净，卫生费200元可以退还。"},
        {"q": "卫生费200元能退吗？", "a": "可以退。如果退房时房屋卫生干净，200元卫生费可以全额退还。"},
        {"q": "离租卫生干净退200元是什么意思？", "a": "意思是：如果退房时房屋打扫干净，200元卫生费可以退还。"},
    ],
    
    # === 第6类：综合问答 ===
    "comprehensive": [
        {"q": "退房时能退多少钱？", "a": "退房时可以退还押金2400元和卫生费200元（如果卫生干净），共2600元。"},
        {"q": "租房需要注意什么？", "a": "租房注意事项：1.查看房屋状况 2.确认租金和押金 3.签订正规合同 4.检查水电设施。"},
    ],
}


def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    send_telegram("🧠 *开始均衡训练*\n\n目标：解决过拟合问题，增加通用对话能力")
    
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True)
    
    # 冻结90%权重
    all_params = list(model.named_parameters())
    for i, (name, param) in enumerate(all_params):
        if i < int(len(all_params) * 0.9):
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数: {trainable/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    
    # 准备均衡数据
    all_data = []
    for category, items in TRAINING_DATA.items():
        all_data.extend(items)
        logger.info(f"  {category}: {len(items)}条")
    logger.info(f"总数据: {len(all_data)}条")
    
    # 训练
    best_loss = float('inf')
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        
        for item in all_data:
            prompt = f"问题：{item['q']}\n\n答案：{item['a']}"
            inputs = tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
            
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'].clone())
            
            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            
            total_loss += outputs.loss.item()
            del outputs
            gc.collect()
        
        avg_loss = total_loss / len(all_data)
        if avg_loss < best_loss:
            best_loss = avg_loss
        logger.info(f"Epoch {epoch+1}/5: Loss = {avg_loss:.4f}")
    
    # 测试
    logger.info("\n测试效果:")
    model.eval()
    test_qs = ["你好", "房租1600元20天，月租金是多少？", "押金什么时候可以退？", "卫生费可以退吗？"]
    
    results = []
    for q in test_qs:
        inputs = tokenizer(f"问题：{q}\n\n答案：", return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            out = model.generate(inputs['input_ids'], max_new_tokens=80, do_sample=False)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        if "答案：" in response:
            response = response.split("答案：")[-1]
        logger.info(f"Q: {q}\nA: {response}\n")
        results.append(f"Q: {q}\nA: {response}")
    
    # 保存
    os.makedirs(output_path, exist_ok=True)
    torch.save({n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}, 
               os.path.join(output_path, 'dynamic_weights.pt'))
    
    send_telegram(f"""✅ *均衡训练完成*

📉 Loss: {best_loss:.4f}
📊 数据: {len(all_data)}条

*测试结果:*
{chr(10).join(results[:2])}
""")
    
    logger.info("训练完成！")


if __name__ == "__main__":
    train()
