#!/usr/bin/env python3
"""完整训练 - 优化版"""
import os, sys, json, gc, logging
from pathlib import Path
from datetime import datetime
import torch

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRAINING_DATA = [
    # 房租计算 - 核心数据，多重复
    {"q": "房租1600元租了20天，日租金是多少？", "r": "日租金=1600÷20=80元/天", "a": "日租金是80元/天。"},
    {"q": "房租1600元租了20天，月租金是多少？", "r": "日租金=1600÷20=80元/天\n月租金=80×30=2400元/月", "a": "月租金是2400元/月。"},
    {"q": "房租1600元20天，月租金是多少？", "r": "日租金=1600÷20=80元/天\n月租金=80×30=2400元/月", "a": "月租金是2400元/月。"},
    {"q": "租期1年，3月12日起租，3月份20天房租1600元。押金2400元，卫生费200元。月租金是多少？", "r": "关键信息：房租1600元，20天\n日租金=1600÷20=80元/天\n月租金=80×30=2400元/月", "a": "月租金是2400元/月。"},
    {"q": "房租1600元20天，押金2400元，卫生费200元，合计2600元。月租金是多少？", "r": "关键信息：房租1600元，20天\n日租金=1600÷20=80元/天\n月租金=80×30=2400元/月", "a": "月租金是2400元/月。"},
    # 押金问题
    {"q": "押金什么时候可以退？", "r": "押金在退房时退还，需要房屋无损坏、结清费用", "a": "押金在退房时退还，需要房屋无损坏、结清所有费用。"},
    {"q": "押金2400元什么时候退？", "r": "押金2400元在退房时退还", "a": "押金2400元在退房时退还，需要房屋无损坏。"},
    {"q": "押金可以退吗？", "r": "押金可以退，在退房时退还", "a": "可以退。押金在退房时退还，需要房屋无损坏。"},
    {"q": "押金两千四百元能退吗？", "r": "押金2400元可以退，在退房时退还", "a": "可以退。押金2400元在退房时退还。"},
    # 卫生费问题
    {"q": "卫生费可以退吗？", "r": "卫生费200元，离租卫生干净退200元", "a": "可以退。如果离租时房屋卫生干净，可以退还200元卫生费。"},
    {"q": "卫生费200元能退吗？", "r": "卫生费200元，离租卫生干净退200元", "a": "可以退。如果离租时房屋卫生干净，可以退还200元卫生费。"},
    {"q": "离租卫生干净退200元卫生费是什么意思？", "r": "意思是退房时如果房屋打扫干净，可以退还200元卫生费", "a": "意思是退房时如果房屋打扫干净，可以退还200元卫生费。"},
    # 其他问题
    {"q": "租期是多久？", "r": "租期1年，3月12日起租", "a": "租期是1年，从3月12日起租。"},
    {"q": "起租日期是什么时候？", "r": "起租日期是3月12日", "a": "起租日期是3月12日。"},
    {"q": "你好", "r": "用户打招呼", "a": "你好！我是类人脑AI助手，有什么可以帮助你的吗？"},
    {"q": "谢谢", "r": "用户感谢", "a": "不客气！有其他问题随时问我。"},
]

def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    logger.info(f"训练数据: {len(TRAINING_DATA)}条")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    
    all_params = list(model.named_parameters())
    for i, (name, param) in enumerate(all_params):
        if i < int(len(all_params) * 0.9):
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数: {trainable/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    
    best_loss = float('inf')
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for item in TRAINING_DATA:
            prompt = f"问题：{item['q']}\n\n思考：{item['r']}\n\n答案：{item['a']}"
            inputs = tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'].clone())
            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += outputs.loss.item()
            del outputs
            gc.collect()
        avg_loss = total_loss / len(TRAINING_DATA)
        if avg_loss < best_loss:
            best_loss = avg_loss
        logger.info(f"Epoch {epoch+1}/10: Loss = {avg_loss:.4f}, Best = {best_loss:.4f}")
    
    # 测试
    logger.info("\n测试:")
    model.eval()
    test_qs = ["押金什么时候可以退？", "卫生费可以退吗？", "房租1600元20天，月租金是多少？"]
    for q in test_qs:
        inputs = tokenizer(f"问题：{q}\n\n请回答：", return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            out = model.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
        logger.info(f"Q: {q}")
        logger.info(f"A: {tokenizer.decode(out[0], skip_special_tokens=True)}\n")
    
    # 保存
    os.makedirs(output_path, exist_ok=True)
    torch.save({n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}, os.path.join(output_path, 'dynamic_weights.pt'))
    tokenizer.save_pretrained(output_path)
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump({'training_time': datetime.now().isoformat(), 'data_count': len(TRAINING_DATA), 'best_loss': best_loss}, f)
    logger.info("训练完成！")

if __name__ == "__main__":
    train()
