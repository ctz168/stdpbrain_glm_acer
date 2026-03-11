import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/Administrator/Desktop/stdpbrian_gra")
sys.path.insert(0, str(PROJECT_ROOT))

def test_engine():
    from core.truly_integrated_engine import TrulyIntegratedEngine
    model_path = str(PROJECT_ROOT / "weights/DeepSeek-R1-Distill-Qwen-1.5B")
    engine = TrulyIntegratedEngine(model_path)
    engine.initialize()

    questions = [
        "你好，我3月12日起租，3月份20天房租1600元。押金:两千四百元；卫生费200元。离租卫生干净退200元卫生费。合计2600元。那我的月租金是多少？",
        "卫生费怎样才能退？",
        "那我的押金是多少？"
    ]
    
    with open("c:/Users/Administrator/Desktop/stdpbrian_gra/test_log.txt", "w", encoding="utf-8") as f:
        f.write("=== 开始测试 ===\n")
        f.flush()
        for q in questions:
            f.write(f"\nUser: {q}\nAssistant: ")
            f.flush()
            for token in engine.generate_stream(q, max_new_tokens=200):
                f.write(token)
                f.flush()
            f.write("\n")
            f.flush()

if __name__ == "__main__":
    test_engine()
