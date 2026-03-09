#!/usr/bin/env python3
"""
下载Qwen3.5-0.8B模型权重
Download Qwen3.5-0.8B model weights
"""

import os
import sys

# 设置缓存目录
os.environ['HF_HOME'] = '/home/z/my-project/download/brain_like_ai/hf_cache'

from huggingface_hub import snapshot_download

MODEL_DIR = "/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B"

def download_model():
    """下载模型"""
    print("=" * 60)
    print("开始下载 Qwen3.5-0.8B 模型")
    print("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id="Qwen/Qwen3-0.8B",
            local_dir=MODEL_DIR,
        )
        
        print("\n" + "=" * 60)
        print("模型下载完成！")
        print(f"模型路径: {MODEL_DIR}")
        print("=" * 60)
        
        # 列出文件
        files = os.listdir(MODEL_DIR)
        total_size = 0
        print(f"\n下载的文件 ({len(files)}):")
        for f in sorted(files):
            filepath = os.path.join(MODEL_DIR, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath) / (1024 * 1024)
                total_size += size
                print(f"  - {f} ({size:.2f} MB)")
        
        print(f"\n总大小: {total_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
