"""
类人脑双系统全闭环AI架构 - 主入口
Human-Like Brain Dual-System Full-Loop AI Architecture - Main Entry

提供完整的API接口和命令行工具
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import BrainLikeAIEngine, GenerationConfig
from core.config import BrainLikeConfig


class BrainLikeAI:
    """
    类人脑双系统全闭环AI架构主类
    
    整合所有模块，提供统一的API接口
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        初始化
        
        Args:
            config_path: 配置文件路径
            model_path: 模型路径
            device: 设备类型（cuda/cpu）
        """
        # 加载配置
        if config_path and os.path.exists(config_path):
            self.config = BrainLikeConfig.load(config_path)
        else:
            self.config = BrainLikeConfig()
        
        # 模型路径
        self.model_path = model_path or os.environ.get(
            "MODEL_PATH",
            str(PROJECT_ROOT / "weights" / "DeepSeek-R1-Distill-Qwen-1.5B")
        )
        
        # 初始化引擎
        self.engine = BrainLikeAIEngine(
            model_path=self.model_path,
            device=device
        )
    
    def initialize(self) -> bool:
        """初始化引擎"""
        return self.engine.initialize()
    
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        生成文本
        
        Args:
            input_text: 输入文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            
        Returns:
            生成结果
        """
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        output = self.engine.generate(input_text, config)
        
        return {
            'input': input_text,
            'output': output,
            'config': {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p
            }
        }
    
    def generate_stream(self, input_text: str, **kwargs):
        """流式生成"""
        config = GenerationConfig(**kwargs)
        yield from self.engine.generate_stream(input_text, config)
    
    def train(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行训练
        
        Args:
            train_data_path: 训练数据路径
            eval_data_path: 评估数据路径
            
        Returns:
            训练结果
        """
        # TODO: 实现训练逻辑
        return {
            'status': 'not_implemented',
            'message': '训练功能开发中'
        }
    
    def evaluate(
        self,
        test_data_path: str,
        baseline_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行测评
        
        Args:
            test_data_path: 测试数据路径
            baseline_model_path: 基线模型路径
            
        Returns:
            测评结果
        """
        # TODO: 实现测评逻辑
        return {
            'status': 'not_implemented',
            'message': '测评功能开发中'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return self.engine.get_statistics()
    
    def clear_memory(self):
        """清空海马体记忆"""
        self.engine.clear_memory()
    
    def offline_consolidation(self) -> Dict[str, Any]:
        """执行离线记忆巩固"""
        return self.engine.offline_consolidation()
    
    def save(self, path: str):
        """保存模型和状态"""
        # TODO: 实现保存逻辑
        pass
    
    def load(self, path: str):
        """加载模型和状态"""
        # TODO: 实现加载逻辑
        pass


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='类人脑双系统全闭环AI架构'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 生成命令
    gen_parser = subparsers.add_parser('generate', help='生成文本')
    gen_parser.add_argument('--input', '-i', type=str, required=True, help='输入文本')
    gen_parser.add_argument('--max-tokens', '-m', type=int, default=512, help='最大生成token数')
    gen_parser.add_argument('--temperature', '-t', type=float, default=0.7, help='温度参数')
    gen_parser.add_argument('--model-path', type=str, help='模型路径')
    
    # 统计命令
    stats_parser = subparsers.add_parser('stats', help='显示统计信息')
    stats_parser.add_argument('--model-path', type=str, help='模型路径')
    
    # 清空记忆命令
    clear_parser = subparsers.add_parser('clear', help='清空记忆')
    clear_parser.add_argument('--model-path', type=str, help='模型路径')
    
    # 离线巩固命令
    consolidate_parser = subparsers.add_parser('consolidate', help='离线巩固')
    consolidate_parser.add_argument('--model-path', type=str, help='模型路径')
    
    # Bot命令
    bot_parser = subparsers.add_parser('bot', help='启动Telegram Bot')
    bot_parser.add_argument('--token', type=str, help='Telegram Bot Token')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 执行命令
    if args.command == 'generate':
        ai = BrainLikeAI(model_path=getattr(args, 'model_path', None))
        
        print(f"输入: {args.input}")
        print("输出: ", end="", flush=True)
        
        for text in ai.generate_stream(
            args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        ):
            print(text, end="", flush=True)
        print()
    
    elif args.command == 'stats':
        ai = BrainLikeAI(model_path=getattr(args, 'model_path', None))
        if ai.initialize():
            stats = ai.get_statistics()
            print(json.dumps(stats, indent=2, default=str))
    
    elif args.command == 'clear':
        ai = BrainLikeAI(model_path=getattr(args, 'model_path', None))
        ai.clear_memory()
        print("记忆已清空")
    
    elif args.command == 'consolidate':
        ai = BrainLikeAI(model_path=getattr(args, 'model_path', None))
        result = ai.offline_consolidation()
        print(json.dumps(result, indent=2))
    
    elif args.command == 'bot':
        if args.token:
            os.environ['TELEGRAM_BOT_TOKEN'] = args.token
        
        from bot.telegram_bot import main as bot_main
        bot_main()


if __name__ == '__main__':
    main()
