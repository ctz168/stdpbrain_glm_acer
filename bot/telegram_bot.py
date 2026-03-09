"""
类人脑双系统全闭环AI架构 - Telegram Bot服务
Human-Like Brain Dual-System Full-Loop AI Architecture - Telegram Bot Service

生产级实现 - 整合所有核心模块，支持流式输出
"""

import asyncio
import os
import sys
import logging
from typing import Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入引擎
from core.engine import BrainLikeAIEngine, GenerationConfig

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8534413276:AAHzqgxVTOL2fapd8NV7UjppF4NXr1zSUek"
)

# 模型路径
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "Qwen3.5-0.8B")
)

# 全局引擎实例
_engine: Optional[BrainLikeAIEngine] = None


def get_engine() -> BrainLikeAIEngine:
    """获取引擎实例"""
    global _engine
    if _engine is None:
        _engine = BrainLikeAIEngine(MODEL_PATH)
    return _engine


async def run_bot():
    """运行Telegram Bot"""
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            MessageHandler,
            filters,
            ContextTypes
        )
        from telegram.constants import ParseMode
    except ImportError:
        logger.error("请安装 python-telegram-bot: pip install python-telegram-bot")
        return
    
    # 初始化引擎
    engine = get_engine()
    
    logger.info("正在初始化引擎...")
    if not engine.initialize():
        logger.error("引擎初始化失败")
        return
    
    logger.info("引擎初始化成功！")
    
    # 创建应用
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # /start 命令
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 *欢迎使用类人脑双系统AI架构！*\n\n"
            "我是基于Qwen大模型的类脑AI助手。\n\n"
            "核心特性：\n"
            "• 海马体记忆系统 - 长期记忆存储与召回\n"
            "• STDP在线学习 - 时序可塑性权重更新\n"
            "• 自闭环优化 - 自生成、自博弈、自评判\n"
            "• 流式输出 - 实时响应\n\n"
            "命令：\n"
            "/start - 开始使用\n"
            "/help - 帮助信息\n"
            "/stats - 系统统计\n"
            "/clear - 清空记忆\n"
            "/consolidate - 离线巩固\n\n"
            "发送任意消息开始对话！",
            parse_mode=ParseMode.MARKDOWN
        )
    
    # /help 命令
    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """🧠 *类人脑AI使用指南*

*核心架构：*
• 海马体记忆系统 (EC/DG/CA3/CA1/SWR)
• STDP时序可塑性学习
• 100Hz高刷新推理引擎
• 自闭环优化系统

*命令列表：*
/start - 开始使用
/help - 显示帮助
/stats - 系统统计
/clear - 清空记忆
/consolidate - 离线巩固

*特性说明：*
• 记忆系统会自动存储对话
• 可以召回相关历史记忆
• 支持在线学习优化
• 流式输出实时响应

直接发送消息即可开始对话！"""
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    # /stats 命令
    async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = engine.get_statistics()
        
        stats_text = f"""📊 *系统统计*

*基本信息：*
• 初始化状态: {'✅' if stats['initialized'] else '❌'}
• 设备: {stats['device']}
• 生成次数: {stats['generation_count']}

*海马体系统：*
"""
        if 'hippocampus' in stats:
            hc = stats['hippocampus']
            stats_text += f"• 编码次数: {hc.get('encode_count', 0)}\n"
            stats_text += f"• 召回次数: {hc.get('recall_count', 0)}\n"
            stats_text += f"• 记忆数量: {hc.get('memory_count', 0)}\n"
        
        if 'stdp' in stats:
            stdp = stats['stdp']
            stats_text += f"\n*STDP系统：*\n"
            stats_text += f"• 总更新次数: {stdp.get('total_updates', 0)}\n"
            stats_text += f"• LTP次数: {stdp.get('ltp_count', 0)}\n"
            stats_text += f"• LTD次数: {stdp.get('ltd_count', 0)}\n"
        
        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
    
    # /clear 命令
    async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        engine.clear_memory()
        await update.message.reply_text("✅ 海马体记忆已清空！")
    
    # /consolidate 命令
    async def consolidate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        result = engine.offline_consolidation()
        status = result.get('status', 'unknown')
        
        if status == 'completed':
            text = f"""✅ *离线巩固完成*

• 回放记忆数: {result.get('replay_count', 0)}
• 修剪记忆数: {result.get('pruned_count', 0)}"""
        else:
            text = f"离线巩固状态: {status}"
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    
    # 消息处理
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message = update.message.text
        
        if not user_message:
            return
        
        # 发送"正在输入"状态
        await update.message.chat.send_action("typing")
        
        try:
            # 流式生成
            response_text = ""
            chunk_size = 20
            last_sent_len = 0
            message = None
            
            for token in engine.generate_stream(user_message):
                response_text += token
                
                # 当累积足够字符时更新消息
                if len(response_text) - last_sent_len >= chunk_size:
                    try:
                        if message is None:
                            message = await update.message.reply_text(response_text)
                        else:
                            await message.edit_text(response_text)
                        last_sent_len = len(response_text)
                    except Exception:
                        pass
            
            # 发送最终响应
            if response_text:
                if message is None:
                    await update.message.reply_text(response_text)
                else:
                    try:
                        await message.edit_text(response_text)
                    except Exception:
                        await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("抱歉，我无法生成回复。请稍后再试。")
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            import traceback
            traceback.print_exc()
            await update.message.reply_text(f"❌ 处理失败: {str(e)}")
    
    # 添加处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("consolidate", consolidate_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # 启动Bot
    logger.info("🧠 类人脑AI Telegram Bot 启动中...")
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
    logger.info("Bot启动成功！等待消息...")
    
    # 保持运行
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("正在停止Bot...")
        await application.updater.stop()
        await application.stop()


def main():
    """主入口"""
    print("=" * 60)
    print("🧠 类人脑双系统AI架构 - Telegram Bot")
    print("=" * 60)
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 60)
    
    # 运行Bot
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
