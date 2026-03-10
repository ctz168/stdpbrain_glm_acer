"""
类人脑双系统全闭环AI架构 - Telegram Bot服务 (完整集成版V2)
"""

import asyncio
import os
import sys
import logging
from typing import Optional
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8534413276:AAHzqgxVTOL2fapd8NV7UjppF4NXr1zSUek"
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "Qwen3.5-0.8B")
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from core.complete_integrated_engine import CompleteIntegratedEngine, BrainLikeConfig
        config = BrainLikeConfig()
        _engine = CompleteIntegratedEngine(MODEL_PATH, config)
    return _engine


async def run_bot():
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes
    )
    from telegram.constants import ParseMode
    
    engine = get_engine()
    
    logger.info("正在初始化完整集成引擎...")
    if not engine.initialize():
        logger.error("引擎初始化失败")
        return
    
    logger.info("引擎初始化成功！")
    
    stats = engine.get_statistics()
    logger.info(f"引擎统计: {stats}")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 *类人脑双系统AI架构 (完整集成版V2)*\n\n"
            "核心模块：\n"
            "• 100Hz高刷新 - 每10ms推理周期\n"
            "• STDP在线学习 - 边推理边更新\n"
            "• 海马体记忆 - 长期记忆存储\n"
            "• 自优化闭环 - 三种模式自动切换\n"
            "• 输入预处理 - 提取关键信息\n\n"
            "命令：\n"
            "/start - 开始\n"
            "/stats - 统计\n"
            "/clear - 清空记忆",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = engine.get_statistics()
        
        text = "📊 *系统统计*\n\n"
        text += f"*总周期数:* {stats.get('cycle_count', 0)}\n\n"
        
        if 'stdp' in stats:
            stdp = stats['stdp']
            text += f"*STDP学习:*\n"
            text += f"• 更新次数: {stdp.get('total_updates', 0)}\n"
            text += f"• LTP: {stdp.get('ltp_count', 0)}\n"
            text += f"• LTD: {stdp.get('ltd_count', 0)}\n\n"
        
        if 'hippocampus' in stats:
            hc = stats['hippocampus']
            text += f"*海马体记忆:*\n"
            text += f"• 记忆数量: {hc.get('memory_count', 0)}\n"
            text += f"• 编码次数: {hc.get('encode_count', 0)}\n\n"
        
        if 'self_optimization' in stats:
            opt = stats['self_optimization']
            text += f"*自优化闭环:*\n"
            text += f"• 自生成: {opt.get('generation_count', 0)}\n"
            text += f"• 自博弈: {opt.get('play_count', 0)}\n"
            text += f"• 自评判: {opt.get('judgment_count', 0)}\n"
        
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    
    async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        engine.clear_memory()
        await update.message.reply_text("✅ 记忆已清空！")
    
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message = update.message.text
        
        if not user_message:
            return
        
        await update.message.chat.send_action("typing")
        
        try:
            response_text = ""
            chunk_size = 20
            last_sent_len = 0
            message = None
            
            for token in engine.generate_stream(user_message, max_new_tokens=200):
                response_text += token
                
                if len(response_text) - last_sent_len >= chunk_size:
                    try:
                        if message is None:
                            message = await update.message.reply_text(response_text)
                        else:
                            await message.edit_text(response_text)
                        last_sent_len = len(response_text)
                    except Exception:
                        pass
            
            if response_text:
                if message is None:
                    await update.message.reply_text(response_text)
                else:
                    try:
                        await message.edit_text(response_text)
                    except Exception:
                        await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("抱歉，我无法生成回复。")
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            import traceback
            traceback.print_exc()
            await update.message.reply_text(f"❌ 处理失败: {str(e)}")
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("🧠 类人脑AI Telegram Bot (完整集成版V2) 启动中...")
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
    logger.info("Bot启动成功！等待消息...")
    
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
    print("=" * 60)
    print("🧠 类人脑双系统AI架构 - Telegram Bot (完整集成版V2)")
    print("=" * 60)
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 60)
    
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
