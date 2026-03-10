#!/bin/bash
# 24小时持续训练启动脚本

cd /home/z/my-project/download/brain_like_ai
source venv/bin/activate

echo "========================================"
echo "  启动24小时持续训练"
echo "========================================"
echo ""
echo "开始时间: $(date)"
echo "预计结束: $(date -d '+24 hours')"
echo ""
echo "日志文件: /tmp/training_24h.log"
echo ""
echo "查看进度: tail -f /tmp/training_24h.log"
echo ""
echo "========================================"

# 启动训练
nohup python training/continuous_24h.py > /tmp/training_24h_stdout.log 2>&1 &

echo "训练已启动，PID: $!"
echo ""
echo "使用以下命令查看进度:"
echo "  tail -f /tmp/training_24h.log"
echo ""
echo "使用以下命令检查状态:"
echo "  cat output/integrated_trained/training_report.json"
