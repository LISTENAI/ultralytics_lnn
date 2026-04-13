#!/bin/bash
# ============================================================
# 第三方依赖下载脚本
# 用于克隆 LISTENAI 工具链 (linger + thinker)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRDPARTY_DIR="$SCRIPT_DIR"

echo "============================================================"
echo "📦 克隆 LISTENAI 第三方依赖"
echo "============================================================"

# 克隆 linger (量化训练框架)
if [ -d "$THIRDPARTY_DIR/linger" ] && [ -n "$(ls -A $THIRDPARTY_DIR/linger 2>/dev/null)" ]; then
    echo "✅ linger 已存在，跳过克隆"
else
    echo "📥 克隆 linger (量化训练框架)..."
    git clone https://github.com/LISTENAI/linger.git "$THIRDPARTY_DIR/linger"
    echo "✅ linger 克隆完成"
fi

# 克隆 thinker (模型打包框架)
if [ -d "$THIRDPARTY_DIR/thinker" ] && [ -n "$(ls -A $THIRDPARTY_DIR/thinker 2>/dev/null)" ]; then
    echo "✅ thinker 已存在，跳过克隆"
else
    echo "📥 克隆 thinker (模型打包框架)..."
    git clone https://github.com/LISTENAI/thinker.git "$THIRDPARTY_DIR/thinker"
    echo "✅ thinker 克隆完成"
fi

echo ""
echo "============================================================"
echo "📋 安装说明"
echo "============================================================"
echo ""
echo "如需安装 linger:"
echo "  cd thirdparty/linger && pip install -e ."
echo ""
echo "如需安装 thinker:"
echo "  cd thirdparty/thinker && pip install -e ."
echo "  cd thirdparty/thinker/tools && pip install -e ."
echo ""
echo "============================================================"
echo "✅ 第三方依赖准备完成!"
echo "============================================================"