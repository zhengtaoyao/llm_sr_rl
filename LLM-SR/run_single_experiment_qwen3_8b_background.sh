#!/usr/bin/env bash
# ========================================================================
# LLM-SR 单个实验脚本 - 使用本地 Qwen3-8B 模型 (后台运行版本)
# 此脚本允许运行单个实验，采用进化搜索方法（非RL）
# 🔥 特点：终端关闭后实验继续运行
# 
# 用法:
#   ./run_single_experiment_qwen3_8b_background.sh oscillator1
#   ./run_single_experiment_qwen3_8b_background.sh oscillator2  
#   ./run_single_experiment_qwen3_8b_background.sh bactgrow
#   ./run_single_experiment_qwen3_8b_background.sh stressstrain
# ========================================================================

set -e  # 出错时停止执行

# 🔧 配置参数
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据您的GPU数量调整
export PORT_LLM=5011
export MODEL_PATH="/storage/home/westlakeLab/zhangjunlei/Qwen3-8B"

# 📋 检查命令行参数
if [ $# -eq 0 ]; then
    echo "❌ 错误: 请指定实验名称"
    echo ""
    echo "用法: $0 <实验名称>"
    echo ""
    echo "可用实验:"
    echo "  oscillator1   - 振荡器1实验"
    echo "  oscillator2   - 振荡器2实验" 
    echo "  bactgrow      - 细菌生长实验"
    echo "  stressstrain  - 应力应变实验"
    echo ""
    echo "示例: $0 oscillator1"
    exit 1
fi

EXPERIMENT_NAME=$1

# 🧪 实验配置映射
case ${EXPERIMENT_NAME} in
    "oscillator1")
        SPEC_PATH="./specs/specification_oscillator1_numpy.txt"
        DISPLAY_NAME="振荡器1"
        ;;
    "oscillator2")
        SPEC_PATH="./specs/specification_oscillator2_numpy.txt"
        DISPLAY_NAME="振荡器2"
        ;;
    "bactgrow")
        SPEC_PATH="./specs/specification_bactgrow_numpy.txt"
        DISPLAY_NAME="细菌生长"
        ;;
    "stressstrain")
        SPEC_PATH="./specs/specification_stressstrain_numpy.txt"
        DISPLAY_NAME="应力应变"
        ;;
    *)
        echo "❌ 错误: 未知实验名称 '${EXPERIMENT_NAME}'"
        echo "可用实验: oscillator1, oscillator2, bactgrow, stressstrain"
        exit 1
        ;;
esac

# 🌍 激活conda环境
echo "🔄 激活conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl_v2 || { echo "❌ conda环境激活失败"; exit 1; }

# 📁 创建日志目录
LOGDIR=./llmsr_logs
mkdir -p "${LOGDIR}"

echo "=========================================="
echo "🧬 LLM-SR 进化搜索 + 本地Qwen3-8B (后台模式)"
echo "🧪 实验: ${DISPLAY_NAME} (${EXPERIMENT_NAME})"
echo "=========================================="
echo "📍 模型路径: ${MODEL_PATH}"
echo "🌐 服务端口: ${PORT_LLM}"
echo "📄 规格文件: ${SPEC_PATH}"
echo "🎯 GPU设备: ${CUDA_VISIBLE_DEVICES}"

# 检查所需文件是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ 错误: 模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${SPEC_PATH}" ]; then
    echo "❌ 错误: 规格文件不存在: ${SPEC_PATH}"
    exit 1
fi

DATA_PATH="./data/${EXPERIMENT_NAME}/train.csv"
if [ ! -f "${DATA_PATH}" ]; then
    echo "❌ 错误: 数据文件不存在: ${DATA_PATH}"
    exit 1
fi

# ================================
# 1️⃣ 启动本地LLM服务器
# ================================
echo ""
echo "🔥 步骤1: 启动本地LLM服务器..."

# 检查端口是否被占用
if netstat -tuln | grep -q ":${PORT_LLM} "; then
    echo "⚠️  警告: 端口${PORT_LLM}已被占用，尝试终止现有进程..."
    pkill -f "engine.py.*port.*${PORT_LLM}" || true
    sleep 3
fi

# 启动LLM服务器 (后台 + nohup 双重保险)
ENGINE_LOG="${LOGDIR}/qwen3_8b_engine_${PORT_LLM}_$(date +%Y%m%d_%H%M%S).log"
nohup python llm_engine/engine.py \
    --model_path "${MODEL_PATH}" \
    --host "127.0.0.3" \
    --port ${PORT_LLM} \
    --temperature 0.8 \
    --do_sample true \
    --max_new_tokens 8192 \
    --top_k 30 \
    --top_p 0.9 \
    > "${ENGINE_LOG}" 2>&1 &

ENGINE_PID=$!
echo "✅ LLM服务器已启动 (PID=${ENGINE_PID})"
echo "📋 服务器日志: ${ENGINE_LOG}"

# 保存PID到文件，方便后续管理
echo ${ENGINE_PID} > "${LOGDIR}/engine_${PORT_LLM}.pid"
echo "💾 服务器PID已保存到: ${LOGDIR}/engine_${PORT_LLM}.pid"

# ⏰ 等待模型加载
echo ""
echo "⏳ 等待模型加载中..."
echo "   (Qwen3-8B需要几分钟来初始化，请耐心等待...)"

# 检查服务器是否就绪的函数
check_server_ready() {
    local max_attempts=60  # 最多等待10分钟 (120 * 10秒)
    for i in $(seq 1 ${max_attempts}); do
        if curl -s -m 5 "http://127.0.0.1:${PORT_LLM}/" > /dev/null 2>&1; then
            echo "✅ LLM服务器就绪！"
            return 0
        fi
        echo "   等待中... (${i}/${max_attempts}) - 正在加载Qwen3-8B模型"
        sleep 10
    done
    echo "❌ 超时: LLM服务器未能在10分钟内启动"
    return 1
}

if ! check_server_ready; then
    echo "❌ 请检查服务器日志以了解详情:"
    echo "   tail -f ${ENGINE_LOG}"
    kill ${ENGINE_PID} 2>/dev/null || true
    rm -f "${LOGDIR}/engine_${PORT_LLM}.pid"
    exit 1
fi

# 🧪 测试服务器响应
echo ""
echo "🧪 测试LLM服务器响应..."
TEST_RESPONSE=$(curl -s -X POST "http://127.0.0.1:${PORT_LLM}/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello, are you working?",
        "repeat_prompt": 1,
        "params": {
            "max_new_tokens": 50,
            "temperature": 0.8,
            "do_sample": true
        }
    }' | head -c 200)

if [ -n "${TEST_RESPONSE}" ]; then
    echo "✅ 服务器响应正常"
else
    echo "❌ 服务器响应异常"
    kill ${ENGINE_PID} 2>/dev/null || true
    rm -f "${LOGDIR}/engine_${PORT_LLM}.pid"
    exit 1
fi

# ================================
# 2️⃣ 运行LLM-SR进化搜索实验 (后台)
# ================================
echo ""
echo "🧬 步骤2: 在后台启动 ${DISPLAY_NAME} 进化搜索实验..."

EXP_LOG="${LOGDIR}/${EXPERIMENT_NAME}_qwen3_8b_evolution_$(date +%Y%m%d_%H%M%S).log"

# 🔥 使用 nohup 让主实验也在后台运行
nohup python main.py \
    --problem_name ${EXPERIMENT_NAME} \
    --spec_path ${SPEC_PATH} \
    --log_path ./logs/${EXPERIMENT_NAME}_qwen3_8b_evolution \
    > "${EXP_LOG}" 2>&1 &

EXP_PID=$!

# 保存实验PID
echo ${EXP_PID} > "${LOGDIR}/experiment_${EXPERIMENT_NAME}.pid"

echo "🎯 实验已在后台启动！"
echo "📊 实验进程PID: ${EXP_PID}"
echo "📋 实验日志: ${EXP_LOG}"
echo "💾 实验PID已保存到: ${LOGDIR}/experiment_${EXPERIMENT_NAME}.pid"

echo ""
echo "=========================================="
echo "🎉 后台实验启动成功！"
echo "=========================================="
echo "📊 监控信息:"
echo "   LLM服务器PID: ${ENGINE_PID} (日志: ${ENGINE_LOG})"
echo "   实验进程PID: ${EXP_PID} (日志: ${EXP_LOG})"
echo ""
echo "📋 实时查看实验进度:"
echo "   tail -f ${EXP_LOG}"
echo ""
echo "🔍 检查进程状态:"
echo "   ps aux | grep ${EXP_PID}"
echo "   ps aux | grep ${ENGINE_PID}"
echo ""
echo "🛑 如需停止实验:"
echo "   kill ${EXP_PID}  # 停止实验"
echo "   kill ${ENGINE_PID}  # 停止LLM服务器"
echo ""
echo "   或使用保存的PID文件:"
echo "   kill \$(cat ${LOGDIR}/experiment_${EXPERIMENT_NAME}.pid)"
echo "   kill \$(cat ${LOGDIR}/engine_${PORT_LLM}.pid)"
echo ""
echo "✅ 您现在可以安全关闭终端，实验将继续在后台运行！"
echo "=========================================="
