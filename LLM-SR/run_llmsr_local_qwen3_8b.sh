#!/usr/bin/env bash
# ========================================================================
# LLM-SR 进化搜索脚本 - 使用本地 Qwen3-8B 模型
# 此脚本不使用RL，采用原版进化搜索方法，但使用本地Qwen3-8B模型
# ========================================================================

set -e  # 出错时停止执行

# 🔧 配置参数
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据您的GPU数量调整
export PORT_LLM=6000
export MODEL_PATH="/storage/home/westlakeLab/zhangjunlei/Qwen3-8B"

# 🌍 激活conda环境
echo "🔄 激活conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl_v2 || { echo "❌ conda环境激活失败"; exit 1; }

# 📁 创建日志目录
LOGDIR=./llmsr_logs
mkdir -p "${LOGDIR}"

echo "=========================================="
echo "🚀 LLM-SR 进化搜索 + 本地Qwen3-8B"
echo "=========================================="
echo "📍 模型路径: ${MODEL_PATH}"
echo "🌐 服务端口: ${PORT_LLM}"
echo "📂 日志目录: ${LOGDIR}"
echo "🎯 GPU设备: ${CUDA_VISIBLE_DEVICES}"

# ================================
# 1️⃣ 启动本地LLM服务器
# ================================
echo ""
echo "🔥 步骤1: 启动本地LLM服务器..."

# 检查模型路径是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ 错误: 模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# 启动LLM服务器
nohup python llm_engine/engine.py \
    --model_path "${MODEL_PATH}" \
    --host "127.0.0.2" \
    --port ${PORT_LLM} \
    --temperature 0.8 \
    --do_sample true \
    --max_new_tokens 10240 \
    --top_k 30 \
    --top_p 0.9 \
    > "${LOGDIR}/qwen3_8b_engine_${PORT_LLM}.log" 2>&1 &

ENGINE_PID=$!
echo "✅ LLM服务器已启动 (PID=${ENGINE_PID})"
echo "📋 服务器日志: ${LOGDIR}/qwen3_8b_engine_${PORT_LLM}.log"

# ⏰ 等待模型加载
echo ""
echo "⏳ 等待模型加载中..."
echo "   (Qwen3-8B需要一些时间来初始化，请耐心等待)"

# 检查服务器是否就绪
check_server_ready() {
    for i in {1..60}; do  # 最多等待10分钟
        if curl -s http://127.0.0.1:${PORT_LLM}/ > /dev/null 2>&1; then
            echo "✅ LLM服务器就绪！"
            return 0
        fi
        echo "   等待中... (${i}/60)"
        sleep 10
    done
    echo "❌ 超时: LLM服务器未能在10分钟内启动"
    return 1
}

if ! check_server_ready; then
    echo "❌ 请检查服务器日志: ${LOGDIR}/qwen3_8b_engine_${PORT_LLM}.log"
    kill ${ENGINE_PID} 2>/dev/null || true
    exit 1
fi

# ================================
# 2️⃣ 运行LLM-SR进化搜索实验
# ================================
echo ""
echo "🧬 步骤2: 开始LLM-SR进化搜索实验..."

# 🎯 oscillator1 实验
echo ""
echo "📊 运行 oscillator1 实验..."
python main.py \
    --problem_name oscillator1 \
    --spec_path ./specs/specification_oscillator1_numpy.txt \
    --log_path ./logs/oscillator1_qwen3_8b_evolution \
    2>&1 | tee "${LOGDIR}/oscillator1_qwen3_8b_evolution_$(date +%Y%m%d_%H%M%S).log"

# 🎯 oscillator2 实验  
echo ""
echo "📊 运行 oscillator2 实验..."
python main.py \
    --problem_name oscillator2 \
    --spec_path ./specs/specification_oscillator2_numpy.txt \
    --log_path ./logs/oscillator2_qwen3_8b_evolution \
    2>&1 | tee "${LOGDIR}/oscillator2_qwen3_8b_evolution_$(date +%Y%m%d_%H%M%S).log"

# 🎯 bacterial-growth 实验
echo ""
echo "📊 运行 bacterial-growth 实验..."
python main.py \
    --problem_name bactgrow \
    --spec_path ./specs/specification_bactgrow_numpy.txt \
    --log_path ./logs/bactgrow_qwen3_8b_evolution \
    2>&1 | tee "${LOGDIR}/bactgrow_qwen3_8b_evolution_$(date +%Y%m%d_%H%M%S).log"

# 🎯 stress-strain 实验
echo ""
echo "📊 运行 stress-strain 实验..."
python main.py \
    --problem_name stressstrain \
    --spec_path ./specs/specification_stressstrain_numpy.txt \
    --log_path ./logs/stressstrain_qwen3_8b_evolution \
    2>&1 | tee "${LOGDIR}/stressstrain_qwen3_8b_evolution_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "🎉 所有实验完成！"

# ================================
# 3️⃣ 清理：关闭LLM服务器
# ================================
echo ""
echo "🧹 关闭LLM服务器..."
kill ${ENGINE_PID} 2>/dev/null || true
echo "✅ 清理完成"

echo ""
echo "=========================================="
echo "✅ LLM-SR 进化搜索实验全部完成"
echo "📊 查看实验结果:"
echo "   - 日志目录: ${LOGDIR}/"
echo "   - 结果目录: ./logs/"
echo "=========================================="
