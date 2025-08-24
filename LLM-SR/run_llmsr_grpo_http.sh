#!/usr/bin/env bash

# 🔧 设置可见的GPU卡为5和6
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6}
export GPUS=${GPUS:-2}

# 🔥 修改为直连模式，不再依赖HTTP服务
echo "🚀 Starting GRPO training in DIRECT mode (no HTTP needed)..."
echo "🎯 Using GPU cards: ${CUDA_VISIBLE_DEVICES} (${GPUS} cards)"

PROBLEM_NAME=${PROBLEM_NAME:-oscillator1}
MODEL_PATH=${MODEL_PATH:-"/storage/home/westlakeLab/zhangjunlei/Qwen3-8B"}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-3}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./llmsr_grpo_outputs/${PROBLEM_NAME}_qwen8b_direct_${TIMESTAMP}"
LOG_DIR="./llmsr_logs"
LOG_FILE="${LOG_DIR}/grpo_direct_${PROBLEM_NAME}_qwen8b_${TIMESTAMP}.log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 将输出目录导出，便于奖励函数写入 sample.jsonl
export LLMSR_OUTPUT_DIR="${OUTPUT_DIR}"

echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📋 Log file: ${LOG_FILE}"

# 🔥 使用直连模式，真正微调权重
nohup python main.py \
    --use_rl \
    --problem_name "${PROBLEM_NAME}" \
    --spec_path "./specs/specification_${PROBLEM_NAME}_numpy.txt" \
    --model_path "${MODEL_PATH}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --rollout_n "${ROLLOUT_N}" \
    --gpus "${GPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_path "${LOG_FILE}" \
    >> "${LOG_FILE}" 2>&1 &

GRPO_PID=$!
echo "✅ GRPO 直连训练已启动 (PID: $GRPO_PID)"
echo "📋 日志文件: ${LOG_FILE}"
echo "💡 监控命令: tail -f ${LOG_FILE}"

sleep 5
if ps -p $GRPO_PID > /dev/null 2>&1; then
  echo "✅ 训练进程存活"
else
  echo "❌ 训练进程未存活，请查看日志: ${LOG_FILE}"
  exit 1
fi
