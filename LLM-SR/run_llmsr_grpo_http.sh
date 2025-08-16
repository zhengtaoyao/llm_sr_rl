#!/usr/bin/env bash

# 🔧 设置可见的GPU卡为5和6
export CUDA_VISIBLE_DEVICES=5,6

# 检查HTTP服务是否运行
if ! curl -s http://localhost:5000/ > /dev/null; then
    echo "❌ HTTP LLM service not running. Please start it first:"
    echo "   ./run_llmsr_engine.sh"
    exit 1
fi

echo "✅ HTTP LLM service is running"
echo "🚀 Starting GRPO training with Mixtral HTTP backend..."
echo "🎯 Using GPU cards: 5,6"

PROBLEM_NAME=${PROBLEM_NAME:-oscillator1}

python main.py \
    --use_rl \
    --use_http \
    --problem_name "${PROBLEM_NAME}" \
    --spec_path "./specs/specification_${PROBLEM_NAME}_numpy.txt" \
    --http_url "${HTTP_URL:-http://localhost:5000}" \
    --tokenizer_path "${TOKENIZER_PATH:-mistralai/Mixtral-8x7B-Instruct-v0.1}" \
    --epochs "${EPOCHS:-5}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --learning_rate "${LEARNING_RATE:-1e-6}" \
    --rollout_n "${ROLLOUT_N:-3}" \
    --gpus 2  # 🔥 修改为2卡
