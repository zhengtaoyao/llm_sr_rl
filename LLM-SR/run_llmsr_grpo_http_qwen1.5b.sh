#!/usr/bin/env bash

# 🔧 设置可见的GPU卡为2和3（后两张卡）
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 🔧 设置内存优化环境变量
# 🔥 修复 vLLM 内存池兼容性问题：移除 expandable_segments 配置
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.6"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

# 检查HTTP服务是否运行
if ! curl -s http://localhost:5000/ > /dev/null; then
    echo "❌ HTTP LLM service not running. Please start it first:"
    echo "   ./run_llmsr_engine_qwen1.5b.sh"
    exit 1
fi

echo "✅ HTTP LLM service is running"
echo "🚀 Starting GRPO training with Qwen 1.5B HTTP backend..."
echo "🎯 Using GPU cards: 2,3,4,5"
echo "🤖 Model: Qwen/Qwen2.5-1.5B-Instruct"

PROBLEM_NAME=${PROBLEM_NAME:-oscillator1}

python main.py \
    --use_rl \
    --use_http \
    --problem_name "${PROBLEM_NAME}" \
    --spec_path "./specs/specification_${PROBLEM_NAME}_numpy.txt" \
    --http_url "${HTTP_URL:-http://localhost:5000}" \
    --tokenizer_path "${TOKENIZER_PATH:-Qwen/Qwen2.5-1.5B-Instruct}" \
    --epochs "${EPOCHS:-5}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --learning_rate "${LEARNING_RATE:-1e-6}" \
    --rollout_n "${ROLLOUT_N:-3}" \
    --gpus 4

echo "🏁 GRPO training completed or stopped"