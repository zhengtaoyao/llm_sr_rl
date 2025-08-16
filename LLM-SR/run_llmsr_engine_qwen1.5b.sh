#!/usr/bin/env bash
# ---------------------------------
# Launch Qwen 1.5B LLM Engine
# Using GPU 0,1 (前两张卡)
# ---------------------------------

# 🔧 环境配置
export CUDA_VISIBLE_DEVICES=0,1          # 使用前两张卡
export PORT_LLM=5000
export HF_TOKEN=""                        # 如果需要的话设置HuggingFace token

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmsr || { echo "❌ Failed to activate llmsr environment"; exit 1; }

# 创建日志目录
LOGDIR=./llmsr_logs
mkdir -p "$LOGDIR"

echo "🚀 Starting Qwen 1.5B LLM Engine"
echo "🎯 Using GPU cards: 0,1"
echo "🤖 Model: Qwen/Qwen2.5-1.5B-Instruct"
echo "🌐 Port: $PORT_LLM"

# 启动LLM引擎
nohup python llm_engine/engine.py \
    --model_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --gpu_ids 0 1 \
    --host "0.0.0.0" \
    --port "$PORT_LLM" \
    --temperature 0.8 \
    --max_new_tokens 512 \
    --top_k 30 \
    --top_p 0.9 \
    --do_sample true \
    > "$LOGDIR/engine_qwen1.5b_${PORT_LLM}.log" 2>&1 &

ENGINE_PID=$!
echo "✅ Qwen 1.5B Engine started (PID=$ENGINE_PID)"
echo "📝 Log file: $LOGDIR/engine_qwen1.5b_${PORT_LLM}.log"

# 等待模型加载
echo "⏳ Waiting for model to load..."
sleep 30

# 测试连接
echo "🔍 Testing engine connection..."
if curl -s http://localhost:$PORT_LLM/ > /dev/null; then
    echo "✅ Engine is responding on port $PORT_LLM"
else
    echo "❌ Engine not responding. Check logs:"
    echo "   tail -f $LOGDIR/engine_qwen1.5b_${PORT_LLM}.log"
fi

echo "🎯 Engine ready for GRPO training!"
echo "💡 To stop the engine: kill $ENGINE_PID"