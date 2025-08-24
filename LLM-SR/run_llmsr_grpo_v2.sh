#!/usr/bin/env bash
set -euo pipefail

# 🔥 v2 训练脚本（nohup 后台执行）- 大token长度优化版本

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
GPUS=${GPUS:-8}

PROBLEM_NAME=${PROBLEM_NAME:-"oscillator1"}
SPEC_PATH=${SPEC_PATH:-"./specs/specification_${PROBLEM_NAME}_numpy.txt"}
MODEL_PATH=${MODEL_PATH:-"/storage/home/westlakeLab/zhangjunlei/Qwen3-8B"}
OUT_DIR=${OUT_DIR:-"./llmsr_grpo_outputs/${PROBLEM_NAME}_v2"}

EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-4}
KL_COEF=${KL_COEF:-1e-3}

# 🔥 大token长度配置 - 基于46GB/81GB显存使用情况优化（v2增强版）
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}      # 提示长度：4K tokens
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}      # 生成长度：8K tokens  
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}       # 模型最大长度：16K tokens
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}  # 批量token数：8K

GRID_TRAIN=${GRID_TRAIN:-0}
NUM_GROUPS=${NUM_GROUPS:-10}
FEW_SHOT_K=${FEW_SHOT_K:-3}

LOG_DIR=${LOG_DIR:-"./llmsr_logs"}
mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG_FILE="${LOG_DIR}/grpo_v2_${PROBLEM_NAME}_$(date +%Y%m%d_%H%M%S).log"

# 将输出目录导出，便于奖励函数写入 sample.jsonl
export LLMSR_OUTPUT_DIR="${OUT_DIR}"

# 🔥 显存优化配置 - 适配大token长度
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# 🔥 vLLM 大token优化
export VLLM_USE_MODELSCOPE=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1800

# 设置 PYTHONPATH 确保能找到 verl 模块
export PYTHONPATH="/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/verl:/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR:$PYTHONPATH"

# 显示配置信息 - 大token长度优化版本
echo "🔥 v2 训练配置 (大Token优化版本 + 截断修复)"
echo "📋 基础配置:"
echo "  问题: $PROBLEM_NAME"
echo "  模型: Qwen3-8B (v2模式)"
echo "  模型路径: $MODEL_PATH"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LR"
echo "  组大小: $ROLLOUT_N"
echo "  GPU: ${CUDA_VISIBLE_DEVICES} ($GPUS 张卡)"
echo "🚀 Token长度配置 (大幅提升):"
echo "  提示长度: $MAX_PROMPT_LEN tokens"
echo "  生成长度: $MAX_NEW_TOKENS tokens"
echo "  模型最大长度: $MAX_MODEL_LEN tokens"
echo "  批量Token数: $MAX_NUM_BATCHED_TOKENS tokens"
echo "  输出目录: $OUT_DIR"
echo "  训练模式: 🔥 v2模式 - 大Token支持 + 截断修复"

CMD="python main.py \
  --use_rl_v2 \
  --problem_name ${PROBLEM_NAME} \
  --spec_path ${SPEC_PATH} \
  --model_path ${MODEL_PATH} \
  --epochs ${EPOCHS} \
  --learning_rate ${LR} \
  --rollout_n ${ROLLOUT_N} \
  --gpus ${GPUS} \
  --output_dir ${OUT_DIR} \
  --kl_coef ${KL_COEF} \
  --max_prompt_length ${MAX_PROMPT_LEN} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --max_model_len ${MAX_MODEL_LEN} \
  --max_num_batched_tokens ${MAX_NUM_BATCHED_TOKENS} \
  --few_shot_k ${FEW_SHOT_K} \
  $( [[ "$GRID_TRAIN" == "1" ]] && echo "--grid_train_data --num_grid_groups ${NUM_GROUPS}" ) \
  --log_path ${LOG_FILE}"

echo "🚀 [NOHUP] 启动v2训练 (大Token优化版本)"
echo "📋 日志文件: ${LOG_FILE}"

nohup bash -lc "${CMD}" >> "$LOG_FILE" 2>&1 &
GRPO_PID=$!

echo "✅ GRPO v2 训练已启动 (PID: $GRPO_PID)"
echo "📋 日志文件: ${LOG_FILE}"
echo "💡 监控命令: tail -f ${LOG_FILE}"
echo "🎯 大Token配置已启用 - 支持完整思考过程输出 + 截断恢复"

sleep 5
if ps -p $GRPO_PID > /dev/null 2>&1; then
  echo "✅ 训练进程存活"
else
  echo "❌ 训练进程未存活，请查看日志: ${LOG_FILE}"
  exit 1
fi


