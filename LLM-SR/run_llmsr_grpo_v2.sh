#!/usr/bin/env bash
set -euo pipefail

# 🔥 v2 训练脚本（nohup 后台执行）- 大token长度优化版本

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
GPUS=${GPUS:-8}

PROBLEM_NAME=${PROBLEM_NAME:-"oscillator1"}
SPEC_PATH=${SPEC_PATH:-"./specs/specification_${PROBLEM_NAME}_numpy.txt"}
MODEL_PATH=${MODEL_PATH:-"/storage/home/westlakeLab/zhangjunlei/Qwen3-8B"}
# 🔥 修复输出目录命名，使其与v1版本一致包含时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${OUT_DIR:-"./llmsr_grpo_outputs/${PROBLEM_NAME}_qwen8b_v2_${TIMESTAMP}"}

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
# 若同名文件已存在且不是目录，则使用备用目录
if [[ -e "$LOG_DIR" && ! -d "$LOG_DIR" ]]; then
  ALT_DIR="./llmsr_logs_dir"
  echo -e "⚠️  检测到 ${LOG_DIR} 不是目录，切换到 ${ALT_DIR}"
  LOG_DIR="$ALT_DIR"
fi
mkdir -p "$LOG_DIR" "$OUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/grpo_v2_${PROBLEM_NAME}_qwen8b_${TIMESTAMP}.log"

# 🔥 将输出目录导出，便于奖励函数写入 sample.jsonl（与v1版本一致）
export LLMSR_OUTPUT_DIR="${OUT_DIR}"

# 🔥 显存优化配置 - 适配大token长度
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# 🔥 vLLM 大token优化
export VLLM_USE_MODELSCOPE=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1800

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔥 LLM-SR GRPO v2模式训练 (8卡)${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查必要文件
if [[ ! -f "$SPEC_PATH" ]]; then
    echo -e "${RED}❌ 错误: 未找到规范文件: $SPEC_PATH${NC}"
    exit 1
fi

if [[ ! -f "./data/${PROBLEM_NAME}/train.csv" ]]; then
    echo -e "${RED}❌ 错误: 未找到数据文件: ./data/${PROBLEM_NAME}/train.csv${NC}"
    exit 1
fi

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "verl" ]]; then
    echo -e "${YELLOW}⚠️  激活 conda 环境 (verl)...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate verl || { 
        echo -e "${RED}❌ 无法激活 verl 环境${NC}"; 
        exit 1; 
    }
fi

# 设置 PYTHONPATH 确保能找到 verl 模块
export PYTHONPATH="/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/verl:/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR:${PYTHONPATH:-}"

# 显示配置信息 - 大token长度优化版本
echo -e "${PURPLE}🔥 v2训练配置 (大Token优化 + 截断修复):${NC}"
echo -e "${GREEN}📋 基础配置:${NC}"
echo -e "  问题: ${YELLOW}$PROBLEM_NAME${NC}"
echo -e "  模型: ${YELLOW}Qwen3-8B (v2模式)${NC}"
echo -e "  模型路径: ${YELLOW}$MODEL_PATH${NC}"
echo -e "  训练轮数: ${YELLOW}$EPOCHS${NC}"
echo -e "  学习率: ${YELLOW}$LR${NC}"
echo -e "  组大小: ${YELLOW}$ROLLOUT_N${NC}"
echo -e "  GPU: ${YELLOW}${CUDA_VISIBLE_DEVICES} (${GPUS} 张卡)${NC}"
echo -e "${GREEN}🚀 Token长度配置 (大幅提升):${NC}"
echo -e "  提示长度: ${YELLOW}$MAX_PROMPT_LEN${NC} tokens"
echo -e "  生成长度: ${YELLOW}$MAX_NEW_TOKENS${NC} tokens"
echo -e "  模型最大长度: ${YELLOW}$MAX_MODEL_LEN${NC} tokens"
echo -e "  批量Token数: ${YELLOW}$MAX_NUM_BATCHED_TOKENS${NC} tokens"
echo -e "  输出目录: ${YELLOW}$OUT_DIR${NC}"
echo -e "  训练模式: ${YELLOW}🔥 v2模式 - 大Token支持 + 截断修复${NC}"

echo -e "${GREEN}✅ 环境变量设置完成${NC}"

# 检查训练 GPU 可用性
echo -e "${BLUE}🎮 检查训练 GPU 状态 (${GPUS}卡)...${NC}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits || true
else
  echo -e "${YELLOW}⚠️  未找到 nvidia-smi，跳过GPU状态检查${NC}"
fi

# 🔥 设置环境变量进行优化 - 大token长度优化
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=WARN

# NCCL 优化配置
NETWORK_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1 2>/dev/null || true)
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0
export NCCL_TIMEOUT=1800
export NCCL_TREE_THRESHOLD=0

# Ray/DeepSpeed 配置
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 以 nohup 后台运行
echo -e "${BLUE}🚀 启动 GRPO v2训练 (nohup 后台)...${NC}"
echo -e "${YELLOW}日志文件: ${LOG_FILE}${NC}"
CMD=(
  python main.py \
    --use_rl_v2 \
    --problem_name "$PROBLEM_NAME" \
    --spec_path "$SPEC_PATH" \
    --model_path "$MODEL_PATH" \
    --epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --rollout_n "$ROLLOUT_N" \
    --gpus "$GPUS" \
    --output_dir "$OUT_DIR" \
    --kl_coef "$KL_COEF" \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
    --few_shot_k "$FEW_SHOT_K" \
    --log_path "$LOG_FILE"
)

# 添加网格训练参数（如果启用）
if [[ "$GRID_TRAIN" == "1" ]]; then
  CMD+=(--grid_train_data --num_grid_groups "$NUM_GROUPS")
fi

echo -e "${BLUE}[NOHUP] ${CMD[*]}${NC}"
nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
GRPO_PID=$!

echo -e "${GREEN}✅ GRPO v2训练已启动 (PID: $GRPO_PID)${NC}"
echo -e "${YELLOW}📋 日志文件: ${LOG_FILE}${NC}"
echo -e "${YELLOW}💡 监控命令: tail -f ${LOG_FILE}${NC}"

echo -e "${BLUE}⏳ 等待 8 秒后检查进程...${NC}"
sleep 8
if ps -p $GRPO_PID > /dev/null 2>&1; then
  echo -e "${GREEN}✅ 训练进程存活 (PID: $GRPO_PID)${NC}"
else
  echo -e "${RED}❌ 训练进程未存活，请查看日志: ${LOG_FILE}${NC}"
  exit 1
fi


