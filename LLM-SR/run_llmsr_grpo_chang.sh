#!/usr/bin/env bash
# ========================================
# LLM-SR GRPO Training Launch Script
# ========================================
# 🔥 这个脚本默认使用直连模式启动LLM-SR GRPO训练
# 直连模式会真正微调LLM权重，而不是仅进行策略优化

set -e  # Exit on any error

# 🔥 训练模式配置
export WANDB_API_KEY="0824c860323f310aa17f7f55675f94200d116cfd"
export WANDB_PROJECT=${WANDB_PROJECT:-"llm_sr_grpo_direct"}
export WANDB_ENTITY="changma"
export WANDB_MODE="online"  # 可选值: online

TRAINING_MODE=${TRAINING_MODE:-"direct"}  # direct: 直连模式(微调权重) | http: HTTP模式(不更新权重)

# 默认配置
PROBLEM_NAME=${PROBLEM_NAME:-"oscillator1"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-16}  # 🔥 减小batch size以适应直连模式
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-5}
GPUS=${GPUS:-4}  # 🔥 默认4张GPU进行FSDP训练

# 🔥 直连模式优化参数
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # 每个GPU的micro batch size
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}    # PPO mini batch size
SAVE_FREQ=${SAVE_FREQ:-2}                # 每2个epoch保存一次

# HTTP模式配置（如果需要）
HTTP_URL=${HTTP_URL:-"http://localhost:5000"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}

# 路径配置
SPEC_PATH="./specs/specification_${PROBLEM_NAME}_numpy.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./llmsr_grpo_outputs/${PROBLEM_NAME}_${TRAINING_MODE}_${TIMESTAMP}"
LOG_DIR="./llmsr_logs"
LOG_FILE="${LOG_DIR}/grpo_${PROBLEM_NAME}_${TRAINING_MODE}_${TIMESTAMP}.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔥 LLM-SR GRPO Training (直连模式)${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查必要文件
if [[ ! -f "$SPEC_PATH" ]]; then
    echo -e "${RED}❌ 错误: 未找到规范文件: $SPEC_PATH${NC}"
    echo -e "${YELLOW}可用的规范文件:${NC}"
    ls -1 ./specs/specification_*_numpy.txt 2>/dev/null || echo "未找到"
    exit 1
fi

if [[ ! -f "./data/${PROBLEM_NAME}/train.csv" ]]; then
    echo -e "${RED}❌ 错误: 未找到数据文件: ./data/${PROBLEM_NAME}/train.csv${NC}"
    echo -e "${YELLOW}可用的数据集:${NC}"
    ls -1 ./data/*/train.csv 2>/dev/null || echo "未找到"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 🔥 显示训练模式信息
echo -e "${PURPLE}🔥 训练模式: ${TRAINING_MODE}${NC}"
if [[ "$TRAINING_MODE" == "direct" ]]; then
    echo -e "${GREEN}✅ 直连模式 - Actor进程直接加载模型，通过FSDP真正微调权重${NC}"
elif [[ "$TRAINING_MODE" == "http" ]]; then
    echo -e "${YELLOW}⚠️  HTTP模式 - 通过HTTP调用外部LLM，权重不会更新${NC}"
    echo -e "${YELLOW}   如需微调权重，请设置 TRAINING_MODE=direct${NC}"
else
    echo -e "${RED}❌ 错误: 不支持的训练模式: $TRAINING_MODE${NC}"
    echo -e "${YELLOW}支持的模式: direct, http${NC}"
    exit 1
fi

# 显示配置
echo -e "${GREEN}📋 配置信息:${NC}"
echo -e "  问题: ${YELLOW}$PROBLEM_NAME${NC}"
echo -e "  模型: ${YELLOW}$MODEL_PATH${NC}"
echo -e "  训练轮数: ${YELLOW}$EPOCHS${NC}"
echo -e "  批次大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  学习率: ${YELLOW}$LEARNING_RATE${NC}"
echo -e "  组大小: ${YELLOW}$ROLLOUT_N${NC}"
echo -e "  GPU数量: ${YELLOW}$GPUS${NC}"
echo -e "  Micro批次大小: ${YELLOW}$MICRO_BATCH_SIZE${NC}"
echo -e "  Mini批次大小: ${YELLOW}$MINI_BATCH_SIZE${NC}"
echo -e "  输出目录: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "llmsr" ]]; then
    echo -e "${YELLOW}⚠️  警告: conda环境 'llmsr' 未激活${NC}"
    echo -e "   请运行: conda activate llmsr"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查VERL安装
echo -e "${BLUE}🔍 检查VERL安装...${NC}"
python -c "import verl; print('✅ VERL已安装')" 2>/dev/null || {
    echo -e "${RED}❌ 未找到VERL，请先安装VERL${NC}"
    echo -e "${YELLOW}安装指南: https://github.com/volcengine/verl${NC}"
    exit 1
}

# 检查GPU可用性
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}🎮 找到 $GPU_COUNT 张GPU${NC}"
    if [[ $GPUS -gt $GPU_COUNT ]]; then
        echo -e "${YELLOW}⚠️  警告: 请求 $GPUS 张GPU，但只有 $GPU_COUNT 张可用${NC}"
        echo -e "   将使用所有可用GPU: $GPU_COUNT 张"
        GPUS=$GPU_COUNT
    fi
    
    # 显示GPU信息
    echo -e "${BLUE}💻 GPU信息:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | head -n $GPUS
else
    echo -e "${YELLOW}⚠️  警告: 未找到nvidia-smi，跳过GPU检测${NC}"
fi

# 🔥 设置环境变量进行内存优化
echo -e "${BLUE}🔧 设置训练环境变量...${NC}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)))
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
# 🔥 修复 vLLM 内存池兼容性问题：移除 expandable_segments 配置
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.6"

# 🔥 FSDP优化环境变量
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo -e "${GREEN}✅ 环境变量设置完成${NC}"
echo -e "  CUDA_VISIBLE_DEVICES: ${YELLOW}$CUDA_VISIBLE_DEVICES${NC}"
echo -e "  PYTORCH_CUDA_ALLOC_CONF: ${YELLOW}$PYTORCH_CUDA_ALLOC_CONF${NC}"

echo -e "${BLUE}🚀 启动GRPO训练...${NC}"
echo -e "${YELLOW}日志文件: ${LOG_FILE}${NC}"
echo ""

# 🔥 根据训练模式构建不同的命令
if [[ "$TRAINING_MODE" == "direct" ]]; then
    # 直连模式：真正微调权重
    echo -e "${GREEN}🔥 启动直连模式训练 - 真正微调LLM权重${NC}"
    python main.py \
        --use_rl \
        --rl_mode "direct" \
        --problem_name "$PROBLEM_NAME" \
        --spec_path "$SPEC_PATH" \
        --model_path "$MODEL_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --micro_batch_size "$MICRO_BATCH_SIZE" \
        --mini_batch_size "$MINI_BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --rollout_n "$ROLLOUT_N" \
        --gpus "$GPUS" \
        --save_freq "$SAVE_FREQ" \
        --output_dir "$OUTPUT_DIR" \
        --log_path "$LOG_FILE" \
elif [[ "$TRAINING_MODE" == "http" ]]; then
    # HTTP模式：不更新权重
    echo -e "${YELLOW}🌐 启动HTTP模式训练 - 权重不会更新${NC}"
    python main.py \
        --use_rl \
        --rl_mode "http" \
        --use_http \
        --http_url "$HTTP_URL" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --problem_name "$PROBLEM_NAME" \
        --spec_path "$SPEC_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --micro_batch_size "$MICRO_BATCH_SIZE" \
        --mini_batch_size "$MINI_BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --rollout_n "$ROLLOUT_N" \
        --gpus "$GPUS" \
        --output_dir "$OUTPUT_DIR" \
        --log_path "$LOG_FILE" \
fi

# 检查训练结果
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ 训练完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}📁 结果保存到: $OUTPUT_DIR${NC}"
    echo -e "${YELLOW}📋 日志文件: ${LOG_FILE}${NC}"
    
    # 显示结果统计
    if [[ -f "$OUTPUT_DIR/grpo_config_direct.yaml" ]]; then
        echo -e "${BLUE}📊 直连模式训练总结:${NC}"
        echo -e "  配置文件: $OUTPUT_DIR/grpo_config_direct.yaml"
        echo -e "  数据集: $OUTPUT_DIR/llmsr_train.parquet"
        echo -e "  奖励函数: $OUTPUT_DIR/llmsr_reward.py"
        echo -e "${GREEN}🔥 模型权重已通过FSDP微调更新！${NC}"
    elif [[ -f "$OUTPUT_DIR/grpo_config_http.yaml" ]]; then
        echo -e "${BLUE}📊 HTTP模式训练总结:${NC}"
        echo -e "  配置文件: $OUTPUT_DIR/grpo_config_http.yaml"
        echo -e "  数据集: $OUTPUT_DIR/llmsr_train.parquet"
        echo -e "  奖励函数: $OUTPUT_DIR/llmsr_reward.py"
        echo -e "${YELLOW}⚠️  注意: HTTP模式下权重未更新${NC}"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ 训练失败！${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${YELLOW}📋 检查日志文件: ${LOG_FILE}${NC}"
    echo -e "${YELLOW}🔧 常见问题:${NC}"
    echo -e "  1. GPU内存不足 - 尝试减小batch_size或micro_batch_size"
    echo -e "  2. 模型下载失败 - 检查网络连接或使用本地模型"
    echo -e "  3. VERL配置错误 - 检查VERL安装和版本"
    exit 1
fi 