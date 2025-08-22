#!/usr/bin/env bash
# ========================================
# 🔥 LLM-SR GRPO 直连模式训练脚本 
# 使用GRPO替换LLM-SR的进化搜索，直接加载模型真正微调权重
# ========================================

set -e

# 🔧 GPU 配置：直连模式使用所有 8 张 GPU 进行训练
# 直连模式不需要单独的LLM服务GPU，Actor进程直接加载模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS=8

# 🎯 训练配置 - 参照HTTP模式但使用直连
PROBLEM_NAME=${PROBLEM_NAME:-"oscillator1"}
MODEL_PATH="/storage/home/westlakeLab/zhangjunlei/Qwen/Qwen2.5-Coder-7B-Instruct"
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-24}       # 24 * 6 = 144, 适合6卡训练
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-4}          # 4 个 rollout
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # 减少微批量大小

# 📁 路径配置 - 完全参照HTTP模式
SPEC_PATH="./specs/specification_${PROBLEM_NAME}_numpy.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./llmsr_grpo_outputs/${PROBLEM_NAME}_qwen7b_direct_${TIMESTAMP}"
LOG_DIR="./llmsr_logs"
LOG_FILE="${LOG_DIR}/grpo_direct_${PROBLEM_NAME}_qwen7b_${TIMESTAMP}.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔥 LLM-SR GRPO 直连模式训练 (8卡)${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查必要文件 - 参照HTTP模式
if [[ ! -f "$SPEC_PATH" ]]; then
    echo -e "${RED}❌ 错误: 未找到规范文件: $SPEC_PATH${NC}"
    exit 1
fi

if [[ ! -f "./data/${PROBLEM_NAME}/train.csv" ]]; then
    echo -e "${RED}❌ 错误: 未找到数据文件: ./data/${PROBLEM_NAME}/train.csv${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 显示配置信息 - 参照HTTP模式
echo -e "${PURPLE}🔥 训练配置 (6卡全参数微调):${NC}"
echo -e "${GREEN}📋 配置信息:${NC}"
echo -e "  问题: ${YELLOW}$PROBLEM_NAME${NC}"
echo -e "  模型: ${YELLOW}Qwen2.5-Coder-7B-Instruct (直连)${NC}"
echo -e "  模型路径: ${YELLOW}$MODEL_PATH${NC}"
echo -e "  训练轮数: ${YELLOW}$EPOCHS${NC}"
echo -e "  批次大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  学习率: ${YELLOW}$LEARNING_RATE${NC}"
echo -e "  组大小: ${YELLOW}$ROLLOUT_N${NC}"
echo -e "  GPU: ${YELLOW}0,1,2,3,4,5,6,7 (8张卡)${NC}"
echo -e "  输出目录: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "  训练模式: ${YELLOW}🔥 直连模式 - 真正微调权重${NC}"

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "verl" ]]; then
    echo -e "${YELLOW}⚠️  激活 conda 环境 (verl)...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate verl || { 
        echo -e "${RED}❌ 无法激活 verl 环境${NC}"; 
        exit 1; 
    }
fi

# 检查训练 GPU 可用性
echo -e "${BLUE}🎮 检查训练 GPU 状态 (8卡)...${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# 🔥 设置环境变量进行优化
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=WARN
# 🔥 修复 vLLM 内存池兼容性问题：移除 expandable_segments 配置
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.6"

# NCCL 优化配置 - 参照HTTP模式
NETWORK_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
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

echo -e "${GREEN}✅ 环境变量设置完成${NC}"



# 🚀 启动 GRPO 直连模式训练 - 参照HTTP模式调用main.py
echo -e "${BLUE}🚀 启动 GRPO 直连训练 (后台运行)...${NC}"
echo -e "${YELLOW}日志文件: ${LOG_FILE}${NC}"
echo -e "${GREEN}✅ 使用直连模式 - 真正微调权重${NC}"
echo -e "${GREEN}✅ 不依赖HTTP服务，直接加载模型${NC}"

# 🔥 使用和HTTP模式相同的main.py调用，但不使用--use_http
nohup python main.py \
    --use_rl \
    --problem_name "$PROBLEM_NAME" \
    --spec_path "$SPEC_PATH" \
    --model_path "$MODEL_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --rollout_n "$ROLLOUT_N" \
    --gpus "$GPUS" \
    --output_dir "$OUTPUT_DIR" \
    --log_path "$LOG_FILE" \
    > "${LOG_FILE}" 2>&1 &

GRPO_PID=$!
echo -e "${GREEN}✅ GRPO 直连训练已启动 (PID: $GRPO_PID)${NC}"
echo -e "${YELLOW}📋 日志文件: ${LOG_FILE}${NC}"
echo -e "${YELLOW}💡 监控命令: tail -f ${LOG_FILE}${NC}"
echo -e "${YELLOW}🛑 停止命令: kill $GRPO_PID${NC}"

# 等待训练启动
echo -e "${BLUE}⏳ 等待训练启动...${NC}"
sleep 10

# 检查进程状态
if ps -p $GRPO_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ GRPO 直连训练正常运行中${NC}"
    echo -e "${BLUE}📊 进程状态:${NC}"
    echo -e "  🔧 PID: $GRPO_PID"
    echo -e "  📋 日志: ${LOG_FILE}"
    echo -e "  🎮 GPU: 0,1,2,3,4,5,6,7"
    echo -e "  ⚡ 模式: 直连模式 - 真正微调权重"
    echo -e "  ⏰ 预计训练时间: ${EPOCHS} 轮次"
    
    # 等待训练完成
    wait $GRPO_PID
    TRAIN_STATUS=$?
    
    if [[ $TRAIN_STATUS -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✅ GRPO 直连训练完成！${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo -e "${YELLOW}📁 结果保存到: $OUTPUT_DIR${NC}"
        echo -e "${YELLOW}📋 日志文件: ${LOG_FILE}${NC}"
        echo -e "${GREEN}🔥 权重已真正更新！${NC}"
        
        # 显示最终GPU使用情况
        echo -e "${BLUE}📊 最终GPU状态:${NC}"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
        
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}❌ 直连训练失败！${NC}"
        echo -e "${RED}========================================${NC}"
        echo -e "${YELLOW}📋 检查日志: ${LOG_FILE}${NC}"
        exit 1
    fi
    
else
    echo -e "${RED}❌ GRPO 直连训练启动失败${NC}"
    echo -e "${YELLOW}📋 检查日志: ${LOG_FILE}${NC}"
    exit 1
fi 