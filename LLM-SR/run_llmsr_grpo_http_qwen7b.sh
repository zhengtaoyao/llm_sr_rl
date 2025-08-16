#!/usr/bin/env bash
# ========================================
# 启动 GRPO 训练使用 Qwen2.5-Coder-7B
# 使用 GPU 1-4 进行全参数训练 (4张卡)
# ========================================

set -e

# 🔧 GPU 配置：使用空闲的 GPU 1-4 进行 4 卡训练
export CUDA_VISIBLE_DEVICES=1,2,3,4
export GPUS=4

# 🌐 HTTP 服务配置
HTTP_URL="http://localhost:5000"
TOKENIZER_PATH="/storage/home/westlakeLab/zhangjunlei/Qwen/Qwen2.5-Coder-7B-Instruct"

# 🎯 训练配置 - 针对 4 张卡优化
PROBLEM_NAME=${PROBLEM_NAME:-"oscillator1"}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-16}       # 16 * 4 = 64, 能被 5 整除
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-4}          # 4 个 rollout
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # 减少微批量大小
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-10}

# 📁 路径配置
SPEC_PATH="./specs/specification_${PROBLEM_NAME}_numpy.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./llmsr_grpo_outputs/${PROBLEM_NAME}_qwen7b_4gpu_${TIMESTAMP}"
LOG_DIR="./llmsr_logs"
LOG_FILE="${LOG_DIR}/grpo_${PROBLEM_NAME}_qwen7b_4gpu_${TIMESTAMP}.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔥 GRPO 训练 - Qwen2.5-Coder-7B (4卡)${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 HTTP 服务
echo -e "${BLUE}🔍 检查 HTTP LLM 服务...${NC}"
if ! curl -s "$HTTP_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ HTTP LLM 服务未运行${NC}"
    echo -e "${YELLOW}请先启动引擎: ./run_llmsr_engine_qwen7b.sh${NC}"
    exit 1
fi

# 测试模型响应
echo -e "${BLUE}🔍 测试模型响应...${NC}"
MODEL_RESPONSE=$(curl -s "$HTTP_URL/v1/models" | grep -o "Qwen" | head -1)
if [[ "$MODEL_RESPONSE" == "Qwen" ]]; then
    echo -e "${GREEN}✅ Qwen 模型服务正常${NC}"
else
    echo -e "${RED}❌ 模型服务异常，请检查引擎状态${NC}"
    exit 1
fi

# 检查必要文件
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

# 显示配置信息
echo -e "${PURPLE}🔥 训练配置 (4卡全参数微调):${NC}"
echo -e "${GREEN}📋 配置信息:${NC}"
echo -e "  问题: ${YELLOW}$PROBLEM_NAME${NC}"
echo -e "  模型: ${YELLOW}Qwen2.5-Coder-7B-Instruct (HTTP)${NC}"
echo -e "  HTTP服务: ${YELLOW}$HTTP_URL${NC}"
echo -e "  训练轮数: ${YELLOW}$EPOCHS${NC}"
echo -e "  批次大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  学习率: ${YELLOW}$LEARNING_RATE${NC}"
echo -e "  组大小: ${YELLOW}$ROLLOUT_N${NC}"
echo -e "  GPU: ${YELLOW}1,2,3,4 (4张卡)${NC}"
echo -e "  输出目录: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "  训练模式: ${YELLOW}全参数微调 (非LoRA)${NC}"

# 检查 conda 环境 - 修改为使用 verl 环境
if [[ "$CONDA_DEFAULT_ENV" != "verl" ]]; then
    echo -e "${YELLOW}⚠️  激活 conda 环境 (verl)...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate verl || { 
        echo -e "${RED}❌ 无法激活 verl 环境${NC}"; 
        exit 1; 
    }
fi

# 检查训练 GPU 可用性
echo -e "${BLUE}🎮 检查训练 GPU 状态...${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | sed -n '2,6p'

# 检查每张训练卡的显存
for gpu_id in {1..5}; do
    GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$gpu_id)
    if [[ $GPU_FREE -lt 15000 ]]; then
        echo -e "${YELLOW}⚠️  警告: GPU $gpu_id 可用显存较少 (${GPU_FREE}MB)${NC}"
    fi
done

# 🔥 设置环境变量进行优化
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=INFO  # 改为 INFO 以获取更多调试信息
export VLLM_LOGGING_LEVEL=WARN
# 🔥 修复 vLLM 内存池兼容性问题：移除 expandable_segments 配置
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.6"

# FSDP 优化 - 4卡配置 (修改网络配置)
# 自动检测网络接口
NETWORK_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
# 或者注释掉让 NCCL 自动选择
# export NCCL_SOCKET_IFNAME=eth0

# 禁用 InfiniBand 和 P2P（如果硬件不支持）
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 强制使用 TCP 通信
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0

# 设置超时时间
export NCCL_TIMEOUT=1800

# 降低 NCCL 树阈值
export NCCL_TREE_THRESHOLD=0

# DeepSpeed 优化（如果使用）
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo -e "${GREEN}✅ 环境变量设置完成${NC}"
echo -e "  CUDA_VISIBLE_DEVICES: ${YELLOW}$CUDA_VISIBLE_DEVICES${NC}"
echo -e "  训练卡数: ${YELLOW}4${NC}"
echo -e "  推理卡: ${YELLOW}GPU 0 (独立)${NC}"

# 🚀 启动 GRPO 训练 - 修改为后台运行
echo -e "${BLUE}🚀 启动 GRPO 4卡训练 (后台运行)...${NC}"
echo -e "${YELLOW}日志文件: ${LOG_FILE}${NC}"
echo -e "${GREEN}✅ 使用全参数微调模式 (非LoRA)${NC}"
echo -e "${GREEN}✅ 推理与训练分离，避免显存冲突${NC}"

# 使用 nohup 让训练在后台持续运行
nohup python main.py \
    --use_rl \
    --use_http \
    --problem_name "$PROBLEM_NAME" \
    --spec_path "$SPEC_PATH" \
    --http_url "$HTTP_URL" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --rollout_n "$ROLLOUT_N" \
    --gpus "$GPUS" \
    --output_dir "$OUTPUT_DIR" \
    --log_path "$LOG_FILE" \
    > "${LOG_FILE}" 2>&1 &

GRPO_PID=$!
echo -e "${GREEN}✅ GRPO 训练已启动 (PID: $GRPO_PID)${NC}"
echo -e "${YELLOW}📋 日志文件: ${LOG_FILE}${NC}"
echo -e "${YELLOW}💡 监控命令: tail -f ${LOG_FILE}${NC}"
echo -e "${YELLOW}🛑 停止命令: kill $GRPO_PID${NC}"

# 等待一段时间确保训练正常启动
echo -e "${BLUE}⏳ 等待训练启动...${NC}"
sleep 10

# 检查进程是否正常运行
if ps -p $GRPO_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ GRPO 训练正常运行中${NC}"
    echo -e "${BLUE}📊 进程状态:${NC}"
    echo -e "  🔧 PID: $GRPO_PID"
    echo -e "  📋 日志: ${LOG_FILE}"
    echo -e "  🎮 GPU: 1,2,3,4"
    echo -e "  ⏰ 预计训练时间: ${EPOCHS} 轮次"
else
    echo -e "${RED}❌ GRPO 训练启动失败${NC}"
    echo -e "${YELLOW}📋 检查日志: ${LOG_FILE}${NC}"
    exit 1
fi

# 检查训练结果
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ GRPO 4卡训练完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}📁 结果保存到: $OUTPUT_DIR${NC}"
    echo -e "${YELLOW}📋 日志文件: ${LOG_FILE}${NC}"
    echo -e "${GREEN}🔥 全参数微调完成，模型权重已更新${NC}"
    
    # 显示最终GPU使用情况
    echo -e "${BLUE}📊 最终GPU状态:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
    
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ 训练失败！${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${YELLOW}📋 检查日志: ${LOG_FILE}${NC}"
    echo -e "${YELLOW}🔧 常见问题:${NC}"
    echo -e "  1. GPU 显存不足 - 尝试减小 batch_size"
    echo -e "  2. HTTP 服务中断 - 检查推理服务状态"
    echo -e "  3. NCCL 通信错误 - 检查网络配置"
    echo -e "  4. 模型加载失败 - 检查模型路径"
    exit 1
fi
