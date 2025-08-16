#!/usr/bin/env bash
# ========================================
# 启动 Qwen2.5-Coder-7B-Instruct 引擎
# 使用 GPU 0 托管 7B 模型
# ========================================

set -e

# 🔧 GPU 配置：使用 0 号卡托管 7B 模型
export CUDA_VISIBLE_DEVICES=0
export PORT_LLM=5000

# 🤖 模型配置
MODEL_PATH="/storage/home/westlakeLab/zhangjunlei/Qwen/Qwen2.5-Coder-7B-Instruct"
TENSOR_PARALLEL_SIZE=1  # 单卡推理

# 📁 日志配置
LOGDIR=./llmsr_logs
mkdir -p "$LOGDIR"
LOG_FILE="$LOGDIR/engine_qwen7b_${PORT_LLM}.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 启动 Qwen2.5-Coder-7B 引擎${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🤖 模型: ${MODEL_PATH}${NC}"
echo -e "${GREEN}🎮 GPU: 0 (单卡推理)${NC}"
echo -e "${GREEN}🌐 端口: ${PORT_LLM}${NC}"
echo -e "${GREEN}📋 日志: ${LOG_FILE}${NC}"

# 检查模型路径
if [[ ! -d "$MODEL_PATH" ]]; then
    echo -e "${RED}❌ 错误: 模型路径不存在: $MODEL_PATH${NC}"
    exit 1
fi

# 检查必要文件
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo -e "${RED}❌ 错误: 模型配置文件不存在: $MODEL_PATH/config.json${NC}"
    exit 1
fi

# 检查 conda 环境 - 使用 verl 环境
if [[ "$CONDA_DEFAULT_ENV" != "verl" ]]; then
    echo -e "${YELLOW}⚠️  激活 conda 环境 (verl)...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate verl || { 
        echo -e "${RED}❌ 无法激活 verl 环境${NC}"; 
        exit 1; 
    }
fi

# 检查端口是否被占用
if lsof -Pi :$PORT_LLM -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ 端口 $PORT_LLM 已被占用${NC}"
    echo "请先停止现有服务或更改端口"
    exit 1
fi

# 检查 GPU 可用性
echo -e "${BLUE}🔍 检查 GPU 状态...${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -n 1

# 检查 GPU 0 显存
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0)
if [[ $GPU_FREE -lt 20000 ]]; then
    echo -e "${YELLOW}⚠️  警告: GPU 0 可用显存较少 (${GPU_FREE}MB)，可能影响 7B 模型加载${NC}"
fi

# 🚀 启动引擎
echo -e "${BLUE}🚀 启动 LLM 引擎...${NC}"
echo -e "${YELLOW}日志文件: ${LOG_FILE}${NC}"
echo -e "${YELLOW}🔧 禁用 torch.compile 以避免 Triton 编译错误${NC}"

# 使用 vLLM 启动高性能推理服务（单卡优化）
# 添加 --enforce-eager 禁用 torch.compile
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT_LLM \
    --host 0.0.0.0 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-requests \
    --swap-space 4 \
    --max-num-seqs 64 \
    --enforce-eager \
    > "$LOG_FILE" 2>&1 &

ENGINE_PID=$!
echo -e "${GREEN}✅ 引擎已启动 (PID: $ENGINE_PID)${NC}"

# 等待服务启动
echo -e "${YELLOW}⏳ 等待服务启动...${NC}"
sleep 15

# 测试服务连接
echo -e "${BLUE}🔍 测试服务连接...${NC}"
for i in {1..40}; do
    if curl -s http://localhost:$PORT_LLM/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 服务启动成功！${NC}"
        echo -e "${BLUE}🌐 服务地址: http://localhost:$PORT_LLM${NC}"
        echo -e "${BLUE}📊 模型信息: http://localhost:$PORT_LLM/v1/models${NC}"
        break
    else
        echo -e "${YELLOW}⏳ 等待服务启动... ($i/40)${NC}"
        sleep 5
    fi
    
    if [[ $i -eq 40 ]]; then
        echo -e "${RED}❌ 服务启动超时${NC}"
        echo -e "${YELLOW}请检查日志: ${LOG_FILE}${NC}"
        echo -e "${YELLOW}可能的原因:${NC}"
        echo -e "  1. GPU 0 显存不足"
        echo -e "  2. 模型文件损坏"
        echo -e "  3. vLLM 版本不兼容"
        exit 1
    fi
done

# 显示服务信息
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 Qwen2.5-Coder-7B 引擎就绪！${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}📊 服务状态:${NC}"
echo -e "  🌐 URL: http://localhost:$PORT_LLM"
echo -e "  🔧 PID: $ENGINE_PID"
echo -e "  📋 日志: $LOG_FILE"
echo -e "  🎮 GPU: 0 (单卡推理)"
echo -e "  🤖 模型: Qwen2.5-Coder-7B-Instruct"
echo -e "  💾 显存利用率: 85%"
echo -e "  ⚡ 模式: Eager (禁用编译优化)"
echo ""
echo -e "${YELLOW}💡 使用方法:${NC}"
echo -e "  • 测试: curl http://localhost:$PORT_LLM/v1/models"
echo -e "  • 停止: kill $ENGINE_PID"
echo -e "  • 日志: tail -f $LOG_FILE"
echo -e "  • 监控: watch -n 1 'nvidia-smi --id=0'"
echo ""
echo -e "${GREEN}🚀 现在可以启动 GRPO 训练了！${NC}"
echo -e "${BLUE}建议使用: ./run_llmsr_grpo_qwen7b.sh${NC}"
