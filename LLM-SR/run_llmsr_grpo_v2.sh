#!/usr/bin/env bash
set -euo pipefail

# 基于 SLURM 的 v2 训练脚本（使用 srun 按用户规范提交作业）

PARTITION=${PARTITION:-a6000_xgpan}
JOB_NAME=${JOB_NAME:-llmsr_grpo_v2}
GPUS=${GPUS:-1}
SRUN_EXTRAS=${SRUN_EXTRAS:-}

PROBLEM_NAME=${PROBLEM_NAME:-"oscillator1"}
SPEC_PATH=${SPEC_PATH:-"./specs/specification_${PROBLEM_NAME}_numpy.txt"}
MODEL_PATH=${MODEL_PATH:-"/storage/home/westlakeLab/zhangjunlei/Qwen/Qwen2.5-Coder-7B-Instruct"}
OUT_DIR=${OUT_DIR:-"./llmsr_grpo_outputs/${PROBLEM_NAME}_v2"}

EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-4}
KL_COEF=${KL_COEF:-1e-3}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GRID_TRAIN=${GRID_TRAIN:-0}
NUM_GROUPS=${NUM_GROUPS:-10}
FEW_SHOT_K=${FEW_SHOT_K:-3}

LOG_DIR=${LOG_DIR:-"./llmsr_logs"}
mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG_FILE="${LOG_DIR}/grpo_v2_${PROBLEM_NAME}_$(date +%Y%m%d_%H%M%S).log"

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
  --few_shot_k ${FEW_SHOT_K} \
  $( [[ "$GRID_TRAIN" == "1" ]] && echo "--grid_train_data --num_grid_groups ${NUM_GROUPS}" ) \
  --log_path ${LOG_FILE}"

echo "[SRUN] ${CMD}" | tee -a "$LOG_FILE"

srun -p ${PARTITION} -n 1 --gres=gpu:${GPUS} --job-name=${JOB_NAME} --kill-on-bad-exit=1 ${SRUN_EXTRAS} bash -lc "${CMD} |& cat" |& tee -a "$LOG_FILE"


