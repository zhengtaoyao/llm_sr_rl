# #!/usr/bin/env bash
# # -----------------------------
# # Launch the local LLM engine,
# # then reproduce the oscillator1
# # experiment of LLM-SR.
# # -----------------------------
# # Example – change if busy
# export PORT_LLM=5000
# export CUDA_VISIBLE_DEVICES=0          # or 0,1,2…
# conda init

# source ~/miniconda3/etc/profile.d/conda.sh   # ← adjust if miniconda lives elsewhere
# conda activate llmsr
# LOGDIR=./llmsr_logs  
# ################################
# # 2) run LLM-SR on oscillator1
# ################################
# nohup python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_local
#         > "${LOGDIR}/oscillator1.log" 2>&1 &

# echo "LLM-SR run started – log: ${LOGDIR}/oscillator1.log"


#!/usr/bin/env bash
# ----------------------------- #
#  Run LLM-SR Oscillator-1 job  #
# ----------------------------- #

export PORT_LLM=5000                 # MUST match engine
export CUDA_VISIBLE_DEVICES=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmsr || { echo "conda activate failed"; exit 1; }

LOGDIR=./llmsr_logs
mkdir -p "$LOGDIR"

# Wait until engine is ready
echo "Waiting for engine on port $PORT_LLM …"
until nc -z localhost $PORT_LLM; do sleep 2; done
echo "Engine is up. Starting search."

# Launch evolutionary search


# python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_local > "$LOGDIR/oscillator1.log" 2>&1 &
python main.py --problem_name stressstrain --spec_path ./specs/specification_stressstrain_numpy.txt --log_path ./logs/stresstrain_local > "$LOGDIR/stresstrain.log" 2>&1 &

echo "Search PID=$!  log=$LOGDIR/stresstrain.log"
