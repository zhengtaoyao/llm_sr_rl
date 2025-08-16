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
# mkdir -p "${LOGDIR}"

# ################################
# # 1) start the local LLM server
# ################################
# nohup python ./llm_engine/engine.py > "$LOGDIR/engine_${PORT_LLM}.log" 2>&1 &

# ENGINE_PID=$!
# echo "LLM server started (PID=$ENGINE_PID) -- log: ${LOGDIR}/engine_${PORT_LLM}.log"

# # give the model a little time to load
# # sleep 300                        # adjust if your GPU is fast/slow
#!/usr/bin/env bash
# -------------------------- #
#  Start Mixtral on *one* GPU
# -------------------------- #

# 0) env
export CUDA_VISIBLE_DEVICES=0,1,2,3      # ← only GPU-0 is visible
export PORT_LLM=5000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmsr || { echo "activate failed"; exit 1; }

LOGDIR=./llmsr_logs;  mkdir -p "$LOGDIR"

# 1) launch engine
nohup python llm_engine/engine.py \
        > "$LOGDIR/engine_${PORT_LLM}.log" 2>&1 &
echo "Engine PID=$!  log=$LOGDIR/engine_${PORT_LLM}.log"
