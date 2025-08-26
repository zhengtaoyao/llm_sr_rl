## LLM-SR: GRPO v2 vs v1 差异与 v2 设计详解

### 目录
- 1. 总览要点
- 2. 与 v1 的逐项对比
  - 2.1 脚本：run_llmsr_grpo_v2.sh vs run_llmsr_grpo_direct.sh
  - 2.2 训练 Runner：grpo_runner_v2.py vs grpo_runner.py
  - 2.3 奖励函数：simple_verl_reward_v2_fixed.py vs simple_verl_reward_fixed.py
  - 2.4 入口集成：main.py
- 3. v2 算法设计思想（为何这样设计）
  - 3.1 奖励密化与成分化
  - 3.2 受约束的动作/解码与常数分离（含 Grammar/AST 与 EDIT DSL）
  - 3.3 记忆/多样性：轻量岛屿检索 few-shot
  - 3.4 稳健的 GRPO 配置（KL、长度偏置、组内采样）
- 4. v2 核心代码结构与函数职责
  - 4.1 llmsr/rl/grpo_runner_v2.py
  - 4.2 llmsr/rl/simple_verl_reward_v2.py
  - 4.3 main.py 新增参数与分支
  - 4.4 与 v1 Runner 的关系
- 5. 训练配置建议与可调超参
- 6. 运行与产物
- 7. 常见问题与排查

---

### 1. 总览要点
- **训练入口**: v2 通过 `--use_rl_v2` 走全新直连管线，仅直连模式；v1 支持直连与 HTTP 两种模式。
- **奖励设计**: v2 使用“拟合+简洁+物理+过程”的复合奖励并做组内排名归一；v1 以负 MSE 为主的简化奖励。
- **记忆与日志**: v2 引入轻量“记忆/检索 few-shot”，并将样本与指标写入 `sample.jsonl`，自动生成 `best_reward.json`/`best_mse.json`；v1 无少样本检索与训练后汇总。
- **配置稳健性**: v2 强化显式 KL-loss（默认 1e-3）、token-mean 聚合（减轻长度偏置）、同题组内多样采样；v1 也支持 KL/GRPO，但配置分支更多（含 HTTP）。

---

### 2. 与 v1 的逐项对比

#### 2.1 脚本：`run_llmsr_grpo_v2.sh` vs `run_llmsr_grpo_direct.sh`
- **调度方式**
  - v2: 使用 `srun -p <partition> --gres=gpu:<GPUS>` 提交，遵循 SLURM 规范；支持 `PARTITION/GPUS/JOB_NAME/SRUN_EXTRAS` 环境参数。
  - v1: 本地 `nohup python main.py ... &` 后台执行，并显式设置 `CUDA_VISIBLE_DEVICES` 与 NCCL 变量。
- **入口参数**
  - v2: 传递 `--use_rl_v2`、`--kl_coef`、`--max_prompt_length`、`--max_new_tokens`、`--max_model_len`、`--few_shot_k`、`--grid_train_data/--num_grid_groups` 等 v2 专属参数。
  - v1: 走 `--use_rl`，未暴露 v2 特有的长度/KL/检索参数。
- **输出与日志**
  - v2/v1 均导出 `LLMSR_OUTPUT_DIR`；v2 用 `tee -a` 同步落盘日志，v1 自行监控 `nohup` 进程并打印 GPU 状态。

#### 2.2 训练 Runner：`llmsr/rl/grpo_runner_v2.py` vs `llmsr/rl/grpo_runner.py`
- **数据集构建**
  - v2: `_extract_prompt_header` + `MemoryManagerV2` 检索历史 Top-K 多样 few-shot，拼接入 prompt；支持网格/聚类分桶抽样；写出 `llmsr_train_v2.parquet`。
  - v1: 构建基础 prompt，可选网格分桶；写出 `llmsr_train.parquet`。
- **奖励包装**
  - v2: 生成 `llmsr_reward_v2.py`，调用 `simple_verl_reward_v2_fixed.compute_score`，训练前设置 `LLMSR_OUTPUT_DIR` 以便奖励落盘样本。
  - v1: 生成 `llmsr_reward.py`，调用 `simple_verl_reward_fixed.compute_score`（简化版）。
- **GRPO 配置**
  - v2: 显式 KL 加入 loss，`loss_agg_mode: token-mean`，微批默认 1，`rollout.n` 组内多样采样，保存 `grpo_config_v2.yaml`。
  - v1: 提供直连/HTTP 两套配置，含更完整的分布式/日志/Ref 配置，直连保存 `grpo_config_direct.yaml`。
- **记忆与训练后处理**
  - v2: 训练后汇总 `sample.jsonl`，导出 `best_reward.json` 与 `best_mse.json`。
  - v1: 无训练后最佳样本聚合导出。
- **模式支持**
  - v2: 仅直连模式；
  - v1: 直连和 HTTP 模式（HTTP 不更新权重）。

#### 2.3 奖励函数：`llmsr/rl/simple_verl_reward_v2_fixed.py` vs `simple_verl_reward_fixed.py`
- **v2 奖励成分（修复版）**
  - 拟合：`r_fit = exp(-λ_nmse * NMSE)`（缩放提升可学习性）。
  - 简洁：`r_simp = exp(-λ_simp * complexity)`（基于运算符/函数/常数/标识符近似复杂度）。
  - 物理一致：`r_phys`（如 log 正域等基础数值物理检查）。
  - 过程：`r_proc`（常数优化是否收敛等）。
  - 融合：`0.6*r_fit + 0.2*r_simp + 0.15*r_phys + 0.05*r_proc`。
- **共同点（修复版 v1/v2）**
  - 不再做“表达式字符串提取”，而是直接执行 LLM 生成的 Python 函数；内部用 BFGS 优化常数参数得到最小 MSE/NMSE。
  - 若设置 `LLMSR_OUTPUT_DIR`，均写入 `sample.jsonl`（记录时间戳、MSE/NMSE、参数等）。
- **组内排名归一（v2）**
  - 同一组候选按得分排序后做 rank→[1,0] 归一，削弱尺度噪声与长度偏置。
- **样本日志与指标（v2）**
  - 通过 `LLMSR_OUTPUT_DIR` 写入 `sample.jsonl`，逐条记录 `reward/nmse/complexity/params/...`，用于事后分析与最佳样本导出。
- **v1 奖励（修复版）**
  - 直接执行函数 + BFGS 常数优化，奖励为 `-MSE`（带上下限裁剪），无成分化/组内排名；同样写入 `sample.jsonl`。

#### 2.4 入口集成：`main.py`
- 新增 `--use_rl_v2` 分支，进入 v2 直连管线；训练前通过环境变量传递 `LLMSR_OUTPUT_DIR` 给奖励函数。
- v2 分支不支持 HTTP；v1 保留 `--use_http` 支持非微调推理式训练。

---

### 3. v2 算法设计思想（为何这样设计）

#### 3.1 奖励密化与成分化
- 直接用单一误差（如 -MSE）会导致奖励稀疏、高噪声、credit assignment 困难。
- v2 将奖励拆成四类：拟合、简洁（MDL 近似）、物理一致（可检验的软约束）、过程（如常数拟合是否收敛）。
- 将各成分做加权融合并在组内做 rank 归一，弱化尺度与长度偏置，稳定 GRPO 的组内优势估计。

#### 3.2 受约束的动作/解码与常数分离（含 Grammar/AST 与 EDIT DSL）
- v2 沿用 LLM-SR 原有“骨架→常数拟合→再细化”的思想：生成结构表达式，常数由后验拟合评估是否可收敛（计入过程奖励）。
- 新增：
  - 在数据集侧为 prompt 注入 `system` 约束，明确 grammar/AST 规则与 EDIT DSL 用法（ADD/MUL/REPLACE）。
  - 在奖励侧识别 `EDIT` 响应并基于 `extra_info.base_impl` 应用 `_apply_edit_dsl`，随后用 `_ast_is_legal` 做 AST 合法性检查（函数/变量白名单 + 深度约束）。
- 小步编辑可显著改善信用分配与可执行率，降低一次性从零生成复杂表达式的难度。

#### 3.3 记忆/多样性：轻量岛屿检索 few-shot
- FunSearch/LLM-SR 中“候选池→评估→检索再生成”的范式有效提升多样性与样本效率。
- v2 用 `MemoryManagerV2` 从输出目录维护轻量记忆库（JSON 文件），在构建训练数据时检索 Top-K 多样实现拼成 few-shot，鼓励“在好模式附近微调”。

#### 3.4 稳健的 GRPO 配置（KL、长度偏置、组内采样）
- 显式 KL-loss（coef≈1e-3 起步）而非混入奖励，便于监控漂移与稳定优化。
- `loss_agg_mode: token-mean` 缓解长序列的相对优势偏置。
- `rollout.n ≥ 4`，同题组内采样才能形成有意义的组内相对优势估计。

---

### 4. v2 核心代码结构与函数职责

#### 4.1 `llmsr/rl/grpo_runner_v2.py`
- `MemoryManagerV2`：基于 `memory_v2.json` 的轻量记忆管理，提供 `sample_few_shot(k)` 返回多样实现片段。
- `_extract_prompt_header(spec_text)`：从规范模版抽出 `@equation.evolve` 之前的前缀，用作 base prompt。
- `create_llmsr_dataset_v2(...)`：
  - 拼接 few-shot（来自记忆库）到 base prompt；
  - 提示结构包含 `system`（grammar/EDIT 规则）与 `user`（规范+few-shot）；
  - 可选网格/聚类分桶采样；
  - 生成 `llmsr_train_v2.parquet`（VERL 训练数据），并在 `extra_info` 中预留 `base_impl` 接口。
- `create_llmsr_reward_file_v2(...)`：生成 VERL 自定义奖励入口 `llmsr_reward_v2.py`，转发到 `simple_verl_reward_v2.compute_score`。
- `create_grpo_config_v2(...)`：
  - GRPO 直连配置：KL 加显式 loss、token-mean、微批 1、rollout.n 组内多样采样；
  - 保存 `grpo_config_v2.yaml`。
- `train_llmsr_grpo_v2(...)`：
  - 串起数据集→奖励→配置→训练；
  - 设置 `LLMSR_OUTPUT_DIR` 使奖励可写 `sample.jsonl`；
  - 训练后扫描 `sample.jsonl` 导出 `best_reward.json` 与 `best_mse.json`。

#### 4.2 `llmsr/rl/simple_verl_reward_v2_fixed.py`
- `compute_score(...)`：
  - 直接执行 LLM 生成的 Python 函数体（无表达式抽取）→ BFGS 优化常数，得到 MSE/NMSE；
  - 估算复杂度、物理一致、过程项并加权融合；
  - （可选）组内排名归一；
  - 若设置 `LLMSR_OUTPUT_DIR`，将每条样本落盘至 `sample.jsonl`（含 `reward/nmse/complexity/params/...`）。
- 其余辅助函数：
  - `_load_training_data_from_path`：读取 CSV 使用全部样本；
  - `extract_function_body_v2`/`_trim_function_body_v2`：从响应中提取并裁剪函数体；
  - `_estimate_complexity_from_body`：从函数体估算复杂度；
  - `_physical_consistency_v2`：基础物理软约束；
  - `build_executable_program_v2`/`execute_and_compute_mse_v2`：拼装与执行评估程序。

#### 4.3 `main.py` 新增参数与分支
- 新增 `--use_rl_v2`、`--kl_coef`、`--max_prompt_length`、`--max_new_tokens`、`--max_model_len`、`--few_shot_k`、`--grid_train_data/--num_grid_groups`。
- `--use_rl_v2` 分支调用 `train_llmsr_grpo_v2`，并设置 `LLMSR_OUTPUT_DIR`。

#### 4.4 与 v1 Runner 的关系
- v1 的 `grpo_runner.py` 依旧保留，支持直连/HTTP，两者互不影响；
- v2 专注直连+密化奖励+记忆检索与稳健 GRPO 配置。

---

### 5. 训练配置建议与可调超参
- 组内采样：`--rollout_n` 建议 4–8。
- KL：`--kl_coef` 从 `1e-3` 起步网格搜，观察 KL 漂移与奖励分布。
- 长度：`--loss_agg_mode token-mean` 已内置；必要时控制 `--max_new_tokens` 防冗长解释。
- few-shot：`--few_shot_k` 2–6 之间调节，过多可能引入模式偏置。
- 数据分桶：开启 `--grid_train_data` 并调整 `--num_grid_groups`，提升覆盖与稳健性。

---

### 6. 运行与产物
- 运行（SLURM）：
  - `bash run_llmsr_grpo_v2.sh`
  - 或自定义：`PARTITION=a6000_xgpan GPUS=1 PROBLEM_NAME=oscillator1 ... run_llmsr_grpo_v2.sh`
- 主要产物：
  - `llmsr_grpo_outputs/<exp>/llmsr_train_v2.parquet`
  - `llmsr_grpo_outputs/<exp>/grpo_config_v2.yaml`
  - `llmsr_grpo_outputs/<exp>/sample.jsonl`
  - `llmsr_grpo_outputs/<exp>/best_reward.json`、`best_mse.json`
  - EDIT/AST 相关字段：`sample.jsonl` 中包含 `edit_mode`、`ast_ok`、`base_expr`。

---

### 7. 常见问题与排查
- 奖励始终很低/样本大量不可解析：
  - 检查表达式抽取是否失败；缩短 `max_new_tokens` 限制冗余；提高 few-shot 质量；检查 `ast_ok` 字段是否大量为 false。
- KL 漂移过快或发散：
  - 减小 `--learning_rate` 或增大 `--kl_coef`；观察 `rollout.n` 是否足够。
- 多样性塌缩：
  - 提升 `rollout.n`、提高采样温度，或丰富 few-shot 记忆（跨结构样例）。
- 训练过慢/显存不足：
  - 维持微批 1，必要时下调 `max_model_len` 与 `max_new_tokens`；确保 `bf16`、FSDP offload 打开。


- v2 用的 LLM
  - 文件位置:
    - /storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/run_llmsr_grpo_v2.sh（变量 MODEL_PATH）
    - /storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/llmsr/rl/grpo_runner_v2.py（配置里 actor_rollout_ref.model.path）
  - 默认值: 由 run_llmsr_grpo_v2.sh 的 MODEL_PATH 决定（当前为 Qwen/Qwen2.5-Coder-7B-Instruct 路径）。
  - 修改方式:
    - 最简单: 调用前覆写环境变量 MODEL_PATH=... bash run_llmsr_grpo_v2.sh
    - 或直接改脚本 run_llmsr_grpo_v2.sh 里的 MODEL_PATH
    - 代码层: grpo_runner_v2.py 里传入的 model_path 会写到 config["actor_rollout_ref"]["model"]["path"]

- v1 用的 LLM（直连模式）
  - 文件位置:
    - /storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/run_llmsr_grpo_direct.sh（变量 MODEL_PATH）
    - /storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/llmsr/rl/grpo_runner.py（配置里 actor_rollout_ref.model.path；train_llmsr_grpo_direct 的 model_path 参数默认 Qwen/Qwen2.5-1.5B-Instruct）
  - 默认值: 以 run_llmsr_grpo_direct.sh 的 MODEL_PATH 为准（脚本里指定的 Qwen 7B Coder）；若未传，grpo_runner.py 的函数默认是 Qwen/Qwen2.5-1.5B-Instruct。
  - 修改方式:
    - 覆写环境变量 MODEL_PATH=... bash run_llmsr_grpo_direct.sh 或直接改脚本里的 MODEL_PATH
    - 代码层: grpo_runner.py 里传入的 model_path 会写到 config["actor_rollout_ref"]["model"]["path"]

- v1 的 HTTP 模式（仅当 --use_http）
  - 文件位置:
    - /storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/llmsr/rl/grpo_runner.py（create_grpo_config_http）
  - 相关字段:
    - tokenizer（仅分词器）: config["actor_rollout_ref"]["model"]["path"] ← tokenizer_path
    - 外部服务: rollout.http_url（指向你的推理服务）
  - 修改方式:
    - 在 main.py 调用时传 --use_http 并设置 --tokenizer_path 与 --http_url，或在 grpo_runner.py 的 create_grpo_config_http 参数处改

简要提示
- 推荐用“改脚本变量 MODEL_PATH 或运行时覆写 MODEL_PATH”的方式切换模型；这会透传到 config 的 actor_rollout_ref.model.path。
- 两套（v1/v2）都是读取 HuggingFace 名称或本地权重目录路径作为 model_path。