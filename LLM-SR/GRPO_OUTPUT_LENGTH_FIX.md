# 🔥 LLM输出长度和重复内容修复

## 问题分析

从日志 `grpo_direct_oscillator1_qwen8b_20250824_003914.log` 中发现两个主要问题：

1. **LLM输出被截断**：在第988行就停止了，equation函数没有完整输出
2. **重复内容过多**：输出中包含大量重复的`<think>`标签内容，占用了大量token

## 修复方案

### 1. 增加Token长度配置

**修改文件**: `llmsr/rl/grpo_runner.py`

```python
# 🔥 增加默认token长度
prompt_len_cfg = kwargs.get('max_prompt_length', 4096)  # 2048 -> 4096
response_len_cfg = kwargs.get('max_new_tokens', 8192)   # 4096 -> 8192
safe_max_token_len = max(16384, int(prompt_len_cfg + response_len_cfg + 512))  # 12288 -> 16384

# 数据配置中的长度
"max_prompt_length": kwargs.get('max_prompt_length', 4096),     # 2048 -> 4096
"max_response_length": kwargs.get('max_response_length', 8192), # 2048 -> 8192

# Rollout配置中的长度
"prompt_length": kwargs.get('max_prompt_length', 4096),         # 2048 -> 4096
"response_length": kwargs.get('max_new_tokens', 8192),          # 4096 -> 8192
"max_num_batched_tokens": kwargs.get('max_num_batched_tokens', 8192),  # 4096 -> 8192
"max_model_len": kwargs.get('max_model_len', 16384),            # 8192 -> 16384
```

### 2. 优化生成参数减少重复

**修改文件**: `llmsr/rl/grpo_runner.py`

```python
# 🔥 优化生成参数
"temperature": kwargs.get('temperature', 0.7),    # 0.8 -> 0.7 降低温度减少重复
"top_p": kwargs.get('top_p', 0.95),              # 0.9 -> 0.95 增加多样性
"top_k": kwargs.get('top_k', 50),                # 30 -> 50 增加选择范围
"repetition_penalty": 1.1,                       # 新增：重复惩罚
"frequency_penalty": 0.1,                        # 新增：频率惩罚
```

### 3. 更新main.py默认参数

**修改文件**: `main.py`

```python
# 🔥 更新默认参数
parser.add_argument('--max_prompt_length', type=int, default=4096)    # 2048 -> 4096
parser.add_argument('--max_new_tokens', type=int, default=8192)       # 1024 -> 8192
parser.add_argument('--max_model_len', type=int, default=16384)       # 8192 -> 16384
parser.add_argument('--max_num_batched_tokens', type=int, default=8192) # 4096 -> 8192
```

### 4. 增强表达式提取逻辑

**修改文件**: `simple_verl_reward.py`

新增功能：
- 处理截断的`<think>`标签内容
- 更强大的正则表达式模式匹配
- 从截断内容中组合数学表达式
- 更好的表达式清理逻辑

```python
# 🔥 新增截断处理逻辑
elif "<think>" in solution_str:
    # 处理截断情况：如果有<think>但没有</think>，可能被截断了
    think_parts = solution_str.split("<think>")
    if len(think_parts) > 1:
        after_think = think_parts[-1].strip()
        # 查找可能的代码模式...

# 🔥 增强的表达式模式匹配
simple_patterns = [
    r'return\s+([^;}\n]+)',
    r'a\s*=\s*([^;}\n]+)',
    r'result\s*=\s*([^;}\n]+)',
    r'acceleration\s*=\s*([^;}\n]+)',
    # 新增：处理截断情况的模式
    r'[-+]?\s*params\[\d+\]\s*\*\s*[xv](?:\*\*\d+)?(?:\s*[-+]\s*params\[\d+\]\s*\*\s*[xv](?:\*\*\d+)?)*',
    r'[-+]?\s*\d*\.?\d*\s*\*?\s*[xv](?:\*\*\d+)?(?:\s*[-+]\s*\d*\.?\d*\s*\*?\s*[xv](?:\*\*\d+)?)*',
]
```

## 预期效果

1. **更长的输出**：从8K tokens增加到16K tokens，足够完整生成equation函数
2. **减少重复**：通过调整温度和添加重复惩罚，减少重复内容
3. **更好的提取**：即使输出被截断，也能从部分内容中提取有效的数学表达式
4. **更高的成功率**：提高奖励函数计算的成功率，避免-100.0的默认奖励

## 使用方法

修改后的配置会自动生效，重新运行训练脚本即可：

```bash
./run_llmsr_grpo_direct.sh
```

脚本中的环境变量设置会被正确传递：
- `MAX_PROMPT_LENGTH=4096`
- `MAX_NEW_TOKENS=8192` 
- `MAX_MODEL_LEN=16384`
- `MAX_NUM_BATCHED_TOKENS=8192`

## 注意事项

1. **显存使用**：增加token长度会增加显存使用，但在8卡配置下应该可以承受
2. **生成时间**：更长的输出可能需要更多生成时间
3. **批量大小**：如果显存不足，可能需要进一步减少批量大小

## 验证方法

运行后检查日志中的：
1. LLM输出是否完整（包含完整的equation函数）
2. 奖励计算是否成功（不再是-100.0）
3. 表达式提取是否成功（有具体的数学表达式输出）
