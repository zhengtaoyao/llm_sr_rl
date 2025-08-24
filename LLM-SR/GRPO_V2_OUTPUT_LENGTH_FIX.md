# 🔥 LLM-SR GRPO v2版本输出长度和重复内容修复

## 修复概述

基于v1版本的修复经验，对v2版本进行了相同的优化，解决LLM输出截断和重复内容问题。

## 修复内容

### 1. **增加Token长度配置**

**修改文件**: `llmsr/rl/grpo_runner_v2.py`

```python
# 🔥 函数签名中的默认参数更新
def create_grpo_config_v2(
    max_prompt_length: int = 4096,     # 2048 -> 4096
    max_new_tokens: int = 8192,        # 1024 -> 8192
    max_model_len: int = 16384,        # 8192 -> 16384
    max_num_batched_tokens: int = 8192, # 4096 -> 8192
)

def train_llmsr_grpo_v2(
    max_prompt_length: int = 4096,     # 2048 -> 4096
    max_new_tokens: int = 8192,        # 1024 -> 8192
    max_model_len: int = 16384,        # 8192 -> 16384
    max_num_batched_tokens: int = 8192, # 4096 -> 8192
)

# 🔥 安全阈值更新
safe_max_token_len = max(16384, int(max_prompt_length + max_new_tokens + 512))  # 12288 -> 16384
```

### 2. **优化生成参数减少重复**

**修改文件**: `llmsr/rl/grpo_runner_v2.py`

```python
# 🔥 rollout配置优化
"temperature": 0.7,           # 0.8 -> 0.7 降低温度减少重复
"top_p": 0.95,               # 0.9 -> 0.95 增加多样性
"top_k": 50,                 # 30 -> 50 扩大选择范围
"repetition_penalty": 1.1,   # 新增：重复惩罚
"frequency_penalty": 0.1,    # 新增：频率惩罚

# 🔥 val_kwargs配置同步优化
"val_kwargs": {
    "temperature": 0.7,       # 0.8 -> 0.7
    "top_p": 0.95,           # 0.9 -> 0.95
    "top_k": 50,             # 30 -> 50
}
```

### 3. **增强表达式提取逻辑**

**修改文件**: `llmsr/rl/simple_verl_reward_v2.py`

新增功能：
- 处理截断的`<think>`标签内容
- 更强大的正则表达式模式匹配
- 从截断内容中组合数学表达式
- V2专用的调试信息输出

```python
# 🔥 新增截断处理逻辑
elif "<think>" in code:
    # 处理截断情况：如果有<think>但没有</think>，可能被截断了
    think_parts = code.split("<think>")
    if len(think_parts) > 1:
        after_think = think_parts[-1].strip()
        # 查找可能的代码模式...
        if _valid_expr(candidate):
            print(f"🔧 V2从截断的<think>内容中提取表达式: {candidate}")
            return candidate

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

# 🔥 专门处理截断情况的简单表达式提取
math_like_patterns = [
    r'[-+]?\s*[xv](?:\*\*\d+)?',
    r'[-+]?\s*\d+\.?\d*\s*\*?\s*[xv](?:\*\*\d+)?',
    r'[-+]?\s*params\[\d+\]',
]
```

### 4. **更新脚本配置**

**修改文件**: `run_llmsr_grpo_v2.sh`

```bash
# 🔥 大token长度配置 - 基于46GB/81GB显存使用情况优化（v2增强版）
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}      # 提示长度：4K tokens
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}      # 生成长度：8K tokens  
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}       # 模型最大长度：16K tokens
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}  # 批量token数：8K

# 更新显示信息
echo "🔥 v2 训练配置 (大Token优化版本 + 截断修复)"
echo "  训练模式: 🔥 v2模式 - 大Token支持 + 截断修复"
echo "🎯 大Token配置已启用 - 支持完整思考过程输出 + 截断恢复"
```

## v2版本的特殊优势

相比v1版本，v2版本还具有以下额外特性：

### 1. **密化多成分奖励**
- 拟合奖励：`r_fit = exp(-lambda * NMSE)`
- 简洁奖励：基于AST复杂度
- 物理一致性：维度/边界检查
- 过程奖励：语法校验和常数优化

### 2. **记忆与多样性**
- 轻量"岛屿/记忆库"检索few-shot
- 跨进程安全的文件级共享
- 动态few-shot示例注入

### 3. **EDIT DSL支持**
- `EDIT ADD <expr>` → `(base) + (expr)`
- `EDIT MUL <expr>` → `(base) * (expr)`
- `EDIT REPLACE <old> => <new>` → 字符串替换

### 4. **组内排名归一**
- 同一提示组内对候选加权求和
- List-wise归一化降低尺度噪声
- 更稳定的训练信号

## 预期效果

1. **完整输出**：16K tokens足够生成完整的equation函数和思考过程
2. **减少重复**：通过参数调优和惩罚机制减少重复内容
3. **更好提取**：即使截断也能提取有效表达式，V2专用调试信息
4. **多成分奖励**：更细粒度的奖励信号，提高训练效果
5. **记忆增强**：few-shot示例提高生成质量

## 使用方法

运行v2版本的训练脚本：

```bash
./run_llmsr_grpo_v2.sh
```

或者直接调用Python：

```bash
python main.py --use_rl_v2 \
  --problem_name oscillator1 \
  --spec_path ./specs/specification_oscillator1_numpy.txt \
  --model_path /storage/home/westlakeLab/zhangjunlei/Qwen3-8B \
  --max_prompt_length 4096 \
  --max_new_tokens 8192 \
  --max_model_len 16384 \
  --max_num_batched_tokens 8192
```

## 验证方法

运行后检查日志中的：
1. **V2专用标识**：`🔥🔥🔥 V2 REWARD FUNCTION CALLED! 🔥🔥🔥`
2. **完整输出**：包含完整的equation函数和思考过程
3. **多成分奖励**：`r_fit`, `r_simp`, `r_phys`, `r_proc`各项分数
4. **表达式提取**：`🔧 V2从截断的<think>内容中提取表达式`等调试信息
5. **记忆系统**：few-shot示例的注入和使用情况

## 与v1版本的区别

| 特性 | v1版本 | v2版本 |
|------|--------|--------|
| 奖励函数 | 单一NMSE | 多成分密化奖励 |
| 表达式提取 | 基础模式匹配 | 增强模式匹配 + 截断恢复 |
| 记忆系统 | 无 | 岛屿记忆库 + few-shot |
| EDIT支持 | 无 | 支持EDIT DSL |
| 排名归一 | 无 | 组内排名归一 |
| 调试信息 | 基础 | V2专用详细信息 |

v2版本在保持v1版本所有修复的基础上，提供了更强大的功能和更稳定的训练效果。
