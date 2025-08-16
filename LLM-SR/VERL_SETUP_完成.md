# 🔥 LLM-SR VERL GRPO 设置完成

> **成功解决了 KeyError: 'ground_truth' 问题，现在可以运行VERL微调！**

## 📋 完成的工作

### 1. 数据集处理 ✅
- **创建了 `process_data_for_verl.py`** - 将所有LLM-SR数据转换为VERL兼容格式
- **生成了4个VERL数据集**:
  - `oscillator1_train_verl.parquet` (37.6 KB)
  - `oscillator2_train_verl.parquet` (47.3 KB) 
  - `bactgrow_train_verl.parquet` (57.2 KB)
  - `stressstrain_train_verl.parquet` (29.0 KB)
- **解决了KeyError问题**: 每个数据集都包含必需的 `reward_model.ground_truth` 字段

### 2. 奖励函数优化 ✅
- **创建了 `simple_verl_reward.py`** - 简化的奖励函数，专门处理新数据格式
- **修改了 `grpo_runner.py`** - 自动检测并使用预生成的VERL数据集
- **避免了复杂依赖**: 不再依赖复杂的LLM-SR评估器和Ray Workers

### 3. 配置优化 ✅
- **修改了直连模式训练函数** - 自动检测预生成的数据集
- **优化了GPU内存配置** - 针对6卡训练进行了优化
- **简化了奖励计算** - 使用更高效的MSE计算

### 4. 测试验证 ✅
- **创建了 `test_verl_setup.py`** - 全面测试所有组件
- **所有测试通过**: 数据集格式、奖励函数、数据兼容性、配置生成

## 🚀 如何使用

### 快速开始
```bash
# 1. 确保在正确的conda环境中
conda activate verl

# 2. 直接运行GRPO训练（推荐）
./run_llmsr_grpo_direct.sh
```

### 自定义训练
```bash
# 训练不同的问题
PROBLEM_NAME=bactgrow ./run_llmsr_grpo_direct.sh

# 使用不同的训练参数
EPOCHS=15 BATCH_SIZE=32 LEARNING_RATE=5e-7 ./run_llmsr_grpo_direct.sh

# 使用不同GPU数量
GPUS=4 ./run_llmsr_grpo_direct.sh
```

### 手动调用Python
```bash
python main.py \
    --use_rl \
    --problem_name oscillator1 \
    --spec_path ./specs/specification_oscillator1_numpy.txt \
    --model_path /storage/home/westlakeLab/zhangjunlei/Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs 10 \
    --batch_size 24 \
    --learning_rate 1e-6 \
    --rollout_n 4 \
    --gpus 6
```

## 📊 支持的数据集

| 数据集 | 输入变量 | 输出变量 | 理论方程 | 样本数 |
|--------|----------|----------|----------|--------|
| oscillator1 | x, v | a | a = -x | 1000 |
| oscillator2 | x, v | a | a = -x - 0.1*v | 1000 |
| bactgrow | b, s, temp, pH | db | db = r×b×s/(K+s)×f(temp)×g(pH) | 1000 |
| stressstrain | strain, temp | stress | stress = E×strain | 1000 |

## 🔧 关键修复

### 解决的主要问题
1. **KeyError: 'ground_truth'** - 在 `reward_model` 字段中添加了 `ground_truth`
2. **数据格式不兼容** - 转换为标准的VERL parquet格式
3. **复杂的奖励计算** - 简化为直接的MSE计算
4. **内存效率** - 优化了批量大小和GPU使用

### 数据格式示例
```python
{
    "prompt": [{"role": "user", "content": "..."}],
    "data_source": "llm_sr_oscillator1_train", 
    "ability": "symbolic_regression_oscillator1",
    "extra_info": {
        "problem_type": "oscillator1",
        "task": "symbolic_regression",
        "data_sample": {"x": -0.251, "v": -0.173, "a": 0.028}
    },
    "reward_model": {
        "style": "rule",
        "ground_truth": "a = -x",  # 🔥 关键：避免KeyError
        "evaluation_type": "mse_based"
    }
}
```

## 🎯 训练配置

### 默认配置（6卡7B模型）
- **模型**: Qwen2.5-Coder-7B-Instruct
- **GPU**: 6张卡 (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5)
- **批量大小**: 24 × 6 = 144
- **学习率**: 1e-6
- **训练轮数**: 10
- **组大小**: 4个rollout

### 内存优化
- **FSDP策略**: 参数和优化器offload
- **GPU内存利用率**: 40%
- **微批量大小**: 1每GPU
- **梯度检查点**: 启用

## 📁 文件结构

```
LLM-SR/
├── verl_datasets/                    # 🔥 新生成的VERL数据集
│   ├── oscillator1_train_verl.parquet
│   ├── oscillator2_train_verl.parquet  
│   ├── bactgrow_train_verl.parquet
│   └── stressstrain_train_verl.parquet
├── process_data_for_verl.py         # 🔥 数据处理脚本
├── simple_verl_reward.py            # 🔥 简化奖励函数
├── test_verl_setup.py               # 🔥 测试脚本
├── run_llmsr_grpo_direct.sh         # 训练脚本
└── llmsr/rl/grpo_runner.py          # 🔥 修改的运行器
```

## 🔍 监控训练

### 查看训练日志
```bash
# 实时监控训练
tail -f llmsr_logs/grpo_direct_oscillator1_qwen7b_*.log

# 检查GPU使用情况
nvidia-smi

# 检查训练进程
ps aux | grep python
```

### 训练输出
- **模型权重**: `llmsr_grpo_outputs/*/` 
- **配置文件**: `grpo_config_direct.yaml`
- **日志文件**: `llmsr_logs/grpo_direct_*.log`
- **检查点**: 每2个epoch保存一次

## ⚠️ 重要说明

### 训练模式
- **直连模式**: 真正微调LLM权重（推荐）
- **HTTP模式**: 仅策略优化，不更新权重

### 资源需求
- **内存**: 至少 80GB GPU内存（6×A100）
- **存储**: 至少 50GB 可用空间
- **时间**: 每个epoch约20-30分钟

### 依赖环境
```bash
# 必需的包
pip install verl
pip install pyarrow
pip install pandas
```

## 🎉 成功标志

如果看到以下信息，说明训练成功启动：
```
✅ 找到预生成的VERL数据集: ./verl_datasets/oscillator1_train_verl.parquet
✅ 创建奖励函数: llmsr_grpo_outputs/.../llmsr_reward.py  
✅ GRPO 直连训练已启动 (PID: xxxxx)
🔥 模型权重将通过 FSDP 进行真正的微调更新
```

---

**现在你可以运行 `./run_llmsr_grpo_direct.sh` 开始VERL微调训练了！** 🚀 