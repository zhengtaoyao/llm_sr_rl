# 🏝️ 群岛机制超参数配置总结

## 📊 Score、MSE、Reward 关系链条

```
原始数据 → MSE (拟合误差) → NMSE (标准化) → r_fit (拟合奖励) → base_reward → final_reward → memory.score
   ↓           ↓ 越小越好         ↓ 越小越好        ↓ 越大越好        ↓ 越大越好      ↓ 越大越好       ↓ 作为选入标准
```

### 🔥 详细计算流程：

1. **MSE**: `np.mean((predictions - outputs) ** 2)` - 均方误差
2. **NMSE**: `MSE / var(y)` - 标准化均方误差 
3. **r_fit**: `exp(-λ_nmse × NMSE)` - 拟合奖励分量
4. **base_reward**: `w_fit×r_fit + w_simp×r_simp + w_phys×r_phys + w_proc×r_proc`
5. **final_reward**: `base_reward - length_penalty + parse_reward`
6. **memory.score**: `= final_reward` (用于岛屿选择和排序)

## 🏝️ 群岛机制超参数

### 默认配置
- **群岛数量**: 4 个岛屿 (num_islands=4)
- **每岛top样本**: 8 个样本 (top_k_per_island=8)
- **Few-shot采样**: 3 个样本 (few_shot_k=3)

### 岛屿分配策略 (基于final_reward)
```
🏝️ 岛屿0 (score >= 0.7): 顶级岛屿 - 高质量解，MSE很小，表达式简洁有效
🏝️ 岛屿1 (0.4 <= score < 0.7): 优质岛屿 - 中高质量解，MSE较小，表达式合理  
🏝️ 岛屿2 (0.2 <= score < 0.4): 中等岛屿 - 中等质量解，MSE中等，可作为参考
🏝️ 岛屿3 (0.1 <= score < 0.2): 基础岛屿 - 低质量但可执行的解
```

### 选入memory标准
- ✅ **execution_success** = True (代码可执行)
- ✅ **final_reward** >= 0.1 (最低质量门槛)
- ✅ **function_body** 不为空 (有有效的函数体)

## 🔧 配置方式

### 1. Shell脚本配置 (`run_llmsr_grpo_v2.sh`)
```bash
# 🏝️ 群岛机制超参数配置
NUM_ISLANDS=${NUM_ISLANDS:-4}           # 群岛数量（默认4个）
TOP_K_PER_ISLAND=${TOP_K_PER_ISLAND:-8} # 每个岛屿保存的top样本数（默认8个）
FEW_SHOT_K=${FEW_SHOT_K:-3}             # Few-shot样本数（默认3个）
```

### 2. 命令行参数 (`main.py`)
```bash
python main.py --use_rl_v2 \
  --num_islands 4 \
  --top_k_per_island 8 \
  --few_shot_k 3 \
  ...
```

### 3. 环境变量方式
```bash
export NUM_ISLANDS=6           # 增加岛屿数量到6个
export TOP_K_PER_ISLAND=12     # 每个岛屿保存12个样本
export FEW_SHOT_K=5            # Few-shot使用5个样本
```

## 🎯 优化建议

### 针对不同训练规模的推荐配置

**小规模训练** (epochs <= 3):
```bash
NUM_ISLANDS=2        # 2个岛屿就够
TOP_K_PER_ISLAND=5   # 每岛5个样本
FEW_SHOT_K=2         # Few-shot 2个样本
```

**中等规模训练** (3 < epochs <= 7):
```bash
NUM_ISLANDS=4        # 4个岛屿 (默认)
TOP_K_PER_ISLAND=8   # 每岛8个样本 (默认)
FEW_SHOT_K=3         # Few-shot 3个样本 (默认)
```

**大规模训练** (epochs > 7):
```bash
NUM_ISLANDS=6        # 6个岛屿，更精细分层
TOP_K_PER_ISLAND=12  # 每岛12个样本，更多选择
FEW_SHOT_K=5         # Few-shot 5个样本，更丰富示例
```

## 🔍 调试检查

### Memory文件检查
```bash
# 检查memory是否为空
cat ./llmsr_grpo_outputs/*/memory_v2/memory_v2.json | jq '.'

# 统计各岛屿样本数量
cat ./llmsr_grpo_outputs/*/memory_v2/memory_v2.json | jq 'to_entries | map({island: .key, count: (.value | length)})'
```

### 实时监控Memory更新
```bash
# 监控奖励函数日志中的Memory更新信息
tail -f ./llmsr_logs/*.log | grep -E "(添加样本到memory|Memory管理器|岛屿)"
```

### 检查Few-shot是否生效
```bash
# 检查训练数据集中是否包含Few-shot示例
grep -A 10 -B 5 "Few-shot program skeletons" ./llmsr_grpo_outputs/*/llmsr_train_v2.parquet 2>/dev/null || echo "需要转换parquet格式检查"
```

## 🚨 已修复的Bug

1. **缺少add_sample方法** → ✅ 已添加完整的样本添加逻辑
2. **硬编码超参数** → ✅ 所有超参数可通过脚本/命令行配置
3. **Memory更新缺失** → ✅ 奖励函数现在会自动更新Memory
4. **参数传递链断裂** → ✅ 完整的参数传递路径：脚本→main.py→训练函数→奖励函数

## 🎉 修复验证

修复后，你应该能看到：
- ✅ `memory_v2.json` 不再是空的
- ✅ 各岛屿中有按score分层的样本
- ✅ Few-shot示例能够在训练过程中生效
- ✅ 日志中能看到"添加样本到memory"的信息
