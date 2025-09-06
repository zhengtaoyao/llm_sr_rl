"""
GRPO v2 Training Runner for LLM-SR

本文件实现一版更稳健的 LLM+GRPO→SR 流程（v2）：
- 奖励密化与成分化：拟合/复杂度/物理一致性/过程奖励 + 组内排名归一
- 解码与动作约束：沿用原有骨架模板；可扩展常数占位符<C*>与后验拟合
- 记忆与多样性：轻量“岛屿/记忆库”检索 few-shot（文件级共享，跨进程安全）
- 稳健 GRPO 配置：同一提示组内采样、显式 KL 损失、token-mean 聚合

说明：为避免干扰旧实现，所有逻辑独立于 v1 文件，main.py 分支进入本文件。
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

from omegaconf import OmegaConf, DictConfig

# 将 VERL 加入路径
verl_path = str(Path(__file__).parent.parent.parent.parent / "verl")
if verl_path not in sys.path:
    sys.path.append(verl_path)

from verl.trainer.main_ppo import run_ppo


# 轻量记忆管理（文件共享，跨进程安全）- V2改进版：自适应分位数分桶
class MemoryManagerV2:
    def __init__(self, memory_dir: str, top_k_per_island: int = 8, num_islands: int = 4,
                 update_frequency: int = 50, recent_samples_window: int = 200) -> None:
        os.makedirs(memory_dir, exist_ok=True)
        self._memory_dir = memory_dir  # 🔥 保存目录用于创建版本化文件
        self._path = os.path.join(memory_dir, "memory_v2.json")
        self._history_path = os.path.join(memory_dir, "sample_history.json")  # 🔥 新增：样本历史
        self._top_k = top_k_per_island
        self._num_islands = num_islands
        self._update_frequency = update_frequency  # 每N次写库后重新计算分位数
        self._recent_samples_window = recent_samples_window  # 最近M条样本用于计算分位数
        self._samples_since_update = 0  # 自上次更新后的样本计数
        self._update_version = 0  # 🔥 新增：版本计数器
        self._samples_since_last_dataset_refresh = 0  # 🔥 新增：自上次数据集刷新后的新样本数
        self._dataset_refresh_threshold = 8  # 🔥 新增：触发数据集刷新的新样本阈值

        if not os.path.exists(self._path):
            # 初始化空结构
            import json
            init = {
                "islands": {str(i): [] for i in range(self._num_islands)},
                "adaptive_thresholds": None,  # 🔥 自适应阈值，初始为None使用默认分桶
                "last_update_count": 0,
                "version": 0,  # 🔥 新增：版本号
                "last_dataset_refresh_version": 0,  # 🔥 记录上次数据集刷新时的版本号
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(init, f)
        
        # 初始化样本历史文件
        if not os.path.exists(self._history_path):
            import json
            with open(self._history_path, "w", encoding="utf-8") as f:
                json.dump({"samples": []}, f)

    def load(self) -> Dict[str, Any]:
        import json
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 🔥 兼容旧格式：如果是旧格式，转换为新格式
                if "islands" not in data:
                    old_data = data
                    data = {
                        "islands": old_data,
                        "adaptive_thresholds": None,
                        "last_update_count": 0,
                        "version": 0,  # 🔥 新增版本号
                        "last_dataset_refresh_version": 0  # 🔥 新增数据集刷新版本号
                    }
                # 🔥 兼容缺失字段的情况
                elif "version" not in data:
                    data["version"] = 0
                if "last_dataset_refresh_version" not in data:
                    data["last_dataset_refresh_version"] = 0
                return data
        except Exception:
            return {
                "islands": {str(i): [] for i in range(self._num_islands)},
                "adaptive_thresholds": None,
                "last_update_count": 0,
                "version": 0,  # 🔥 新增版本号
                "last_dataset_refresh_version": 0  # 🔥 新增数据集刷新版本号
            }

    def save(self, data: Dict[str, Any], create_versioned_copy: bool = False) -> None:
        """
        保存memory数据
        Args:
            data: 要保存的数据
            create_versioned_copy: 是否创建带时间戳的版本化副本
        """
        import json
        import time
        from datetime import datetime
        
        # 🔥 更新版本号
        current_version = data.get("version", 0)
        data["version"] = current_version + 1
        self._update_version = data["version"]
        
        # 保存主文件
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 🔥 创建带时间戳的版本化副本
        if create_versioned_copy:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_filename = f"memory_v2_version_{data['version']:04d}_{timestamp}.json"
            versioned_path = os.path.join(self._memory_dir, versioned_filename)
            
            with open(versioned_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"🗃️ 创建版本化memory副本: {versioned_filename}")
    
    def _load_sample_history(self) -> List[Dict[str, Any]]:
        """加载样本历史"""
        import json
        try:
            with open(self._history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                return history.get("samples", [])
        except Exception:
            return []
    
    def _save_sample_history(self, samples: List[Dict[str, Any]]) -> None:
        """保存样本历史（保留最近N条）"""
        import json
        # 只保留最近的样本
        recent_samples = samples[-self._recent_samples_window:] if len(samples) > self._recent_samples_window else samples
        try:
            with open(self._history_path, "w", encoding="utf-8") as f:
                json.dump({"samples": recent_samples}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def _compute_adaptive_thresholds(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """基于最近样本计算自适应分位数阈值"""
        if len(samples) < 20:  # 样本太少时使用默认阈值
            return None
        
        import numpy as np
        scores = [s.get("score", 0.0) for s in samples if "score" in s]
        if len(scores) < 10:
            return None
        
        scores = np.array(scores)
        # 🔥 计算分位数：P90→岛0(high), P70→岛1(mid-high), P40→岛2(mid), 其余→岛3(low)
        try:
            p90 = np.percentile(scores, 90)
            p70 = np.percentile(scores, 70) 
            p40 = np.percentile(scores, 40)
            
            thresholds = {
                "p90": float(p90),  # 岛屿0阈值（high quality）
                "p70": float(p70),  # 岛屿1阈值（mid-high quality）
                "p40": float(p40),  # 岛屿2阈值（mid quality）
                # 岛屿3：剩余样本（low quality）
            }
            
            print(f"🏝️ 自适应阈值更新: P90={p90:.3f}, P70={p70:.3f}, P40={p40:.3f}")
            return thresholds
        except Exception as e:
            print(f"⚠️ 计算自适应阈值失败: {e}")
            return None

    def sample_few_shot(self, k: int = 3, use_random_sampling: bool = True) -> List[str]:
        """
        🔥 V2改进版：跨岛屿采样多样few-shot，带质量标签/元信息
        
        Args:
            k: 最大采样数量（如果use_random_sampling=True则忽略）
            use_random_sampling: 是否使用新的随机采样逻辑（每个岛屿随机一个）
        
        Returns:
            格式化的示例列表，包含质量标签和使用指导
        """
        data = self.load()
        islands = data.get("islands", {})
        examples: List[str] = []
        
        if not islands:
            return examples
        
        quality_levels = ["high", "mid-high", "mid", "low"]
        
        if use_random_sampling:
            # 🔥 新的采样逻辑：从四个岛屿中每个岛屿随机抽取一个样本
            import random
            selected_samples = []
            
            for island_id in range(self._num_islands):
                island_id_str = str(island_id)
                items = islands.get(island_id_str, [])
                
                if not items:
                    continue  # 该岛屿为空，跳过
                
                # 🔥 从该岛屿随机选择一个样本
                selected_item = random.choice(items)
                
                # 确定质量等级
                quality = quality_levels[min(island_id, 3)]
                
                impl = selected_item.get("implementation", "")
                if impl:
                    score = selected_item.get("score", 0.0)
                    mse = selected_item.get("mse")
                    complexity = selected_item.get("complexity")
                    
                    # 🔥 构建质量标签注释
                    metadata = f"island={island_id}, quality={quality}, reward={score:.2f}"
                    if mse is not None:
                        metadata += f", mse={mse:.3f}"
                    if complexity is not None:
                        metadata += f", complexity={complexity:.1f}"
                    
                    # 🔥 添加使用指导（根据质量级别）
                    if quality == "high":
                        guidance = ""  # 高质量样本无需额外指导
                    elif quality == "mid-high":
                        guidance = "  # good pattern, minor refinements may help"
                    elif quality == "mid":
                        guidance = "  # exploratory pattern only; prefer smoother/parsimonious forms"
                    else:  # low quality
                        guidance = "  # counter-example; avoid this pattern; analyze why quality is low"
                    
                    # 🔥 格式化示例
                    formatted_example = f"# Example [{metadata}]{guidance}\n{impl.rstrip()}"
                    examples.append(formatted_example)
            
            print(f"🎯 随机Few-shot采样完成: {len(examples)}个示例，来自{len(examples)}个不同岛屿")
            return examples
        
        else:
            # 🔥 保留原有的采样逻辑（兼容性）
            # 构建所有候选样本，按岛屿分组
            all_candidates = []
            
            for island_id, items in islands.items():
                if not items:
                    continue
                
                # 确定质量等级
                island_idx = int(island_id) if island_id.isdigit() else 3
                quality = quality_levels[min(island_idx, 3)]
                
                # 优先score高的前几项
                items_sorted = sorted(items, key=lambda x: x.get("score", -1.0), reverse=True)
                
                for it in items_sorted[:2]:  # 每个岛屿最多取2个
                    impl = it.get("implementation", "")
                    if impl:
                        score = it.get("score", 0.0)
                        mse = it.get("mse")
                        complexity = it.get("complexity")
                        
                        # 🔥 构建质量标签注释
                        metadata = f"island={island_id}, quality={quality}, reward={score:.2f}"
                        if mse is not None:
                            metadata += f", mse={mse:.3f}"
                        if complexity is not None:
                            metadata += f", complexity={complexity:.1f}"
                        
                        # 🔥 添加使用指导（根据质量级别）
                        if quality == "high":
                            guidance = ""  # 高质量样本无需额外指导
                        elif quality == "mid-high":
                            guidance = "  # good pattern, minor refinements may help"
                        elif quality == "mid":
                            guidance = "  # exploratory pattern only; prefer smoother/parsimonious forms"
                        else:  # low quality
                            guidance = "  # counter-example; avoid this pattern; analyze why quality is low"
                        
                        # 🔥 格式化示例
                        formatted_example = f"# Example [{metadata}]{guidance}\n{impl.rstrip()}"
                        
                        candidate = {
                            "content": formatted_example,
                            "island": island_idx,
                            "quality": quality,
                            "score": score
                        }
                        all_candidates.append(candidate)
            
            if not all_candidates:
                return examples
            
            # 🔥 按质量和分数排序：高质量在前，同质量内按分数排序
            all_candidates.sort(key=lambda x: (x["island"], -x["score"]))
            
            # 🔥 优先选择高质量示例，确保多样性
            selected = []
            
            # 首先尽量选择高质量样本（岛屿0和1）
            high_quality = [c for c in all_candidates if c["island"] <= 1]
            selected.extend(high_quality[:max(1, k//2)])
            
            # 然后添加中等质量样本保持多样性
            mid_quality = [c for c in all_candidates if c["island"] > 1]
            remaining = k - len(selected)
            if remaining > 0:
                selected.extend(mid_quality[:remaining])
            
            # 提取内容
            examples = [c["content"] for c in selected[:k]]
            
            print(f"🎯 Few-shot采样完成: {len(examples)}个示例，质量分布: {[c['quality'] for c in selected[:k]]}")
            return examples
    
    def add_sample(self, function_body: str, score: float, mse: float = None, complexity: float = None) -> bool:
        """
        🔥 V2改进版：添加优秀样本到记忆库（自适应分位数分桶）
        
        Returns:
            bool: 是否需要刷新数据集（当新样本数>=8时）
        """
        if not function_body or score < 0.1:  # 过滤低质量样本
            return False
        
        try:
            data = self.load()
            islands = data.get("islands", {})
            adaptive_thresholds = data.get("adaptive_thresholds")
            
            # 🔥 基于自适应阈值选择目标岛屿
            if adaptive_thresholds and all(k in adaptive_thresholds for k in ["p90", "p70", "p40"]):
                # 使用自适应分位数阈值
                if score >= adaptive_thresholds["p90"]:
                    target_island = "0"  # 高质量岛屿（P90以上）
                elif score >= adaptive_thresholds["p70"]:
                    target_island = "1"  # 中高质量岛屿（P70-P90）
                elif score >= adaptive_thresholds["p40"]:
                    target_island = "2"  # 中质量岛屿（P40-P70）
                else:
                    target_island = "3"  # 低质量岛屿（P40以下）
            else:
                # 兜底：使用固定阈值（向后兼容）
                if score >= 0.8:
                    target_island = "0"
                elif score >= 0.5:
                    target_island = "1"
                elif score >= 0.3:
                    target_island = "2"
                else:
                    target_island = "3"
            
            # 构建样本记录
            sample = {
                "implementation": function_body,
                "score": float(score),
                "mse": float(mse) if mse is not None else None,
                "complexity": float(complexity) if complexity is not None else None,
                "timestamp": time.time()
            }
            
            # 添加到目标岛屿
            if target_island not in islands:
                islands[target_island] = []
            
            islands[target_island].append(sample)
            
            # 保持每个岛屿最多top_k个样本（按score排序）
            islands[target_island] = sorted(islands[target_island], key=lambda x: x.get("score", 0), reverse=True)[:self._top_k]
            
            # 🔥 更新样本历史
            sample_history = self._load_sample_history()
            sample_history.append(sample)
            self._save_sample_history(sample_history)
            
            # 🔥 检查是否需要更新自适应阈值
            self._samples_since_update += 1
            create_version = False  # 🔥 新增：是否创建版本化副本
            
            if self._samples_since_update >= self._update_frequency:
                recent_samples = sample_history[-self._recent_samples_window:] if len(sample_history) > self._recent_samples_window else sample_history
                new_thresholds = self._compute_adaptive_thresholds(recent_samples)
                if new_thresholds:
                    data["adaptive_thresholds"] = new_thresholds
                    create_version = True  # 🔥 阈值更新时创建版本
                data["last_update_count"] = len(sample_history)
                self._samples_since_update = 0
                print(f"🔄 已触发自适应阈值更新，基于{len(recent_samples)}个最近样本")
            
            # 更新数据结构
            data["islands"] = islands
            
            # 🔥 增加新样本计数
            self._samples_since_last_dataset_refresh += 1
            
            # 🔥 检查是否需要刷新数据集（基于新样本数）
            last_refresh_version = data.get("last_dataset_refresh_version", 0)
            current_version = data.get("version", 0)
            
            # 计算自上次刷新以来的版本差异（近似为新样本数的指标）
            version_diff = current_version - last_refresh_version
            need_refresh = self._samples_since_last_dataset_refresh >= self._dataset_refresh_threshold
            
            if need_refresh:
                print(f"🔄 达到数据集刷新阈值：新样本数={self._samples_since_last_dataset_refresh} >= {self._dataset_refresh_threshold}")
                # 🔥 记录刷新时的版本号，重置计数器
                data["last_dataset_refresh_version"] = current_version + 1  # +1因为下面会再增加版本号
                self._samples_since_last_dataset_refresh = 0
                create_version = True  # 强制创建版本化副本
            
            # 🔥 保存更新后的数据，在重要更新时创建版本化副本
            self.save(data, create_versioned_copy=create_version)
            
            threshold_info = ""
            if adaptive_thresholds:
                threshold_info = f" (自适应阈值: P90={adaptive_thresholds['p90']:.3f})"
            
            refresh_info = f" [触发数据集刷新]" if need_refresh else ""
            print(f"✅ 成功添加样本到岛屿{target_island}，score: {score:.3f}{threshold_info}{refresh_info}")
            
            return need_refresh
            
        except Exception as e:
            print(f"⚠️ 添加样本到memory失败: {e}")
            return False
    

    
    def get_island_stats(self) -> Dict[str, Any]:
        """
        🔥 获取群岛统计信息（用于监控）
        """
        data = self.load()
        islands = data.get("islands", {})
        stats = {
            "version": data.get("version", 0),
            "total_samples": sum(len(items) for items in islands.values()),
            "islands_info": {}
        }
        
        quality_levels = ["high", "mid-high", "mid", "low"]
        for island_id, items in islands.items():
            island_idx = int(island_id) if island_id.isdigit() else 3
            quality = quality_levels[min(island_idx, 3)]
            
            if items:
                scores = [item.get("score", 0.0) for item in items]
                stats["islands_info"][island_id] = {
                    "quality": quality,
                    "count": len(items),
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores)
                }
            else:
                stats["islands_info"][island_id] = {
                    "quality": quality,
                    "count": 0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "min_score": 0.0
                }
        
        return stats





def _extract_prompt_header(spec_text: str) -> str:
    lines = spec_text.split("\n")
    prompt_lines: List[str] = []
    in_evolve = False
    for line in lines:
        if "@equation.evolve" in line:
            in_evolve = True
            continue
        if in_evolve and line.strip().startswith("def "):
            prompt_lines.append(line.rstrip())
            break
        if not in_evolve:
            prompt_lines.append(line.rstrip())
    return "\n".join(prompt_lines).strip()


def refresh_dataset_with_islands(
    dataset_path: str,
    template_path: str,
    data_path: str,
    memory_dir: str,
    output_dir: str,
    grid_train_data: bool = False,
    num_grid_groups: int = 10,
    few_shot_k: int = 3,
    num_islands: int = 4,
    top_k_per_island: int = 8,
) -> str:
    """
    🔥 新增：动态刷新数据集，从群岛中重新采样few-shot examples并生成新的数据集
    
    Args:
        dataset_path: 现有数据集路径
        template_path: 模板文件路径
        data_path: 原始数据路径
        memory_dir: 群岛memory目录
        output_dir: 输出目录
        其他参数: 与create_llmsr_dataset_v2相同
        
    Returns:
        新数据集文件路径
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import datetime
    
    print(f"🔄 开始刷新数据集，从群岛重新采样few-shot examples...")
    
    # 🔥 从群岛中重新采样few-shot examples（使用随机采样）
    memory = MemoryManagerV2(memory_dir, top_k_per_island=top_k_per_island, num_islands=num_islands)
    examples = memory.sample_few_shot(k=few_shot_k, use_random_sampling=True)
    
    if not examples:
        print(f"⚠️ 群岛中没有样本，保持原数据集不变")
        return dataset_path
    
    # 🔥 重新构建带few-shot的prompt
    with open(template_path, "r", encoding="utf-8") as f:
        spec_text = f.read()
    base_prompt = _extract_prompt_header(spec_text)
    
    # 🔥 添加护栏说明（英文）
    guardrail_instruction = """
# === Few-shot Examples from Memory (Quality-Guided) ===
# PRIORITY GUIDANCE: 
# - Follow examples marked as "quality=high" primarily
# - Examples with "quality=mid-high" show good patterns with minor refinements needed
# - Examples with "quality=mid" are exploratory only; prefer smoother/parsimonious forms
# - Examples with "quality=low" are counter-examples; analyze why quality is low and avoid similar patterns
# - Focus on mathematical correctness, simplicity, and numerical stability
"""
    
    # 🔥 格式化few-shot块
    few_shot_content = guardrail_instruction + "\n" + "\n\n".join(examples)
    few_shot_block = few_shot_content
    composed_prompt = (base_prompt + few_shot_block).strip()
    
    # 🔥 读取原有数据集的数据部分（保持相同的数据分布）
    try:
        # 尝试读取原始parquet文件的数据
        original_table = pq.read_table(dataset_path)
        original_data = original_table.to_pylist()
        
        print(f"✅ 从原数据集读取{len(original_data)}条记录")
        
        # 🔥 更新每条记录的prompt部分
        updated_entries = []
        for entry in original_data:
            # 保持原有的系统消息
            original_prompt = entry.get("prompt", [])
            if len(original_prompt) >= 2:
                system_msg = original_prompt[0]  # 保持系统消息不变
                # 更新用户消息为新的few-shot prompt
                updated_prompt = [
                    system_msg,
                    {"role": "user", "content": composed_prompt}
                ]
                entry["prompt"] = updated_prompt
            
            updated_entries.append(entry)
        
        # 🔥 生成新的数据集文件（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dataset_path = os.path.join(output_dir, f"llmsr_train_v2_refreshed_{timestamp}.parquet")
        
        table = pa.Table.from_pylist(updated_entries)
        pq.write_table(table, new_dataset_path)
        
        print(f"✅ 数据集刷新完成: {new_dataset_path}")
        print(f"🎯 新few-shot样本数: {len(examples)}")
        
        return new_dataset_path
        
    except Exception as e:
        print(f"❌ 数据集刷新失败: {e}")
        # 如果失败，重新创建数据集
        print(f"🔄 回退到重新创建数据集...")
        return create_llmsr_dataset_v2(
            template_path=template_path,
            data_path=data_path,
            output_dir=output_dir,
            memory_dir=memory_dir,
            grid_train_data=grid_train_data,
            num_grid_groups=num_grid_groups,
            few_shot_k=few_shot_k,
            num_islands=num_islands,
            top_k_per_island=top_k_per_island,
        )


def create_llmsr_dataset_v2(
    template_path: str,
    data_path: str,
    output_dir: str,
    memory_dir: str,
    grid_train_data: bool = False,
    num_grid_groups: int = 10,
    few_shot_k: int = 3,
    # 🏝️ 群岛机制超参数
    num_islands: int = 4,           # 群岛数量
    top_k_per_island: int = 8,      # 每个岛屿保存的top样本数
) -> str:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import KBinsDiscretizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    with open(template_path, "r", encoding="utf-8") as f:
        spec_text = f.read()
    base_prompt = _extract_prompt_header(spec_text)

    # 🔥 V2改进版：few-shot 拼接 + 护栏提示（英文）- 使用随机采样
    memory = MemoryManagerV2(memory_dir, top_k_per_island=top_k_per_island, num_islands=num_islands)
    examples = memory.sample_few_shot(k=few_shot_k, use_random_sampling=True)
    
    if examples:
        # 🔥 添加护栏说明（英文）
        guardrail_instruction = """
# === Few-shot Examples from Memory (Quality-Guided) ===
# PRIORITY GUIDANCE: 
# - Follow examples marked as "quality=high" primarily
# - Examples with "quality=mid-high" show good patterns with minor refinements needed
# - Examples with "quality=mid" are exploratory only; prefer smoother/parsimonious forms
# - Examples with "quality=low" are counter-examples; analyze why quality is low and avoid similar patterns
# - Focus on mathematical correctness, simplicity, and numerical stability
"""
        
        # 🔥 格式化few-shot块
        few_shot_content = guardrail_instruction + "\n" + "\n\n".join(examples)
        few_shot_block = few_shot_content
    else:
        few_shot_block = ""
    
    composed_prompt = (base_prompt + few_shot_block).strip()

    df = pd.read_csv(data_path)
    max_samples = len(df)

    # Grid-based sampling（按问题自动检测输入/输出列）
    if grid_train_data:
        input_cols: List[str] = []
        output_col: str | None = None
        if "oscillator" in data_path:
            input_cols = [c for c in ["x", "v"] if c in df.columns] or df.columns[:-1].tolist()
            output_col = "a" if "a" in df.columns else df.columns[-1]
        elif "bactgrow" in data_path:
            input_cols = [c for c in ["b", "s", "temp", "pH"] if c in df.columns] or df.columns[:-1].tolist()
            output_col = "db" if "db" in df.columns else df.columns[-1]
        elif "stressstrain" in data_path:
            input_cols = [c for c in ["strain", "temp"] if c in df.columns] or df.columns[:-1].tolist()
            output_col = "stress" if "stress" in df.columns else df.columns[-1]
        else:
            input_cols = df.columns[:-1].tolist()
            output_col = df.columns[-1]

        # 1D/2D/ND 分桶
        if len(input_cols) == 1:
            discretizer = KBinsDiscretizer(n_bins=num_grid_groups, encode="ordinal", strategy="uniform")
            df["grid_group"] = discretizer.fit_transform(df[input_cols].values)
        elif len(input_cols) == 2:
            import math
            nbin = int(max(1, round(math.sqrt(num_grid_groups))))
            dx = KBinsDiscretizer(n_bins=nbin, encode="ordinal", strategy="uniform")
            dy = KBinsDiscretizer(n_bins=nbin, encode="ordinal", strategy="uniform")
            gx = dx.fit_transform(df[[input_cols[0]]].values)
            gy = dy.fit_transform(df[[input_cols[1]]].values)
            df["grid_group"] = gx * nbin + gy
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_grid_groups, random_state=42)
            df["grid_group"] = kmeans.fit_predict(df[input_cols].values)

        df["grid_group"] = df["grid_group"].astype(int)
        samples_per_group = max(1, max_samples // num_grid_groups)
        parts = []
        for gid in range(num_grid_groups):
            sub = df[df["grid_group"] == gid]
            if len(sub) == 0:
                continue
            parts.append(sub.sample(n=min(samples_per_group, len(sub)), random_state=42))
        df_s = pd.concat(parts).reset_index(drop=True)
    else:
        df_s = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
        df_s["grid_group"] = 0

    # 自动检测输入/输出列
    input_cols: List[str] = []
    output_col: str | None = None
    
    # 根据问题类型自动检测输入/输出列
    if "oscillator" in data_path:
        input_cols = [c for c in ["x", "v"] if c in df_s.columns] or df_s.columns[:-1].tolist()
        output_col = "a" if "a" in df_s.columns else df_s.columns[-1]
    elif "bactgrow" in data_path:
        input_cols = [c for c in ["b", "s", "temp", "pH"] if c in df_s.columns] or df_s.columns[:-1].tolist()
        output_col = "db" if "db" in df_s.columns else df_s.columns[-1]
    elif "stressstrain" in data_path:
        input_cols = [c for c in ["strain", "temp"] if c in df_s.columns] or df_s.columns[:-1].tolist()
        output_col = "stress" if "stress" in df_s.columns else df_s.columns[-1]
    else:
        # 默认：假设最后一列是输出，其余是输入
        input_cols = df_s.columns[:-1].tolist()
        output_col = df_s.columns[-1]
    
    print(f"🎯 检测到输入列: {input_cols}, 输出列: {output_col}")

    # 组装 VERL 数据
    dataset_entries: List[Dict[str, Any]] = []
    for i in range(len(df_s)):
        row = df_s.iloc[i]
        
        # 🔥 提取真实的 ground truth 值（CSV中的因变量值）
        ground_truth_value = float(row[output_col]) if output_col in row else None
        
        # 增加 system 约束说明；user 中放入规范 + few-shot
        system_prefix = (
            "You generate equation skeletons under grammar/AST constraints.\n"
            "- Prefer valid mathematical expressions over prose.\n"
            "- Optionally use EDIT DSL to modify a provided base expression.\n"
            "EDIT Usage: 'EDIT ADD <expr>' | 'EDIT MUL <expr>' | 'EDIT REPLACE <old> => <new>'\n"
            "Return final Pythonic expression or a minimal function body."
        )
        chat_prompt = [
            {"role": "system", "content": system_prefix},
            {"role": "user", "content": composed_prompt},
        ]

        entry = {
            "prompt": chat_prompt,
            "data_source": "llm_sr_train_v2",
            "reward_model": {
                "style": "rule",
                # 🔥 正确使用CSV中的因变量值作为ground truth
                "ground_truth": ground_truth_value
            },
            "extra_info": {
                "grid_group": int(row["grid_group"]),
                # 记录原始数据点用于潜在物理一致性与过程奖励
                "data_point": row.drop("grid_group").to_dict(),
                # 记录输入/输出列信息
                "input_cols": input_cols,
                "output_col": output_col,
                # 可选传递基底表达式（由后续管线填充），用于 EDIT 模式
                "base_impl": None,
            },
        }
        dataset_entries.append(entry)

    table = pa.Table.from_pylist(dataset_entries)
    out_path = os.path.join(output_dir, "llmsr_train_v2.parquet")
    pq.write_table(table, out_path)
    return out_path


def create_llmsr_reward_file_v2(
    template_path: str, 
    data_path: str, 
    output_dir: str, 
    memory_dir: str, 
    grid_train_data: bool = False,
    length_penalty_alpha: float = 0.03,
    parse_bonus: float = 0.1,
    invalid_penalty: float = -0.5,
    enable_physics_reward: bool = False,
    enable_process_reward: bool = True,  # 🔥 新增过程奖励开关
    # 🏝️ 群岛机制超参数
    num_islands: int = 4,           # 群岛数量
    top_k_per_island: int = 8,      # 每个岛屿保存的top样本数
    # 🔥 数据集刷新参数
    refresh_manager_config: Dict[str, Any] = None,  # 数据集刷新管理器配置
) -> str:
    # 🔥 生成刷新管理器配置字符串
    refresh_config_str = "None"
    if refresh_manager_config:
        refresh_config_str = repr(refresh_manager_config)
    
    code = f'''"""
Wrapper for v2 reward to plug into VERL custom_reward_function.
"""
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from llmsr.rl.simple_verl_reward_v2_fixed import compute_score as compute_score_v2

# 🔥 设置输出目录环境变量，确保sample.jsonl写入正确位置
os.environ["LLMSR_OUTPUT_DIR"] = "{output_dir}"

def compute_score(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
    return compute_score_v2(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        grid_train_data={grid_train_data},
        template_path="{template_path}",
        data_path="{data_path}",
        memory_dir="{memory_dir}",
        length_penalty_alpha={length_penalty_alpha},
        parse_bonus={parse_bonus},
        invalid_penalty={invalid_penalty},
        enable_physics_reward={enable_physics_reward},
        enable_process_reward={enable_process_reward},
        num_islands={num_islands},
        top_k_per_island={top_k_per_island},
        refresh_manager_config={refresh_config_str},
        **kwargs
    )
'''
    out = os.path.join(output_dir, "llmsr_reward_v2.py")
    with open(out, "w", encoding="utf-8") as f:
        f.write(code)
    return out


def create_grpo_config_v2(
    model_path: str,
    dataset_path: str,
    reward_file_path: str,
    output_dir: str,
    *,
    gpus: int = 8,
    rollout_n: int = 4,
    kl_coef: float = 1e-3,
    max_prompt_length: int = 4096,  # 🔥 增加到4096
    max_new_tokens: int = 8192,     # 🔥 增加到8192
    max_model_len: int = 16384,     # 🔥 增加到16384
    max_num_batched_tokens: int = 8192,  # 🔥 增加到8192
    learning_rate: float = 1e-6,
    epochs: int = 5,
) -> DictConfig:
    # token 长度安全阈值
    safe_max_token_len = max(16384, int(max_prompt_length + max_new_tokens + 512))  # 🔥 增加到16384
    micro_bsz_per_gpu = 1
    traj_micro_bsz = micro_bsz_per_gpu * gpus
    traj_mini_bsz = max(2, traj_micro_bsz * 2)
    prompt_mini_bsz = traj_mini_bsz * rollout_n
    prompt_bsz = prompt_mini_bsz

    cfg = {
        "algorithm": {
            "_target_": "verl.trainer.config.AlgoConfig",
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "kl_penalty": "kl",
            "use_pf_ppo": False,
            "pf_ppo": {
                "reweight_method": "pow",
                "weight_pow": 2.0
            },
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "fixed", 
                "kl_coef": float(kl_coef), 
                "horizon": 10000, 
                "target_kl": 0.1
            },
        },
        "data": {
            "tokenizer": None,
            "use_shm": False,
            "train_files": [dataset_path],
            "val_files": [dataset_path],
            "prompt_key": "prompt",
            "reward_fn_key": "data_source",
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_new_tokens,
            "train_batch_size": prompt_bsz,
            "val_batch_size": prompt_mini_bsz,
            "return_raw_input_ids": False,
            "return_raw_chat": False,
            "return_full_prompt": False,
            "shuffle": True,
            "dataloader_num_workers": 8,
            "validation_shuffle": False,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "truncation": "error",
            "image_key": "images",
            "video_key": "videos",
            "trust_remote_code": False,
            "custom_cls": {
                "path": None,
                "name": None
            },
            "return_multi_modal_inputs": True,
            # 🔧 避免 Missing key data.sampler
            "sampler": {
                "class_path": None,
                "class_name": None
            },
            "datagen": {
                "path": None,
                "name": None
            },
            "apply_chat_template_kwargs": {}
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "use_fused_kernels": False,
                "trust_remote_code": False,
                "custom_chat_template": None,
                "external_lib": None,
                "override_config": {},
                "enable_activation_offload": False,
                "lora_rank": 0,
                "lora_alpha": 16,
                "target_modules": "all-linear",
                "exclude_modules": None,
                "use_liger": False,
                "fused_kernel_options": {}
            },
            "actor": {
                "_target_": "verl.workers.config.FSDPActorConfig",
                "strategy": "fsdp",
                "optim": {
                    "_target_": "verl.workers.config.FSDPOptimizerConfig",
                    "lr": learning_rate, 
                    "weight_decay": 0.01,
                    "lr_warmup_steps_ratio": 0.0,
                    "total_training_steps": -1,
                    "lr_warmup_steps": -1,
                    "min_lr_ratio": 0.0,
                    "num_cycles": 0.5,
                    "warmup_style": "constant"
                },
                "ppo_mini_batch_size": prompt_mini_bsz,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": micro_bsz_per_gpu,
                "use_kl_loss": True,
                "kl_loss_coef": float(kl_coef),
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": 0.0,
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",  # 处理长度偏置
                "use_dynamic_bsz": True,
                "ulysses_sequence_parallel_size": 1,
                "ppo_max_token_len_per_gpu": safe_max_token_len,
                "use_remove_padding": True,
                "use_fused_kernels": False,
                "use_torch_compile": False,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,
                "shuffle": False,
                "policy_loss": {
                    "_target_": "verl.workers.config.PolicyLossConfig",
                    "loss_mode": "vanilla",
                    "clip_cov_ratio": 0.0002,
                    "clip_cov_lb": 1.0,
                    "clip_cov_ub": 5.0,
                    "kl_cov_ratio": 0.0002,
                    "ppo_kl_coef": 0.1
                },
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "forward_prefetch": False,
                    "offload_policy": False,
                    "reshard_after_forward": True,
                    "wrap_policy": {"min_num_params": 0},
                },
                "checkpoint": {
                    "_target_": "verl.trainer.config.CheckpointConfig",
                    "save_contents": ["model", "optimizer", "extra"], 
                    "load_contents": ["model", "optimizer", "extra"],
                    "async_save": False
                },
                "profiler": {
                    "_target_": "verl.utils.profiler.ProfilerConfig",
                    "tool": None,
                    "enable": False,
                    "all_ranks": False,
                    "ranks": [],
                    "save_path": None,
                    "tool_config": {
                        "nsys": {
                            "_target_": "verl.utils.profiler.config.NsightToolConfig",
                            "discrete": False
                        },
                        "torch": {
                            "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                            "step_start": -1,
                            "step_end": -1
                        },
                        "torch_memory": {
                            "_target_": "verl.utils.profiler.config.TorchMemoryToolConfig",
                            "trace_alloc_max_entries": 100000,
                            "stack_depth": 32
                        },
                        "npu": {
                            "_target_": "verl.utils.profiler.config.NPUToolConfig",
                            "contents": [],
                            "level": "level1",
                            "analysis": False,
                            "discrete": False
                        }
                    },
                    "global_tool_config": None
                },
            },
            "rollout": {
                "_target_": "verl.workers.config.RolloutConfig",
                "name": "vllm",
                "mode": "sync",
                "n": rollout_n,
                "temperature": 0.7,  # 🔥 降低温度减少重复
                "top_p": 0.95,       # 🔥 增加top_p提高多样性
                "top_k": 50,         # 🔥 增加top_k
                "do_sample": True,
                "over_sample_rate": 0,
                "prompt_length": max_prompt_length,
                "response_length": max_new_tokens,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.6,
                "ignore_eos": False,
                "enforce_eager": True,
                "cudagraph_capture_sizes": None,
                "free_cache_engine": True,
                "tensor_model_parallel_size": 1,
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_model_len": max_model_len,
                "max_num_seqs": 1024,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_bsz_per_gpu,
                "log_prob_use_dynamic_bsz": True,
                "log_prob_max_token_len_per_gpu": safe_max_token_len,
                "disable_log_stats": False,
                "multi_stage_wake_up": False,
                "engine_kwargs": {
                    "vllm": {},  # 🔥 penalty参数已移到rollout配置中
                    "sglang": {}
                },
                "calculate_log_probs": False,
                "agent": {
                    "_target_": "verl.workers.config.AgentLoopConfig",
                    "num_workers": 8,
                    "agent_loop_config_path": None,
                    "custom_async_server": {
                        "_target_": "verl.workers.config.CustomAsyncServerConfig",
                        "path": None,
                        "name": None
                    }
                },
                "trace": {
                    "_target_": "verl.workers.config.TraceConfig",
                    "backend": None,
                    "token2text": False
                },
                "update_weights_bucket_megabytes": 512,
                "skip_rollout": False,
                "skip_dump_dir": "/tmp/rollout_dump",
                "profiler": {
                    "_target_": "verl.utils.profiler.ProfilerConfig",
                    "tool": None,
                    "enable": False,
                    "all_ranks": False,
                    "ranks": [],
                    "save_path": None,
                    "tool_config": {
                        "nsys": {
                            "_target_": "verl.utils.profiler.config.NsightToolConfig",
                            "discrete": False
                        }
                    },
                    "global_tool_config": None
                },
                "enable_chunked_prefill": False,
                "load_format": "auto",
                "layered_summon": False,
                "layer_name_map": {},
                "val_kwargs": {
                    "_target_": "verl.workers.config.SamplingConfig",
                    "do_sample": True, 
                    "temperature": 0.7,  # 🔥 降低温度减少重复
                    "top_p": 0.95,       # 🔥 增加top_p提高多样性
                    "top_k": 50,         # 🔥 增加top_k
                    "n": 1  # 🔥 CRITICAL: 验证时采样数量必须为1
                },
                # 🔥 CRITICAL: 添加缺失的rollout字段 (v2模式)
                "calculate_log_probs": False,  # 用于调试的rollout概率记录
                "free_cache_engine": True,  # 生成后释放KV缓存引擎
                "ignore_eos": False,
                "over_sample_rate": 0,
                "multi_stage_wake_up": False,
                "engine_kwargs": {
                    "vllm": {},  # 🔥 penalty参数已移到rollout配置中
                    "sglang": {}
                },
                "update_weights_bucket_megabytes": 512,
                "trace": {
                    "_target_": "verl.workers.config.TraceConfig",
                    "backend": None,
                    "token2text": False
                },
                "skip_rollout": False,
                "skip_dump_dir": "/tmp/rollout_dump",
                "profiler": {
                    "_target_": "verl.utils.profiler.ProfilerConfig",
                    "tool": None,
                    "enable": False,
                    "all_ranks": False,
                    "ranks": [],
                    "save_path": None,
                    "tool_config": {
                        "nsys": {
                            "_target_": "verl.utils.profiler.config.NsightToolConfig",
                            "discrete": False
                        }
                    },
                    "global_tool_config": None
                },
                "enable_chunked_prefill": False,
                "load_format": "auto",
                "layered_summon": False,
                "layer_name_map": {},
                "multi_turn": {
                    "_target_": "verl.workers.config.MultiTurnConfig",
                    "enable": False, 
                    "max_assistant_turns": None, 
                    "tool_config_path": None, 
                    "max_user_turns": None,
                    "max_parallel_calls": 1,
                    "max_tool_response_length": 256,
                    "tool_response_truncate_side": "middle",
                    "interaction_config_path": None,
                    "use_inference_chat_template": False, 
                    "tokenization_sanity_check_mode": "strict", 
                    "format": "hermes"
                },
            },
            "ref": {
                "log_prob_micro_batch_size": None,
                # 🔥 CRITICAL: ref模型必须有正确的微批量大小 (不能为0!)
                "log_prob_micro_batch_size_per_gpu": micro_bsz_per_gpu,
                # 🔥 CRITICAL: 添加缺失的 ulysses_sequence_parallel_size 字段
                "ulysses_sequence_parallel_size": 1,  # 与 actor 保持一致
                "log_prob_use_dynamic_bsz": True,
                # 参考模型 log_prob 的最大 token 长度同样使用安全阈值
                "log_prob_max_token_len_per_gpu": safe_max_token_len,
                # 🔥 添加 DataParallelPPOActor 需要的其他字段
                "use_remove_padding": True,  # 与 actor 保持一致
                "use_fused_kernels": False,  # 禁用融合内核
                "entropy_from_logits_with_chunking": False,
                "use_torch_compile": False,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,  # 梯度裁剪，即使ref不优化也需要
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "fsdp_size": gpus, 
                    "param_offload": True, 
                    "optimizer_offload": True,
                    "forward_prefetch": False,  # 🔥 禁用前向预取以节省内存
                    "offload_policy": False,
                    "reshard_after_forward": True,
                    "wrap_policy": {"min_num_params": 0}
                },
            },
            "hybrid_engine": True,
        },
        "critic": {
            "_target_": "verl.workers.config.FSDPCriticConfig",
            "enable": True,  # 🔧 修复 Missing key critic.enable
            "strategy": "fsdp",
            "rollout_n": rollout_n,
            "ppo_mini_batch_size": prompt_mini_bsz,
            "ppo_micro_batch_size": None,
            "ppo_micro_batch_size_per_gpu": micro_bsz_per_gpu,
            "use_dynamic_bsz": True,
            "forward_micro_batch_size": None,
            "forward_micro_batch_size_per_gpu": micro_bsz_per_gpu,
            "ulysses_sequence_parallel_size": 1,
            "grad_clip": 1.0,
            "optim": {
                "_target_": "verl.workers.config.FSDPOptimizerConfig",
                "lr": learning_rate,
                "weight_decay": 0.01,
                "lr_warmup_steps_ratio": 0.0,
                "total_training_steps": -1,
                "lr_warmup_steps": -1,
                "min_lr_ratio": None,
                "warmup_style": "constant"
            },
            "model": {
                "_target_": "verl.workers.config.FSDPCriticModelCfg",
                "path": model_path,
                "tokenizer_path": model_path,
                "override_config": {},
                "external_lib": None,
                "trust_remote_code": False,
                "use_shm": False,
                "enable_gradient_checkpointing": True,
                "enable_activation_offload": False,
                "use_remove_padding": True,
                "lora_rank": 0,
                "lora_alpha": 16,
                "target_modules": "all-linear",
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "param_offload": True,
                    "optimizer_offload": True,
                    "offload_policy": False,
                    "reshard_after_forward": True,
                    "wrap_policy": {"min_num_params": 0},
                    "fsdp_size": gpus,
                    "forward_prefetch": False
                }
            },
            "ppo_epochs": 1,
            "shuffle": False,
            "cliprange_value": 0.5,
            "loss_agg_mode": "token-mean",
            "ppo_max_token_len_per_gpu": safe_max_token_len,
            "forward_max_token_len_per_gpu": safe_max_token_len,
            "checkpoint": {
                "_target_": "verl.trainer.config.CheckpointConfig",
                "save_contents": ["model", "optimizer", "extra"],
                "load_contents": ["model", "optimizer", "extra"],
                "async_save": False
            },
            "profiler": {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": None,
                "enable": False,
                "all_ranks": False,
                "ranks": [],
                "save_path": None,
                "tool_config": {
                    "nsys": {
                        "_target_": "verl.utils.profiler.config.NsightToolConfig",
                        "discrete": False
                    },
                    "torch": {
                        "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                        "step_start": -1,
                        "step_end": -1
                    },
                    "torch_memory": {
                        "_target_": "verl.utils.profiler.config.TorchMemoryToolConfig",
                        "trace_alloc_max_entries": 100000,
                        "stack_depth": 32
                    },
                    "npu": {
                        "_target_": "verl.utils.profiler.config.NPUToolConfig",
                        "contents": [],
                        "level": "level1",
                        "analysis": False,
                        "discrete": False
                    }
                },
                "global_tool_config": None
            },

        },
        "reward_model": {"enable": False, "launch_reward_fn_async": False},
        "custom_reward_function": {"path": reward_file_path, "name": "compute_score"},
        "trainer": {
            "total_epochs": epochs,
            "total_training_steps": None,
            "project_name": "llm_sr_grpo_v2",
            "experiment_name": "direct_weight_tuning_v2",
            "logger": ["console", "wandb"],
            "n_gpus_per_node": gpus,
            "nnodes": 1,
            "save_freq": 2,
            "test_freq": 5,
            "val_before_train": True,
            "default_local_dir": output_dir,
            "device": "cuda",
            "resume_mode": "disable",
            "critic_warmup": 0,
            "balance_batch": True,
            "log_val_generations": 10,
            "log_train_generations": 5,
            "profile_steps": None,
            "default_hdfs_dir": None,
            "log_prob_max_token_len_per_gpu": safe_max_token_len,
            
            # 🔧 添加缺失的 ESI 配置项
            "esi_redundant_time": 0.0,
            "esi_enable": False,
        },
        "ray_kwargs": {
            "ray_init": {
                "num_cpus": None,
                "runtime_env": {
                    "env_vars": {
                        "PYTHONPATH": os.environ.get("LOCAL_PYTHON_PATH", "/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/verl:/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR")
                    }
                }
            }
        },
        "global_profiler": {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": None,
            "steps": None,
            "profile_continuous_steps": False,
            "save_path": "outputs/profile",
            "global_tool_config": {
                "nsys": {
                    "_target_": "verl.utils.profiler.config.NsightToolConfig",
                    "discrete": False
                },
                "torch_memory": {
                    "trace_alloc_max_entries": 100000,
                    "stack_depth": 32,
                    "context": "all",
                    "stacks": "all",
                    "kw_args": {}
                }
            }
        },
    }
    return OmegaConf.create(cfg)


def train_llmsr_grpo_v2(
    template_path: str,
    data_path: str,
    model_path: str,
    output_dir: str,
    *,
    grid_train_data: bool = False,
    num_grid_groups: int = 10,
    gpus: int = 8,
    rollout_n: int = 4,
    kl_coef: float = 1e-3,
    max_prompt_length: int = 4096,  # 🔥 增加到4096
    max_new_tokens: int = 8192,     # 🔥 增加到8192
    max_model_len: int = 16384,     # 🔥 增加到16384
    max_num_batched_tokens: int = 8192,  # 🔥 增加到8192
    learning_rate: float = 1e-6,
    epochs: int = 5,
    few_shot_k: int = 3,
    # 🏝️ 群岛机制超参数
    num_islands: int = 4,           # 群岛数量
    top_k_per_island: int = 8,      # 每个岛屿保存的top样本数
    # 🔥 新增长度惩罚和解析奖励参数
    length_penalty_alpha: float = 0.03,  # 长度惩罚系数，建议0.02-0.05
    parse_bonus: float = 0.1,            # 解析成功奖励
    invalid_penalty: float = -0.5,       # 无效样本惩罚
    # 🔥 物理一致性奖励开关（默认关闭）
    enable_physics_reward: bool = False,  # 是否启用物理一致性奖励
    # 🔥 过程奖励开关（默认开启）
    enable_process_reward: bool = True,   # 是否启用真过程奖励
) -> None:
    # 🔥 修复输出目录命名，使其与v1版本一致包含时间戳
    import time
    from datetime import datetime
    
    # 从data_path提取问题名称
    problem_name = "unknown"
    if "oscillator1" in data_path:
        problem_name = "oscillator1"
    elif "oscillator2" in data_path:
        problem_name = "oscillator2"
    elif "bactgrow" in data_path:
        problem_name = "bactgrow"
    elif "stressstrain" in data_path:
        problem_name = "stressstrain"
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 🔥 使用与v1一致的命名格式：{problem}_qwen8b_v2_{timestamp}
    if output_dir.endswith("_v2") or output_dir.endswith("/oscillator1_v2"):
        # 如果传入的是简单的v2目录，则生成完整的带时间戳的目录名
        base_dir = os.path.dirname(output_dir) if "/" in output_dir else "./llmsr_grpo_outputs"
        output_dir = os.path.join(base_dir, f"{problem_name}_qwen8b_v2_{timestamp}")
        print(f"🔥 V2输出目录已更新为: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    memory_dir = os.path.join(output_dir, "memory_v2")
    os.makedirs(memory_dir, exist_ok=True)

    # 1) 数据集
    dataset_path = create_llmsr_dataset_v2(
        template_path=template_path,
        data_path=data_path,
        output_dir=output_dir,
        memory_dir=memory_dir,
        grid_train_data=grid_train_data,
        num_grid_groups=num_grid_groups,
        few_shot_k=few_shot_k,
        num_islands=num_islands,
        top_k_per_island=top_k_per_island,
    )

    # 2) 奖励文件
    reward_file_path = create_llmsr_reward_file_v2(
        template_path=template_path,
        data_path=data_path,
        output_dir=output_dir,
        memory_dir=memory_dir,
        grid_train_data=grid_train_data,
        length_penalty_alpha=length_penalty_alpha,
        parse_bonus=parse_bonus,
        invalid_penalty=invalid_penalty,
        enable_physics_reward=enable_physics_reward,
        enable_process_reward=enable_process_reward,
        num_islands=num_islands,
        top_k_per_island=top_k_per_island,
        # 🔥 传递数据集刷新参数（简化版）
        refresh_manager_config={
            "output_dir": output_dir,
            "refresh_params": {
                "current_dataset_path": dataset_path,
                "template_path": template_path,
                "data_path": data_path,
                "memory_dir": memory_dir,
                "grid_train_data": grid_train_data,
                "num_grid_groups": num_grid_groups,
                "few_shot_k": few_shot_k,
                "num_islands": num_islands,
                "top_k_per_island": top_k_per_island,
            }
        }
    )

    # 3) 配置
    config = create_grpo_config_v2(
        model_path=model_path,
        dataset_path=dataset_path,
        reward_file_path=reward_file_path,
        output_dir=output_dir,
        gpus=gpus,
        rollout_n=rollout_n,
        kl_coef=kl_coef,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # 4) 训练
    cfg_path = os.path.join(output_dir, "grpo_config_v2.yaml")
    OmegaConf.save(config, cfg_path)
    # 让奖励函数能写入样本文件
    os.environ["LLMSR_OUTPUT_DIR"] = output_dir
    run_ppo(config)

    # 5) 训练结束后，从 sample.jsonl 中选出最优
    try:
        import json
        best_reward = None
        best_mse = None
        best_reward_rec = None
        best_mse_rec = None
        jsonl_path = os.path.join(output_dir, "sample.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    r = rec.get("reward")
                    # 🔥 优先使用mse，如果没有则使用nmse（兼容v1和v2格式）
                    m = rec.get("mse") or rec.get("nmse")
                    if isinstance(r, (int, float)) and (best_reward is None or r > best_reward):
                        best_reward = r
                        best_reward_rec = rec
                    if isinstance(m, (int, float)) and (best_mse is None or m < best_mse):
                        best_mse = m
                        best_mse_rec = rec
        if best_reward_rec:
            with open(os.path.join(output_dir, "best_reward.json"), "w", encoding="utf-8") as f:
                json.dump(best_reward_rec, f, ensure_ascii=False, indent=2)
        if best_mse_rec:
            with open(os.path.join(output_dir, "best_mse.json"), "w", encoding="utf-8") as f:
                json.dump(best_mse_rec, f, ensure_ascii=False, indent=2)
    except Exception:
        pass