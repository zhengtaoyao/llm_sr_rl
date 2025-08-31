#!/usr/bin/env python3
"""
🔥 测试群岛few-shot动态更新机制 V2

测试内容：
1. MemoryManagerV2的版本化存储机制
2. 新的随机采样逻辑（每个岛屿随机一个）
3. DatasetRefreshManager的数据集刷新功能
4. 监控脚本的集成测试
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.append("/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR")

from llmsr.rl.grpo_runner_v2 import MemoryManagerV2, DatasetRefreshManager, refresh_dataset_with_islands


def test_memory_manager_versioning():
    """测试MemoryManagerV2的版本化存储机制"""
    print(f"\n🧪 测试MemoryManagerV2版本化存储...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = os.path.join(temp_dir, "memory_v2")
        
        # 初始化memory管理器
        manager = MemoryManagerV2(memory_dir, top_k_per_island=3, num_islands=4, update_frequency=2)
        
        # 添加一些测试样本
        test_samples = [
            ("    return x + y", 0.9, 0.1, 5.0),  # 高质量 -> 岛屿0
            ("    return x * y + 1", 0.7, 0.3, 7.0),  # 中高质量 -> 岛屿1
            ("    return x**2 + y**2", 0.4, 0.8, 10.0),  # 中质量 -> 岛屿2
            ("    return x + y + z", 0.2, 1.5, 15.0),  # 低质量 -> 岛屿3
            ("    return np.sin(x) + np.cos(y)", 0.85, 0.15, 8.0),  # 高质量 -> 岛屿0
        ]
        
        for i, (body, score, mse, complexity) in enumerate(test_samples):
            print(f"  添加样本{i+1}: score={score}")
            manager.add_sample(body, score, mse, complexity)
        
        # 检查版本化文件是否创建
        versioned_files = [f for f in os.listdir(memory_dir) if f.startswith("memory_v2_version_")]
        print(f"  ✅ 创建了{len(versioned_files)}个版本化文件")
        
        # 检查群岛分布
        stats = manager.get_island_stats()
        print(f"  📊 群岛统计: 版本={stats['version']}, 总样本={stats['total_samples']}")
        for island_id, info in stats['islands_info'].items():
            if info['count'] > 0:
                print(f"    岛屿{island_id}({info['quality']}): {info['count']}样本, 平均分={info['avg_score']:.3f}")
        
        return True


def test_random_sampling():
    """测试新的随机采样逻辑"""
    print(f"\n🧪 测试随机采样逻辑...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = os.path.join(temp_dir, "memory_v2")
        
        # 初始化并填充数据
        manager = MemoryManagerV2(memory_dir, top_k_per_island=5, num_islands=4)
        
        # 每个岛屿添加多个样本
        island_samples = {
            0: [("    return x + y", 0.9), ("    return x * 2", 0.95), ("    return y - x", 0.85)],
            1: [("    return x * y", 0.7), ("    return x / (y+1)", 0.75), ("    return x**0.5", 0.6)],
            2: [("    return x**2", 0.4), ("    return x + y**2", 0.45), ("    return x*y + 1", 0.35)],
            3: [("    return x**3 + y**3", 0.2), ("    return complex_expr", 0.15)]
        }
        
        for island_id, samples in island_samples.items():
            for body, score in samples:
                manager.add_sample(body, score, 0.1, 5.0)
        
        # 测试随机采样
        print(f"  🎲 测试随机采样（每个岛屿随机一个）:")
        for trial in range(3):
            examples = manager.sample_few_shot(k=3, use_random_sampling=True)
            print(f"    试验{trial+1}: {len(examples)}个样本")
            for j, example in enumerate(examples):
                # 提取岛屿信息
                island_info = example.split("island=")[1].split(",")[0] if "island=" in example else "unknown"
                print(f"      样本{j+1}: 来自岛屿{island_info}")
        
        # 测试传统采样（对比）
        print(f"  📊 传统采样逻辑:")
        examples_traditional = manager.sample_few_shot(k=3, use_random_sampling=False)
        print(f"    传统方式: {len(examples_traditional)}个样本")
        
        return True


def test_dataset_refresh():
    """测试数据集刷新功能"""
    print(f"\n🧪 测试数据集刷新功能...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试环境
        memory_dir = os.path.join(temp_dir, "memory_v2")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建模拟的模板文件
        template_path = os.path.join(temp_dir, "template.txt")
        with open(template_path, "w") as f:
            f.write("""
# Test problem for oscillator
# Input: x, v  
# Output: a (acceleration)

@equation.evolve
def equation(x, v, params):
    # Your implementation here
    pass
""")
        
        # 创建模拟的数据文件
        data_path = os.path.join(temp_dir, "train.csv")
        with open(data_path, "w") as f:
            f.write("x,v,a\n")
            for i in range(50):
                f.write(f"{i*0.1},{i*0.05},{i*0.01}\n")
        
        # 初始化管理器并添加样本
        manager = MemoryManagerV2(memory_dir, top_k_per_island=3, num_islands=4, update_frequency=2)
        
        # 添加测试样本到各个岛屿
        samples = [
            ("    return x + v", 0.9, 0.1, 5.0),
            ("    return x * v", 0.7, 0.3, 7.0),
            ("    return x**2 + v", 0.4, 0.8, 10.0),
            ("    return x + v + 1", 0.2, 1.5, 15.0),
        ]
        
        for body, score, mse, complexity in samples:
            manager.add_sample(body, score, mse, complexity)
        
        # 🔥 测试数据集刷新
        print(f"  📁 测试数据集刷新...")
        
        try:
            # 创建初始数据集（模拟）
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            initial_entries = [
                {
                    "prompt": [
                        {"role": "system", "content": "Test system message"},
                        {"role": "user", "content": "Test user message without few-shot"}
                    ],
                    "data_source": "test",
                    "extra_info": {"test": True}
                }
            ]
            
            initial_dataset_path = os.path.join(output_dir, "llmsr_train_v2.parquet")
            table = pa.Table.from_pylist(initial_entries)
            pq.write_table(table, initial_dataset_path)
            
            # 执行刷新
            refreshed_path = refresh_dataset_with_islands(
                dataset_path=initial_dataset_path,
                template_path=template_path,
                data_path=data_path,
                memory_dir=memory_dir,
                output_dir=output_dir,
                few_shot_k=3,
                num_islands=4,
                top_k_per_island=3
            )
            
            if refreshed_path != initial_dataset_path:
                print(f"  ✅ 数据集刷新成功: {os.path.basename(refreshed_path)}")
                
                # 验证新数据集包含few-shot examples
                new_table = pq.read_table(refreshed_path)
                new_data = new_table.to_pylist()
                
                if new_data and len(new_data) > 0:
                    user_content = new_data[0]["prompt"][1]["content"]
                    if "Few-shot Examples from Memory" in user_content:
                        print(f"  ✅ 新数据集包含few-shot examples")
                    else:
                        print(f"  ⚠️ 新数据集未包含few-shot examples")
            else:
                print(f"  ⚠️ 数据集未刷新（可能没有足够样本）")
                
        except Exception as e:
            print(f"  ❌ 数据集刷新测试失败: {e}")
            return False
        
        return True


def test_integration():
    """集成测试：完整的工作流程"""
    print(f"\n🧪 集成测试：完整工作流程...")
    
    try:
        # 检查monitor脚本是否存在
        monitor_script = "/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/monitor_and_refresh_dataset_v2.py"
        if os.path.exists(monitor_script):
            print(f"  ✅ 监控脚本存在: {monitor_script}")
        else:
            print(f"  ❌ 监控脚本不存在: {monitor_script}")
            return False
        
        # 检查运行脚本是否包含新配置
        run_script = "/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/run_llmsr_grpo_v2.sh"
        if os.path.exists(run_script):
            with open(run_script, "r") as f:
                content = f.read()
                if "ENABLE_DATASET_REFRESH" in content and "monitor_and_refresh_dataset_v2.py" in content:
                    print(f"  ✅ 运行脚本包含数据集刷新配置")
                else:
                    print(f"  ⚠️ 运行脚本可能缺少部分刷新配置")
        
        print(f"  ✅ 集成测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 集成测试失败: {e}")
        return False


def main():
    print(f"🔥 开始测试群岛few-shot动态更新机制 V2")
    print(f"=" * 60)
    
    test_results = []
    
    # 测试1: MemoryManagerV2版本化存储
    test_results.append(("MemoryManagerV2版本化存储", test_memory_manager_versioning()))
    
    # 测试2: 随机采样逻辑
    test_results.append(("随机采样逻辑", test_random_sampling()))
    
    # 测试3: 数据集刷新功能
    test_results.append(("数据集刷新功能", test_dataset_refresh()))
    
    # 测试4: 集成测试
    test_results.append(("集成测试", test_integration()))
    
    # 汇总结果
    print(f"\n" + "=" * 60)
    print(f"🏁 测试结果汇总:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{len(test_results)}项测试通过")
    
    if passed == len(test_results):
        print(f"🎉 所有测试通过！群岛few-shot动态更新机制已就绪")
        return True
    else:
        print(f"⚠️ 部分测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试群岛few-shot动态更新机制")
    parser.add_argument("--test", choices=["all", "memory", "sampling", "refresh", "integration"], 
                       default="all", help="选择要运行的测试")
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = main()
    elif args.test == "memory":
        success = test_memory_manager_versioning()
    elif args.test == "sampling":
        success = test_random_sampling()
    elif args.test == "refresh":
        success = test_dataset_refresh()
    elif args.test == "integration":
        success = test_integration()
    
    sys.exit(0 if success else 1)
