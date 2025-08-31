#!/usr/bin/env python3
"""
🔥 群岛few-shot机制演示脚本 V2

演示如何使用新的群岛few-shot动态更新机制：
1. 模拟样本添加到不同质量的岛屿
2. 演示随机采样逻辑
3. 演示数据集刷新过程
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.append("/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR")

from llmsr.rl.grpo_runner_v2 import MemoryManagerV2


def demo_islands_mechanism():
    """演示群岛机制"""
    print(f"🏝️ 群岛few-shot机制演示")
    print(f"=" * 50)
    
    # 使用实际的输出目录进行演示
    demo_output_dir = "/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/demo_islands_output"
    memory_dir = os.path.join(demo_output_dir, "memory_v2")
    os.makedirs(memory_dir, exist_ok=True)
    
    # 初始化memory管理器
    manager = MemoryManagerV2(
        memory_dir, 
        top_k_per_island=5, 
        num_islands=4, 
        update_frequency=3,  # 每3个样本更新阈值
        recent_samples_window=20
    )
    
    print(f"📁 演示目录: {demo_output_dir}")
    print(f"🗃️ Memory目录: {memory_dir}")
    
    # 🔥 模拟不同质量的样本
    print(f"\n1️⃣ 添加不同质量的样本到群岛...")
    
    samples = [
        # 高质量样本（应该进入岛屿0）
        ("    return -params[0] * x - params[1] * v", 0.92, 0.05, 6.0, "高质量：物理准确的振荡器方程"),
        ("    return -params[0] * x", 0.88, 0.08, 4.0, "高质量：简化的弹簧振荡"),
        
        # 中高质量样本（应该进入岛屿1）
        ("    return -params[0] * x - params[1] * v + params[2]", 0.75, 0.15, 8.0, "中高质量：含偏移项"),
        ("    return -params[0] * (x + params[1] * v)", 0.72, 0.18, 7.5, "中高质量：耦合形式"),
        
        # 中质量样本（应该进入岛屿2）
        ("    return -params[0] * x**2 - params[1] * v", 0.45, 0.45, 12.0, "中质量：非线性项"),
        ("    return -params[0] * np.sin(x) - params[1] * v", 0.40, 0.50, 15.0, "中质量：三角函数"),
        
        # 低质量样本（应该进入岛屿3）
        ("    return params[0] * x**3 + params[1] * v**3", 0.25, 1.2, 25.0, "低质量：过度非线性"),
        ("    return complex_function_with_many_terms", 0.15, 2.0, 35.0, "低质量：过于复杂"),
    ]
    
    for i, (body, score, mse, complexity, description) in enumerate(samples):
        print(f"  添加样本{i+1}: {description}")
        print(f"    代码: {body.strip()}")
        print(f"    质量: score={score}, mse={mse}, complexity={complexity}")
        manager.add_sample(body, score, mse, complexity)
        print()
    
    # 🔥 显示群岛分布
    print(f"2️⃣ 群岛分布状态...")
    stats = manager.get_island_stats()
    print(f"📊 Memory版本: {stats['version']}, 总样本: {stats['total_samples']}")
    
    quality_levels = ["高质量(high)", "中高质量(mid-high)", "中质量(mid)", "低质量(low)"]
    for island_id in range(4):
        island_id_str = str(island_id)
        info = stats['islands_info'].get(island_id_str, {})
        quality = quality_levels[island_id]
        
        if info.get('count', 0) > 0:
            print(f"  🏝️ 岛屿{island_id} [{quality}]: {info['count']}样本")
            print(f"     平均分: {info['avg_score']:.3f}, 最高分: {info['max_score']:.3f}")
        else:
            print(f"  🏝️ 岛屿{island_id} [{quality}]: 空")
    
    # 🔥 演示随机采样
    print(f"\n3️⃣ Few-shot随机采样演示...")
    print(f"🎲 进行3次随机采样，每次从每个非空岛屿随机选择一个样本:")
    
    for trial in range(3):
        print(f"\n  试验{trial+1}:")
        examples = manager.sample_few_shot(k=4, use_random_sampling=True)
        
        for j, example in enumerate(examples):
            lines = example.split('\n')
            # 提取元信息
            header = lines[0] if lines else ""
            code_line = lines[1] if len(lines) > 1 else ""
            
            island_info = "unknown"
            quality_info = "unknown"
            score_info = "unknown"
            
            if "island=" in header:
                try:
                    parts = header.split(", ")
                    for part in parts:
                        if part.startswith("island="):
                            island_info = part.split("=")[1]
                        elif part.startswith("quality="):
                            quality_info = part.split("=")[1]
                        elif part.startswith("reward="):
                            score_info = part.split("=")[1]
                except:
                    pass
            
            print(f"    样本{j+1}: 岛屿{island_info}({quality_info}), 分数={score_info}")
            print(f"           代码: {code_line.strip()}")
    
    # 🔥 检查版本化文件
    print(f"\n4️⃣ 版本化文件检查...")
    versioned_files = [f for f in os.listdir(memory_dir) if f.startswith("memory_v2_version_")]
    print(f"📁 创建了{len(versioned_files)}个版本化文件:")
    for vfile in sorted(versioned_files):
        print(f"  - {vfile}")
    
    print(f"\n🎉 演示完成！")
    print(f"💡 提示：在实际训练中，这些few-shot examples会动态地影响模型的下一轮生成")
    
    return demo_output_dir


def cleanup_demo(demo_dir: str):
    """清理演示文件"""
    try:
        if os.path.exists(demo_dir):
            import shutil
            shutil.rmtree(demo_dir)
            print(f"🧹 已清理演示目录: {demo_dir}")
    except Exception as e:
        print(f"⚠️ 清理演示目录失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="群岛few-shot机制演示")
    parser.add_argument("--cleanup", action="store_true", help="清理演示文件")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_demo("/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/demo_islands_output")
    else:
        demo_dir = demo_islands_mechanism()
        print(f"\n💡 如需清理演示文件，运行: python {__file__} --cleanup")
