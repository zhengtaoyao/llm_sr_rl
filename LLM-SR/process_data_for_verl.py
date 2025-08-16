#!/usr/bin/env python3
"""
🔥 LLM-SR数据集处理器 - 为VERL GRPO训练准备数据

这个脚本处理@/data目录中的所有数据集，将其转换为符合VERL格式要求的数据，
确保包含必需的`reward_model.ground_truth`字段以避免KeyError。

用法:
    python process_data_for_verl.py
    
输出:
    在verl_datasets/目录中生成所有问题的parquet格式数据集
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import re


def extract_function_names(specification: str) -> Tuple[str, str]:
    """从specification模板中提取函数名"""
    
    # 查找 @evaluate.run 装饰的函数
    run_pattern = r'@evaluate\.run\s+def\s+(\w+)'
    run_match = re.search(run_pattern, specification)
    run_function = run_match.group(1) if run_match else "evaluate_function"
    
    # 查找 @equation.evolve 装饰的函数  
    evolve_pattern = r'@equation\.evolve\s+def\s+(\w+)'
    evolve_match = re.search(evolve_pattern, specification)
    evolve_function = evolve_match.group(1) if evolve_match else "symbolic_regression"
    
    return evolve_function, run_function


def build_problem_prompt(template_content: str, problem_name: str) -> str:
    """根据模板构建问题的提示词"""
    
    # 提取模板中@equation.evolve函数之前的部分作为基础提示
    lines = template_content.split('\n')
    prompt_lines = []
    in_evolve_function = False
    
    for line in lines:
        if '@equation.evolve' in line:
            in_evolve_function = True
            continue
        if in_evolve_function and line.strip().startswith('def '):
            # 找到函数定义，包含它并停止
            prompt_lines.append(line.rstrip())
            break
        if not in_evolve_function:
            prompt_lines.append(line.rstrip())
    
    base_prompt = '\n'.join(prompt_lines).strip()
    
    # 根据问题添加特定的指导
    if 'oscillator' in problem_name:
        task_description = """
你的任务是发现描述简谐振荡器运动的物理方程。
给定位置x、速度v，预测加速度a。
请使用物理学知识，考虑胡克定律和简谐运动的特性。
"""
    elif 'bactgrow' in problem_name:
        task_description = """
你的任务是发现描述细菌生长速率的生物学方程。
给定细菌浓度b、底物浓度s、温度temp、pH值，预测细菌生长速率db。
请考虑微生物学中的Monod方程、温度和pH对生长的影响。
"""
    elif 'stressstrain' in problem_name:
        task_description = """
你的任务是发现描述材料应力-应变关系的工程方程。
请考虑材料力学中的胡克定律、塑性变形等概念。
"""
    else:
        task_description = """
你的任务是通过符号回归发现数据中隐藏的数学方程。
请分析输入变量和输出变量之间的关系，提出合理的数学表达式。
"""
    
    return base_prompt + task_description


def get_ground_truth_equations() -> Dict[str, str]:
    """获取各个问题的理论方程（作为ground truth）"""
    
    return {
        'oscillator1': 'a = -x',  # 简谐振荡器: F = -kx, a = F/m = -kx (假设k/m=1)
        'oscillator2': 'a = -x - 0.1*v',  # 带阻尼的振荡器
        'bactgrow': 'db = r * b * s / (K + s) * f(temp) * g(pH)',  # Monod方程
        'stressstrain': 'stress = E * strain'  # 胡克定律
    }


def create_verl_dataset_entry(
    prompt: str, 
    problem_name: str, 
    data_sample: Dict[str, float], 
    ground_truth: str,
    data_source: str = "llm_sr_train"
) -> Dict[str, Any]:
    """创建单个VERL数据集条目"""
    
    # 构建对话格式的提示（LLM需要chat格式）
    chat_messages = [
        {
            "role": "user", 
            "content": prompt
        }
    ]
    
    # 创建VERL格式的数据条目，确保包含所有必需字段
    entry = {
        "prompt": chat_messages,
        "data_source": data_source,
        "ability": f"symbolic_regression_{problem_name}",  # 添加能力标识
        "extra_info": {
            "problem_type": problem_name,
            "task": "symbolic_regression",
            "domain": "scientific_equation_discovery",
            "data_sample": data_sample  # 包含具体的数据样本
        },
        "reward_model": {
            "style": "rule",  # 使用基于规则的评估
            "ground_truth": ground_truth,  # 🔥 关键：提供ground truth避免KeyError
            "evaluation_type": "mse_based",  # 基于MSE的评估
            "target_variables": list(data_sample.keys())  # 目标变量
        }
    }
    
    return entry


def process_single_dataset(
    problem_name: str, 
    data_dir: Path, 
    spec_dir: Path,
    output_dir: Path,
    max_samples: int = 1000
) -> bool:
    """处理单个数据集"""
    
    print(f"\n🔄 处理数据集: {problem_name}")
    
    # 检查必要文件
    train_file = data_dir / problem_name / "train.csv"
    spec_file = spec_dir / f"specification_{problem_name}_numpy.txt"
    
    if not train_file.exists():
        print(f"❌ 训练数据不存在: {train_file}")
        return False
        
    if not spec_file.exists():
        print(f"❌ 规范文件不存在: {spec_file}")
        return False
    
    # 读取数据和模板
    try:
        df = pd.read_csv(train_file)
        with open(spec_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
            
        print(f"📊 数据形状: {df.shape}")
        print(f"📊 数据列: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return False
    
    # 提取函数名和构建提示
    evolve_function, run_function = extract_function_names(template_content)
    prompt = build_problem_prompt(template_content, problem_name)
    
    # 获取ground truth
    ground_truths = get_ground_truth_equations()
    ground_truth = ground_truths.get(problem_name, f"未知方程，需要通过{evolve_function}函数发现")
    
    print(f"🎯 目标函数: {evolve_function}")
    print(f"🔍 评估函数: {run_function}")
    print(f"📜 理论方程: {ground_truth}")
    
    # 限制样本数量以提高训练效率
    num_samples = min(max_samples, len(df))
    df_sampled = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    print(f"📊 选择样本数: {num_samples}")
    
    # 创建VERL格式的数据条目
    dataset_entries = []
    
    for idx, row in df_sampled.iterrows():
        # 将数据行转换为字典
        data_sample = row.to_dict()
        
        # 创建数据条目
        entry = create_verl_dataset_entry(
            prompt=prompt,
            problem_name=problem_name,
            data_sample=data_sample,
            ground_truth=ground_truth,
            data_source=f"llm_sr_{problem_name}_train"
        )
        
        dataset_entries.append(entry)
    
    # 保存为parquet格式
    output_file = output_dir / f"{problem_name}_train_verl.parquet"
    
    try:
        # 转换为PyArrow表格并保存
        table = pa.Table.from_pylist(dataset_entries)
        pq.write_table(table, output_file)
        
        print(f"✅ 成功保存: {output_file}")
        print(f"📁 文件大小: {output_file.stat().st_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
        return False


def main():
    """主函数：处理所有数据集"""
    
    print("🔥 LLM-SR 数据集处理器 - 为VERL GRPO训练准备数据")
    print("=" * 70)
    
    # 设置路径
    base_dir = Path(".")
    data_dir = base_dir / "data"
    spec_dir = base_dir / "specs"
    output_dir = base_dir / "verl_datasets"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    print(f"📁 输出目录: {output_dir.absolute()}")
    
    # 检查输入目录
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
        
    if not spec_dir.exists():
        print(f"❌ 规范目录不存在: {spec_dir}")
        return
    
    # 获取所有可用的问题
    available_problems = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "train.csv").exists():
            available_problems.append(item.name)
    
    print(f"🔍 发现的数据集: {available_problems}")
    
    if not available_problems:
        print("❌ 未找到任何可用的数据集")
        return
    
    # 处理每个数据集
    success_count = 0
    total_count = len(available_problems)
    
    for problem_name in available_problems:
        try:
            success = process_single_dataset(
                problem_name=problem_name,
                data_dir=data_dir,
                spec_dir=spec_dir,
                output_dir=output_dir,
                max_samples=1000  # 限制每个数据集的样本数
            )
            
            if success:
                success_count += 1
                
        except Exception as e:
            print(f"❌ 处理{problem_name}时出错: {e}")
            continue
    
    # 总结
    print(f"\n{'='*70}")
    print(f"🎉 数据处理完成！")
    print(f"✅ 成功处理: {success_count}/{total_count} 个数据集")
    print(f"📁 输出目录: {output_dir.absolute()}")
    
    if success_count > 0:
        print(f"\n📋 可用的VERL数据集:")
        for file in output_dir.glob("*_train_verl.parquet"):
            file_size = file.stat().st_size / 1024
            print(f"  - {file.name} ({file_size:.1f} KB)")
        
        print(f"\n🚀 下一步: 修改run_llmsr_grpo_direct.sh使用新的数据集路径")
        print(f"   例如: --dataset_path verl_datasets/oscillator1_train_verl.parquet")
    else:
        print(f"❌ 没有成功处理任何数据集")


if __name__ == "__main__":
    main() 