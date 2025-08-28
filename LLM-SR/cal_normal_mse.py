#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 从 JSONL 样本中筛选最佳方程 V2
借鉴 reward 函数的执行方式，从包含大量样本的 jsonl 文件中
筛选出在 test_id 和 test_ood 上 NMSE 最小的两个样本
"""

import json, re, textwrap, types, math, ast
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ─────────────── 基础配置 ───────────────────────────────────────────────────
ROOT = "/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR/"


def nmse(pred, true):
    """Normalized MSE – identical to evaluator.py (+1e-12 for safety)."""
    return np.mean((pred - true) ** 2) / (np.mean(true ** 2) + 1e-12)


def load_csv(csv_path):
    """加载CSV文件并返回输入输出数据"""
    if not Path(csv_path).exists():
        print(f"⚠️ 文件不存在: {csv_path}")
        return None, None
    try:
        arr = pd.read_csv(csv_path).to_numpy(float)
        return arr[:, :-1], arr[:, -1]  # X matrix, y vector
    except Exception as e:
        print(f"❌ 加载CSV失败 {csv_path}: {e}")
        return None, None


def get_var_names_from_problem(problem_type):
    """根据问题类型获取变量名"""
    if problem_type in ["oscillator1", "oscillator2"]:
        return ['x', 'v']
    elif problem_type == "bactgrow":
        return ['b', 's', 'temp', 'pH']
    elif problem_type == "stressstrain":
        return ['strain', 'temp']
    else:
        print(f"⚠️ 未知问题类型: {problem_type}，使用默认变量名")
        return ['x', 'v']


# ═══════════════════════════════════════════════════════════════════
# 🔥 借鉴 cal_normal_mse_first2500.py 和 evaluator.py 的函数执行方法
# ═══════════════════════════════════════════════════════════════════

def compile_equation_from_body(function_body: str, var_names: list) -> tuple[callable, int]:
    """
    从函数体编译方程，返回可调用函数和参数数量。
    参考cal_normal_mse_first2500.py的compile_equation方法。
    """
    try:
        # 🔥 清理函数体：移除不可打印字符和特殊空格
        if function_body:
            # 替换各种特殊空格字符
            function_body = function_body.replace('\u00A0', ' ')  # 不间断空格
            function_body = function_body.replace('\u200B', '')   # 零宽空格
            function_body = function_body.replace('\u200C', '')   # 零宽非连接符
            function_body = function_body.replace('\u200D', '')   # 零宽连接符
            function_body = function_body.replace('\uFEFF', '')   # 字节顺序标记
            
            # 规范化空白字符，但保持行结构
            lines = function_body.split('\n')
            cleaned_lines = []
            for line in lines:
                if line.strip():  # 非空行
                    # 确保每行都有合适的缩进
                    cleaned_line = line.strip()
                    if cleaned_line:
                        cleaned_lines.append('    ' + cleaned_line)
                else:
                    cleaned_lines.append('')
            function_body = '\n'.join(cleaned_lines)
        
        if not function_body:
            print("❌ 函数体为空或清理后为空")
            return None, 0
        
        # 🔥 智能计算参数数量
        n_params = 10  # 默认值
        
        # 1. 从参数解包中推断，如：k, gamma, delta = params[:6]
        unpack_matches = re.findall(r'[a-zA-Z_][a-zA-Z0-9_,\s]*\s*=\s*params\s*\[\s*:\s*(\d+)\s*\]', function_body)
        if unpack_matches:
            n_params = max(int(m) for m in unpack_matches)
        
        # 2. 从直接索引访问中推断，如：params[5]
        elif re.search(r'params\s*\[\s*\d+\s*\]', function_body):
            param_indices = re.findall(r'params\s*\[\s*(\d+)\s*\]', function_body)
            if param_indices:
                n_params = max(int(idx) for idx in param_indices) + 1
        
        # 构建完整的函数定义
        params_str = ', '.join(var_names) + ', params'
        full_function = f"def equation({params_str}):\n{function_body}"
        
        # 🔥 更鲁棒的编译
        try:
            # 创建模块并执行
            mod = types.ModuleType("mod")
            mod.__dict__["np"] = np
            mod.__dict__["math"] = math
            
            exec(textwrap.dedent(full_function), mod.__dict__)
            eq = mod.equation
            
            return eq, n_params
            
        except SyntaxError as syntax_err:
            # 如果语法错误，尝试修复常见问题
            print(f"⚠️ 语法错误，尝试修复: {syntax_err}")
            return try_fix_syntax_errors(function_body, var_names)
        
    except Exception as e:
        print(f"❌ 编译函数失败: {e}")
        # 🔥 尝试更宽松的清理和修复
        return try_aggressive_cleanup(function_body, var_names)


def try_fix_syntax_errors(function_body: str, var_names: list) -> tuple[callable, int]:
    """尝试修复常见的语法错误"""
    try:
        # 修复1：确保所有行都有正确缩进
        lines = function_body.split('\n')
        fixed_lines = []
        for line in lines:
            if line.strip():
                # 确保非空行至少有4个空格缩进
                stripped = line.lstrip()
                if stripped and not line.startswith('    '):
                    fixed_lines.append('    ' + stripped)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append('')
        
        fixed_body = '\n'.join(fixed_lines)
        
        # 修复2：如果缺少return语句，添加默认return
        if 'return' not in fixed_body:
            fixed_body += '\n    return 0.0'
        
        params_str = ', '.join(var_names) + ', params'
        full_function = f"def equation({params_str}):\n{fixed_body}"
        
        mod = types.ModuleType("mod")
        mod.__dict__["np"] = np
        mod.__dict__["math"] = math
        
        exec(textwrap.dedent(full_function), mod.__dict__)
        eq = mod.equation
        
        return eq, 10  # 使用默认参数数量
        
    except Exception as e:
        print(f"❌ 语法修复失败: {e}")
        return None, 0


def try_aggressive_cleanup(function_body: str, var_names: list) -> tuple[callable, int]:
    """更激进的清理和修复"""
    try:
        if not function_body:
            return None, 0
            
        # 更激进的清理：移除所有非ASCII字符
        cleaned_body = ''.join(char for char in function_body if ord(char) < 128)
        
        if cleaned_body != function_body:
            print(f"⚠️ 尝试清理特殊字符后重新编译...")
            return compile_equation_from_body(cleaned_body, var_names)
        
        # 最后的尝试：构建一个最简单的占位函数
        print(f"⚠️ 使用占位函数")
        simple_body = "    return 0.0"
        params_str = ', '.join(var_names) + ', params'
        full_function = f"def equation({params_str}):\n{simple_body}"
        
        mod = types.ModuleType("mod")
        mod.__dict__["np"] = np
        mod.__dict__["math"] = math
        
        exec(full_function, mod.__dict__)
        eq = mod.equation
        
        return eq, 10
        
    except Exception:
        return None, 0


def evaluate_sample_with_params(function_body: str, params: list, inputs: np.ndarray, 
                               outputs: np.ndarray, var_names: list) -> float:
    """使用已有参数评估样本性能，不再进行优化"""
    
    try:
        # 编译函数
        eq, n_params = compile_equation_from_body(function_body, var_names)
        
        if eq is None:
            return float('inf')
        
        # 确保参数数组长度足够
        params_array = np.array(params)
        if len(params_array) < n_params:
            # 如果参数不够，用1填充
            padded_params = np.ones(n_params)
            padded_params[:len(params_array)] = params_array
            params_array = padded_params
        
        # 计算预测值
        if len(var_names) == 2:  # oscillator: x, v 或 stressstrain: strain, temp
            predictions = eq(inputs[:, 0], inputs[:, 1], params_array)
        elif len(var_names) == 4:  # bactgrow: b, s, temp, pH
            predictions = eq(inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], params_array)
        else:
            # 通用处理
            args = [inputs[:, j] for j in range(inputs.shape[1])] + [params_array]
            predictions = eq(*args)
        
        # 确保predictions是数组
        predictions = np.asarray(predictions, dtype=np.float64)
        
        # 处理标量返回值
        if predictions.ndim == 0:
            predictions = np.full_like(outputs, float(predictions))
        
        # 计算MSE
        mse = np.mean((predictions - outputs) ** 2)
        
        return float(mse) if np.isfinite(mse) else float('inf')
        
    except Exception as e:
        print(f"❌ 评估样本失败: {e}")
        return float('inf')


def load_jsonl_samples(jsonl_path: str) -> List[Dict]:
    """从jsonl文件加载所有样本"""
    samples = []
    
    if not Path(jsonl_path).exists():
        print(f"❌ JSONL文件不存在: {jsonl_path}")
        return samples
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 第{line_no}行JSON解析失败: {e}")
        
        print(f"✅ 成功加载 {len(samples)} 个样本")
        return samples
        
    except Exception as e:
        print(f"❌ 读取JSONL文件失败: {e}")
        return samples


def load_single_json_sample(json_path: str) -> Dict:
    """从单个JSON文件加载样本"""
    if not Path(json_path).exists():
        print(f"❌ JSON文件不存在: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        print(f"✅ 成功加载单个样本")
        return sample
    except Exception as e:
        print(f"❌ 读取JSON文件失败: {e}")
        return {}


def evaluate_sample_on_testsets(sample: Dict, problem_type: str, 
                               data_dir: str) -> Dict[str, float]:
    """评估单个样本在测试集上的性能，使用已有参数"""
    
    # 检查样本是否执行成功
    if not sample.get('execution_success', False):
        return {'test_id_mse': float('inf'), 'test_ood_mse': float('inf'), 
                'test_id_nmse': float('inf'), 'test_ood_nmse': float('inf')}
    
    function_body = sample.get('function_body')
    params = sample.get('params')
    
    if not function_body or not params:
        return {'test_id_mse': float('inf'), 'test_ood_mse': float('inf'),
                'test_id_nmse': float('inf'), 'test_ood_nmse': float('inf')}
    
    # 获取变量名
    var_names = get_var_names_from_problem(problem_type)
    
    # 测试集路径
    test_id_path = f"{data_dir}/test_id.csv"
    test_ood_path = f"{data_dir}/test_ood.csv"
    
    results = {}
    
    # 评估在test_id上的性能
    X_id, y_id = load_csv(test_id_path)
    if X_id is not None and y_id is not None:
        try:
            mse_id = evaluate_sample_with_params(function_body, params, X_id, y_id, var_names)
            results['test_id_mse'] = float(mse_id)
            # 计算NMSE
            if mse_id < float('inf'):
                nmse_id = mse_id / (np.var(y_id) + 1e-12)
                results['test_id_nmse'] = float(nmse_id)
            else:
                results['test_id_nmse'] = float('inf')
        except Exception as e:
            print(f"⚠️ test_id评估失败: {e}")
            results['test_id_mse'] = float('inf')
            results['test_id_nmse'] = float('inf')
    else:
        results['test_id_mse'] = float('inf')
        results['test_id_nmse'] = float('inf')
    
    # 评估在test_ood上的性能
    X_ood, y_ood = load_csv(test_ood_path)
    if X_ood is not None and y_ood is not None:
        try:
            mse_ood = evaluate_sample_with_params(function_body, params, X_ood, y_ood, var_names)
            results['test_ood_mse'] = float(mse_ood)
            # 计算NMSE
            if mse_ood < float('inf'):
                nmse_ood = mse_ood / (np.var(y_ood) + 1e-12)
                results['test_ood_nmse'] = float(nmse_ood)
            else:
                results['test_ood_nmse'] = float('inf')
        except Exception as e:
            print(f"⚠️ test_ood评估失败: {e}")
            results['test_ood_mse'] = float('inf')
            results['test_ood_nmse'] = float('inf')
    else:
        results['test_ood_mse'] = float('inf')
        results['test_ood_nmse'] = float('inf')
    
    return results


def generate_output_filename(base_name: str, extension: str = 'json') -> str:
    """生成带时间戳的输出文件名，保存在test_results文件夹中"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name_without_ext = Path(base_name).stem
    
    # 确保test_results文件夹存在
    test_results_dir = Path(ROOT) / "test_results"
    test_results_dir.mkdir(exist_ok=True)
    
    return str(test_results_dir / f"{name_without_ext}_{timestamp}.{extension}")


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description='🔥 从JSONL样本中筛选最佳方程 V2 / 评估单个JSON样本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', type=str, required=True, choices=['jsonl', 'json', 'best_from_jsonl'],
                       help='处理模式：jsonl（筛选最佳样本）、json（评估单个样本）或best_from_jsonl（从jsonl中选择训练NMSE最优样本评估）')
    parser.add_argument('--problem_type', '-p', type=str, required=True,
                       choices=['oscillator1', 'oscillator2', 'bactgrow', 'stressstrain'],
                       help='问题类型')
    parser.add_argument('--input_path', '-i', type=str, required=True,
                       help='输入文件路径（JSONL或JSON文件）')
    parser.add_argument('--output_json', '-o', type=str, default='result.json',
                       help='输出JSON文件基础名（会自动添加时间戳并保存到test_results文件夹）')
    parser.add_argument('--max_samples', '-m', type=int, default=0,
                       help='最大处理样本数量（jsonl和best_from_jsonl模式有效，0表示处理全部）')
    
    args = parser.parse_args()
    
    # 生成带时间戳的输出文件名
    output_filename = generate_output_filename(args.output_json)
    
    mode_names = {
        'jsonl': '筛选最佳样本',
        'json': '评估单个样本', 
        'best_from_jsonl': '从训练NMSE选择最优样本评估'
    }
    print(f"🔥 {mode_names.get(args.mode, args.mode)}")
    print(f"📊 问题类型: {args.problem_type}")
    print(f"🎯 处理模式: {args.mode}")
    print(f"📄 输入文件: {args.input_path}")
    print(f"💾 输出文件: {output_filename}")
    if args.mode in ['jsonl', 'best_from_jsonl']:
        print(f"🔢 最大样本数: {args.max_samples if args.max_samples > 0 else '无限制'}")
    print("=" * 60)
    
    # 数据目录
    data_dir = f"{ROOT}data/{args.problem_type}"
    print(f"📁 数据目录: {data_dir}")
    
    # 检查数据文件是否存在
    test_id_path = f"{data_dir}/test_id.csv"
    test_ood_path = f"{data_dir}/test_ood.csv"
    
    if not Path(test_id_path).exists():
        print(f"❌ test_id.csv 不存在: {test_id_path}")
        return
    if not Path(test_ood_path).exists():
        print(f"❌ test_ood.csv 不存在: {test_ood_path}")
        return
    
    if args.mode == 'jsonl':
        # JSONL模式：筛选最佳样本
        print("\n📖 加载JSONL样本...")
        samples = load_jsonl_samples(args.input_path)
        
        if not samples:
            print("❌ 没有加载到任何样本")
            return
        
        # 限制样本数量
        if args.max_samples > 0 and len(samples) > args.max_samples:
            print(f"⚠️ 样本数量过多，只处理前 {args.max_samples} 个")
            samples = samples[:args.max_samples]
        
        print(f"\n🔬 开始评估 {len(samples)} 个样本在测试集上的性能...")
        
        # 用于记录最佳样本（基于NMSE）
        best_id_sample = None
        best_id_nmse = float('inf')
        best_ood_sample = None
        best_ood_nmse = float('inf')
        
        valid_samples = 0
        
        for i, sample in enumerate(samples, 1):
            if i % 100 == 0 or i <= 10:
                print(f"🔍 处理第 {i}/{len(samples)} 个样本...")
            
            # 评估样本性能
            test_results = evaluate_sample_on_testsets(sample, args.problem_type, data_dir)
            
            # 检查是否为有效样本
            if (test_results['test_id_nmse'] < float('inf') or 
                test_results['test_ood_nmse'] < float('inf')):
                valid_samples += 1
                
                # 更新test_id最佳样本（基于NMSE）
                if test_results['test_id_nmse'] < best_id_nmse:
                    best_id_nmse = test_results['test_id_nmse']
                    best_id_sample = {
                        **sample,
                        'sample_index': i - 1,
                        **test_results
                    }
                
                # 更新test_ood最佳样本（基于NMSE）
                if test_results['test_ood_nmse'] < best_ood_nmse:
                    best_ood_nmse = test_results['test_ood_nmse']
                    best_ood_sample = {
                        **sample,
                        'sample_index': i - 1,
                        **test_results
                    }
        
        print(f"\n✅ 评估完成！")
        print(f"📊 有效样本数: {valid_samples}/{len(samples)}")
        print(f"🏆 test_id 最佳NMSE: {best_id_nmse:.6e}")
        print(f"🏆 test_ood 最佳NMSE: {best_ood_nmse:.6e}")
        
        # 准备输出结果
        output_data = {
            'mode': 'jsonl',
            'evaluation_info': {
                'problem_type': args.problem_type,
                'input_path': args.input_path,
                'total_samples': len(samples),
                'valid_samples': valid_samples,
                'data_dir': data_dir,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'best_test_id_sample': best_id_sample,
            'best_test_ood_sample': best_ood_sample
        }
        
        # 保存结果并打印总结
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {output_filename}")
            
            # 打印总结
            print("\n📋 最佳样本总结:")
            print("-" * 40)
            if best_id_sample:
                print(f"🎯 test_id 最佳样本:")
                print(f"   索引: {best_id_sample['sample_index']}")
                print(f"   NMSE: {best_id_sample['test_id_nmse']:.6e}")
                print(f"   MSE: {best_id_sample['test_id_mse']:.6e}")
                print(f"   原始奖励: {best_id_sample.get('reward', 'N/A')}")
                print(f"   复杂度: {best_id_sample.get('complexity', 'N/A')}")
                print(f"   执行成功: {best_id_sample.get('execution_success', 'N/A')}")
            else:
                print("❌ 未找到test_id有效样本")
                
            print()
            if best_ood_sample:
                print(f"🎯 test_ood 最佳样本:")
                print(f"   索引: {best_ood_sample['sample_index']}")
                print(f"   NMSE: {best_ood_sample['test_ood_nmse']:.6e}")
                print(f"   MSE: {best_ood_sample['test_ood_mse']:.6e}")
                print(f"   原始奖励: {best_ood_sample.get('reward', 'N/A')}")
                print(f"   复杂度: {best_ood_sample.get('complexity', 'N/A')}")
                print(f"   执行成功: {best_ood_sample.get('execution_success', 'N/A')}")
            else:
                print("❌ 未找到test_ood有效样本")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    elif args.mode == 'json':
        # JSON模式：评估单个样本
        print("\n📖 加载单个JSON样本...")
        sample = load_single_json_sample(args.input_path)
        
        if not sample:
            print("❌ 没有加载到有效样本")
            return
        
        print("\n🔬 开始评估样本在测试集上的性能...")
        
        # 评估样本性能
        test_results = evaluate_sample_on_testsets(sample, args.problem_type, data_dir)
        
        print(f"\n✅ 评估完成！")
        print(f"🎯 test_id NMSE: {test_results['test_id_nmse']:.6e}")
        print(f"🎯 test_ood NMSE: {test_results['test_ood_nmse']:.6e}")
        print(f"📊 test_id MSE: {test_results['test_id_mse']:.6e}")
        print(f"📊 test_ood MSE: {test_results['test_ood_mse']:.6e}")
        
        # 准备输出结果（只包含NMSE信息）
        output_data = {
            'mode': 'json',
            'evaluation_info': {
                'problem_type': args.problem_type,
                'input_path': args.input_path,
                'data_dir': data_dir,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'sample_evaluation': {
                'test_id_nmse': test_results['test_id_nmse'],
                'test_ood_nmse': test_results['test_ood_nmse'],
                'test_id_mse': test_results['test_id_mse'],
                'test_ood_mse': test_results['test_ood_mse'],
                'execution_success': sample.get('execution_success', False)
            },
            'original_sample': sample
        }
        
        # 保存结果
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {output_filename}")
            
            # 打印总结
            print("\n📋 样本评估总结:")
            print("-" * 30)
            print(f"🎯 test_id NMSE: {test_results['test_id_nmse']:.6e}")
            print(f"🎯 test_ood NMSE: {test_results['test_ood_nmse']:.6e}")
            print(f"📊 test_id MSE: {test_results['test_id_mse']:.6e}")
            print(f"📊 test_ood MSE: {test_results['test_ood_mse']:.6e}")
            print(f"🏆 原始奖励: {sample.get('reward', 'N/A')}")
            print(f"🔧 复杂度: {sample.get('complexity', 'N/A')}")
            print(f"✅ 执行成功: {sample.get('execution_success', 'N/A')}")
        
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    elif args.mode == 'best_from_jsonl':
        # best_from_jsonl模式：从jsonl中选择训练NMSE最优样本并评估
        print("\n📖 加载JSONL样本...")
        samples = load_jsonl_samples(args.input_path)
        
        if not samples:
            print("❌ 没有加载到任何样本")
            return
        
        # 限制样本数量
        if args.max_samples > 0 and len(samples) > args.max_samples:
            print(f"⚠️ 样本数量过多，只处理前 {args.max_samples} 个")
            samples = samples[:args.max_samples]
        
        print(f"\n🔍 在 {len(samples)} 个样本中寻找训练NMSE最优样本...")
        
        # 找到训练NMSE最小的样本
        best_sample = None
        best_train_nmse = float('inf')
        
        for i, sample in enumerate(samples):
            # 尝试从样本中获取训练NMSE
            train_nmse = None
            
            # 方法1：直接从样本中获取nmse字段
            if 'nmse' in sample and sample['nmse'] is not None:
                train_nmse = float(sample['nmse'])
            # 方法2：从mse和输出方差计算nmse（如果有mse字段）
            elif 'mse' in sample and sample['mse'] is not None:
                # 这里需要加载训练数据来计算输出方差
                try:
                    train_data_path = f"{data_dir}/train.csv"
                    X_train, y_train = load_csv(train_data_path)
                    if X_train is not None and y_train is not None:
                        var_y = np.var(y_train) + 1e-12
                        train_nmse = float(sample['mse']) / var_y
                except Exception as e:
                    print(f"⚠️ 无法计算第{i+1}个样本的训练NMSE: {e}")
                continue
            
            # 如果找到了训练NMSE且更优，更新最佳样本
            if train_nmse is not None and train_nmse < best_train_nmse:
                best_train_nmse = train_nmse
                best_sample = sample
                best_sample['sample_index'] = i
        
        if best_sample is None:
            print("❌ 未找到包含有效训练NMSE的样本")
            return
        
        print(f"✅ 找到最优样本:")
        print(f"   索引: {best_sample['sample_index']}")
        print(f"   训练NMSE: {best_train_nmse:.6e}")
        print(f"   训练MSE: {best_sample.get('mse', 'N/A')}")
        print(f"   原始奖励: {best_sample.get('reward', 'N/A')}")
        
        print(f"\n🔬 评估该样本在测试集上的性能...")
        
        # 评估最优样本在测试集上的性能
        test_results = evaluate_sample_on_testsets(best_sample, args.problem_type, data_dir)
        
        print(f"\n✅ 评估完成！")
        print(f"🎯 test_id NMSE: {test_results['test_id_nmse']:.6e}")
        print(f"🎯 test_ood NMSE: {test_results['test_ood_nmse']:.6e}")
        print(f"📊 test_id MSE: {test_results['test_id_mse']:.6e}")
        print(f"📊 test_ood MSE: {test_results['test_ood_mse']:.6e}")
        
        # 准备输出结果
        output_data = {
            'mode': 'best_from_jsonl',
            'evaluation_info': {
                'problem_type': args.problem_type,
                'input_path': args.input_path,
                'total_samples': len(samples),
                'selected_sample_index': best_sample['sample_index'],
                'train_nmse': best_train_nmse,
                'data_dir': data_dir,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'sample_evaluation': {
                'test_id_nmse': test_results['test_id_nmse'],
                'test_ood_nmse': test_results['test_ood_nmse'],
                'test_id_mse': test_results['test_id_mse'],
                'test_ood_mse': test_results['test_ood_mse'],
                'train_nmse': best_train_nmse,
                'execution_success': best_sample.get('execution_success', False)
            },
            'selected_sample': best_sample
        }
        
        # 保存结果
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {output_filename}")
            
            # 打印总结
            print("\n📋 最优样本评估总结:")
            print("-" * 40)
            print(f"🏆 选中样本索引: {best_sample['sample_index']}")
            print(f"📊 训练NMSE: {best_train_nmse:.6e}")
            print(f"🎯 test_id NMSE: {test_results['test_id_nmse']:.6e}")
            print(f"🎯 test_ood NMSE: {test_results['test_ood_nmse']:.6e}")
            print(f"🏆 原始奖励: {best_sample.get('reward', 'N/A')}")
            print(f"🔧 复杂度: {best_sample.get('complexity', 'N/A')}")
            print(f"✅ 执行成功: {best_sample.get('execution_success', 'N/A')}")
        
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


if __name__ == "__main__":
    main()