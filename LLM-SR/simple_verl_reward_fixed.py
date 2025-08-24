#!/usr/bin/env python3
"""
🔥 修复版VERL奖励函数 - 采用无RL版本的正确方法

关键改进：
1. 不再提取数学表达式字符串
2. 直接执行LLM生成的完整Python函数
3. 模仿无RL版本的处理流程：_extract_body -> _sample_to_program -> exec -> 调用函数

这样完全避免了表达式提取的问题。
"""

import numpy as np
import pandas as pd
import re
import ast
import copy
import multiprocessing
import time
import math
from pathlib import Path
from typing import Any, List, Dict, Tuple
import os
import json


def compute_score(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
    """
    🔥 修复版计算LLM-SR符号回归的奖励分数 - 直接执行Python函数
    
    Args:
        data_sources: 数据源列表 (默认: None)
        solution_strs: 模型生成的解决方案字符串列表 (默认: None)
        ground_truths: 参考答案列表 (默认: None)
        extra_infos: 额外信息列表，包含problem_type等 (默认: None)
        **kwargs: 其他参数，用于兼容VERL的各种调用方式
        
    Returns:
        rewards: 奖励分数列表（浮点数）
    """
    
    print(f"🔥🔥🔥 FIXED REWARD FUNCTION CALLED! 🔥🔥🔥")
    print(f"🔧 修复版奖励函数被调用，参数: data_sources={type(data_sources)}, solution_strs={type(solution_strs)}, ground_truths={type(ground_truths)}, extra_infos={type(extra_infos)}")
    print(f"🔧 kwargs: {list(kwargs.keys())}")
    print(f"🔧 Solution strings count: {len(solution_strs) if solution_strs else 0}")
    if solution_strs and len(solution_strs) > 0:
        print(f"🔧 First solution preview: {solution_strs[0][:200] if solution_strs[0] else 'None'}...")
    
    # 🔧 处理默认参数值，避免 TypeError
    if data_sources is None:
        data_sources = []
    if solution_strs is None:
        solution_strs = []
    if ground_truths is None:
        ground_truths = []
    if extra_infos is None:
        extra_infos = []
    
    # 🔧 尝试从kwargs中提取数据（兼容VERL的不同调用方式）
    if not solution_strs and 'responses' in kwargs:
        solution_strs = kwargs['responses']
    if not solution_strs and 'generated_texts' in kwargs:
        solution_strs = kwargs['generated_texts']
    if not extra_infos and 'batch' in kwargs:
        extra_infos = [{'problem_type': 'oscillator1'}]  # 默认问题类型
    
    # 🔧 重要：处理VERL的单数形式参数
    if not solution_strs and 'solution_str' in kwargs:
        solution_strs = [kwargs['solution_str']]  # 转换为列表
    if not data_sources and 'data_source' in kwargs:
        data_sources = [kwargs['data_source']]
    if not ground_truths and 'ground_truth' in kwargs:
        ground_truths = [kwargs['ground_truth']]
    if not extra_infos and 'extra_info' in kwargs:
        extra_infos = [kwargs['extra_info']]
    
    # 如果没有解决方案字符串，返回空列表
    if not solution_strs:
        print("⚠️  没有解决方案字符串，返回空奖励列表")
        return []
    
    print(f"🔍 奖励计算开始，处理{len(solution_strs)}个解决方案")
    
    # 初始化奖励列表
    rewards = []
    
    # 加载训练数据（从第一个extra_info推断问题类型）
    problem_type = None
    if extra_infos and len(extra_infos) > 0 and extra_infos[0]:
        if isinstance(extra_infos[0], dict):
            if 'problem_type' in extra_infos[0]:
                problem_type = extra_infos[0]['problem_type']
            elif 'extra_info' in extra_infos[0] and isinstance(extra_infos[0]['extra_info'], dict) and 'problem_type' in extra_infos[0]['extra_info']:
                problem_type = extra_infos[0]['extra_info']['problem_type']
    
    # 从问题类型加载对应的训练数据
    train_data = load_training_data(problem_type)
    if train_data is None:
        print(f"❌ 无法加载训练数据，问题类型: {problem_type}")
        return [0.0] * len(solution_strs)
    
    inputs, outputs, var_names = train_data
    print(f"📊 训练数据: {inputs.shape}, 变量: {var_names}")
    
    # 评估每个解决方案
    for i, solution_str in enumerate(solution_strs):
        try:
            # 🔥 使用新的方法：直接执行Python函数
            reward = evaluate_single_solution_fixed(solution_str, inputs, outputs, var_names)
            rewards.append(reward)
            
            if i < 3:  # 只显示前3个的详细信息
                print(f"✅ 解决方案 {i+1}: 奖励 = {reward:.4f}")
                
        except Exception as e:
            print(f"❌ 评估解决方案 {i+1} 时出错: {e}")
            rewards.append(0.0)
    
    print(f"🎯 奖励计算完成，平均奖励: {np.mean(rewards):.4f}")
    
    # 🔧 VERL兼容性修复：确保返回正确的数据类型
    if not rewards:
        print("⚠️  没有计算出奖励，返回默认值 0.0")
        return 0.0
    
    # 🔧 处理单个样本的情况（VERL验证时经常如此）
    if len(rewards) == 1:
        reward_value = float(rewards[0])
        print(f"🎯 返回单个奖励值: {reward_value}")
        return reward_value
    
    # 多个样本的情况
    rewards_array = np.array(rewards, dtype=np.float32)
    print(f"🎯 返回奖励数组，长度: {len(rewards_array)}")
    return rewards_array.tolist()


def load_training_data(problem_type):
    """根据问题类型加载训练数据"""
    
    if not problem_type:
        print("⚠️  问题类型未知，尝试使用oscillator1作为默认")
        problem_type = "oscillator1"
    
    # 数据文件路径
    data_file = f"data/{problem_type}/train.csv"
    if not Path(data_file).exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    
    try:
        df = pd.read_csv(data_file)
        
        if problem_type == "oscillator1":
            if all(col in df.columns for col in ['x', 'v', 'a']):
                inputs = df[['x', 'v']].values[:100]  # 限制样本数以提高速度
                outputs = df['a'].values[:100]
                var_names = ['x', 'v']
                return inputs, outputs, var_names
                
        elif problem_type == "oscillator2":
            if all(col in df.columns for col in ['t', 'x', 'v', 'a']):
                inputs = df[['x', 'v']].values[:100]  # 只用x,v，忽略t
                outputs = df['a'].values[:100]
                var_names = ['x', 'v']
                return inputs, outputs, var_names
                
        elif problem_type == "bactgrow":
            if all(col in df.columns for col in ['b', 's', 'temp', 'pH', 'db']):
                inputs = df[['b', 's', 'temp', 'pH']].values[:100]
                outputs = df['db'].values[:100]
                var_names = ['b', 's', 'temp', 'pH']
                return inputs, outputs, var_names
                
        elif problem_type == "stressstrain":
            if all(col in df.columns for col in ['strain', 'temp', 'stress']):
                inputs = df[['strain', 'temp']].values[:100]
                outputs = df['stress'].values[:100]
                var_names = ['strain', 'temp']
                return inputs, outputs, var_names
        
        print(f"❌ 不支持的问题类型或数据格式: {problem_type}, 列: {list(df.columns)}")
        return None
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        return None


def evaluate_single_solution_fixed(solution_str: str, inputs: np.ndarray, outputs: np.ndarray, var_names: list) -> float:
    """
    🔥 修复版：使用无RL版本的方法直接执行Python函数
    """
    
    # 记录到jsonl的信息
    log_info = {
        "solution_length": len(solution_str) if solution_str else 0,
        "timestamp": time.time(),
        "execution_success": False,
        "function_body": "",
        "params": None,  # 将记录params数组的具体数值列表
        "mse": float('inf'),
        "reward": 0.0,
        "error": None
    }
    
    try:
        # 🔥 步骤1：从LLM输出中提取函数体（模仿sampler.py的_extract_body）
        function_body = extract_function_body(solution_str)
        log_info["function_body"] = function_body
        
        if not function_body:
            log_info["error"] = "函数体提取失败"
            print(f"❌ 函数体提取失败，解决方案长度: {len(solution_str)}")
            _log_to_jsonl(log_info)
            return 0.0
        
        print(f"✅ 成功提取函数体，长度: {len(function_body)}")
        
        # 🔥 步骤2：构建完整的可执行程序（模仿evaluator.py的_sample_to_program）
        program = build_executable_program(function_body, var_names)
        
        # 🔥 步骤3：在安全环境中执行程序并计算MSE
        mse, params_used = execute_and_compute_mse(program, inputs, outputs, var_names)
        log_info["mse"] = float(mse)
        log_info["params"] = params_used.tolist() if params_used is not None else None
        log_info["execution_success"] = True
        
        # 返回负MSE作为奖励（MSE越小，奖励越高）
        reward = -mse
        
        # 限制奖励范围，避免数值不稳定
        reward = max(min(reward, 10.0), -100.0)
        log_info["reward"] = float(reward)
        
        print(f"✅ 计算完成 - MSE: {mse:.6f}, 奖励: {reward:.6f}")
        
        # 记录成功的评估
        _log_to_jsonl(log_info)
        
        return reward
        
    except Exception as e:
        error_msg = f"执行Python函数时出错: {e}"
        log_info["error"] = error_msg
        print(f"❌ {error_msg}")
        
        # 即使出错也要记录
        _log_to_jsonl(log_info)
        
        return 0.0


class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None 
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        
        except SyntaxError as e:
            if e.lineno is None: # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def extract_function_body(solution_str: str) -> str:
    """
    从LLM输出中提取函数体，完全模仿sampler.py的_extract_body函数
    """
    if not solution_str or not isinstance(solution_str, str):
        return ""
    
    # 处理<think>标签
    if "</think>" in solution_str:
        parts = solution_str.split("</think>")
        if len(parts) > 1:
            solution_str = parts[-1].strip()
    
    # 查找Python代码块
    import re
    code_block_patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```'
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, solution_str, re.DOTALL)
        if matches:
            solution_str = matches[0]  # 使用第一个代码块
            break
    
    # 🔥 完全模仿sampler.py的_extract_body逻辑
    lines = solution_str.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':  # 🔥 使用无RL版本的精确匹配
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # 🔥 模仿无RL版本的缩进处理逻辑
        code = ''
        indent = '    '
        for line in lines[func_body_lineno + 1:]:
            if line[:4] != indent:
                line = indent + line
            code += line + '\n'
        
        # 🔥 使用无RL版本的_trim_function_body确保语法正确
        return _trim_function_body(code)
    
    return solution_str  # 🔥 如果没找到def，返回原始sample（无RL版本的行为）


def build_executable_program(function_body: str, var_names: list) -> str:
    """
    构建完整的可执行程序，模仿evaluator.py的_sample_to_program函数
    """
    
    # 构建函数签名
    params_str = ', '.join(var_names) + ', params'
    
    # 构建完整的程序
    program = f"""
import numpy as np
import math

def equation({params_str}):
{function_body}

def evaluate_function(inputs, outputs, var_names):
    \"\"\"评估函数性能\"\"\"
    try:
        # 准备参数
        params = np.ones(10)  # 默认参数
        
        # 🔥 按照无RL版本的方式，直接传递整个数组
        if len(var_names) == 2:  # x, v (oscillator)
            x_data = inputs[:, 0]
            v_data = inputs[:, 1] 
            predictions = equation(x_data, v_data, params)
        elif len(var_names) == 4:  # b, s, temp, pH (bactgrow)
            b_data = inputs[:, 0]
            s_data = inputs[:, 1]
            temp_data = inputs[:, 2]
            pH_data = inputs[:, 3]
            predictions = equation(b_data, s_data, temp_data, pH_data, params)
        elif len(var_names) == 2 and 'strain' in var_names:  # strain, temp (stressstrain)
            strain_data = inputs[:, 0]
            temp_data = inputs[:, 1]
            predictions = equation(strain_data, temp_data, params)
        else:
            # 通用处理：传递所有列作为参数
            args = [inputs[:, j] for j in range(inputs.shape[1])]
            predictions = equation(*args, params)
        
        # 确保predictions是numpy数组
        predictions = np.asarray(predictions, dtype=np.float64)
        
        # 处理标量返回值
        if predictions.ndim == 0:
            predictions = np.full_like(outputs, float(predictions))
        
        # 计算MSE
        mse = np.mean((predictions - outputs) ** 2)
        return float(mse) if np.isfinite(mse) else 1e6, params
        
    except Exception as e:
        print(f"❌ 函数执行错误: {{e}}")
        return 1e6, params
"""
    
    return program


def execute_and_compute_mse(program: str, inputs: np.ndarray, outputs: np.ndarray, var_names: list) -> tuple[float, np.ndarray]:
    """
    在安全环境中执行程序并计算MSE，模仿evaluator.py的执行逻辑
    """
    
    try:
        # 🔥 步骤3：执行程序（模仿evaluator.py的exec逻辑）
        all_globals_namespace = {
            'np': np,
            'numpy': np,
            'math': math
        }
        
        # 执行程序
        exec(program, all_globals_namespace)
        
        # 获取评估函数
        evaluate_function = all_globals_namespace['evaluate_function']
        
        # 调用评估函数
        mse, params_used = evaluate_function(inputs, outputs, var_names)
        
        return mse, params_used
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1e6, None


def _log_to_jsonl(log_info: dict):
    """记录评估信息到jsonl文件"""
    try:
        # 获取输出目录
        output_dir = os.environ.get('LLMSR_OUTPUT_DIR', './llmsr_grpo_outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        jsonl_path = os.path.join(output_dir, 'sample.jsonl')
        
        # 追加写入jsonl文件
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_info, ensure_ascii=False) + '\n')
            
        print(f"📝 已记录到 {jsonl_path}")
        
    except Exception as e:
        print(f"❌ 记录jsonl失败: {e}")


# 为了兼容VERL的调用方式，创建data_proto接口
def compute_score_data_proto(data_proto, **kwargs):
    """兼容data_proto格式的接口"""
    
    # 从data_proto提取信息
    try:
        batch = data_proto.batch
        responses = batch.get('responses', [])
        
        # 将responses转换为字符串
        solution_strs = []
        for response in responses:
            if isinstance(response, str):
                solution_strs.append(response)
            else:
                # 如果是token，尝试解码
                solution_strs.append(str(response))
        
        # 调用主要的compute_score函数
        rewards = compute_score(
            data_sources=[], 
            solution_strs=solution_strs,
            ground_truths=[],
            extra_infos=[{'problem_type': 'oscillator1'}]  # 默认
        )
        
        return {"reward": np.array(rewards, dtype=np.float32)}
        
    except Exception as e:
        print(f"❌ data_proto接口错误: {e}")
        return {"reward": np.array([0.0], dtype=np.float32)}


# 🔧 通用的奖励函数包装器，处理所有可能的调用方式
def reward_function(*args, **kwargs):
    """
    通用奖励函数包装器 - 处理VERL的各种调用方式
    这个函数会尝试将任何调用方式转换为我们期望的格式
    """
    try:
        print(f"🔧 通用包装器被调用，args={len(args)}, kwargs={list(kwargs.keys())}")
        
        # 如果没有位置参数，直接使用关键字参数
        if len(args) == 0:
            return compute_score(**kwargs)
        
        # 如果有位置参数，尝试映射到正确的参数名
        elif len(args) == 4:
            return compute_score(
                data_sources=args[0],
                solution_strs=args[1], 
                ground_truths=args[2],
                extra_infos=args[3],
                **kwargs
            )
        
        # 如果只有一个参数，可能是batch数据
        elif len(args) == 1:
            batch_data = args[0]
            if hasattr(batch_data, 'responses') or isinstance(batch_data, dict):
                return compute_score(solution_strs=getattr(batch_data, 'responses', batch_data.get('responses', [])), **kwargs)
            else:
                return compute_score(solution_strs=[str(batch_data)], **kwargs)
        
        # 其他情况，尝试最灵活的处理
        else:
            return compute_score(*args, **kwargs)
            
    except Exception as e:
        print(f"❌ 通用包装器错误: {e}")
        # 返回默认奖励
        return [0.0]


# 🔧 为了最大兼容性，创建多个别名
score_function = compute_score
reward_fn = compute_score
compute_reward = compute_score


if __name__ == "__main__":
    # 测试函数
    test_solution = """
    ```python
    def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        k = params[0]  # Linear spring constant
        c = params[1]  # Linear damping coefficient
        d = params[2]  # Nonlinear restoring coefficient
        F0 = params[3]  # Constant driving force
        m = params[4]  # Mass

        # Compute acceleration using the derived formula
        a = (-k * x - c * v - d * x**3 + F0) / m
        return a
    ```
    """
    
    print("🧪 测试修复版奖励函数:")
    rewards = compute_score(
        solution_strs=[test_solution],
        extra_infos=[{'problem_type': 'oscillator1'}]
    )
    print(f"测试奖励: {rewards}")