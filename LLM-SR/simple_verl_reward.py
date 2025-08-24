#!/usr/bin/env python3
"""
🔥 简化的VERL奖励函数 - 专为新数据格式设计

此函数直接处理VERL的数据格式，避免KeyError: 'ground_truth'问题。
它从data_item中提取必要信息并计算奖励。
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path


def compute_score(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
    """
    🔥 计算LLM-SR符号回归的奖励分数 - 支持灵活参数调用
    
    Args:
        data_sources: 数据源列表 (默认: None)
        solution_strs: 模型生成的解决方案字符串列表 (默认: None)
        ground_truths: 参考答案列表 (默认: None)
        extra_infos: 额外信息列表，包含problem_type等 (默认: None)
        **kwargs: 其他参数，用于兼容VERL的各种调用方式
        
    Returns:
        rewards: 奖励分数列表（浮点数）
    """
    
    print(f"🔥🔥🔥 SIMPLE REWARD FUNCTION CALLED! 🔥🔥🔥")
    print(f"🔧 奖励函数被调用，参数: data_sources={type(data_sources)}, solution_strs={type(solution_strs)}, ground_truths={type(ground_truths)}, extra_infos={type(extra_infos)}")
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
            # 评估单个解决方案
            reward = evaluate_single_solution(solution_str, inputs, outputs, var_names)
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
    return rewards_array.tolist()  # 返回Python列表


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


def evaluate_single_solution(solution_str: str, inputs: np.ndarray, outputs: np.ndarray, var_names: list) -> float:
    """评估单个解决方案"""
    
    # 记录到jsonl的信息
    log_info = {
        "solution_str": solution_str,
        "solution_length": len(solution_str) if solution_str else 0,
        "timestamp": __import__('time').time(),
        "extraction_success": False,
        "expression": "",
        "mse": float('inf'),
        "reward": 0.0,
        "error": None
    }
    
    try:
        # 提取数学表达式
        expression = extract_mathematical_expression(solution_str)
        log_info["expression"] = expression
        
        if not expression:
            log_info["error"] = "表达式提取失败"
            print(f"❌ 表达式提取失败，解决方案长度: {len(solution_str)}")
            _log_to_jsonl(log_info)
            return 0.0
        
        log_info["extraction_success"] = True
        print(f"✅ 成功提取表达式: {expression}")
        
        # 计算MSE
        mse = compute_mse(expression, inputs, outputs, var_names)
        log_info["mse"] = float(mse)
        
        # 返回负MSE作为奖励（MSE越小，奖励越高）
        reward = -mse
        
        # 限制奖励范围，避免数值不稳定
        reward = max(min(reward, 10.0), -100.0)
        log_info["reward"] = float(reward)
        
        print(f"✅ 计算完成 - 表达式: {expression}, MSE: {mse:.6f}, 奖励: {reward:.6f}")
        
        # 记录成功的评估
        _log_to_jsonl(log_info)
        
        return reward
        
    except Exception as e:
        error_msg = f"评估表达式时出错: {e}"
        log_info["error"] = error_msg
        print(f"❌ {error_msg}")
        
        # 即使出错也要记录
        _log_to_jsonl(log_info)
        
        return 0.0


def _log_to_jsonl(log_info: dict):
    """记录评估信息到jsonl文件"""
    try:
        import json
        import os
        
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


def extract_mathematical_expression(solution_str: str) -> str:
    """从LLM生成的Python代码中提取数学表达式，专注于代码块解析"""
    
    if not solution_str or not isinstance(solution_str, str):
        print("❌ 表达式提取失败: 输入为空或非字符串")
        return ""
    
    # 清理输入
    solution_str = solution_str.strip()
    original_length = len(solution_str)
    print(f"🔍 开始提取表达式，输入长度: {original_length}")
    
    # 🔧 处理包含 <think> 标签的情况，提取实际代码部分
    if "</think>" in solution_str:
        parts = solution_str.split("</think>")
        if len(parts) > 1:
            solution_str = parts[-1].strip()  # 取最后一部分
            print(f"🔧 从</think>后提取内容，长度: {len(solution_str)}")
    
    import re
    
    # 🔥 主策略：从Python代码块中提取表达式
    print("🔍 主策略: 解析Python代码块")
    
    # 1. 查找Python代码块
    python_code_patterns = [
        r'```python\s*\n(.*?)\n```',  # 标准Python代码块
        r'```\s*\n(.*?)\n```',       # 通用代码块
    ]
    
    for pattern in python_code_patterns:
        code_blocks = re.findall(pattern, solution_str, re.DOTALL)
        if code_blocks:
            print(f"🔧 找到{len(code_blocks)}个代码块")
            for i, code_block in enumerate(code_blocks):
                print(f"🔧 处理代码块 {i+1}")
                # 从代码块中提取表达式
                extracted = _extract_from_python_code(code_block.strip())
                if extracted:
                    print(f"✅ 从代码块{i+1}中提取到表达式: {extracted}")
                    return extracted
    
    # 2. 如果没有找到代码块，尝试直接解析函数定义
    print("🔍 备用策略: 直接解析函数定义")
    function_pattern = r'def\s+equation\s*\([^)]+\)\s*:\s*(.*?)(?=def|\Z)'
    func_matches = re.findall(function_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if func_matches:
        for func_body in func_matches:
            print(f"🔧 找到函数体，长度: {len(func_body)}")
            extracted = _extract_from_python_code(func_body)
            if extracted:
                print(f"✅ 从函数体提取表达式: {extracted}")
                return extracted
    
    # 3. 最后尝试查找简单的return语句或赋值语句
    print("🔍 最后策略: 查找简单表达式")
    simple_patterns = [
        r'return\s+([^.\n]+?)(?=\s*$|\s*\n|\s*#|$)',
        r'a\s*=\s*([^.\n]+?)(?=\s*$|\s*\n|\s*#|$)',
        r'acceleration\s*=\s*([^.\n]+?)(?=\s*$|\s*\n|\s*#|$)',
    ]
    
    for pattern in simple_patterns:
        matches = re.findall(pattern, solution_str, re.IGNORECASE | re.MULTILINE)
        if matches:
            for match in matches:
                cleaned = match.strip().rstrip('.,;:')
                if _is_valid_math_expression(cleaned):
                    print(f"✅ 从简单模式提取表达式: {cleaned}")
                    return cleaned
    
    # 如果都失败了，记录详细信息
    print("❌ 所有表达式提取策略都失败了")
    print(f"❌ 输入文本前500字符: {solution_str[:500]}")
    print(f"❌ 输入文本后500字符: {solution_str[-500:]}")
    return ""


def _extract_from_python_code(code_block: str) -> str:
    """从Python代码中提取数学表达式，专门处理LLM生成的equation函数"""
    
    if not code_block:
        return ""
    
    print(f"🔧 解析Python代码，长度: {len(code_block)}")
    
    lines = code_block.split('\n')
    
    # 1. 首先查找return语句（最直接的方式）
    print("🔧 查找return语句")
    for line in lines:
        line = line.strip()
        if line.startswith('return '):
            expr = line.replace('return ', '').strip()
            # 清理可能的注释
            if '#' in expr:
                expr = expr.split('#')[0].strip()
            if _is_valid_math_expression(expr):
                print(f"🔧 从return语句提取: {expr}")
                return expr
    
    # 2. 查找加速度相关的赋值语句
    print("🔧 查找赋值语句")
    assignments = {}
    for line in lines:
        line = line.strip()
        # 跳过注释和函数定义
        if line.startswith('#') or line.startswith('def') or not line:
            continue
        
        if '=' in line and not line.startswith('def'):
            parts = line.split('=', 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                expr = parts[1].strip()
                
                # 清理可能的注释
                if '#' in expr:
                    expr = expr.split('#')[0].strip()
                
                # 检查是否是有效的变量名和表达式
                if var_name.replace('_', '').isalpha() and _is_valid_math_expression(expr):
                    assignments[var_name] = expr
                    print(f"🔧 找到赋值: {var_name} = {expr}")
    
    # 3. 优先返回acceleration相关的赋值
    priority_vars = ['a', 'acceleration', 'result', 'output', 'acc']
    for var in priority_vars:
        if var in assignments:
            print(f"🔧 选择优先变量 {var}: {assignments[var]}")
            return assignments[var]
    
    # 4. 返回最后一个有效赋值（通常是最终结果）
    if assignments:
        last_var = list(assignments.keys())[-1]
        last_expr = assignments[last_var]
        print(f"🔧 选择最后赋值 {last_var}: {last_expr}")
        return last_expr
    
    # 5. 如果没有找到赋值，尝试查找包含params的表达式行
    print("🔧 查找包含params的表达式")
    for line in lines:
        line = line.strip()
        if 'params[' in line and not line.startswith('#') and not line.startswith('def'):
            # 清理可能的注释
            if '#' in line:
                line = line.split('#')[0].strip()
            
            # 如果这行看起来像一个表达式
            if any(op in line for op in ['+', '-', '*', '/', '**']) and _is_valid_math_expression(line):
                print(f"🔧 从params表达式提取: {line}")
                return line
    
    print("🔧 Python代码解析失败")
    return ""


def _extract_from_function_body(func_body: str) -> str:
    """从函数体中提取最终的表达式，重用Python代码解析逻辑"""
    return _extract_from_python_code(func_body)


def _is_valid_math_expression(expr: str) -> bool:
    """检查字符串是否是有效的数学表达式"""
    if not expr or not isinstance(expr, str):
        return False
    
    expr = expr.strip()
    if not expr:
        return False
    
    # 基本验证：包含变量或数字
    import re
    
    # 检查是否包含变量、数字、运算符或函数
    patterns = [
        r'[a-zA-Z_][a-zA-Z0-9_]*',  # 变量名
        r'-?[0-9]*\.?[0-9]+',        # 数字
        r'[\+\-\*/\(\)]',           # 基本运算符
        r'(sin|cos|tan|exp|log|sqrt|abs|tanh)\(',  # 数学函数
    ]
    
    has_valid_content = any(re.search(pattern, expr) for pattern in patterns)
    
    # 排除明显无效的情况
    invalid_patterns = [
        r'print\s*\(',              # print语句
        r'import\s+',               # import语句  
        r'def\s+',                  # 函数定义
        r'class\s+',                # 类定义
        r'if\s+',                   # if语句
        r'for\s+',                  # for循环
        r'while\s+',                # while循环
    ]
    
    has_invalid_content = any(re.search(pattern, expr) for pattern in invalid_patterns)
    
    return has_valid_content and not has_invalid_content


def compute_mse(expression: str, inputs: np.ndarray, outputs: np.ndarray, var_names: list) -> float:
    """计算表达式的MSE"""
    
    if not expression:
        return 1e6
    
    # 安全的数学环境
    safe_dict = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs, "tanh": np.tanh,
        "pi": np.pi, "e": np.e,
        "np": np, "__builtins__": {}
    }
    
    try:
        # 设置变量
        for i, var_name in enumerate(var_names):
            if i < inputs.shape[1]:
                safe_dict[var_name] = inputs[:, i]
        
        # 清理表达式
        cleaned_expr = expression.replace('^', '**')  # 幂运算
        cleaned_expr = cleaned_expr.replace(' ', '')   # 移除空格
        
        # 尝试计算
        predictions = eval(cleaned_expr, safe_dict)
        
        # 确保predictions是合适的数组
        if isinstance(predictions, (int, float)):
            predictions = np.full(len(outputs), predictions)
        elif not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # 处理长度不匹配
        if len(predictions) != len(outputs):
            if len(predictions) > 0:
                predictions = np.full(len(outputs), predictions[0])
            else:
                predictions = np.zeros(len(outputs))
        
        # 计算MSE
        mse = np.mean((predictions - outputs) ** 2)
        
        # 处理无效值
        if np.isnan(mse) or np.isinf(mse) or mse < 0:
            return 1e6
            
        return float(mse)
        
    except Exception as e:
        # 如果计算失败，返回大的MSE值
        return 1e6


# 为了兼容VERL的调用方式，也提供data_proto接口
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
    test_solutions = [
        "return -x",
        "return -1.0 * x", 
        "return x + v"
    ]
    
    print("🧪 测试标准调用:")
    rewards = compute_score(
        data_sources=["test"],
        solution_strs=test_solutions,
        ground_truths=["a = -x"],
        extra_infos=[{'problem_type': 'oscillator1'}] * len(test_solutions)
    )
    print(f"标准调用奖励: {rewards}")
    
    print("\n🧪 测试默认参数调用:")
    rewards2 = compute_score(solution_strs=test_solutions)
    print(f"默认参数奖励: {rewards2}")
    
    print("\n🧪 测试通用包装器:")
    rewards3 = reward_function(solution_strs=test_solutions)
    print(f"包装器奖励: {rewards3}") 