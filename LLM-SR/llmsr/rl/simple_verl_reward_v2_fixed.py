#!/usr/bin/env python3
"""
🔥 修复版VERL奖励函数 V2 - 采用无RL版本的正确方法 + 多成分奖励

关键改进：
1. 不再提取数学表达式字符串
2. 直接执行LLM生成的完整Python函数
3. 保持V2版本的多成分奖励计算
4. 模仿无RL版本的处理流程
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple
import numpy as np
import os, json, time
import ast
import pandas as pd
import warnings

# 🔥 数值保护：抑制数值计算警告，避免日志污染
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
np.seterr(all='ignore')  # 忽略numpy的数值错误警告
from pathlib import Path


def compute_score(
    data_sources: List[Any] | None = None,
    solution_strs: List[str] | None = None,
    ground_truths: List[Any] | None = None,
    extra_infos: List[Dict[str, Any]] | None = None,
    *,
    grid_train_data: bool = False,
    template_path: str | None = None,
    data_path: str | None = None,
    memory_dir: str | None = None,
    lambda_nmse: float = 3.0,
    lambda_simp: float = 0.1,
    w_fit: float = 0.6,
    w_simp: float = 0.2,
    w_phys: float = 0.15,
    w_proc: float = 0.05,
    groupwise_rank_norm: bool = True,
    **kwargs,
):
    print(f"🔥🔥🔥 FIXED V2 REWARD FUNCTION CALLED! 🔥🔥🔥")
    print(f"🔧 V2修复版参数类型: data_sources={type(data_sources)}, solution_strs={type(solution_strs)}, ground_truths={type(ground_truths)}, extra_infos={type(extra_infos)}")
    print(f"🔧 V2 Solution strings count: {len(solution_strs) if solution_strs else 0}")
    if solution_strs and len(solution_strs) > 0:
        print(f"🔧 V2 First solution preview: {solution_strs[0][:200] if solution_strs[0] else 'None'}...")
    print(f"🔧 V2 LLMSR_OUTPUT_DIR env: {os.environ.get('LLMSR_OUTPUT_DIR', 'NOT_SET')}")
    
    # 输入兜底
    solution_strs = solution_strs or []
    extra_infos = extra_infos or [{} for _ in range(len(solution_strs))]
    if len(solution_strs) == 0:
        return []

    # 加载数据（最多 256 个样本，提升速度）
    inputs, outputs, var_names = _load_training_data_from_path(data_path)
    if inputs is None:
        # 返回惩罚
        return [float(-1.0)] * len(solution_strs)

    # 计算各项 reward
    out_dir = os.environ.get("LLMSR_OUTPUT_DIR")
    jsonl_path = None
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            jsonl_path = os.path.join(out_dir, "sample.jsonl")
        except Exception:
            jsonl_path = None

    rewards: List[float] = []
    for i, code in enumerate(solution_strs):
        base_impl = None
        if i < len(extra_infos) and isinstance(extra_infos[i], dict):
            base_impl = extra_infos[i].get("base_impl")

        # 支持 EDIT DSL：若包含 EDIT 指令则基于 base_impl 生成表达式
        edit_mode = False
        
        # 🔥 使用新的方法：直接执行Python函数
        reward, execution_success, mse, complexity, params_used = evaluate_single_solution_v2_fixed(
            code, inputs, outputs, var_names, lambda_nmse, lambda_simp, w_fit, w_simp, w_phys, w_proc
        )
        
        rewards.append(float(reward))

        # 记录样本
        if jsonl_path:
            try:
                rec = {
                    "timestamp": time.time(),
                    "expr": "直接执行Python函数",
                    "params": params_used.tolist() if params_used is not None else None,
                    "reward": float(reward),
                    "nmse": float(mse) if mse is not None else None,
                    "complexity": float(complexity) if complexity is not None else None,
                    "r_fit": None,
                    "r_simp": None,
                    "r_phys": None,
                    "r_proc": None,
                    "edit_mode": edit_mode,
                    "ast_ok": execution_success,
                    "data_path": data_path,
                }
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    # 组内排名归一（若 VERL 批次来自同一提示组，可降低尺度噪声）
    if groupwise_rank_norm and len(rewards) >= 2:
        order = np.argsort(rewards)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(rewards))
        rewards = (1.0 - ranks / max(1, len(rewards) - 1)).astype(np.float32).tolist()

    # VERL 兼容：返回 float 或 list
    if len(rewards) == 1:
        return float(rewards[0])
    return rewards


def evaluate_single_solution_v2_fixed(
    solution_str: str, 
    inputs: np.ndarray, 
    outputs: np.ndarray, 
    var_names: list,
    lambda_nmse: float = 3.0,
    lambda_simp: float = 0.1,
    w_fit: float = 0.6,
    w_simp: float = 0.2,
    w_phys: float = 0.15,
    w_proc: float = 0.05
) -> Tuple[float, bool, float, float, np.ndarray]:
    """
    🔥 V2修复版：使用无RL版本的方法直接执行Python函数 + 多成分奖励
    
    Returns:
        reward, execution_success, mse, complexity, params_used
    """
    
    try:
        # 🔥 步骤1：从LLM输出中提取函数体
        function_body = extract_function_body_v2(solution_str)
        
        if not function_body:
            print(f"❌ V2函数体提取失败")
            return -1.0, False, 1e6, 0.0, None
        
        print(f"✅ V2成功提取函数体，长度: {len(function_body)}")
        
        # 🔥 步骤2：构建完整的可执行程序
        program = build_executable_program_v2(function_body, var_names)
        
        # 🔥 步骤3：在安全环境中执行程序并计算MSE
        mse, params_used = execute_and_compute_mse_v2(program, inputs, outputs, var_names)
        
        if mse >= 1e6:
            return -1.0, False, mse, 0.0, params_used
        
        # 计算NMSE
        var_y = float(np.var(outputs) + 1e-9)
        nmse = mse / var_y
        
        # 计算各项奖励
        r_fit = math.exp(-lambda_nmse * nmse)
        
        # 估算复杂度（基于函数体长度）
        complexity = _estimate_complexity_from_body(function_body)
        r_simp = math.exp(-lambda_simp * complexity)
        
        # 物理一致性（简化版）
        r_phys = _physical_consistency_v2(function_body, var_names, inputs, outputs)
        
        # 过程奖励（简化版）
        r_proc = 0.5 if mse < 1.0 else 0.0
        
        # 综合奖励
        reward = w_fit * r_fit + w_simp * r_simp + w_phys * r_phys + w_proc * r_proc
        
        print(f"✅ V2计算完成 - MSE: {mse:.6f}, 奖励: {reward:.6f}")
        
        return reward, True, mse, complexity, params_used
        
    except Exception as e:
        print(f"❌ V2执行Python函数时出错: {e}")
        return -1.0, False, 1e6, 0.0, None


class _FunctionLineVisitorV2(ast.NodeVisitor):
    """ V2版本：Visitor that finds the last line number of a function with a given name."""

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


def _trim_function_body_v2(generated_code: str) -> str:
    """ V2版本：Extract the body of the generated function, trimming anything after it.
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

    visitor = _FunctionLineVisitorV2('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def extract_function_body_v2(solution_str: str) -> str:
    """V2版本：从LLM输出中提取函数体，完全模仿sampler.py的_extract_body函数"""
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
        return _trim_function_body_v2(code)
    
    return solution_str  # 🔥 如果没找到def，返回原始sample（无RL版本的行为）


def build_executable_program_v2(function_body: str, var_names: list) -> str:
    """V2版本：构建完整的可执行程序"""
    
    # 构建函数签名
    params_str = ', '.join(var_names) + ', params'
    
    # 构建完整的程序
    program = f"""
import numpy as np
import math
from scipy.optimize import minimize

def equation({params_str}):
{function_body}

def evaluate_function(inputs, outputs, var_names):
    \"\"\"V2版本：评估函数性能 - 使用BFGS优化参数\"\"\"
    try:
        def loss_function(params):
            try:
                # 🔥 数值保护：限制参数范围，防止溢出
                params = np.clip(params, -100, 100)
                
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
                
                # 🔥 数值保护：检查预测值是否有效
                if not np.all(np.isfinite(predictions)):
                    return 1e6
                
                # 🔥 数值保护：限制预测值范围，防止极端值
                predictions = np.clip(predictions, -1e6, 1e6)
                
                # 处理标量返回值
                if predictions.ndim == 0:
                    predictions = np.full_like(outputs, float(predictions))
                
                # 计算MSE
                mse = np.mean((predictions - outputs) ** 2)
                
                # 🔥 数值保护：确保MSE有效且不会太大
                if not np.isfinite(mse) or mse > 1e10:
                    return 1e6
                    
                return float(mse)
                
            except (RuntimeWarning, FloatingPointError, OverflowError, ZeroDivisionError):
                return 1e6
            except Exception as e:
                return 1e6
        
        # 🔥 BFGS参数优化（模仿无RL版本）
        initial_params = np.ones(10)
        
        # 🔥 数值保护：添加参数边界约束
        from scipy.optimize import Bounds
        bounds = Bounds(-100, 100)  # 限制参数在[-100, 100]范围内
        
        # 🔥 数值保护：设置优化选项，增加数值稳定性
        options = {
            'maxiter': 100,  # 限制最大迭代次数
            'ftol': 1e-6,    # 函数容差
            'gtol': 1e-6     # 梯度容差
        }
        
        try:
            # 使用L-BFGS-B方法，支持边界约束
            result = minimize(loss_function, initial_params, method='L-BFGS-B', 
                            bounds=bounds, options=options)
            
            # 获取优化后的参数和损失
            optimized_params = result.x
            optimized_loss = result.fun
            
            # 处理优化失败的情况
            if (np.isnan(optimized_loss) or np.isinf(optimized_loss) or 
                not result.success or optimized_loss > 1e6):
                print(f"⚠️ V2 BFGS优化失败，使用默认参数")
                optimized_params = initial_params
                optimized_loss = loss_function(initial_params)
                
        except Exception as e:
            print(f"⚠️ V2 BFGS优化异常: {e}，使用默认参数")
            optimized_params = initial_params
            optimized_loss = loss_function(initial_params)
        
        return float(optimized_loss), optimized_params
        
    except Exception as e:
        print(f"❌ V2函数执行错误: {{e}}")
        return 1e6, np.ones(10)
"""
    
    return program


def execute_and_compute_mse_v2(program: str, inputs: np.ndarray, outputs: np.ndarray, var_names: list) -> tuple[float, np.ndarray]:
    """V2版本：在安全环境中执行程序并计算MSE"""
    
    try:
        # 执行程序
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
        print(f"❌ V2程序执行失败: {e}")
        return 1e6, None


def _load_training_data_from_path(data_path: str | None) -> Tuple[np.ndarray | None, np.ndarray | None, List[str] | None]:
    if not data_path:
        return None, None, None
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        data = df.values
        # 假设最后一列为输出
        X = data[:256, :-1]
        y = data[:256, -1].reshape(-1)
        var_names = df.columns[:-1].tolist()
        return X, y, var_names
    except Exception:
        return None, None, None


def _estimate_complexity_from_body(function_body: str) -> float:
    """从函数体估算复杂度"""
    # 基于代码长度和运算符数量
    ops = len(re.findall(r"[\+\-\*/]", function_body))
    funcs = len(re.findall(r"(sin|cos|tan|exp|log|sqrt|abs|tanh)\(", function_body))
    nums = len(re.findall(r"-?[0-9]*\.?[0-9]+", function_body))
    tokens = max(1, len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", function_body)))
    return 0.5 * ops + 0.8 * funcs + 0.2 * nums + 0.1 * tokens


def _physical_consistency_v2(function_body: str, var_names: List[str], X: np.ndarray, y: np.ndarray) -> float:
    """V2版本：物理一致性检查"""
    try:
        # 简化的物理一致性检查
        if 'log(' in function_body:
            # log 需要正输入，检查是否可能出现非正数
            if np.any(X <= 0):
                return 0.6
        return 1.0
    except Exception:
        return 0.2