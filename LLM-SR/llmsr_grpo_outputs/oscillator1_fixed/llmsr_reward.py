
"""
LLM-SR符号回归奖励函数

对生成的函数进行BFGS优化并返回负MSE作为奖励。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import re

def compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    计算LLM-SR符号回归任务的奖励
    
    Args:
        data_sources: 数据源列表
        solution_strs: 模型生成的解决方案字符串列表
        ground_truths: 参考答案列表（可能为None）
        extra_infos: 额外信息列表
        
    Returns:
        rewards: 奖励分数列表
    """
    rewards = []
    
    # 加载训练数据
    data_path = "data/oscillator1/train.csv"
    df = pd.read_csv(data_path)
    
    # 提取输入输出
    x_data = df['x'].values
    v_data = df['v'].values  
    a_data = df['a'].values
    
    for i, solution_str in enumerate(solution_strs):
        try:
            # 提取函数体
            reward = evaluate_solution(solution_str, x_data, v_data, a_data)
            rewards.append(reward)
        except Exception as e:
            print(f"评估解决方案时出错: {e}")
            rewards.append(0.0)  # 错误情况给0分
    
    return rewards


def evaluate_solution(solution_str: str, x_data: np.ndarray, v_data: np.ndarray, a_data: np.ndarray) -> float:
    """评估单个解决方案的质量"""
    
    try:
        # 提取函数体（简单的正则表达式提取）
        # 查找 return 语句之前的内容
        lines = solution_str.strip().split('\n')
        function_body = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('def ') or line.startswith('"""') or line == '"""':
                continue
            if line and not line.startswith('#'):
                function_body.append(line)
        
        if not function_body:
            return 0.0
            
        # 构建完整的函数代码
        full_function = f"""
import numpy as np

def equation(x, v, params):
    {chr(10).join(['    ' + line for line in function_body])}

def evaluate_equation(params, x_data, v_data, a_data):
    try:
        pred = equation(x_data, v_data, params)
        mse = np.mean((pred - a_data) ** 2)
        return mse
    except:
        return 1e6  # 大的惩罚值
"""
        
        # 执行代码并优化参数
        namespace = {}
        exec(full_function, namespace)
        
        # BFGS优化
        initial_params = np.ones(10)  # 10个参数
        result = minimize(
            lambda p: namespace['evaluate_equation'](p, x_data, v_data, a_data),
            initial_params,
            method='BFGS'
        )
        
        final_mse = result.fun
        
        # 返回负MSE作为奖励（MSE越小，奖励越大）
        if np.isnan(final_mse) or np.isinf(final_mse) or final_mse > 1e3:
            return 0.0
        else:
            return -final_mse  # 负数，因为MSE越小越好
            
    except Exception as e:
        return 0.0  # 出错时返回0
