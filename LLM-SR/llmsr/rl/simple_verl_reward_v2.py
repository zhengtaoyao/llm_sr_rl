"""
VERL 自定义奖励（v2）：符号回归的密化多成分奖励 + 组内排名归一

组成：
- 拟合：NMSE→r_fit = exp(-lambda * NMSE)
- 简洁：基于表达式 AST 复杂度（长度/节点/算子加权）
- 物理一致性：可选维度/边界/单调性等软约束通过计分
- 过程奖励：常数拟合是否成功、是否通过语法校验
- 排名归一：同一组内对候选加权求和后做 list-wise 归一，降低尺度噪声

默认对 code 片段提取数学表达式并在安全环境中 eval 计算；
如出现异常则返回强惩罚。
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple
import numpy as np


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
    rewards: List[float] = []
    for i, code in enumerate(solution_strs):
        expr = _extract_math_expr(code)
        if not expr:
            rewards.append(-1.0)
            continue

        nmse = _compute_nmse(expr, inputs, outputs, var_names)
        r_fit = math.exp(-lambda_nmse * nmse)

        complexity = _estimate_ast_complexity(expr)
        r_simp = math.exp(-lambda_simp * complexity)

        r_phys = _physical_consistency(expr, var_names, inputs, outputs)

        r_proc = 0.0
        if _constants_optimized(expr, inputs, outputs, var_names):
            r_proc += 0.5

        reward = w_fit * r_fit + w_simp * r_simp + w_phys * r_phys + w_proc * r_proc
        rewards.append(float(reward))

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


# ============== 辅助函数 ==============

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


def _extract_math_expr(code: str) -> str:
    if not code or not isinstance(code, str):
        return ""
    code = code.strip()
    lines = code.split("\n")
    assigns: Dict[str, str] = {}
    ret_var: str | None = None

    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('return '):
            val = s[len('return '):].strip()
            if val.isidentifier():
                ret_var = val
            else:
                if _valid_expr(val):
                    return val
        elif '=' in s and not s.startswith('def'):
            left, right = s.split('=', 1)
            left = left.strip()
            right = right.strip()
            if left.isidentifier() and _valid_expr(right):
                assigns[left] = right

    if ret_var and ret_var in assigns:
        return assigns[ret_var]
    if assigns:
        #回退：取最后一个或常见名称
        for k in ["result", "output", "y", "a", "value"]:
            if k in assigns:
                return assigns[k]
        return list(assigns.values())[-1]
    # 直接匹配
    for line in lines:
        s = line.strip()
        if _valid_expr(s):
            return s
    return ""


def _valid_expr(expr: str) -> bool:
    if not expr:
        return False
    invalid = [r'print\s*\(', r'import\s+', r'def\s+', r'class\s+', r'if\s+', r'for\s+', r'while\s+']
    if any(re.search(p, expr) for p in invalid):
        return False
    has = [r'[a-zA-Z_][a-zA-Z0-9_]*', r'-?[0-9]*\.?[0-9]+', r'[\+\-\*/\(\)]', r'(sin|cos|tan|exp|log|sqrt|abs|tanh)\(']
    return any(re.search(p, expr) for p in has)


def _compute_nmse(expr: str, X: np.ndarray, y: np.ndarray, var_names: List[str]) -> float:
    if not expr:
        return 1.0
    safe = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs, "tanh": np.tanh,
        "pi": np.pi, "e": np.e, "np": np, "__builtins__": {},
    }
    try:
        for i, vn in enumerate(var_names):
            if i < X.shape[1]:
                safe[vn] = X[:, i]
        cleaned = expr.replace('^', '**').replace(' ', '')
        pred = eval(cleaned, safe)
        if isinstance(pred, (int, float)):
            pred = np.full_like(y, float(pred), dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape[0] != y.shape[0]:
            pred = np.full_like(y, float(pred[0]) if pred.size > 0 else 0.0, dtype=np.float64)
        mse = float(np.mean((pred - y) ** 2))
        var = float(np.var(y) + 1e-9)
        nmse = mse / var
        if not np.isfinite(nmse) or nmse < 0:
            return 1.0
        return min(10.0, nmse)
    except Exception:
        return 1.0


def _estimate_ast_complexity(expr: str) -> float:
    # 近似：长度 + 运算符/函数计数
    ops = len(re.findall(r"[\+\-\*/]", expr))
    funcs = len(re.findall(r"(sin|cos|tan|exp|log|sqrt|abs|tanh)\(", expr))
    nums = len(re.findall(r"-?[0-9]*\.?[0-9]+", expr))
    tokens = max(1, len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)))
    return 0.5 * ops + 0.8 * funcs + 0.2 * nums + 0.1 * tokens


def _physical_consistency(expr: str, var_names: List[str], X: np.ndarray, y: np.ndarray) -> float:
    # 占位：通过基本数值稳定性检查 + 边界粗检
    # 若需要维度分析，可在外部传入单位表；此处返回 {1.0, 0.6, 0.2}
    try:
        # 值域合理性粗检
        if re.search(r"log\(", expr):
            # log 需要正输入，检查是否可能出现非正数
            if np.any(X <= 0):
                return 0.6
        return 1.0
    except Exception:
        return 0.2


def _constants_optimized(expr: str, X: np.ndarray, y: np.ndarray, var_names: List[str]) -> bool:
    # 简易：若存在数字常数，尝试一轮线性缩放拟合 y≈a*pred+b 是否收敛
    try:
        safe = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh, "pi": np.pi, "e": np.e, "np": np, "__builtins__": {}}
        for i, vn in enumerate(var_names):
            if i < X.shape[1]:
                safe[vn] = X[:, i]
        cleaned = expr.replace('^', '**').replace(' ', '')
        pred = eval(cleaned, safe)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        A = np.stack([pred, np.ones_like(pred)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        recon = A @ coef
        rel = float(np.mean((recon - y) ** 2) / (np.var(y) + 1e-9))
        return rel < 0.9
    except Exception:
        return False



