#!/usr/bin/env python3
"""
🔥 修复版VERL奖励函数 V2 - 采用无RL版本的正确方法 + 多成分奖励

关键改进：
1. 不再提取数学表达式字符串
2. 直接执行LLM生成的完整Python函数
3. 保持V2版本的多成分奖励计算
4. 模仿无RL版本的处理流程
"""

import math
import re
from typing import Any, Dict, List, Tuple
import numpy as np
import os, json, time
import ast
import pandas as pd
from pathlib import Path


def compute_score(
    data_sources=None,  # 移除类型注解，保持与v1版本一致
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    # 移除 * 使所有参数都可以位置传递，同时也支持关键字传递
    grid_train_data=False,
    template_path=None,
    data_path=None,
    memory_dir=None,
    lambda_nmse=3.0,
    lambda_simp=0.1,
    w_fit=0.75,
    w_simp=0.2,
    w_phys=0.15,
    w_proc=0.05,
    groupwise_rank_norm=True,
    # 🔥 新增长度惩罚和解析奖励参数
    length_penalty_alpha=0.03,  # 长度惩罚系数，建议0.02-0.05
    parse_bonus=0.1,            # 解析成功奖励
    invalid_penalty=-0.5,       # 无效样本惩罚
    # 🔥 物理一致性奖励开关（默认关闭）
    enable_physics_reward=False,  # 是否启用物理一致性奖励
    # 🔥 过程奖励系统开关和参数
    enable_process_reward=True,  # 是否启用真过程奖励（默认开启）
    process_reward_weights=None, # 过程奖励各成分权重，None则使用默认值
    # 🏝️ 群岛机制超参数
    num_islands=4,              # 群岛数量
    top_k_per_island=8,         # 每个岛屿保存的top样本数
    # 🔥 数据集刷新管理器配置
    refresh_manager_config=None, # 数据集刷新管理器配置
    **kwargs,
):
    print(f"🔥🔥🔥 FIXED V2 REWARD FUNCTION CALLED! 🔥🔥🔥")
    print(f"🔧 V2修复版参数类型: data_sources={type(data_sources)}, solution_strs={type(solution_strs)}, ground_truths={type(ground_truths)}, extra_infos={type(extra_infos)}")
    print(f"🔧 V2 kwargs: {list(kwargs.keys())}")
    print(f"🔧 V2 Solution strings count: {len(solution_strs) if solution_strs else 0}")
    if solution_strs and len(solution_strs) > 0:
        print(f"🔧 V2 First solution preview: {solution_strs[0][:200] if solution_strs[0] else 'None'}...")
    print(f"🔧 V2 LLMSR_OUTPUT_DIR env: {os.environ.get('LLMSR_OUTPUT_DIR', 'NOT_SET')}")
    
    # 🔥 兼容v1版本的参数处理方式
    # 尝试从kwargs中提取数据（兼容VERL的不同调用方式）
    if not solution_strs and 'responses' in kwargs:
        solution_strs = kwargs['responses']
        print(f"🔧 V2 从kwargs['responses']获取solution_strs: {len(solution_strs) if solution_strs else 0}个")
    if not solution_strs and 'generated_texts' in kwargs:
        solution_strs = kwargs['generated_texts']
        print(f"🔧 V2 从kwargs['generated_texts']获取solution_strs: {len(solution_strs) if solution_strs else 0}个")
    if not extra_infos and 'batch' in kwargs:
        extra_infos = [{'problem_type': 'oscillator1'}]  # 默认问题类型
        print(f"🔧 V2 使用默认extra_infos: {extra_infos}")
    
    # 🔥 重要：处理VERL的单数形式参数（来自v1版本的经验）
    if not solution_strs and 'solution_str' in kwargs:
        solution_strs = [kwargs['solution_str']]  # 转换为列表
        print(f"🔧 V2 从kwargs['solution_str']获取单个solution")
    if not data_sources and 'data_source' in kwargs:
        data_sources = [kwargs['data_source']]
    if not ground_truths and 'ground_truth' in kwargs:
        ground_truths = [kwargs['ground_truth']]
    if not extra_infos and 'extra_info' in kwargs:
        extra_infos = [kwargs['extra_info']]
    
    # 🔥 调试：检查所有参数是否为None的情况
    if data_sources is None and solution_strs is None and ground_truths is None and extra_infos is None:
        print("⚠️ V2 所有参数都为None，这通常发生在VERL验证阶段，返回默认标量值")
        return 0.0
    
    # 🔥 输入兜底 - 与v1版本保持一致
    solution_strs = solution_strs or []
    extra_infos = extra_infos or [{} for _ in range(len(solution_strs))]
    if len(solution_strs) == 0:
        print("⚠️ V2 没有解决方案字符串，返回默认值 0.0")
        return 0.0  # 🔥 返回标量而不是空列表，与v1版本一致

    # 🔥 修复：优先从extra_infos提取problem_type（与v1版本一致）
    problem_type = None
    if extra_infos and len(extra_infos) > 0 and extra_infos[0]:
        if isinstance(extra_infos[0], dict):
            problem_type = extra_infos[0].get('problem_type')
            if not problem_type and 'extra_info' in extra_infos[0]:
                # 处理嵌套的extra_info
                nested_info = extra_infos[0]['extra_info']
                if isinstance(nested_info, dict):
                    problem_type = nested_info.get('problem_type')
    
    print(f"🔧 V2 提取的problem_type: {problem_type}")
    
    # 如果没有提供data_path，尝试从problem_type构建路径
    if not data_path and problem_type:
        data_path = f"data/{problem_type}/train.csv"
        print(f"🔧 V2 从problem_type构建data_path: {data_path}")
    
    # 加载数据（使用全部样本）
    inputs, outputs, var_names = _load_training_data_from_path(data_path)
    if inputs is None:
        # 🔥 如果从data_path加载失败，尝试使用v1版本的load_training_data方法
        print(f"⚠️ V2 从data_path加载失败，尝试使用v1版本的方法")
        train_data = load_training_data_v1(problem_type)
        if train_data is not None:
            inputs, outputs, var_names = train_data
    
    if inputs is None:
        # 🔥 返回惩罚 - 与v1版本保持一致的返回值处理
        print(f"❌ V2 无法加载训练数据，返回惩罚值")
        if len(solution_strs) == 1:
            return 0.0  # 单个样本返回标量
        else:
            return [float(-1.0)] * len(solution_strs)  # 多个样本返回列表

    # 计算各项 reward
    out_dir = os.environ.get("LLMSR_OUTPUT_DIR")
    jsonl_path = None
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            jsonl_path = os.path.join(out_dir, "sample.jsonl")
        except Exception:
            jsonl_path = None

    # 🔥 初始化memory管理器（用于更新memory）- V2改进版：自适应分位数
    memory_manager = None
    
    # 🔥 简化：准备数据集刷新参数（仅在需要时使用）
    dataset_refresh_params = None
    if refresh_manager_config and isinstance(refresh_manager_config, dict):
        refresh_params = refresh_manager_config.get("refresh_params", {})
        if refresh_params:
            dataset_refresh_params = refresh_params.copy()
            dataset_refresh_params["output_dir"] = out_dir or refresh_manager_config.get("output_dir", "./")
    
    if memory_dir and os.path.exists(memory_dir):
        try:
            # 导入MemoryManagerV2（需要确保路径正确）
            import sys
            sys.path.append("/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR")
            from llmsr.rl.grpo_runner_v2 import MemoryManagerV2, refresh_dataset_with_islands
            
            # 🔥 V2改进版：添加自适应参数
            memory_manager = MemoryManagerV2(
                memory_dir, 
                top_k_per_island=top_k_per_island, 
                num_islands=num_islands,
                update_frequency=50,  # 每50个样本更新一次阈值
                recent_samples_window=200  # 基于最近200个样本计算分位数
            )
            print(f"✅ V2 成功初始化自适应memory管理器: {memory_dir} (岛屿:{num_islands}, 每岛top-k:{top_k_per_island}, 自适应更新)")
            print(f"🔄 数据集刷新阈值: 每{memory_manager._dataset_refresh_threshold}个新群岛样本触发一次刷新")
            
        except Exception as e:
            print(f"⚠️ V2 memory管理器初始化失败: {e}")
            memory_manager = None

    rewards: List[float] = []
    for i, code in enumerate(solution_strs):
        base_impl = None
        if i < len(extra_infos) and isinstance(extra_infos[i], dict):
            base_impl = extra_infos[i].get("base_impl")

        # 支持 EDIT DSL：若包含 EDIT 指令则基于 base_impl 生成表达式
        edit_mode = False
        
        # 🔥 使用新的方法：直接执行Python函数
        base_reward, execution_success, mse, complexity, params_used, opt_info = evaluate_single_solution_v2_fixed(
            code, inputs, outputs, var_names, lambda_nmse, lambda_simp, w_fit, w_simp, w_phys, w_proc, 
            enable_physics_reward, enable_process_reward, process_reward_weights
        )
        
        # 🔥 计算长度惩罚：reward := base_reward - α·(len_tokens/1k)
        len_tokens = _estimate_token_length(code)
        length_penalty = length_penalty_alpha * (len_tokens / 1000.0)
        
        # 🔥 解析奖励和无效惩罚
        if execution_success:
            parse_reward = parse_bonus  # 解析成功奖励
        else:
            parse_reward = invalid_penalty  # 无效样本惩罚
        
        # 🔥 最终奖励 = 基础奖励 - 长度惩罚 + 解析奖励/惩罚
        final_reward = base_reward - length_penalty + parse_reward
        
        rewards.append(float(final_reward))

        # 🔥 记录样本到sample.jsonl（模仿v1版本格式）
        if jsonl_path:
            try:
                # 🔥 提取函数体用于记录（模仿v1版本）
                function_body = extract_function_body_v2(code)
                
                rec = {
                    "solution_length": len(code) if code else 0,  # 🔥 v1版本字段
                    "timestamp": time.time(),
                    "execution_success": execution_success,  # 🔥 v1版本字段
                    "function_body": function_body,  # 🔥 v1版本字段
                    "params": params_used.tolist() if params_used is not None else None,  # 🔥 v1版本字段
                    "mse": float(mse) if mse is not None else None,  # 🔥 v1版本字段
                    "reward": float(final_reward),  # 🔥 v1版本字段
                    "error": None if execution_success else "执行失败",  # 🔥 v1版本字段
                    # 🔥 v2版本额外字段
                    "base_reward": float(base_reward),
                    "length_penalty": float(length_penalty),
                    "parse_reward": float(parse_reward),
                    "len_tokens": int(len_tokens),
                    "nmse": float(mse / (np.var(outputs) + 1e-9)) if mse is not None and mse < 1e6 and outputs is not None else None,
                    "complexity": float(complexity) if complexity is not None else None,
                    "r_fit": None,
                    "r_simp": None,
                    "r_phys": None,
                    "r_proc": None,
                    "edit_mode": edit_mode,
                    "ast_ok": execution_success,
                    "data_path": data_path,
                }
                
                # 🔥 添加过程奖励详情（如果可用）
                if enable_process_reward and opt_info is not None:
                    rec["process_reward_details"] = {
                        "optimizer_success": opt_info.get('success', False),
                        "iterations": opt_info.get('nit', 0),
                        "initial_loss": opt_info.get('initial_loss'),
                        "final_loss": opt_info.get('final_loss'),
                        "improvement": opt_info.get('improvement'),
                        "grad_norm": opt_info.get('grad_norm'),
                        "params_norm": opt_info.get('params_norm'),
                        "has_nan_inf": opt_info.get('has_nan_inf', False),
                        "enabled": True
                    }
                else:
                    rec["process_reward_details"] = {"enabled": False}
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ 记录sample.jsonl失败: {e}")
                pass

        # 🔥 V2改进版：更合理的进岛门槛 - 允许执行成功的样本都进入，让自适应分位数机制充分发挥
        if memory_manager and execution_success:  # 只要执行成功就考虑入库
            try:
                function_body = extract_function_body_v2(code)
                if function_body:  # 确保函数体不为空
                    # 🔥 调用add_sample并检查是否需要刷新数据集
                    need_refresh = memory_manager.add_sample(
                        function_body=function_body,
                        score=final_reward,
                        mse=mse,
                        complexity=complexity
                    )
                    print(f"🎯 V2 成功添加样本到memory，score: {final_reward:.3f}")
                    
                    # 🔥 如果需要刷新数据集，立即执行
                    if need_refresh and dataset_refresh_params:
                        try:
                            print(f"🔄 触发数据集刷新（新样本数>=8）...")
                            
                            # 获取当前数据集路径
                            current_dataset = dataset_refresh_params.get("current_dataset_path")
                            if current_dataset and os.path.exists(current_dataset):
                                new_dataset_path = refresh_dataset_with_islands(
                                    dataset_path=current_dataset,
                                    template_path=dataset_refresh_params.get("template_path"),
                                    data_path=dataset_refresh_params.get("data_path"),
                                    memory_dir=memory_dir,
                                    output_dir=dataset_refresh_params["output_dir"],
                                    grid_train_data=dataset_refresh_params.get("grid_train_data", False),
                                    num_grid_groups=dataset_refresh_params.get("num_grid_groups", 10),
                                    few_shot_k=dataset_refresh_params.get("few_shot_k", 3),
                                    num_islands=num_islands,
                                    top_k_per_island=top_k_per_island,
                                )
                                
                                if new_dataset_path != current_dataset:
                                    print(f"✅ 数据集刷新完成: {new_dataset_path}")
                                    # 🔥 创建信号文件，通知可能的外部监控
                                    signal_file = os.path.join(dataset_refresh_params["output_dir"], "dataset_refreshed.signal")
                                    with open(signal_file, "w") as f:
                                        json.dump({
                                            "timestamp": time.time(),
                                            "new_dataset_path": new_dataset_path,
                                            "trigger": "new_samples_threshold_reached",
                                            "new_samples_count": 8
                                        }, f, indent=2)
                                
                        except Exception as refresh_e:
                            print(f"❌ 数据集刷新失败: {refresh_e}")
            except Exception as e:
                print(f"⚠️ V2 添加样本到memory失败: {e}")
                pass

    # 组内排名归一（若 VERL 批次来自同一提示组，可降低尺度噪声）
    if groupwise_rank_norm and len(rewards) >= 2:
        order = np.argsort(rewards)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(rewards))
        rewards = (1.0 - ranks / max(1, len(rewards) - 1)).astype(np.float32).tolist()

    # 🔥 VERL 兼容：返回值处理与v1版本保持一致
    if not rewards:
        print("⚠️ V2 没有计算出奖励，返回默认值 0.0")
        return 0.0
    
    # 🔥 处理单个样本的情况（VERL验证时经常如此）
    if len(rewards) == 1:
        reward_value = float(rewards[0])
        print(f"🎯 V2 返回单个奖励值: {reward_value}")
        return reward_value
    
    # 多个样本的情况
    rewards_array = np.array(rewards, dtype=np.float32)
    print(f"🎯 V2 返回奖励数组，长度: {len(rewards_array)}")
    return rewards_array.tolist()


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
    w_proc: float = 0.05,
    enable_physics_reward: bool = False,
    enable_process_reward: bool = True,
    process_reward_weights: dict = None
):
    """
    🔥 V2修复版：使用无RL版本的方法直接执行Python函数 + 多成分奖励
    
    Returns:
        reward, execution_success, mse, complexity, params_used, opt_info
    """
    
    try:
        # 🔥 步骤1：从LLM输出中提取函数体
        function_body = extract_function_body_v2(solution_str)
        
        if not function_body:
            print(f"❌ V2函数体提取失败")
            return -1.0, False, 1e6, 0.0, None, None
        
        print(f"✅ V2成功提取函数体，长度: {len(function_body)}")
        
        # 🔥 步骤2：构建完整的可执行程序
        program = build_executable_program_v2(function_body, var_names)
        
        # 🔥 步骤3：在安全环境中执行程序并计算MSE
        mse, params_used, opt_info = execute_and_compute_mse_v2(program, inputs, outputs, var_names)
        
        if mse >= 1e6:
            return -1.0, False, mse, 0.0, params_used, opt_info
        
        # 计算NMSE
        var_y = float(np.var(outputs) + 1e-9)
        nmse = mse / var_y
        
        # 计算各项奖励
        r_fit = math.exp(-lambda_nmse * nmse)
        
        # 估算复杂度（基于函数体长度）
        complexity = _estimate_complexity_from_body(function_body)
        r_simp = math.exp(-lambda_simp * complexity)
        
        # 物理一致性（可选，默认关闭）
        if enable_physics_reward:
            r_phys = _physical_consistency_v2(function_body, var_names, inputs, outputs)
        else:
            r_phys = 1.0  # 默认不惩罚
            w_phys = 0.0  # 权重设为0，不影响总奖励
        
        # 🔥 过程奖励：真正的过程监督 vs 占位式
        if enable_process_reward and opt_info is not None:
            # 真正的过程奖励：基于优化器状态
            r_proc = _compute_true_process_reward(
                opt_info, mse, nmse, complexity, process_reward_weights
            )
        else:
            # 占位式过程奖励（向后兼容）
            r_proc = 0.5 if mse < 1.0 else 0.0
        
        # 综合奖励
        reward = w_fit * r_fit + w_simp * r_simp + w_phys * r_phys + w_proc * r_proc
        
        print(f"✅ V2计算完成 - MSE: {mse:.6f}, 奖励: {reward:.6f}")
        
        return reward, True, mse, complexity, params_used, opt_info
        
    except Exception as e:
        print(f"❌ V2执行Python函数时出错: {e}")
        return -1.0, False, 1e6, 0.0, None, None


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
    """V2版本：从LLM输出中提取函数体，改进处理各种格式"""
    if not solution_str or not isinstance(solution_str, str):
        return ""
    
    # 处理<think>标签
    if "</think>" in solution_str:
        parts = solution_str.split("</think>")
        if len(parts) > 1:
            solution_str = parts[-1].strip()
    
    # 🔥 改进：优先查找完整的函数定义
    lines = solution_str.splitlines()
    
    # 首先尝试找到完整的函数定义（def equation...到函数体结束）
    func_start = -1
    func_end = -1
    in_function = False
    func_indent = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 查找函数定义开始
        if stripped.startswith("def equation") and ":" in stripped:
            func_start = i
            in_function = True
            func_indent = len(line) - len(line.lstrip())
            continue
        
        # 如果在函数内部
        if in_function and line.strip():
            current_indent = len(line) - len(line.lstrip())
            # 如果缩进小于或等于函数定义的缩进，说明函数结束
            if current_indent <= func_indent:
                func_end = i
                break
    
    # 如果没有找到函数结束，设置到最后
    if func_start >= 0 and func_end == -1:
        func_end = len(lines)
    
    # 如果找到了完整的函数定义，提取函数体
    if func_start >= 0 and func_end > func_start:
        func_lines = lines[func_start+1:func_end]
        # 过滤掉EDIT指令（如果有的话）
        body_lines = []
        for line in func_lines:
            if not line.strip().startswith("EDIT"):
                body_lines.append(line)
        
        # 如果有函数体内容
        if body_lines:
            # 规范化缩进
            normalized_body = []
            for line in body_lines:
                if line.strip():
                    # 确保缩进为4个空格
                    normalized_body.append('    ' + line.lstrip())
                else:
                    normalized_body.append('')
            
            body_str = '\n'.join(normalized_body) + '\n\n'
            print(f"✅ V2成功提取完整函数体，长度: {len(body_str)}")
            return body_str
    
    # 🔥 如果没有找到完整函数，尝试处理EDIT DSL格式
    if "EDIT ADD" in solution_str:
        edit_terms = []
        
        for line in lines:
            line_stripped = line.strip()
            # 收集EDIT ADD指令
            if line_stripped.startswith("EDIT ADD"):
                term = line_stripped.replace("EDIT ADD", "").strip()
                edit_terms.append(term)
        
        # 如果找到了EDIT指令，构建函数体
        if edit_terms:
            # 构建return语句，正确处理运算符间距
            # 将连续的项用加号连接（如果项本身不是以运算符开始）
            return_parts = []
            for term in edit_terms:
                # 如果项以运算符开始（如 -params[0]），直接添加
                if term.startswith(('+', '-')):
                    if return_parts:  # 如果不是第一项，添加空格
                        return_parts.append(' ')
                    return_parts.append(term)
                else:
                    # 否则添加加号
                    if return_parts:
                        return_parts.append(' + ')
                    return_parts.append(term)
            
            return_expr = ''.join(return_parts)
            function_body = f"    return {return_expr}\n\n"
            print(f"✅ V2成功从EDIT DSL构建函数体: {return_expr}")
            return function_body
    
    # 🔥 如果前面的方法都失败了，查找Python代码块
    import re
    code_block_patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```'
    ]
    
    code_block_content = None
    for pattern in code_block_patterns:
        matches = re.findall(pattern, solution_str, re.DOTALL)
        if matches:
            code_block_content = matches[0]
            break
    
    # 如果找到代码块，递归调用自己处理代码块内容
    if code_block_content:
        print(f"🔍 V2在代码块中查找函数体")
        return extract_function_body_v2(code_block_content)
    
    # 🔥 完全模仿sampler.py的_extract_body逻辑
    lines = solution_str.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        # 🔥 改进：处理可能的前导空格
        stripped_line = line.lstrip()
        if stripped_line.startswith('def equation'):  # 更具体的匹配
            func_body_lineno = lineno
            find_def_declaration = True
            break
        elif line[:3] == 'def':  # 🔥 保留原始的精确匹配作为后备
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # 🔥 改进：更智能地处理函数体
        # 从函数定义的下一行开始收集函数体
        body_lines = []
        base_indent = None
        
        for i in range(func_body_lineno + 1, len(lines)):
            line = lines[i]
            
            # 跳过空行
            if not line.strip():
                continue
            
            # 检测基础缩进级别
            if base_indent is None and line.strip():
                # 计算第一个非空行的缩进
                base_indent = len(line) - len(line.lstrip())
            
            # 如果遇到缩进级别小于基础缩进的行，说明函数体结束
            if base_indent is not None and line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_indent < base_indent:
                    break
            
            body_lines.append(line)
        
        # 构建函数体
        if body_lines:
            # 规范化缩进为4个空格
            normalized_lines = []
            for line in body_lines:
                if line.strip():  # 非空行
                    # 移除原有缩进，添加标准4空格缩进
                    normalized_lines.append('    ' + line.lstrip())
                else:
                    normalized_lines.append('')
            
            function_body = '\n'.join(normalized_lines) + '\n'
            return function_body
        
    # 🔥 如果上述方法都失败，尝试简单的return语句提取
    # 这是为了处理某些情况下函数体直接作为字符串返回
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('return '):
            # 提取return语句
            return_expr = stripped[7:]  # 移除 'return '
            function_body = f"    return {return_expr}\n"
            print(f"✅ V2从return语句提取函数体: {function_body.strip()}")
            return function_body
    
    # 🔥 如果都失败了，检查是否整个字符串就是一个简单的表达式
    # 去除空白行后，如果剩余内容看起来像一个Python表达式，直接作为return语句
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if non_empty_lines and len(non_empty_lines) <= 3:  # 简单表达式通常不超过3行
        # 检查是否包含Python运算符或函数调用
        expr_text = ' '.join(non_empty_lines)
        if any(op in expr_text for op in ['+', '-', '*', '/', '**', 'np.', 'math.', 'params[']):
            function_body = f"    return {expr_text}\n"
            print(f"✅ V2将表达式作为函数体: {function_body.strip()}")
            return function_body
    
    print(f"⚠️ V2函数体提取失败，返回空字符串。原始内容长度: {len(solution_str)}")
    return ""  # 🔥 保持返回空字符串，这样可以明确知道提取失败


def build_executable_program_v2(function_body: str, var_names: list) -> str:
    """V2版本：构建完整的可执行程序"""
    
    # 构建函数签名
    params_str = ', '.join(var_names) + ', params'
    
    # 🔥 使用字符串拼接而不是f-string，避免格式化问题
    program_template = """
import numpy as np
import math
from scipy.optimize import minimize

def equation(PARAMS_PLACEHOLDER):
FUNCTION_BODY_PLACEHOLDER

def evaluate_function(inputs, outputs, var_names):
    '''V2版本：评估函数性能 - 使用BFGS优化参数，返回详细优化信息'''
    try:
        # 记录初始MSE（用随机参数）
        initial_params = np.ones(10)
        
        def loss_function(params):
            try:
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
                return float(mse) if np.isfinite(mse) else 1e6
                
            except Exception as e:
                return 1e6
        
        # 计算初始损失
        initial_loss = loss_function(initial_params)
        
        # 🔥 BFGS参数优化（模仿无RL版本）
        result = minimize(loss_function, initial_params, method='BFGS', options={'maxiter': 100})
        
        # 获取优化后的参数和损失
        optimized_params = result.x
        optimized_loss = result.fun
        
        # 构建优化信息字典
        opt_info = {
            'success': result.success,
            'nit': result.nit,  # 迭代次数
            'initial_loss': float(initial_loss),
            'final_loss': float(optimized_loss),
            'improvement': float((initial_loss - optimized_loss) / (initial_loss + 1e-9)),
            'message': result.message if hasattr(result, 'message') else '',
            'grad_norm': float(np.linalg.norm(result.jac)) if hasattr(result, 'jac') and result.jac is not None else None,
            'params_norm': float(np.linalg.norm(optimized_params)),
            'has_nan_inf': bool(np.any(np.isnan(optimized_params)) or np.any(np.isinf(optimized_params)))
        }
        
        # 处理优化失败的情况
        if np.isnan(optimized_loss) or np.isinf(optimized_loss) or not result.success:
            print("⚠️ V2 BFGS优化失败，使用默认参数")
            optimized_params = initial_params
            optimized_loss = loss_function(initial_params)
            opt_info['success'] = False
            opt_info['final_loss'] = float(optimized_loss)
        
        return float(optimized_loss), optimized_params, opt_info
        
    except Exception as e:
        print("❌ V2函数执行错误: " + str(e))
        return 1e6, np.ones(10), None
"""
    
    # 安全替换占位符
    program = program_template.replace("PARAMS_PLACEHOLDER", params_str)
    program = program.replace("FUNCTION_BODY_PLACEHOLDER", function_body)
    
    return program


def execute_and_compute_mse_v2(program: str, inputs: np.ndarray, outputs: np.ndarray, var_names: list) -> tuple[float, np.ndarray, dict]:
    """V2版本：在安全环境中执行程序并计算MSE，返回优化信息"""
    
    try:
        # 执行程序
        all_globals_namespace = {
            'np': np,
            'numpy': np,
            'math': math
        }
        
        # 添加scipy.optimize.minimize到命名空间
        from scipy.optimize import minimize
        all_globals_namespace['minimize'] = minimize
        
        # 执行程序
        exec(program, all_globals_namespace)
        
        # 获取评估函数
        evaluate_function = all_globals_namespace['evaluate_function']
        
        # 调用评估函数（现在返回3个值）
        result = evaluate_function(inputs, outputs, var_names)
        
        # 处理返回值
        if isinstance(result, tuple) and len(result) == 3:
            mse, params_used, opt_info = result
        elif isinstance(result, tuple) and len(result) == 2:
            # 向后兼容：如果只返回2个值
            mse, params_used = result
            opt_info = None
        else:
            mse = 1e6
            params_used = None
            opt_info = None
        
        return mse, params_used, opt_info
        
    except Exception as e:
        print(f"❌ V2程序执行失败: {e}")
        return 1e6, None, None


def load_training_data_v1(problem_type):
    """根据问题类型加载训练数据（来自v1版本）"""
    
    if not problem_type:
        print("⚠️ V2 问题类型未知，尝试使用oscillator1作为默认")
        problem_type = "oscillator1"
    
    # 数据文件路径
    data_file = f"data/{problem_type}/train.csv"
    if not Path(data_file).exists():
        print(f"❌ V2 数据文件不存在: {data_file}")
        return None
    
    try:
        df = pd.read_csv(data_file)
        
        if problem_type == "oscillator1":
            if all(col in df.columns for col in ['x', 'v', 'a']):
                inputs = df[['x', 'v']].values  # 使用全部样本
                outputs = df['a'].values
                var_names = ['x', 'v']
                return inputs, outputs, var_names
                
        elif problem_type == "oscillator2":
            if all(col in df.columns for col in ['t', 'x', 'v', 'a']):
                inputs = df[['x', 'v']].values  # 只用x,v，忽略t，使用全部样本
                outputs = df['a'].values
                var_names = ['x', 'v']
                return inputs, outputs, var_names
                
        elif problem_type == "bactgrow":
            if all(col in df.columns for col in ['b', 's', 'temp', 'pH', 'db']):
                inputs = df[['b', 's', 'temp', 'pH']].values  # 使用全部样本
                outputs = df['db'].values
                var_names = ['b', 's', 'temp', 'pH']
                return inputs, outputs, var_names
                
        elif problem_type == "stressstrain":
            if all(col in df.columns for col in ['strain', 'temp', 'stress']):
                inputs = df[['strain', 'temp']].values  # 使用全部样本
                outputs = df['stress'].values
                var_names = ['strain', 'temp']
                return inputs, outputs, var_names
        
        print(f"❌ V2 不支持的问题类型或数据格式: {problem_type}, 列: {list(df.columns)}")
        return None
        
    except Exception as e:
        print(f"❌ V2 加载数据时出错: {e}")
        return None


def _load_training_data_from_path(data_path: str | None) -> Tuple[np.ndarray | None, np.ndarray | None, List[str] | None]:
    if not data_path:
        return None, None, None
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        data = df.values
        # 假设最后一列为输出，使用全部样本
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        var_names = df.columns[:-1].tolist()
        return X, y, var_names
    except Exception:
        return None, None, None


def _estimate_complexity_from_body(function_body: str) -> float:
    """
    🔥 升级版复杂度估计：AST + 子树复用 + 常数MDL + 嵌套深度 + 分段/不可微结构
    基于AST分析而非简单正则，更准确反映表达式的结构复杂度
    """
    return estimate_complexity_from_body_v3(function_body)


# —— 基础权重（参照 PySR: 超越函数更贵） ——
OP_WEIGHTS = {
    'Add': 1.0, 'Sub': 1.0,
    'Mult': 1.5, 'Div': 2.0, 'FloorDiv': 2.0, 'Mod': 2.5,
    'Pow': 3.0,
}
# 常见数学函数代价；可继续扩充
FUNC_WEIGHTS = {
    "sin": 2.0, "cos": 2.0, "tan": 3.0,
    "exp": 4.0, "log": 4.0, "sqrt": 3.0, "abs": 2.0, "tanh": 3.0,
    "sinh": 3.0, "cosh": 3.0, "atan": 3.0, "asin": 3.0, "acos": 3.0,
}


def estimate_complexity_from_body_v3(function_body: str) -> float:
    """
    更精细的复杂度估计（AST + 子树复用 + 常数MDL + 嵌套深度 + 分段/不可微结构）
    返回标量复杂度 C（越大越复杂）
    """
    if not function_body or not isinstance(function_body, str):
        return 0.0

    # 构造可解析的假函数，保证缩进正确
    code = f"def __eq__(x, y, z, params):\n{function_body}"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 语法不通时给高复杂度
        return 100.0

    # 找到目标函数体
    fnode = None
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "__eq__":
            fnode = n
            break
    
    if fnode is None:
        return 100.0

    # —— 状态容器 ——
    stats = {
        "op_cost": 0.0,              # 加权算子成本
        "func_cost": 0.0,            # 加权函数成本
        "depth_max": 0,              # 最大嵌套深度
        "piecewise_cnt": 0,          # 分段/不可微结构出现次数
        "pow_max_k": 1,              # 最大幂阶
        "const_bits": 0.0,           # 常数描述长度（MDL 近似）
        "unique_subtrees": 0,        # DAG 唯一子树计数
        "total_subtrees": 0,         # 子树总数（用于复用率估计）
        "poly_terms": 0,             # 估算多项式项数
    }

    # —— 子树哈希：衡量 DAG 压缩性/唯一子式数量 ——
    from collections import defaultdict
    counter = defaultdict(int)
    
    def hash_subtree(n):
        # 基于节点类型与子结构的递归哈希（文本化）；只做启发式
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return f"Const({repr(n.value)})"
        elif hasattr(ast, 'Num') and isinstance(n, ast.Num):  # Python < 3.8 兼容
            return f"Const({repr(n.n)})"
        
        label = type(n).__name__
        fields = []
        for name, val in ast.iter_fields(n):
            if isinstance(val, ast.AST):
                fields.append((name, hash_subtree(val)))
            elif isinstance(val, list):
                fields.append((name, tuple(hash_subtree(x) for x in val if isinstance(x, ast.AST))))
            elif isinstance(val, (str, int, float, bool, type(None))):
                # 不把行号/列号等元信息纳入
                if name not in ("lineno", "col_offset", "end_lineno", "end_col_offset", "id", "arg"):
                    fields.append((name, val))
        key = f"{label}:{tuple(fields)}"
        counter[key] += 1
        return key

    # —— 遍历：统计各分量 + 记录深度/幂阶/分段结构 ——
    def walk(n, depth=0):
        stats["depth_max"] = max(stats["depth_max"], depth)
        stats["total_subtrees"] += 1
        key = hash_subtree(n)  # 触发计数

        # 二元算子
        if isinstance(n, ast.BinOp):
            op_name = type(n.op).__name__
            stats["op_cost"] += OP_WEIGHTS.get(op_name, 1.5)
            if isinstance(n.op, ast.Pow):
                # 解析幂阶（仅当指数是常数时可靠）
                k = _extract_integer_pow(n)
                if k is not None:
                    stats["pow_max_k"] = max(stats["pow_max_k"], k)

        # 函数调用
        if isinstance(n, ast.Call):
            fname = _get_call_name(n)
            if fname:
                stats["func_cost"] += FUNC_WEIGHTS.get(fname, 3.0)

        # 分段/不可微：if-else, 比较, 条件表达式, abs()
        if isinstance(n, (ast.If, ast.IfExp, ast.Compare)):
            stats["piecewise_cnt"] += 1
        if isinstance(n, ast.Call):
            fname = _get_call_name(n)
            if fname in ("abs",):
                stats["piecewise_cnt"] += 1

        # 常数的 MDL 近似
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            stats["const_bits"] += _constant_description_bits(n.value)
        elif hasattr(ast, 'Num') and isinstance(n, ast.Num):  # Python < 3.8 兼容
            stats["const_bits"] += _constant_description_bits(n.n)

        # 递归子节点
        for child in ast.iter_child_nodes(n):
            walk(child, depth + 1)

    for stmt in fnode.body:
        walk(stmt, depth=1)

    # 统计唯一子树数（DAG）
    stats["unique_subtrees"] = sum(1 for k, c in counter.items() if c >= 1)

    # 估算多项式项数（启发：按 "加法链 + 幂表达式" 粗略估计）
    stats["poly_terms"] = _estimate_poly_terms(counter)

    # —— 组合复杂度（权重可调） ——
    # 深度额外惩罚：深层操作在计算和可解释性上都更难
    depth_cost = 0.5 * stats["depth_max"]
    # DAG：唯一子树越多越复杂；可用 "唯一/总数" 的比值来度量可压缩性
    if stats["total_subtrees"] > 0:
        dag_ratio = stats["unique_subtrees"] / float(stats["total_subtrees"])
    else:
        dag_ratio = 1.0
    dag_cost = 5.0 * dag_ratio

    # 幂阶、分段、项数
    pow_cost = 0.3 * max(0, stats["pow_max_k"] - 1)
    piece_cost = 1.5 * stats["piecewise_cnt"]
    terms_cost = 0.2 * stats["poly_terms"]

    # 合成总复杂度
    C = (
        stats["op_cost"]
        + stats["func_cost"]
        + depth_cost
        + dag_cost
        + 0.05 * stats["const_bits"]  # 常数的描述长度（位数/精度越高越贵）
        + pow_cost
        + piece_cost
        + terms_cost
    )
    return float(C)


# —— 辅助：提取函数名 ——
def _get_call_name(node):
    """提取函数调用的名称"""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# —— 辅助：提取幂指数（若为整数常数） ——
def _extract_integer_pow(binop):
    """提取幂运算的指数（如果是整数常数）"""
    if isinstance(binop.op, ast.Pow):
        if isinstance(binop.right, ast.Constant) and isinstance(binop.right.value, (int, float)):
            try:
                k = int(binop.right.value)
                return k if k >= 1 else None
            except Exception:
                return None
        elif hasattr(ast, 'Num') and isinstance(binop.right, ast.Num):  # Python < 3.8 兼容
            try:
                k = int(binop.right.n)
                return k if k >= 1 else None
            except Exception:
                return None
    return None


# —— 辅助：MDL 近似（常数的描述长度，位数+数量级） ——
def _constant_description_bits(v) -> float:
    """计算常数的描述长度（MDL近似）"""
    v = float(v)
    if v == 0.0:
        return 1.0
    # 位数惩罚：小数的有效数字越多越贵（以十进制近似）
    s = f"{v:.12g}"  # 限12位有效数字，避免科学计数法极端
    digits = len(re.sub(r"[^0-9]", "", s))
    # 数量级惩罚：|log10(|v|)| 越大越贵（防超大/超小常数）
    magnitude = abs(math.log10(abs(v))) if v != 0 else 0.0
    return digits + 2.0 * magnitude


# —— 辅助：估算多项式项数（启发式：基于子树键） ——
def _estimate_poly_terms(subtree_counter) -> int:
    """估算多项式项数（启发式方法）"""
    # 统计出现 "Add:" 的子树个数作为项分裂的粗略度量
    terms = sum(1 for k in subtree_counter if k.startswith("BinOp:") and "Add" in k)
    return max(0, terms)


def _compute_true_process_reward(
    opt_info: dict, 
    mse: float, 
    nmse: float, 
    complexity: float,
    process_reward_weights: dict = None
) -> float:
    """
    🔥 计算真正的过程奖励（基于优化器状态和执行过程）
    
    包含以下成分：
    1. 收敛指示 (r_conv): 优化器是否成功收敛
    2. 改进幅度 (r_impr): 从初始到最终的改进程度
    3. 迭代效率 (r_eff): 收敛速度
    4. 数值健康 (r_num): 无NaN/Inf、参数合理
    5. 约束满足 (r_cons): MSE阈值等约束
    """
    
    # 默认权重
    if process_reward_weights is None:
        process_reward_weights = {
            'conv': 0.3,   # 收敛指示权重
            'impr': 0.25,  # 改进幅度权重
            'eff': 0.15,   # 迭代效率权重
            'num': 0.2,    # 数值健康权重
            'cons': 0.1    # 约束满足权重
        }
    
    # 1. 收敛指示奖励
    if opt_info.get('success', False):
        r_conv = 1.0
    else:
        # 使用sigmoid平滑，基于梯度范数
        grad_norm = opt_info.get('grad_norm', 1e3)
        if grad_norm is not None and np.isfinite(grad_norm):
            # sigmoid(-grad_norm/scale), scale=10使得grad_norm=10时约0.27
            r_conv = 1.0 / (1.0 + np.exp(grad_norm / 10.0))
        else:
            r_conv = 0.0
    
    # 2. 改进幅度奖励
    initial_loss = opt_info.get('initial_loss', 1e6)
    final_loss = opt_info.get('final_loss', 1e6)
    
    if initial_loss > 0 and final_loss > 0 and initial_loss >= final_loss:
        # log-scale improvement reward
        improvement_ratio = (initial_loss + 1e-9) / (final_loss + 1e-9)
        r_impr = np.clip(np.log(improvement_ratio) / 5.0, 0.0, 1.0)  # log(148)≈5 maps to 1.0
    else:
        r_impr = 0.0
    
    # 3. 迭代效率奖励
    nit = opt_info.get('nit', 100)
    max_iter = 100  # BFGS默认最大迭代
    r_eff = 1.0 - min(1.0, nit / max_iter)
    
    # 4. 数值健康奖励
    has_nan_inf = opt_info.get('has_nan_inf', False)
    params_norm = opt_info.get('params_norm', 0.0)
    
    # 基础健康分
    r_num_base = 0.0 if has_nan_inf else 1.0
    
    # 参数范数惩罚（过大的参数通常不稳定）
    if params_norm > 0:
        # 参数范数在10以内给满分，超过100降到0.5
        r_num_norm = 1.0 / (1.0 + (params_norm / 50.0) ** 2)
    else:
        r_num_norm = 1.0
    
    r_num = 0.7 * r_num_base + 0.3 * r_num_norm
    
    # 5. 约束满足奖励
    # MSE阈值约束
    if mse < 0.1:
        r_cons_mse = 1.0
    elif mse < 1.0:
        r_cons_mse = 0.8
    elif mse < 10.0:
        r_cons_mse = 0.5
    else:
        r_cons_mse = 0.0
    
    # 复杂度约束（鼓励简单解）
    if complexity < 5.0:
        r_cons_comp = 1.0
    elif complexity < 10.0:
        r_cons_comp = 0.7
    elif complexity < 20.0:
        r_cons_comp = 0.4
    else:
        r_cons_comp = 0.1
    
    r_cons = 0.6 * r_cons_mse + 0.4 * r_cons_comp
    
    # 综合过程奖励
    r_proc = (
        process_reward_weights['conv'] * r_conv +
        process_reward_weights['impr'] * r_impr +
        process_reward_weights['eff'] * r_eff +
        process_reward_weights['num'] * r_num +
        process_reward_weights['cons'] * r_cons
    )
    
    # 打印详细信息（调试用）
    print(f"🔬 过程奖励详情: conv={r_conv:.3f}, impr={r_impr:.3f}, "
          f"eff={r_eff:.3f}, num={r_num:.3f}, cons={r_cons:.3f}, "
          f"总计={r_proc:.3f}")
    
    return float(r_proc)


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


def _estimate_token_length(
    text: str,
    *,
    # 可选：传入真实 tokenizer（优先使用）
    hf_tokenizer=None,     # 例如 transformers 的 AutoTokenizer() 实例
    tiktoken_encoder=None, # 例如 tiktoken.encoding_for_model(...).encode
    tokenizer_encode_fn=None,  # 任何可调用的 encode 函数
    model_family: str = "qwen",  # 'qwen' | 'openai' | 'llama' | 'generic'
) -> int:
    """
    🔥 升级版token长度估计：精确优先 + 启发式兜底
    1) 若提供真实 tokenizer，直接返回精确长度
    2) 否则采用"类别分段 + 字节/4"的混合启发式，按模型家族调系数
    """
    
    # ---- 1) 精确计数（优先） ----
    if text is None or text == "":
        return 0
    
    # 🔥 尝试自动加载Qwen3-8B的tokenizer
    if hf_tokenizer is None and tiktoken_encoder is None and tokenizer_encode_fn is None:
        hf_tokenizer = _get_qwen_tokenizer()
    
    try:
        # transformers
        if hf_tokenizer is not None and hasattr(hf_tokenizer, "encode"):
            return int(len(hf_tokenizer.encode(text)))
        # tiktoken
        if tiktoken_encoder is not None and hasattr(tiktoken_encoder, "encode"):
            return int(len(tiktoken_encoder.encode(text)))
        # 任意可调用 encode
        if callable(tokenizer_encode_fn):
            return int(len(tokenizer_encode_fn(text)))
    except Exception as e:
        # 若失败，退回启发式
        print(f"⚠️ Tokenizer失败，使用启发式估计: {e}")
        pass

    # ---- 2) 改进启发式（类别分段 + UTF-8字节）----
    return _estimate_token_length_heuristic(text, model_family)


def _get_qwen_tokenizer():
    """尝试加载Qwen3-8B的tokenizer"""
    try:
        from transformers import AutoTokenizer
        # 尝试从本地Qwen3-8B目录加载
        qwen_path = "/storage/home/westlakeLab/zhangjunlei/Qwen3-8B"
        if os.path.exists(qwen_path):
            tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
            print(f"✅ 成功加载Qwen3-8B tokenizer: {qwen_path}")
            return tokenizer
    except Exception as e:
        print(f"⚠️ 无法从 {qwen_path} 加载Qwen3-8B tokenizer: {e}")
    return None


def _estimate_token_length_heuristic(text: str, model_family: str = "qwen") -> int:
    """
    启发式token长度估计：类别分段 + UTF-8字节混合模型
    """
    if not text:
        return 0
    
    # 2.1 类别划分（尽量互斥）
    # CJK 统一表意 & 扩展、假名、韩文音节
    re_cjk = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7A3]")
    # 近似 Emoji / 表情符（覆盖常见区段）
    re_emoji = re.compile(r"[\U0001F000-\U0001FAFF\U00002702-\U000027B0]")
    # URL / Email（URL 先行匹配，避免被按词拆散）
    re_url  = re.compile(r"https?://[^\s]+|www\.[^\s]+")
    re_mail = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    # 数字（含小数/科学计数）
    re_num  = re.compile(r"[+-]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][+-]?\d+)?")
    # 代码/操作符串
    re_ops  = re.compile(r"[+\-*/=<>!%&|^~]+")
    # 蛇形/驼峰词（优先抓长词，避免过度切碎）
    re_word = re.compile(r"[A-Za-z]+(?:[_\-][A-Za-z0-9]+)*|[A-Za-z][a-z0-9]+(?:[A-Z][a-z0-9]+)+")
    # 其它可见 ASCII 标点
    re_punc = re.compile(r"[.,;:!?…—–()\[\]{}\<>\"'`""''#@$]")
    # 空白
    re_space = re.compile(r"\s+")

    text_remaining = text

    def _pop_all(pattern):
        nonlocal text_remaining
        items = pattern.findall(text_remaining)
        text_remaining = pattern.sub(" ", text_remaining)  # 用空格占位，避免连锁影响
        return items

    urls  = _pop_all(re_url)
    mails = _pop_all(re_mail)
    nums  = _pop_all(re_num)
    opss  = _pop_all(re_ops)
    words = _pop_all(re_word)
    # 先弹出 emoji，再匹配 CJK（避免重复计数）
    emojis = _pop_all(re_emoji)
    cjks   = _pop_all(re_cjk)
    puncs  = _pop_all(re_punc)
    spaces = _pop_all(re_space)

    # 剩余零散字符（混合：可能是稀有符号、控制符等）
    leftovers = [c for c in text_remaining if not c.isspace()]

    # 2.2 模型家族系数（可按经验/标定微调）
    if model_family.lower() in ("qwen", "qwen2", "qwen3"):
        coef = dict(
            en_char_per_tok = 4.0,   # 英文：4字符/Token
            digit_char_per_tok = 3.0,# 数字：3字符/Token
            cjk_tok_per_char = 0.65, # 中文日文韩文：~0.6–0.8 Token/字（Qwen BPE 常见区间）
            url_char_per_tok = 3.0,  # URL 更碎：3字符/Token
            mail_char_per_tok= 3.2,  # Email
            ops_char_per_tok = 2.0,  # 操作符：2字符/Token
            punc_char_per_tok= 2.5,  # 标点：2.5字符/Token
            space_char_per_tok=10.0, # 空白：10字符/Token（大多并入相邻 token 的前导空格）
            emoji_tok_per_char=1.3,  # emoji：1.3 Token/字符
            leftover_char_per_tok=3.2,
            mix_byte_weight = 0.30,  # 与"字节/4"融合的权重
        )
    elif model_family.lower() in ("openai", "gpt", "o"):
        coef = dict(
            en_char_per_tok = 4.0,
            digit_char_per_tok = 2.8,
            cjk_tok_per_char = 1.0,  # tiktoken 上中文更接近 1 Token/字
            url_char_per_tok = 2.6,
            mail_char_per_tok= 2.8,
            ops_char_per_tok = 1.8,
            punc_char_per_tok= 2.2,
            space_char_per_tok=12.0,
            emoji_tok_per_char=1.6,
            leftover_char_per_tok=3.0,
            mix_byte_weight = 0.35,
        )
    else:  # 'llama'/'generic' 兜底
        coef = dict(
            en_char_per_tok = 4.0,
            digit_char_per_tok = 3.0,
            cjk_tok_per_char = 0.9,
            url_char_per_tok = 2.8,
            mail_char_per_tok= 3.0,
            ops_char_per_tok = 2.0,
            punc_char_per_tok= 2.5,
            space_char_per_tok=10.0,
            emoji_tok_per_char=1.5,
            leftover_char_per_tok=3.2,
            mix_byte_weight = 0.30,
        )

    # 2.3 子类估计（把"字符/每Token"或"Token/字符"统一换算成 Token 计数）
    # 英文词按字符数估计（拼合蛇形/驼峰后更接近真实 BPE）
    en_chars = sum(len(w) for w in words)
    tokens_en   = en_chars / coef["en_char_per_tok"]
    tokens_num  = sum(len(s) for s in nums)  / coef["digit_char_per_tok"]
    tokens_url  = sum(len(s) for s in urls)  / coef["url_char_per_tok"]
    tokens_mail = sum(len(s) for s in mails) / coef["mail_char_per_tok"]
    tokens_ops  = sum(len(s) for s in opss)  / coef["ops_char_per_tok"]
    tokens_punc = sum(len(s) for s in puncs) / coef["punc_char_per_tok"]
    tokens_space= sum(len(s) for s in spaces)/ coef["space_char_per_tok"]
    tokens_cjk  = sum(len(s) for s in cjks)  * coef["cjk_tok_per_char"]
    tokens_emoji= sum(len(s) for s in emojis)* coef["emoji_tok_per_char"]
    tokens_left = len(leftovers) / coef["leftover_char_per_tok"]

    est_class = (
        tokens_en + tokens_num + tokens_url + tokens_mail +
        tokens_ops + tokens_punc + tokens_space + tokens_cjk +
        tokens_emoji + tokens_left
    )

    # 2.4 字节/4 融合（tiktoken 文档经验：平均每 Token ~4字节）
    est_bytes = len(text.encode("utf-8")) / 4.0
    mix_w = float(coef["mix_byte_weight"])
    est = (1.0 - mix_w) * est_class + mix_w * est_bytes

    # 2.5 保护性约束（避免极端低估/高估）
    # - Token 不可能超过"可见字符数 * 2"（极端碎裂上限，宽松）
    # - 也不应小于"非空字符数 / 8"（极端合并下限，宽松）
    nonspace = len([c for c in text if not c.isspace()])
    upper = 2.0 * nonspace + 16  # 加常数项应对很短文本
    lower = max(1.0, nonspace / 8.0)
    est = max(lower, min(est, upper))

    return int(math.ceil(est))