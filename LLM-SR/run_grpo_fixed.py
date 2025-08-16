#!/usr/bin/env python3
"""
修复版本的LLM-SR GRPO训练脚本

使用符合VERL要求的数据集，避免所有配置字段缺失问题。
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

# 修复 vLLM 内存池兼容性问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.6"

# 添加VERL到路径
verl_path = str(Path(__file__).parent.parent / "verl")
if verl_path not in sys.path:
    sys.path.append(verl_path)

from verl.trainer.main_ppo import run_ppo


def create_llmsr_reward_file(output_dir: str) -> str:
    """创建LLM-SR自定义奖励函数文件"""
    
    reward_function_code = '''
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
        lines = solution_str.strip().split('\\n')
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
'''
    
    output_path = os.path.join(output_dir, "llmsr_reward.py")
    with open(output_path, 'w') as f:
        f.write(reward_function_code)
    
    print(f"✅ 创建奖励函数: {output_path}")
    return output_path


def create_grpo_config_simple(
    model_path: str,
    dataset_path: str,
    reward_file_path: str,
    output_dir: str,
    **kwargs
) -> DictConfig:
    """
    创建简化的GRPO配置，包含所有必需字段
    """
    
    # 简化的批量大小配置
    gpus = kwargs.get('gpus', 6)
    micro_batch_size_per_gpu = 1  # 最小值以节省内存
    rollout_n = kwargs.get('rollout_n', 4)
    
    # 计算批量大小
    train_batch_size = 48  # 固定小批量
    ppo_mini_batch_size = 24  # 固定小批量
    
    print(f"🔧 简化批量配置:")
    print(f"  GPU数量: {gpus}")
    print(f"  微批量/GPU: {micro_batch_size_per_gpu}")
    print(f"  训练批量: {train_batch_size}")
    print(f"  PPO小批量: {ppo_mini_batch_size}")
    
    config = {
        # 算法配置
        "algorithm": {
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.001,
                "horizon": 10000,
                "target_kl": 0.1
            }
        },
        
        # 数据配置
        "data": {
            "train_files": [dataset_path],
            "val_files": [dataset_path],
            "train_batch_size": train_batch_size,
            "val_batch_size": ppo_mini_batch_size,
            "max_prompt_length": 1024,
            "max_response_length": 512,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "reward_fn_key": "data_source",
            "shuffle": True
        },
        
        # Actor-Rollout-Ref配置
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "tokenizer_only": False,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "model_dtype": "bf16"
            },
            "actor": {
                "strategy": "fsdp",
                "optim": {
                    "lr": kwargs.get('learning_rate', 1e-6),
                    "eps": 1e-8,
                    "weight_decay": 0.01
                },
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": 0.0,
                "entropy_from_logits_with_chunking": False,
                "clip_ratio": 0.2,
                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",
                "use_dynamic_bsz": True,
                "ulysses_sequence_parallel_size": 1,
                "use_remove_padding": True,
                "use_fused_kernels": False,
                "use_torch_compile": True,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,
                "shuffle": False,
                "data_loader_seed": None,
                "policy_loss": {"loss_mode": "vanilla"},
                "fsdp_config": {
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "forward_prefetch": False,
                    "wrap_policy": {"min_num_params": 0}
                },
                "checkpoint": {
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"]
                }
            },
            "ref": {
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "ulysses_sequence_parallel_size": 1,
                "log_prob_use_dynamic_bsz": True,
                "log_prob_max_token_len_per_gpu": 4096,
                "use_remove_padding": True,
                "use_fused_kernels": False,
                "entropy_from_logits_with_chunking": False,
                "use_torch_compile": True,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,
                "fsdp_config": {
                    "forward_prefetch": False,
                    "param_offload": True,
                    "wrap_policy": {"min_num_params": 0}
                }
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",
                "n": rollout_n,
                "max_new_tokens": 512,
                "load_format": "auto",
                "dtype": "bfloat16",
                "prompt_length": 1024,
                "response_length": 512,
                "max_model_len": 4096,
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "disable_log_stats": False,
                "enable_chunked_prefill": False,
                "disable_custom_all_reduce": True,
                "gpu_memory_utilization": 0.4,
                "max_num_batched_tokens": 4096,
                "seed": 0,
                "tensor_model_parallel_size": 1,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "val_kwargs": {
                    "do_sample": True,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 30,
                    "max_new_tokens": 512,
                    "n": 1
                },
                # 🔥 添加所有缺失的必需字段
                "calculate_log_probs": False,
                "free_cache_engine": True
            }
        },
        
        # Critic配置
        "critic": {
            "strategy": "fsdp",
            "ppo_mini_batch_size": ppo_mini_batch_size,
            "ppo_micro_batch_size": None,
            "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "use_dynamic_bsz": True
        },
        
        # 自定义奖励函数
        "custom_reward_function": {
            "path": reward_file_path,
            "name": "compute_score"
        },
        
        # 奖励模型配置
        "reward_model": {
            "enable": False,
            "launch_reward_fn_async": False
        },
        
        # 训练器配置
        "trainer": {
            "total_epochs": kwargs.get('epochs', 3),
            "total_training_steps": None,
            "project_name": "llm_sr_grpo_fixed",
            "experiment_name": kwargs.get('experiment_name', 'oscillator1_fixed'),
            "logger": ["console"],
            "n_gpus_per_node": gpus,
            "nnodes": 1,
            "save_freq": 2,
            "test_freq": 1,
            "val_before_train": True,
            "default_local_dir": output_dir,
            "device": "cuda",
            "resume_mode": "disable"
        },
        
        # Ray配置
        "ray_init": {
            "num_cpus": None
        }
    }
    
    return OmegaConf.create(config)


def train_llmsr_grpo_fixed(
    model_path: str = "/storage/home/westlakeLab/zhangjunlei/Qwen/Qwen2.5-Coder-7B-Instruct",
    output_dir: str = "./llmsr_grpo_outputs/oscillator1_fixed",
    **kwargs
):
    """
    修复版本的LLM-SR GRPO训练
    
    使用预生成的符合VERL要求的数据集，避免所有配置问题。
    """
    
    print("🔥 启动修复版本的LLM-SR GRPO训练")
    print(f"🤖 模型: {model_path}")
    print(f"📁 输出: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用预生成的数据集
    dataset_path = "llmsr_grpo_outputs/oscillator1_train.parquet"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}. 请先运行 create_grpo_dataset.py")
    
    print(f"📊 使用数据集: {dataset_path}")
    
    # 创建奖励函数
    print("\n🎯 创建奖励函数...")
    reward_file_path = create_llmsr_reward_file(output_dir)
    
    # 创建GRPO配置
    print("\n⚙️ 创建GRPO配置...")
    config = create_grpo_config_simple(
        model_path=model_path,
        dataset_path=dataset_path,
        reward_file_path=reward_file_path,
        output_dir=output_dir,
        **kwargs
    )
    
    # 保存配置
    config_path = os.path.join(output_dir, "grpo_config_fixed.yaml")
    OmegaConf.save(config, config_path)
    print(f"💾 保存配置: {config_path}")
    
    # 启动训练
    print("\n🚀 启动GRPO训练...")
    try:
        run_ppo(config)
        print("✅ 训练完成！")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise


if __name__ == "__main__":
    train_llmsr_grpo_fixed(
        epochs=2,
        learning_rate=1e-6,
        rollout_n=2,
        gpus=6,
        experiment_name="oscillator1_fixed_test"
    ) 