"""
GRPO Training Runner for LLM-SR Integration

This module provides the main training loop that integrates VERL's GRPO
with LLM-SR's evaluation system for symbolic regression tasks.

🔥 重要说明：
- train_llmsr_grpo_direct(): 直连模式，Actor 进程直接加载模型并更新权重
- train_llmsr_grpo_http(): HTTP 模式，通过 HTTP 调用外部 LLM，不更新权重
- 默认使用直连模式进行真正的权重微调
"""

# 🔥 修复 vLLM 内存池兼容性问题：在导入PyTorch/vLLM之前设置CUDA分配器配置
import os
# 移除 expandable_segments 以避免 vLLM 内存池断言失败
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.6"
import sys
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig
import tempfile

# Add VERL to path if needed
verl_path = str(Path(__file__).parent.parent.parent.parent / "verl")
if verl_path not in sys.path:
    sys.path.append(verl_path)

from verl.trainer.main_ppo import run_ppo
from llmsr.rl.grpo_worker import compute_llmsr_reward
try:
    import wandb  # 可选：启用 Weights & Biases
except Exception:
    wandb = None


def _force_wandb_cloud() -> None:
    """确保使用官方云：移除环境中的 WANDB_BASE_URL（若存在）。"""
    try:
        if "WANDB_BASE_URL" in os.environ:
            os.environ.pop("WANDB_BASE_URL", None)
    except Exception:
        pass


def _is_wandb_enabled() -> bool:
    """读取环境变量 USE_WANDB，默认关闭。支持 1/true/yes/on."""
    val = os.getenv("USE_WANDB", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}



def create_llmsr_dataset(template_path: str, data_path: str, output_dir: str, grid_train_data: bool = False, num_grid_groups: int = 10):
    """
    Create VERL-compatible dataset from LLM-SR specification and data.
    
    Args:
        template_path: Path to LLM-SR specification template
        data_path: Path to training data CSV
        output_dir: Directory to save converted dataset
        grid_train_data: Whether to use grid-based sampling (default: False)
        num_grid_groups: Number of grid groups to divide data into (default: 10)
        
    Returns:
        Path to the created dataset file
    """
    import pandas as pd
    import json
    import numpy as np
    from sklearn.preprocessing import KBinsDiscretizer
    
    # Load the specification template
    with open(template_path, 'r') as f:
        specification = f.read()
    
    # Extract the prompt template (everything before @equation.evolve function)
    lines = specification.split('\n')
    prompt_lines = []
    in_evolve_function = False
    
    for line in lines:
        if '@equation.evolve' in line:
            in_evolve_function = True
            continue
        if in_evolve_function and line.strip().startswith('def '):
            # Found the function definition, include it and stop
            prompt_lines.append(line.rstrip())
            break
        if not in_evolve_function:
            prompt_lines.append(line.rstrip())
    
    # Create the prompt template
    base_prompt = '\n'.join(prompt_lines).strip()
    
    # Load data to determine the task
    df = pd.read_csv(data_path)
    # Limit for training efficiency
    # max_samples = min(5000, len(df))
    max_samples = len(df)

    # Grid-based sampling if enabled
    if grid_train_data:
        print(f"🔢 Using grid-based sampling with {num_grid_groups} groups")
        
        # Find key feature columns (all except for outputs)
        input_cols = []
        output_col = None
        
        # Auto-detect input/output columns based on problem type
        if 'oscillator' in data_path:
            input_cols = ['x', 'v'] if 'v' in df.columns else ['x']
            output_col = 'a'
        elif 'bactgrow' in data_path:
            input_cols = ['b', 's', 'temp', 'pH']
            output_col = 'db'
        elif 'stressstrain' in data_path:
            input_cols = ['strain', 'temp']
            output_col = 'stress'
        else:
            # Default: assume last column is output, rest are inputs
            input_cols = df.columns[:-1].tolist()
            output_col = df.columns[-1]
        
        print(f"📊 Detected input columns: {input_cols}, output column: {output_col}")
        
        # Create grid-based groups
        grid_groups = []
        
        if len(input_cols) == 1:
            # Simple case: one input dimension
            discretizer = KBinsDiscretizer(n_bins=num_grid_groups, encode='ordinal', strategy='uniform')
            df['grid_group'] = discretizer.fit_transform(df[input_cols].values)
            
        elif len(input_cols) == 2:
            # 2D case: use a grid for both dimensions
            discretizer_x = KBinsDiscretizer(n_bins=int(np.sqrt(num_grid_groups)), encode='ordinal', strategy='uniform')
            discretizer_y = KBinsDiscretizer(n_bins=int(np.sqrt(num_grid_groups)), encode='ordinal', strategy='uniform')
            
            group_x = discretizer_x.fit_transform(df[[input_cols[0]]].values)
            group_y = discretizer_y.fit_transform(df[[input_cols[1]]].values)
            
            df['grid_group'] = group_x * int(np.sqrt(num_grid_groups)) + group_y
            
        else:
            # Multi-dimensional case: use k-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_grid_groups, random_state=42)
            df['grid_group'] = kmeans.fit_predict(df[input_cols].values)
            
        # Ensure we have proper group identifiers
        df['grid_group'] = df['grid_group'].astype(int)
        
        # Sample evenly from each group
        samples_per_group = max(1, max_samples // num_grid_groups)
        sampled_dfs = []
        
        for group_id in range(num_grid_groups):
            group_df = df[df['grid_group'] == group_id]
            if len(group_df) > 0:
                group_sample = min(samples_per_group, len(group_df))
                sampled_dfs.append(group_df.sample(n=group_sample, random_state=42))
        
        # Combine all samples
        df_sampled = pd.concat(sampled_dfs).reset_index(drop=True)
        num_samples = len(df_sampled)
        print(f"🔢 Grid sampling complete: {num_samples} samples across {len(sampled_dfs)} groups")
        
    else:
        # Original random sampling
        num_samples = min(max_samples, len(df))
        # num_samples = min(max_samples, 500)

        df_sampled = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        # Add a dummy grid_group for consistency
        df_sampled['grid_group'] = 0
    
    # Create dataset entries in chat format
    dataset_entries = []
    for i in range(num_samples):
        # Get the current row
        row = df_sampled.iloc[i]
        
        # Create a dictionary of the data point
        data_point = row.drop('grid_group').to_dict()
        
        # Format as conversation with user role (required by chat models)
        chat_prompt = [
            {
                "role": "user", 
                "content": base_prompt
            }
        ]
        
        entry = {
            "prompt": chat_prompt,
            "data_source": "llm_sr_train",
            "reward_model": {
                "style": "rule"  # Use rule-based evaluation
            },
            "extra_info": {
                "grid_group": int(row['grid_group']),  # Store the grid group
                "data_point": data_point  # Store the actual data point
            }
        }
        dataset_entries.append(entry)
    
    # Save as parquet file (VERL format)
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    output_path = os.path.join(output_dir, "llmsr_train.parquet")
    table = pa.Table.from_pylist(dataset_entries)
    pq.write_table(table, output_path)
    
    print(f"✅ Created VERL dataset: {output_path} ({len(dataset_entries)} entries)")
    return output_path


def create_llmsr_reward_file(template_path: str, data_path: str, output_dir: str, grid_train_data: bool = False):
    """
    Create a custom reward function file for VERL.
    
    Args:
        template_path: Path to LLM-SR specification template
        data_path: Path to training data CSV  
        output_dir: Directory to save reward function file
        grid_train_data: Whether to use grid-based evaluation
        
    Returns:
        Path to the created reward function file
    """
    
    reward_function_code = f'''
"""
🔥 Simplified VERL reward function - Compatible with new data format
Avoids KeyError: 'ground_truth' problem
"""

import sys
import os
from pathlib import Path

# Import simplified reward function
sys.path.append(str(Path(__file__).parent.parent.parent))
from simple_verl_reward import compute_score as simple_compute_score

def compute_score(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, grid_train_data={grid_train_data}, **kwargs):
    """
    🔥 Use simplified reward calculation, avoid ground_truth KeyError - Support default parameters
    
    Args:
        data_sources: List of data sources (default: None)
        solution_strs: List of model-generated solution strings (default: None)
        ground_truths: List of reference answers (default: None)
        extra_infos: List of extra information (default: None)
        grid_train_data: Whether to use grid-based evaluation (default: {grid_train_data})
        
    Returns:
        rewards: List of reward scores
    """
    
    print(f"🔧 Wrapper called, parameter types: data_sources={{type(data_sources)}}, solution_strs={{type(solution_strs)}}, ground_truths={{type(ground_truths)}}, extra_infos={{type(extra_infos)}}")
    
    # 🔧 Handle None parameters
    if data_sources is None:
        data_sources = []
    if solution_strs is None:
        solution_strs = []
    if ground_truths is None:
        ground_truths = []
    if extra_infos is None:
        extra_infos = []
    
    # Determine problem type
    problem_type = "oscillator1"  # Default value
    data_path = "{data_path}"
    
    if "oscillator1" in data_path:
        problem_type = "oscillator1"
    elif "oscillator2" in data_path:
        problem_type = "oscillator2"
    elif "bactgrow" in data_path:
        problem_type = "bactgrow"
    elif "stressstrain" in data_path:
        problem_type = "stressstrain"
    
    # Build extra_infos to pass problem type
    if not extra_infos:
        extra_infos = [{{'problem_type': problem_type}}] * len(solution_strs)
    else:
        # Ensure each extra_info has problem_type
        for i, extra_info in enumerate(extra_infos):
            if not extra_info:
                extra_infos[i] = {{'problem_type': problem_type}}
            elif 'problem_type' not in extra_info:
                extra_infos[i]['problem_type'] = problem_type
    
    # Call simplified reward function with grid_train_data parameter
    return simple_compute_score(data_sources, solution_strs, ground_truths, extra_infos, grid_train_data={grid_train_data}, **kwargs)
'''
    
    output_path = os.path.join(output_dir, "llmsr_reward.py")
    with open(output_path, 'w') as f:
        f.write(reward_function_code)
    
    print(f"✅ Created reward function: {output_path}")
    return output_path


def create_grpo_config_direct(
    model_path: str,
    dataset_path: str, 
    reward_file_path: str,
    output_dir: str,
    grid_train_data: bool = False,
    **kwargs
) -> DictConfig:
    """
    🔥 修复批量大小配置的直连模式 VERL GRPO 配置
    
    根据VERL官方示例，正确设置批量大小避免除零错误
    """
    
    # 🔥 批量大小配置 - 参照VERL官方示例，针对7B模型优化内存
    gpus = kwargs.get('gpus', 6)
    micro_batch_size_per_gpu = 1  # 🔥 减少到1以节省内存
    rollout_n = kwargs.get('rollout_n', 4)  # 响应数量
    
    # 计算各级批量大小 (参照VERL官方公式，但使用更小的值)
    traj_micro_bsz = micro_batch_size_per_gpu * gpus          # 1 * 6 = 6
    traj_mini_bsz = traj_micro_bsz * 2                        # 6 * 2 = 12  
    prompt_mini_bsz = traj_mini_bsz * rollout_n               # 12 * 4 = 48
    prompt_bsz = prompt_mini_bsz * 1                          # 48 * 1 = 48 (减少倍数)
    
    print(f"🔧 内存优化批量大小配置:")
    print(f"  微批量/GPU: {micro_batch_size_per_gpu}")
    print(f"  GPU数量: {gpus}")
    print(f"  轨迹微批量: {traj_micro_bsz}")
    print(f"  轨迹小批量: {traj_mini_bsz}")
    print(f"  提示小批量: {prompt_mini_bsz}")
    print(f"  训练批量: {prompt_bsz}")
    print(f"  🔥 内存优化: 启用参数/优化器offload")
    
    # 计算安全的 token 长度配置，避免 max_seq_len 超过阈值
    prompt_len_cfg = kwargs.get('max_prompt_length', 1024)
    response_len_cfg = kwargs.get('max_new_tokens', 2048)
    safe_max_token_len = max(4096, int(prompt_len_cfg + response_len_cfg + 256))

    # 直连模式 GRPO 配置
    # 按需选择 logger
    trainer_loggers = ["console"]
    if _is_wandb_enabled():
        trainer_loggers.append("wandb")

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
            "train_batch_size": prompt_bsz,  # 🔥 使用计算得出的训练批量
            "val_batch_size": prompt_mini_bsz,  # 🔥 验证用小批量
            "max_prompt_length": kwargs.get('max_prompt_length', 1024),
            "max_response_length": kwargs.get('max_response_length', 1024),
            "filter_overlong_prompts": True,
            "truncation": "error",
            "reward_fn_key": "data_source",
            "shuffle": True
        },
        
        # 🔥 直连模式 Actor 配置 - 真正加载模型权重
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "tokenizer_only": False,  # 🔥 关键：加载完整模型，不只是tokenizer
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "model_dtype": "bf16"
            },
            "actor": {
                "strategy": "fsdp",  # 🔥 使用 FSDP 进行分布式训练
                "optim": {
                    "lr": kwargs.get('learning_rate', 1e-6),
                    "eps": 1e-8,
                    "weight_decay": 0.01
                },
                # 🔥 CRITICAL: 修复批量大小配置
                "ppo_mini_batch_size": prompt_mini_bsz,  # 96
                "ppo_micro_batch_size": None,  # 废弃字段
                "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 2 (必须>0)
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": 0.0,
                "entropy_from_logits_with_chunking": False,
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,

                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",
                "use_dynamic_bsz": True,
                "ulysses_sequence_parallel_size": 1,  # 禁用序列并行
                # 使用更安全的 token 长度，避免序列长度超过阈值
                "ppo_max_token_len_per_gpu": kwargs.get('ppo_max_token_len_per_gpu', safe_max_token_len),

                # 🔥 添加可能缺失的字段
                "use_remove_padding": True,
                "use_fused_kernels": False,
                "use_torch_compile": False,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,
                "shuffle": False,
                "data_loader_seed": None,
                # 🔥 添加policy_loss配置
                "policy_loss": {
                    "loss_mode": "vanilla"
                },
                "fsdp_config": {
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "forward_prefetch": False,  # 🔥 禁用前向预取以节省内存
                    "wrap_policy": {
                        "min_num_params": 0  # 🔥 更小的包装策略
                    }
                },
                "checkpoint": {
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"]
                }
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",  # 🔥 CRITICAL: 必需字段
                "n": rollout_n,  # 4
                "max_new_tokens": kwargs.get('max_new_tokens', 2048),  # 🔥 增加到原来4倍
                "load_format": "auto",
                "dtype": "bfloat16",
                "prompt_length": kwargs.get('max_prompt_length', 1024),
                "response_length": kwargs.get('max_new_tokens', 2048),  # 🔥 增加到原来4倍
                "max_model_len": kwargs.get('max_model_len', 2048),  # 🔥 减小到2048以匹配max_num_batched_tokens
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "disable_log_stats": False,
                "enable_chunked_prefill": False,
                "disable_custom_all_reduce": True,
                "gpu_memory_utilization": 0.6,
                "max_num_batched_tokens": 2048,  # 🔥 减少批量token数量以适应8卡配置
                "seed": 0,
                "log_prob_use_dynamic_bsz": True,
                "ulysses_sequence_parallel_size": 1,  # 与 actor 保持一致

                "tensor_model_parallel_size": 1,
                "temperature": kwargs.get('temperature', 0.8),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 30),
                "log_prob_micro_batch_size": None,
                # 🔥 CRITICAL: ref模型也需要正确的微批量大小

                # 解耦 log_prob 最大 token 长度，与 rollout 的 max_model_len 独立
                "log_prob_max_token_len_per_gpu": kwargs.get('log_prob_max_token_len_per_gpu', safe_max_token_len),
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 1
                "val_kwargs": {
                    "do_sample": True,
                    "temperature": kwargs.get('temperature', 0.8),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 30),
                    "max_new_tokens": kwargs.get('max_new_tokens', 2048),  # 🔥 增加到原来4倍
                    "n": 1  # 🔥 CRITICAL: 添加缺失的验证时采样数量
                },
                # 🔥 CRITICAL: 添加缺失的rollout字段 (直连模式)
                "calculate_log_probs": False,  # 用于调试的rollout概率记录
                "free_cache_engine": True,  # 生成后释放KV缓存引擎
                # 🔥 CRITICAL: 必需的 multi_turn 配置
                "multi_turn": {
                    "enable": False,
                    "max_turns": None,
                    "tool_config_path": None,
                    "completion_callback": None,
                    "use_inference_chat_template": False,
                    "enable_tokenization_sanity_check": True,
                    "format": "hermes"
                }
            },
            # 🔥 CRITICAL: 修复 ref 配置，添加所有必需字段
            "ref": {
                "log_prob_micro_batch_size": None,
                # 🔥 CRITICAL: ref模型必须有正确的微批量大小 (不能为0!)
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 1
                # 🔥 CRITICAL: 添加缺失的 ulysses_sequence_parallel_size 字段
                "ulysses_sequence_parallel_size": 1,  # 与 actor 保持一致
                "log_prob_use_dynamic_bsz": True,
                # 参考模型 log_prob 的最大 token 长度同样使用安全阈值
                "log_prob_max_token_len_per_gpu": kwargs.get('log_prob_max_token_len_per_gpu', safe_max_token_len),
                # 🔥 添加 DataParallelPPOActor 需要的其他字段
                "use_remove_padding": True,  # 与 actor 保持一致
                "use_fused_kernels": False,  # 禁用融合内核
                "entropy_from_logits_with_chunking": False,
                "use_torch_compile": False,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,  # 梯度裁剪，即使ref不优化也需要
                "fsdp_config": {
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "forward_prefetch": False,  # 🔥 禁用前向预取以节省内存
                    "wrap_policy": {
                        "min_num_params": 0
                    }
                }
            },
            "hybrid_engine": True
        },
        
        # 🔥 CRITICAL: Critic 配置也需要正确的微批量大小
        "critic": {
            "strategy": "fsdp",
            "ppo_mini_batch_size": prompt_mini_bsz,  # 48
            "ppo_micro_batch_size": None,  # 废弃字段
            "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 1 (必须>0)
            "use_dynamic_bsz": True,
            "fsdp_config": {
                "fsdp_size": gpus,
                "param_offload": True,
                "optimizer_offload": True,
                "forward_prefetch": False,
                "wrap_policy": {
                    "min_num_params": 0
                }
            }
        },
        
        # 奖励模型配置
        "reward_model": {
            "enable": False,
            "launch_reward_fn_async": False
        },
        
        # 自定义奖励函数
        "custom_reward_function": {
            "path": reward_file_path,
            "name": "compute_score",
            "kwargs": {
                "grid_train_data": grid_train_data
            }
        },
        
        # 训练器配置
        "trainer": {
            "total_epochs": kwargs.get('epochs', 10),
            "total_training_steps": None,
            "project_name": "llm_sr_grpo_direct",
            "experiment_name": kwargs.get('experiment_name', 'direct_weight_tuning_fixed'),
            "logger": trainer_loggers,
            "n_gpus_per_node": gpus,
            "nnodes": 1,
            "save_freq": kwargs.get('save_freq', 2),
            "test_freq": kwargs.get('test_freq', 5),
            "val_before_train": True,
            "default_local_dir": output_dir,
            "device": "cuda",
            "resume_mode": "disable",
            "default_hdfs_dir": None,

            # 🔧 添加缺失的配置项以避免 ConfigAttributeError
            "log_val_generations": kwargs.get('log_val_generations', 10),  # 验证时记录的生成样本数
            "log_train_generations": kwargs.get('log_train_generations', 5),  # 训练时记录的生成样本数
            "profile_steps": None,
            "balance_batch": None,
            "critic_warmup": 0, # 🔥 FIX: Add missing critic_warmup key

            # 训练器用于对齐/重排的最大 token 长度，同步提升到安全阈值
            "log_prob_max_token_len_per_gpu": kwargs.get('log_prob_max_token_len_per_gpu', safe_max_token_len)
        },
        
        # Ray 初始化
        "ray_init": {
            "num_cpus": None
        }
    }
    
    return OmegaConf.create(config)


def create_grpo_config(
    model_path: str,
    dataset_path: str, 
    reward_file_path: str,
    output_dir: str,
    **kwargs
) -> DictConfig:
    """
    创建 VERL GRPO 配置 - 兼容旧版本，实际调用直连模式
    """
    print("⚠️  使用直连模式配置 (真正微调权重)")
    return create_grpo_config_direct(
        model_path=model_path,
        dataset_path=dataset_path,
        reward_file_path=reward_file_path,
        output_dir=output_dir,
        **kwargs
    )


def create_grpo_config_http(
    http_url: str,
    tokenizer_path: str,
    dataset_path: str, 
    reward_file_path: str,
    output_dir: str,
    **kwargs
) -> DictConfig:
    """
    创建 HTTP 模式的 VERL GRPO 配置 - 不更新权重，仅用于推理
    
    ⚠️  注意：HTTP 模式下权重不会更新！
    """
    
    # 🔥 批量大小配置 - 参照VERL官方示例 (HTTP模式用较小配置)
    gpus = kwargs.get('gpus', 2)
    micro_batch_size_per_gpu = 2  # 每GPU微批量大小 (必须>0)
    rollout_n = kwargs.get('rollout_n', 4)  # 响应数量
    
    # 计算各级批量大小 (参照VERL官方公式)
    traj_micro_bsz = micro_batch_size_per_gpu * gpus          # 2 * 2 = 4
    traj_mini_bsz = traj_micro_bsz * 2                        # 4 * 2 = 8
    prompt_mini_bsz = traj_mini_bsz * rollout_n               # 8 * 4 = 32
    prompt_bsz = prompt_mini_bsz * 2                          # 32 * 2 = 64
    
    # HTTP 模式配置
    # 按需选择 logger
    trainer_loggers = ["console"]
    if _is_wandb_enabled():
        trainer_loggers.append("wandb")

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
            "train_batch_size": prompt_bsz,  # 64
            "val_batch_size": prompt_mini_bsz,  # 32
            "max_prompt_length": kwargs.get('max_prompt_length', 1024),
            "max_response_length": kwargs.get('max_response_length', 512),
            "filter_overlong_prompts": True,
            "truncation": "error",
            "reward_fn_key": "data_source",
            "shuffle": True
        },
        
        # 🔥 HTTP 模式 Actor 配置 - 只加载 tokenizer，不更新权重
        "actor_rollout_ref": {
            "model": {
                "path": tokenizer_path,
                "tokenizer_only": True  # 🔥 关键：只加载tokenizer，不加载模型权重
            },
            "actor": {
                "strategy": "fsdp",
                "optim": {
                    "lr": kwargs.get('learning_rate', 1e-6)
                },
                "ppo_mini_batch_size": prompt_mini_bsz,  # 32
                "ppo_micro_batch_size": None,  # 废弃字段
                "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 2
                "use_kl_loss": False,
                "entropy_coeff": 0,
                "entropy_from_logits_with_chunking": False,
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,

                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",
                "use_dynamic_bsz": True,
                "ulysses_sequence_parallel_size": 1,  # 禁用序列并行
                "fsdp_config": {
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "forward_prefetch": False,
                    "wrap_policy": {
                        "min_num_params": 0
                    }
                },
                "checkpoint": {
                    "save_contents": [],  # HTTP模式不保存权重
                    "load_contents": []   # HTTP模式不加载权重
                }
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",  # 🔥 CRITICAL: 同步模式，必需字段
                "http_url": http_url,
                "n": rollout_n,  # 4
                "max_new_tokens": kwargs.get('max_new_tokens', 2048),  # 🔥 增加到原来4倍
                "load_format": "auto",
                "dtype": "bfloat16",
                "log_prob_use_dynamic_bsz": True,

                "prompt_length": kwargs.get('max_prompt_length', 1024),
                "response_length": kwargs.get('max_new_tokens', 2048),  # 🔥 增加到原来4倍
                "max_model_len": kwargs.get('max_model_len', 2048),  # 🔥 减小到2048以匹配max_num_batched_tokens
                "enforce_eager": True,
                "free_cache_engine": False,
                "enable_prefix_caching": False,
                "disable_log_stats": False,
                "enable_chunked_prefill": False,
                "disable_custom_all_reduce": True,
                "gpu_memory_utilization": 0.60,
                "max_num_batched_tokens": 8192,
                "seed": 0,
                "tensor_model_parallel_size": 1,
                "temperature": kwargs.get('temperature', 0.8),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 30),
                "timeout": kwargs.get('timeout', 60),
                "repeat_prompt": 1,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 2
                "log_prob_max_token_len_per_gpu": kwargs.get('max_model_len', 24000),
                # 🔥 CRITICAL: 添加缺失的验证配置

                "val_kwargs": {
                    "do_sample": True,
                    "n": rollout_n,
                    "max_new_tokens": kwargs.get('max_new_tokens', 2048),  # 🔥 增加到原来4倍
                    "temperature": kwargs.get('temperature', 0.8),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 30)
                },
                # 🔥 CRITICAL: 添加缺失的rollout字段 (HTTP模式)
                "calculate_log_probs": False,  # 用于调试的rollout概率记录
                "free_cache_engine": True,  # 生成后释放KV缓存引擎
                # 🔥 CRITICAL: 添加缺失的 multi_turn 配置
                "multi_turn": {
                    "enable": False,
                    "max_turns": None,
                    "tool_config_path": None,
                    "completion_callback": None,
                    "use_inference_chat_template": False,
                    "enable_tokenization_sanity_check": True,
                    "format": "hermes"
                }
            },
            # 🔥 CRITICAL: 修复HTTP模式的ref配置
            "ref": {
                "disable": True,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 2
                # 🔥 添加必需的配置字段
                "ulysses_sequence_parallel_size": 1,
                "use_remove_padding": False,  # HTTP模式禁用
                "use_fused_kernels": False,
                "entropy_from_logits_with_chunking": False,
                "use_torch_compile": False,  # HTTP模式禁用
                "entropy_checkpointing": False,
                "grad_clip": 1.0,
                "fsdp_config": {
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "forward_prefetch": False,
                    "wrap_policy": {
                        "min_num_params": 0
                    }
                }
            },
            "hybrid_engine": True
        },
        
        # Critic 配置
        "critic": {
            "strategy": "fsdp",
            "ppo_mini_batch_size": prompt_mini_bsz,  # 32
            "ppo_micro_batch_size": None,  # 废弃字段
            "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 2
            "use_dynamic_bsz": True
        },
        
        # 奖励模型配置
        "reward_model": {
            "enable": False,
            "launch_reward_fn_async": False
        },
        
        # 自定义奖励函数
        "custom_reward_function": {
            "path": reward_file_path,
            "name": "compute_score"
        },
        
        # 训练器配置
        "trainer": {
            "total_epochs": kwargs.get('epochs', 10),
            "total_training_steps": None,
            "project_name": "llm_sr_grpo_http",
            "experiment_name": kwargs.get('experiment_name', 'http_mode_no_weight_update'),
            "logger": trainer_loggers,
            "n_gpus_per_node": gpus,
            "nnodes": 1,
            "save_freq": kwargs.get('save_freq', -1),
            "test_freq": kwargs.get('test_freq', 5),
            "val_before_train": True,
            "default_local_dir": output_dir,
            "device": "cuda",
            "critic_warmup": 0, # 🔥 FIX: Add missing critic_warmup key
            "default_hdfs_dir": None,

            "resume_mode": "disable"
        },
        
        # Ray 初始化
        "ray_init": {
            "num_cpus": None
        }
    }
    
    return OmegaConf.create(config)


def train_llmsr_grpo_direct(
    template_path: str,
    data_path: str, 
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./llmsr_grpo_outputs",
    grid_train_data: bool = False,
    num_grid_groups: int = 10,
    **kwargs
):
    """
    🔥 Direct mode LLM-SR GRPO training - Actually fine-tunes LLM weights
    
    Args:
        template_path: Path to LLM-SR specification template
        data_path: Path to training data CSV
        model_path: Model path or name
        output_dir: Output directory
        grid_train_data: Whether to use grid-based training (default: False)
        num_grid_groups: Number of grid groups to divide data into (default: 10)
        **kwargs: Additional training configuration
    """
    
    print("🔥 Starting LLM-SR GRPO direct mode training (fine-tuning weights)")
    print(f"📋 Template: {template_path}")
    print(f"📊 Data: {data_path}")
    print(f"🤖 Model: {model_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🔢 Grid training: {grid_train_data} (groups: {num_grid_groups if grid_train_data else 'N/A'})")
    print(f"🔧 Direct mode: Actor process loads model directly, updates weights via FSDP")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 🔥 Step 1: Check for pre-generated VERL dataset
    print("\n📝 Step 1: Checking VERL dataset...")
    
    # Extract problem_name from data_path
    problem_name = None
    if 'oscillator1' in data_path:
        problem_name = 'oscillator1'
    elif 'oscillator2' in data_path:
        problem_name = 'oscillator2'
    elif 'bactgrow' in data_path:
        problem_name = 'bactgrow'
    elif 'stressstrain' in data_path:
        problem_name = 'stressstrain'
    
    # Check for pre-generated dataset
    verl_dataset_path = None
    if problem_name and not grid_train_data:  # Only use pre-generated datasets for non-grid mode
        potential_path = f"./verl_datasets/{problem_name}_train_verl.parquet"
        if os.path.exists(potential_path):
            verl_dataset_path = potential_path
            print(f"✅ Found pre-generated VERL dataset: {verl_dataset_path}")
        else:
            print(f"❌ Pre-generated VERL dataset not found: {potential_path}")
    
    # If no pre-generated dataset, create new one
    if not verl_dataset_path:
        print("📝 Creating new VERL dataset...")
        dataset_path = create_llmsr_dataset(
            template_path, 
            data_path, 
            output_dir, 
            grid_train_data=grid_train_data, 
            num_grid_groups=num_grid_groups
        )
    else:
        dataset_path = verl_dataset_path
        print(f"📊 Using pre-generated dataset: {dataset_path}")
    
    # Step 2: Create custom reward function
    print("\n🎯 Step 2: Creating reward function...")
    reward_file_path = create_llmsr_reward_file(template_path, data_path, output_dir)
    
    # Step 3: Create direct mode GRPO configuration
    print("\n⚙️ Step 3: Creating direct mode GRPO configuration...")
    config = create_grpo_config_direct(
        model_path=model_path,
        dataset_path=dataset_path,
        reward_file_path=reward_file_path,
        output_dir=output_dir,
        grid_train_data=grid_train_data,  # Pass the grid_train_data parameter
        **kwargs
    )
    
    # Save configuration
    config_path = os.path.join(output_dir, "grpo_config_direct.yaml")
    OmegaConf.save(config, config_path)
    print(f"💾 Saved direct mode configuration: {config_path}")
    
    # Step 4: Start GRPO training
    print("\n🔥 Step 4: Starting GRPO direct mode training...")
    print("⚡ Model weights will be fine-tuned via FSDP")
    # ✅ 初始化 Weights & Biases（若可用）
    try:
        if _is_wandb_enabled() and wandb is not None and getattr(wandb, "run", None) is None:
            _force_wandb_cloud()
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "llm_sr_grpo"),
                name=os.getenv("WANDB_NAME", f"direct_{problem_name}"),
                entity=os.getenv("WANDB_ENTITY"),
                config={
                    "template_path": template_path,
                    "data_path": data_path,
                    "model_path": model_path,
                    "grid_train_data": grid_train_data,
                    **kwargs,
                },
                dir=os.getenv("WANDB_DIR", output_dir),
                reinit=True,
            )
            print("✅ W&B 已初始化")
    except Exception as e:
        print(f"⚠️ W&B 初始化失败: {e}")
    try:
        run_ppo(config)
        print("✅ Direct mode training complete, weights updated!")
    except Exception as e:
        print(f"❌ Direct mode training failed: {e}")
        raise
    finally:
        try:
            if _is_wandb_enabled() and wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.finish()
                print("✅ W&B 已结束记录")
        except Exception:
            pass


def train_llmsr_grpo(
    template_path: str,
    data_path: str, 
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./llmsr_grpo_outputs",
    **kwargs
):
    """
    主要入口点 - 默认使用直连模式进行权重微调
    """
    print("🔄 重定向到直连模式训练...")
    return train_llmsr_grpo_direct(
        template_path=template_path,
        data_path=data_path,
        model_path=model_path,
        output_dir=output_dir,
        **kwargs
    )


def train_llmsr_grpo_http(
    template_path: str,
    data_path: str, 
    http_url: str = "http://localhost:5000",
    tokenizer_path: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    output_dir: str = "./llmsr_grpo_outputs",
    **kwargs
):
    """
    HTTP 模式 LLM-SR GRPO 训练 - 不更新权重，仅用于推理
    
    ⚠️  重要：HTTP 模式下权重不会更新！如需微调权重，请使用 train_llmsr_grpo_direct()
    """
    
    print("🌐 启动 LLM-SR GRPO HTTP 模式训练 (不更新权重)")
    print(f"📋 模板: {template_path}")
    print(f"📊 数据: {data_path}")
    print(f"🌐 HTTP 服务: {http_url}")
    print(f"🔤 分词器: {tokenizer_path}")
    print(f"📁 输出: {output_dir}")
    print(f"⚠️  注意: HTTP 模式下权重不会更新！")
    
    # 测试 HTTP 服务连接
    print(f"\n🔍 测试 HTTP 服务连接...")
    try:
        import requests
        response = requests.get(http_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ HTTP 服务响应正常")
        else:
            print(f"⚠️ HTTP 服务返回状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到 HTTP 服务: {e}")
        print("🔧 请确保 LLM 引擎正在运行:")
        print("   ./run_llmsr_engine.sh")
        raise
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤 1: 创建 VERL 兼容数据集
    print("\n📝 步骤 1: 创建 VERL 数据集...")
    dataset_path = create_llmsr_dataset(template_path, data_path, output_dir)
    
    # 步骤 2: 创建自定义奖励函数
    print("\n🎯 步骤 2: 创建奖励函数...")
    reward_file_path = create_llmsr_reward_file(template_path, data_path, output_dir)
    
    # 步骤 3: 创建 HTTP 模式 GRPO 配置
    print("\n⚙️ 步骤 3: 创建 HTTP 模式 GRPO 配置...")
    config = create_grpo_config_http(
        http_url=http_url,
        tokenizer_path=tokenizer_path,
        dataset_path=dataset_path,
        reward_file_path=reward_file_path,
        output_dir=output_dir,
        **kwargs
    )
    
    # 保存配置
    config_path = os.path.join(output_dir, "grpo_config_http.yaml")
    OmegaConf.save(config, config_path)
    print(f"💾 保存 HTTP 模式配置: {config_path}")
    
    # 步骤 4: 启动 GRPO 训练
    print("\n🔥 步骤 4: 启动 GRPO HTTP 模式训练...")
    print("⚠️  注意: 权重不会更新，仅进行策略优化")
    try:
        run_ppo(config)
        print("✅ HTTP 模式训练完成 (权重未更新)")
    except Exception as e:
        print(f"❌ HTTP 模式训练失败: {e}")
        raise


if __name__ == "__main__":
    # 🔥 默认使用直连模式进行权重微调
    # train_llmsr_grpo_direct(
    #     template_path="./specs/specification_oscillator1_numpy.txt",
    #     data_path="./data/oscillator1/train.csv",
    #     model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
    #     epochs=5,
    #     batch_size=32,
    #     learning_rate=1e-6,
    #     gpus=6
    # )

    # Grid-based training
    train_llmsr_grpo_direct(
        template_path="./specs/specification_oscillator1_numpy.txt",
        data_path="./data/oscillator1/train.csv",
        model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        grid_train_data=False,   # Enable grid-based training
        num_grid_groups=10,     # Divide data into 10 groups
        epochs=5
    )