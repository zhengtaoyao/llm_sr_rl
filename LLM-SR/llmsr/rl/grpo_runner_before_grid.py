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



def create_llmsr_dataset(template_path: str, data_path: str, output_dir: str):
    """
    Create VERL-compatible dataset from LLM-SR specification and data.
    
    Args:
        template_path: Path to LLM-SR specification template
        data_path: Path to training data CSV
        output_dir: Directory to save converted dataset
        
    Returns:
        Path to the created dataset file
    """
    import pandas as pd
    import json
    
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
    # num_samples = min(1000, len(df))  # Limit for training efficiency
    # num_samples = min(5000, len(df))  # Limit for training efficiency
    num_samples = min(len(df), len(df))  # Limit for training efficiency

    # Create dataset entries in chat format
    dataset_entries = []
    for i in range(num_samples):
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


def create_llmsr_reward_file(template_path: str, data_path: str, output_dir: str):
    """
    Create a custom reward function file for VERL.
    
    Args:
        template_path: Path to LLM-SR specification template
        data_path: Path to training data CSV  
        output_dir: Directory to save reward function file
        
    Returns:
        Path to the created reward function file
    """
    
    reward_function_code = f'''
"""
🔥 简化的VERL奖励函数 - 兼容新数据格式
避免KeyError: 'ground_truth'问题
"""

import sys
import os
from pathlib import Path

# 导入简化的奖励函数
sys.path.append(str(Path(__file__).parent.parent.parent))
from simple_verl_reward import compute_score as simple_compute_score

def compute_score(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
    """
    🔥 使用简化的奖励计算，避免ground_truth KeyError - 支持默认参数
    
    Args:
        data_sources: 数据源列表 (默认: None)
        solution_strs: 模型生成的解决方案字符串列表 (默认: None)
        ground_truths: 参考答案列表 (默认: None)
        extra_infos: 额外信息列表 (默认: None)
        
    Returns:
        rewards: 奖励分数列表
    """
    
    print(f"🔧 包装器被调用，参数类型: data_sources={{type(data_sources)}}, solution_strs={{type(solution_strs)}}, ground_truths={{type(ground_truths)}}, extra_infos={{type(extra_infos)}}")
    
    # 🔧 处理None参数
    if data_sources is None:
        data_sources = []
    if solution_strs is None:
        solution_strs = []
    if ground_truths is None:
        ground_truths = []
    if extra_infos is None:
        extra_infos = []
    
    # 确定问题类型
    problem_type = "oscillator1"  # 默认值
    data_path = "{data_path}"
    
    if "oscillator1" in data_path:
        problem_type = "oscillator1"
    elif "oscillator2" in data_path:
        problem_type = "oscillator2"
    elif "bactgrow" in data_path:
        problem_type = "bactgrow"
    elif "stressstrain" in data_path:
        problem_type = "stressstrain"
    
    # 构建extra_infos以传递问题类型
    if not extra_infos:
        extra_infos = [{{'problem_type': problem_type}}] * len(solution_strs)
    else:
        # 确保每个extra_info都有problem_type
        for i, extra_info in enumerate(extra_infos):
            if not extra_info:
                extra_infos[i] = {{'problem_type': problem_type}}
            elif 'problem_type' not in extra_info:
                extra_infos[i]['problem_type'] = problem_type
    
    # 调用简化的奖励函数
    return simple_compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
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
    
    # 直连模式 GRPO 配置
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
            "max_response_length": kwargs.get('max_response_length', 512),
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
                "ppo_max_token_len_per_gpu": kwargs.get('max_model_len', 4096),

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
                "max_new_tokens": kwargs.get('max_new_tokens', 512),
                "load_format": "auto",
                "dtype": "bfloat16",
                "prompt_length": kwargs.get('max_prompt_length', 1024),
                "response_length": kwargs.get('max_new_tokens', 512),
                "max_model_len": kwargs.get('max_model_len', 4096),
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "disable_log_stats": False,
                "enable_chunked_prefill": False,
                "disable_custom_all_reduce": True,
                "gpu_memory_utilization": 0.4,  # 🔥 降低到0.4以避免OOM
                "max_num_batched_tokens": 4096,  # 🔥 减少批量token数量
                "seed": 0,
                "log_prob_use_dynamic_bsz": True,
                "ulysses_sequence_parallel_size": 1,  # 与 actor 保持一致

                "tensor_model_parallel_size": 1,
                "temperature": kwargs.get('temperature', 0.8),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 30),
                "log_prob_micro_batch_size": None,
                # 🔥 CRITICAL: ref模型也需要正确的微批量大小

                "log_prob_max_token_len_per_gpu": kwargs.get('max_model_len', 24000),
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,  # 1
                "val_kwargs": {
                    "do_sample": True,
                    "temperature": kwargs.get('temperature', 0.8),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 30),
                    "max_new_tokens": kwargs.get('max_new_tokens', 512),
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
                "log_prob_max_token_len_per_gpu": kwargs.get('max_model_len', 4096),
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
            "name": "compute_score"
        },
        
        # 训练器配置
        "trainer": {
            "total_epochs": kwargs.get('epochs', 10),
            "total_training_steps": None,
            "project_name": "llm_sr_grpo_direct",
            "experiment_name": kwargs.get('experiment_name', 'direct_weight_tuning_fixed'),
            "logger": ["console"],
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

            "log_prob_max_token_len_per_gpu": kwargs.get('max_model_len', 4096)
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
                "max_new_tokens": kwargs.get('max_new_tokens', 512),
                "load_format": "auto",
                "dtype": "bfloat16",
                "log_prob_use_dynamic_bsz": True,

                "prompt_length": kwargs.get('max_prompt_length', 1024),
                "response_length": kwargs.get('max_new_tokens', 512),
                "max_model_len": kwargs.get('max_model_len', 4096),
                "enforce_eager": True,
                "free_cache_engine": False,
                "enable_prefix_caching": False,
                "disable_log_stats": False,
                "enable_chunked_prefill": False,
                "disable_custom_all_reduce": True,
                "gpu_memory_utilization": 0.85,
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
                    "max_new_tokens": kwargs.get('max_new_tokens', 512),
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
            "logger": ["console"],
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
    **kwargs
):
    """
    🔥 直连模式 LLM-SR GRPO 训练 - 真正微调 LLM 权重
    
    Args:
        template_path: LLM-SR 规范模板路径
        data_path: 训练数据 CSV 路径
        model_path: 模型路径或名称
        output_dir: 输出目录
        **kwargs: 额外训练配置
    """
    
    print("🔥 启动 LLM-SR GRPO 直连模式训练 (真正微调权重)")
    print(f"📋 模板: {template_path}")
    print(f"📊 数据: {data_path}")
    print(f"🤖 模型: {model_path}")
    print(f"📁 输出: {output_dir}")
    print(f"🔧 直连模式: Actor 进程直接加载模型，通过 FSDP 更新权重")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 🔥 步骤 1: 检查预生成的VERL数据集
    print("\n📝 步骤 1: 检查 VERL 数据集...")
    
    # 从data_path提取problem_name
    problem_name = None
    if 'oscillator1' in data_path:
        problem_name = 'oscillator1'
    elif 'oscillator2' in data_path:
        problem_name = 'oscillator2'
    elif 'bactgrow' in data_path:
        problem_name = 'bactgrow'
    elif 'stressstrain' in data_path:
        problem_name = 'stressstrain'
    
    # 检查预生成的数据集
    verl_dataset_path = None
    if problem_name:
        potential_path = f"./verl_datasets/{problem_name}_train_verl.parquet"
        if os.path.exists(potential_path):
            verl_dataset_path = potential_path
            print(f"✅ 找到预生成的VERL数据集: {verl_dataset_path}")
        else:
            print(f"❌ 未找到预生成的VERL数据集: {potential_path}")
    
    # 如果没有预生成的数据集，则创建新的
    if not verl_dataset_path:
        print("📝 创建新的 VERL 数据集...")
        dataset_path = create_llmsr_dataset(template_path, data_path, output_dir)
    else:
        dataset_path = verl_dataset_path
        print(f"📊 使用预生成的数据集: {dataset_path}")
    
    # 步骤 2: 创建自定义奖励函数
    print("\n🎯 步骤 2: 创建奖励函数...")
    reward_file_path = create_llmsr_reward_file(template_path, data_path, output_dir)
    
    # 步骤 3: 创建直连模式 GRPO 配置
    print("\n⚙️ 步骤 3: 创建直连模式 GRPO 配置...")
    config = create_grpo_config_direct(
        model_path=model_path,
        dataset_path=dataset_path,
        reward_file_path=reward_file_path,
        output_dir=output_dir,
        **kwargs
    )
    
    # 保存配置
    config_path = os.path.join(output_dir, "grpo_config_direct.yaml")
    OmegaConf.save(config, config_path)
    print(f"💾 保存直连模式配置: {config_path}")
    
    # 步骤 4: 启动 GRPO 训练
    print("\n🔥 步骤 4: 启动 GRPO 直连模式训练...")
    print("⚡ 模型权重将通过 FSDP 进行真正的微调更新")
    try:
        run_ppo(config)
        print("✅ 直连模式训练完成，权重已更新！")
    except Exception as e:
        print(f"❌ 直连模式训练失败: {e}")
        raise


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
    train_llmsr_grpo_direct(
        template_path="./specs/specification_oscillator1_numpy.txt",
        data_path="./data/oscillator1/train.csv",
        model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        epochs=5,
        batch_size=32,
        learning_rate=1e-6,
        gpus=6
    ) 