"""
GRPO Training Runner for LLM-SR Integration (CSV-Based Reward)

This module provides the main training loop that integrates VERL's GRPO
with a CSV-based reward system for symbolic regression tasks.

🔥 Key Change:
- This version no longer uses hardcoded ground-truth equations.
- It relies on `csv_based_reward.py` which calculates MSE directly from data files.
- The `process_data_for_verl_no_eqtns.py` script must be run first to create compatible datasets.
"""

# 🔥 Set CUDA memory allocator config before importing torch/vllm
import os
import shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.6"
import sys
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add VERL to path if needed
verl_path = str(Path(__file__).parent.parent.parent.parent / "verl")
if verl_path not in sys.path:
    sys.path.append(verl_path)

from verl.trainer.main_ppo import run_ppo



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
    num_samples = min(100, len(df))  # Limit for training efficiency

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
        
        # 🔥 FIX: Add the required ground_truth information for the reward function
        # Determine target variable based on the problem
        target_variable = "y"  # Default for most problems
        if "oscillator1" in data_path:
            target_variable = "y"
        elif "oscillator2" in data_path:
            target_variable = "y"
        elif "bactgrow" in data_path:
            target_variable = "B"
        elif "stressstrain" in data_path:
            target_variable = "stress"
        
        entry = {
            "prompt": chat_prompt,
            "data_source": "llm_sr_train",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "data_path": data_path,
                    "target_variable": target_variable
                }
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


def create_llmsr_reward_file(template_path: str, data_path: str, output_dir: str) -> str:
    """Create a reward‐function adapter for CSV‐only VERL runs."""
    reward_file = os.path.join(output_dir, "llmsr_reward.py")
    with open(reward_file, "w") as f:
        f.write(f"""
import sys, os
from typing import List, Dict, Any, Union
sys.path.append(os.path.dirname(__file__) + "/../..")  # allow import of csv_based_reward
from csv_based_reward import get_reward as original_reward

def get_reward(
    batch: Dict[str, Any] = None,
    return_dict: bool = False,
    **kwargs
) -> Union[List[float], Dict[str, List[float]]]:
    \"\"\"
    Adapter that handles inconsistent VERL calling conventions.
    It can accept either a single 'batch' dictionary or unpacked keyword arguments.
    \"\"\"
    # If 'batch' is not provided, VERL likely passed arguments as kwargs.
    # We reconstruct the batch dictionary from kwargs.
    if batch is None:
        batch = {{
            "data_source": kwargs.get("data_source"),
            "response_str": kwargs.get("response_str"),
            "ground_truth": kwargs.get("ground_truth"),
            "extra_info": kwargs.get("extra_info"),
        }}

    # Extract the required lists from the batch dictionary
    data_sources = batch.get("data_source")
    solution_strs = batch.get("response_str")
    ground_truths = batch.get("ground_truth")
    extra_infos = batch.get("extra_info")

    # Call the original function with the unpacked arguments
    rewards = original_reward(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
    
    if return_dict:
        return {{ "rewards": rewards }}
    return rewards
""")
    print(f"✅ Created reward adapter at {reward_file}")
    return reward_file


def create_grpo_config_direct(
    problem_name: str,
    model_path: str,
    template_path: str,
    data_path: str,
    output_dir: str,
    reward_file_path: str,
    **kwargs
):
    """Builds the dict that VERL will consume for direct GRPO runs."""
    
    gpus = 6  # Fixed for direct mode
    micro_batch_size_per_gpu = 1
    rollout_n = 4  # Fixed rollout steps for direct mode
    
    traj_micro_bsz = micro_batch_size_per_gpu * gpus
    traj_mini_bsz = traj_micro_bsz * 2
    prompt_mini_bsz = traj_mini_bsz * rollout_n
    prompt_bsz = prompt_mini_bsz * 1
    
    logging.info(f"Batch size config: prompt_bsz={prompt_bsz}, prompt_mini_bsz={prompt_mini_bsz}, micro_bsz_per_gpu={micro_batch_size_per_gpu}")
    
    config = {
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
        "data": {
            "train_files": [str(Path(output_dir) / "llmsr_train.parquet")],
            "val_files": [str(Path(output_dir) / "llmsr_train.parquet")],
            "train_batch_size": prompt_bsz,
            "val_batch_size": prompt_mini_bsz,
            "max_prompt_length": kwargs.get('max_prompt_length', 1024),
            "max_response_length": kwargs.get('max_new_tokens', 512),
            "filter_overlong_prompts": True,
            "truncation": "error",
            "reward_fn_key": "reward_model",
            "shuffle": True
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "tokenizer_only": False,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "model_dtype": "bf16"
            },
            "actor": {
                "ulysses_sequence_parallel_size": 1,
                "entropy_from_logits_with_chunking": False,
                "strategy": "fsdp",
                "optim": {"lr": kwargs.get('learning_rate', 1e-6), "eps": 1e-8, "weight_decay": 0.01},
                "ppo_mini_batch_size": prompt_mini_bsz,
                "ppo_micro_batch_size": None,  # 废弃字段
                "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "use_kl_loss": True,
                "entropy_coeff": 0,

                "kl_loss_coef": 0.001,
                "clip_ratio": 0.2,
                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean", # 🔥 FIX: Add missing loss aggregation mode
                "use_dynamic_bsz": True,
                "use_torch_compile": True,
                "grad_clip": 1.0,
                "policy_loss": {"loss_mode": "vanilla"},
                "fsdp_config": {
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "forward_prefetch": False,
                    "wrap_policy": {"min_num_params": 0}
                },
                "checkpoint": {"save_contents": ["model", "optimizer", "extra"], "load_contents": ["model", "optimizer", "extra"]}
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",
                "n": rollout_n,
                "load_format": "auto",
                "response_length": kwargs.get('max_new_tokens', 512),
                "prompt_length": kwargs.get('max_prompt_length', 1024),
                "max_new_tokens": kwargs.get('max_new_tokens', 512),
                "dtype": "bfloat16",
                "max_model_len": kwargs.get('max_model_len', 4096),
                "gpu_memory_utilization": 0.6,
                "tensor_model_parallel_size": 1,
                "temperature": kwargs.get('temperature', 0.8),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 30),
                "timeout": kwargs.get('timeout', 60),
                "repeat_prompt": 1,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "log_prob_use_dynamic_bsz": kwargs.get('log_prob_use_dynamic_bsz', True),
                "log_prob_max_token_len_per_gpu": kwargs.get('log_prob_max_token_len_per_gpu', 24000),
                "val_kwargs": {"do_sample": True, "temperature": 0.8, "top_p": 0.9, "max_new_tokens": 512, "n": 1,"top_k": kwargs.get('top_k', 30)},
                "multi_turn": {"enable": False},
                "free_cache_engine": False,
                "enable_prefix_caching": False,
                "disable_log_stats": False,
                "enable_chunked_prefill": False,
                "disable_custom_all_reduce": True,
                "max_num_batched_tokens": 8192,
                "seed": 0,
                "calculate_log_probs": False,  # 用于调试的rollout概率记录
                "enforce_eager": True

            },
            "ref": {
                "enforce_eager": True,
                "entropy_from_logits_with_chunking": False,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "ulysses_sequence_parallel_size": 1,
                "log_prob_use_dynamic_bsz": True,
                "log_prob_max_token_len_per_gpu": kwargs.get('max_model_len', 4096),
                "use_remove_padding": True,
                "use_torch_compile": True,
                "fsdp_config": {"param_offload": True, "forward_prefetch": False, "wrap_policy": {"min_num_params": 0}}
            },
            "hybrid_engine": True
        },
        "critic": {
            "strategy": "fsdp",
            "ppo_mini_batch_size": prompt_mini_bsz,
            "ppo_micro_batch_size": None,  # 废弃字段
            "ppo_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "use_dynamic_bsz": True
        },
        "reward_model": {"enable": False},
        "custom_reward_function": {
            "name": "get_reward",
            "path": reward_file_path,
        },
        # Trainer config
        "trainer": {
            "total_epochs": kwargs.get('epochs', 10),
            "total_training_steps": None, # 🔥 FIX: Remove this line to use total_epochs
            "project_name": "llm_sr_grpo_csv_reward",
            "experiment_name": kwargs.get('experiment_name', 'llmsr_grpo_oscillator1'),
            "logger": ["console"],
            "n_gpus_per_node": gpus,
            "nnodes": 1,
            "save_freq": kwargs.get('save_freq', 2),
            "test_freq": kwargs.get('test_freq', 5),
            "val_before_train": True,
            "default_local_dir": output_dir,
            "device": "cuda",
            "resume_mode": "disable",
            "log_val_generations": kwargs.get('log_val_generations', 10),
            "log_train_generations": kwargs.get('log_train_generations', 5),
            "log_prob_max_token_len_per_gpu": kwargs.get('max_model_len', 4096)
        },
        "ray_init": {"num_cpus": None}
    }
    
    return OmegaConf.create(config)


def train_llmsr_grpo_direct(
    problem_name: str,
    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    output_dir: str = "./llmsr_grpo_outputs",
    **kwargs
):
    """
    🔥 Main training function for fine-tuning LLM weights using CSV-based rewards.
    
    Args:
        problem_name: The name of the problem (e.g., 'oscillator1').
        model_path: Path or name of the model to fine-tune.
        output_dir: Directory for all outputs.
        **kwargs: Additional training configurations.
    """
    
    logging.info(f"Starting LLM-SR GRPO Training for '{problem_name}' (CSV-based reward)")
    logging.info(f"Model: {model_path}")
    logging.info(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Step 1: Locate the pre-generated VERL dataset ---
    logging.info("Step 1: Locating VERL dataset...")
    base_dir = Path(__file__).parent.parent.parent
    dataset_path = base_dir / f"verl_datasets/{problem_name}_train_verl.parquet"
    
    if not dataset_path.exists():
        error_msg = f"Dataset not found at '{dataset_path}'. Please run 'python {base_dir / 'process_data_for_verl_no_eqtns.py'}' first."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    logging.info(f"Found dataset: {dataset_path}")

    # --- Step 2: Copy the CSV-based reward function ---
    logging.info("Step 2: Preparing reward function...")
    # 🔥 FIX: Don't copy the file, create a proper wrapper
    data_path = base_dir / f"data/{problem_name}/train.csv"
    template_path = kwargs.get('template_path', base_dir / f"specs/specification_{problem_name}_numpy.txt")
    
    # Create the dataset before the reward file, as the config needs the dataset path
    create_llmsr_dataset(str(template_path), str(data_path), output_dir)

    reward_file_path = create_llmsr_reward_file(str(template_path), str(data_path), output_dir)
    logging.info(f"Created CSV-based reward function wrapper at: {reward_file_path}")

    # --- Step 3: Create the GRPO configuration ---
    logging.info("Step 3: Creating GRPO configuration...")
    config = create_grpo_config_direct(
        problem_name,
        model_path,
        template_path,
        str(data_path),
        output_dir,
        reward_file_path,
        **kwargs,
    )
    
    config_path = os.path.join(output_dir, f"grpo_config_{problem_name}.yaml")
    OmegaConf.save(config, config_path)
    logging.info(f"Saved configuration to: {config_path}")
    
    # --- Step 4: Launch GRPO training ---
    logging.info("Step 4: Launching GRPO training...")
    try:
        run_ppo(config)
        logging.info(f"Training for '{problem_name}' completed successfully!")
    except Exception as e:
        logging.error(f"Training for '{problem_name}' failed: {e}", exc_info=True)
        raise

def train_llmsr_grpo(mode: str = "direct", **kwargs):
    """
    Main entry point for GRPO training, called by the pipeline.
    Dispatches to the correct training function based on the mode.
    """
    if mode == "direct":
        logging.info("Switching to GRPO direct training mode...")
        # The pipeline passes 'template_path' and 'data_path', but we only need 'problem_name'.
        # We extract it from the data_path.
        data_path = kwargs.get("data_path")
        if not data_path:
            raise ValueError("Missing 'data_path' in arguments for GRPO training.")
        
        # Extract problem_name from a path like './data/oscillator1/train.csv'
        problem_name = Path(data_path).parent.name
        kwargs['problem_name'] = problem_name
        
        # Remove old arguments that are no longer needed to avoid unexpected keyword argument errors
        kwargs.pop('template_path', None)
        kwargs.pop('data_path', None)
        
        return train_llmsr_grpo_direct(**kwargs)
    else:
        raise ValueError(f"Unsupported GRPO training mode: '{mode}'. This version only supports 'direct' mode.")

if __name__ == "__main__":
    # Example of how to run the training directly from this script
    train_llmsr_grpo(
        mode="direct",
        # Provide the arguments as the pipeline would
        data_path="./data/oscillator1/train.csv",
        template_path="./specs/specification_oscillator1_numpy.txt", # Kept for API compatibility
        model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        output_dir="./llmsr_grpo_outputs/test_run",
        epochs=1,
        gpus=1,
        experiment_name="oscillator1_test_run"
    )