"""
GRPO v2 Training Runner for LLM-SR

本文件实现一版更稳健的 LLM+GRPO→SR 流程（v2）：
- 奖励密化与成分化：拟合/复杂度/物理一致性/过程奖励 + 组内排名归一
- 解码与动作约束：沿用原有骨架模板；可扩展常数占位符<C*>与后验拟合
- 记忆与多样性：轻量“岛屿/记忆库”检索 few-shot（文件级共享，跨进程安全）
- 稳健 GRPO 配置：同一提示组内采样、显式 KL 损失、token-mean 聚合

说明：为避免干扰旧实现，所有逻辑独立于 v1 文件，main.py 分支进入本文件。
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from omegaconf import OmegaConf, DictConfig

# 将 VERL 加入路径
verl_path = str(Path(__file__).parent.parent.parent.parent / "verl")
if verl_path not in sys.path:
    sys.path.append(verl_path)

from verl.trainer.main_ppo import run_ppo


# 轻量记忆管理（文件共享，跨进程安全）
class MemoryManagerV2:
    def __init__(self, memory_dir: str, top_k_per_island: int = 8, num_islands: int = 4) -> None:
        os.makedirs(memory_dir, exist_ok=True)
        self._path = os.path.join(memory_dir, "memory_v2.json")
        self._top_k = top_k_per_island
        self._num_islands = num_islands

        if not os.path.exists(self._path):
            # 初始化空结构
            import json
            init = {str(i): [] for i in range(self._num_islands)}
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(init, f)

    def load(self) -> Dict[str, List[Dict[str, Any]]]:
        import json
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {str(i): [] for i in range(self._num_islands)}

    def save(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        import json
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def sample_few_shot(self, k: int = 3) -> List[str]:
        """跨岛屿采样多样 few-shot（仅返回函数体片段/实现文本）。"""
        data = self.load()
        examples: List[str] = []
        if not data:
            return examples
        # 轮询各岛屿，尽量多样
        for island_id, items in data.items():
            if not items:
                continue
            # 优先 score 高的前几项
            items_sorted = sorted(items, key=lambda x: x.get("score", -1.0), reverse=True)
            for it in items_sorted[:2]:
                impl = it.get("implementation", "")
                if impl:
                    examples.append(impl)
                if len(examples) >= k:
                    return examples
        return examples[:k]


def _extract_prompt_header(spec_text: str) -> str:
    lines = spec_text.split("\n")
    prompt_lines: List[str] = []
    in_evolve = False
    for line in lines:
        if "@equation.evolve" in line:
            in_evolve = True
            continue
        if in_evolve and line.strip().startswith("def "):
            prompt_lines.append(line.rstrip())
            break
        if not in_evolve:
            prompt_lines.append(line.rstrip())
    return "\n".join(prompt_lines).strip()


def create_llmsr_dataset_v2(
    template_path: str,
    data_path: str,
    output_dir: str,
    memory_dir: str,
    grid_train_data: bool = False,
    num_grid_groups: int = 10,
    few_shot_k: int = 3,
) -> str:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import KBinsDiscretizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    with open(template_path, "r", encoding="utf-8") as f:
        spec_text = f.read()
    base_prompt = _extract_prompt_header(spec_text)

    # few-shot 拼接
    memory = MemoryManagerV2(memory_dir)
    examples = memory.sample_few_shot(k=few_shot_k)
    if examples:
        few_shot_block = "\n\n# === Few-shot program skeletons (from memory) ===\n" + "\n\n".join(examples)
    else:
        few_shot_block = ""
    composed_prompt = (base_prompt + few_shot_block).strip()

    df = pd.read_csv(data_path)
    max_samples = len(df)

    # Grid-based sampling（按问题自动检测输入/输出列）
    if grid_train_data:
        input_cols: List[str] = []
        output_col: str | None = None
        if "oscillator" in data_path:
            input_cols = [c for c in ["x", "v"] if c in df.columns] or df.columns[:-1].tolist()
            output_col = "a" if "a" in df.columns else df.columns[-1]
        elif "bactgrow" in data_path:
            input_cols = [c for c in ["b", "s", "temp", "pH"] if c in df.columns] or df.columns[:-1].tolist()
            output_col = "db" if "db" in df.columns else df.columns[-1]
        elif "stressstrain" in data_path:
            input_cols = [c for c in ["strain", "temp"] if c in df.columns] or df.columns[:-1].tolist()
            output_col = "stress" if "stress" in df.columns else df.columns[-1]
        else:
            input_cols = df.columns[:-1].tolist()
            output_col = df.columns[-1]

        # 1D/2D/ND 分桶
        if len(input_cols) == 1:
            discretizer = KBinsDiscretizer(n_bins=num_grid_groups, encode="ordinal", strategy="uniform")
            df["grid_group"] = discretizer.fit_transform(df[input_cols].values)
        elif len(input_cols) == 2:
            import math
            nbin = int(max(1, round(math.sqrt(num_grid_groups))))
            dx = KBinsDiscretizer(n_bins=nbin, encode="ordinal", strategy="uniform")
            dy = KBinsDiscretizer(n_bins=nbin, encode="ordinal", strategy="uniform")
            gx = dx.fit_transform(df[[input_cols[0]]].values)
            gy = dy.fit_transform(df[[input_cols[1]]].values)
            df["grid_group"] = gx * nbin + gy
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_grid_groups, random_state=42)
            df["grid_group"] = kmeans.fit_predict(df[input_cols].values)

        df["grid_group"] = df["grid_group"].astype(int)
        samples_per_group = max(1, max_samples // num_grid_groups)
        parts = []
        for gid in range(num_grid_groups):
            sub = df[df["grid_group"] == gid]
            if len(sub) == 0:
                continue
            parts.append(sub.sample(n=min(samples_per_group, len(sub)), random_state=42))
        df_s = pd.concat(parts).reset_index(drop=True)
    else:
        df_s = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
        df_s["grid_group"] = 0

    # 组装 VERL 数据
    dataset_entries: List[Dict[str, Any]] = []
    for i in range(len(df_s)):
        row = df_s.iloc[i]
        # 增加 system 约束说明；user 中放入规范 + few-shot
        system_prefix = (
            "You generate equation skeletons under grammar/AST constraints.\n"
            "- Prefer valid mathematical expressions over prose.\n"
            "- Optionally use EDIT DSL to modify a provided base expression.\n"
            "EDIT Usage: 'EDIT ADD <expr>' | 'EDIT MUL <expr>' | 'EDIT REPLACE <old> => <new>'\n"
            "Return final Pythonic expression or a minimal function body."
        )
        chat_prompt = [
            {"role": "system", "content": system_prefix},
            {"role": "user", "content": composed_prompt},
        ]

        entry = {
            "prompt": chat_prompt,
            "data_source": "llm_sr_train_v2",
            "reward_model": {"style": "rule"},
            "extra_info": {
                "grid_group": int(row["grid_group"]),
                # 记录原始数据点用于潜在物理一致性与过程奖励
                "data_point": row.drop("grid_group").to_dict(),
                # 可选传递基底表达式（由后续管线填充），用于 EDIT 模式
                "base_impl": None,
            },
        }
        dataset_entries.append(entry)

    table = pa.Table.from_pylist(dataset_entries)
    out_path = os.path.join(output_dir, "llmsr_train_v2.parquet")
    pq.write_table(table, out_path)
    return out_path


def create_llmsr_reward_file_v2(template_path: str, data_path: str, output_dir: str, memory_dir: str, grid_train_data: bool = False) -> str:
    code = f'''"""
Wrapper for v2 reward to plug into VERL custom_reward_function.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from llmsr.rl.simple_verl_reward_v2 import compute_score as compute_score_v2


def compute_score(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
    return compute_score_v2(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        grid_train_data={grid_train_data},
        template_path="{template_path}",
        data_path="{data_path}",
        memory_dir="{memory_dir}",
        **kwargs
    )
'''
    out = os.path.join(output_dir, "llmsr_reward_v2.py")
    with open(out, "w", encoding="utf-8") as f:
        f.write(code)
    return out


def create_grpo_config_v2(
    model_path: str,
    dataset_path: str,
    reward_file_path: str,
    output_dir: str,
    *,
    gpus: int = 8,
    rollout_n: int = 4,
    kl_coef: float = 1e-3,
    max_prompt_length: int = 2048,
    max_new_tokens: int = 1024,
    max_model_len: int = 8192,
    learning_rate: float = 1e-6,
    epochs: int = 5,
) -> DictConfig:
    # token 长度安全阈值
    safe_max_token_len = max(12288, int(max_prompt_length + max_new_tokens + 512))
    micro_bsz_per_gpu = 1
    traj_micro_bsz = micro_bsz_per_gpu * gpus
    traj_mini_bsz = max(2, traj_micro_bsz * 2)
    prompt_mini_bsz = traj_mini_bsz * rollout_n
    prompt_bsz = prompt_mini_bsz

    cfg = {
        "algorithm": {
            "_target_": "verl.trainer.config.AlgoConfig",
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "kl_penalty": "kl",
            "use_pf_ppo": False,
            "pf_ppo": {
                "reweight_method": "pow",
                "weight_pow": 2.0
            },
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "fixed", 
                "kl_coef": float(kl_coef), 
                "horizon": 10000, 
                "target_kl": 0.1
            },
        },
        "data": {
            "tokenizer": None,
            "use_shm": False,
            "train_files": [dataset_path],
            "val_files": [dataset_path],
            "prompt_key": "prompt",
            "reward_fn_key": "data_source",
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_new_tokens,
            "train_batch_size": prompt_bsz,
            "val_batch_size": prompt_mini_bsz,
            "return_raw_input_ids": False,
            "return_raw_chat": False,
            "return_full_prompt": False,
            "shuffle": True,
            "dataloader_num_workers": 8,
            "validation_shuffle": False,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "truncation": "error",
            "image_key": "images",
            "video_key": "videos",
            "trust_remote_code": False,
            "custom_cls": {
                "path": None,
                "name": None
            },
            "return_multi_modal_inputs": True,
            # 🔧 避免 Missing key data.sampler
            "sampler": {
                "class_path": None,
                "class_name": None
            },
            "datagen": {
                "path": None,
                "name": None
            },
            "apply_chat_template_kwargs": {}
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "use_fused_kernels": False,
                "trust_remote_code": False,
                "custom_chat_template": None,
                "external_lib": None,
                "override_config": {},
                "enable_activation_offload": False,
                "lora_rank": 0,
                "lora_alpha": 16,
                "target_modules": "all-linear",
                "exclude_modules": None,
                "use_liger": False,
                "fused_kernel_options": {}
            },
            "actor": {
                "_target_": "verl.workers.config.FSDPActorConfig",
                "strategy": "fsdp",
                "optim": {
                    "_target_": "verl.workers.config.FSDPOptimizerConfig",
                    "lr": learning_rate, 
                    "weight_decay": 0.01,
                    "lr_warmup_steps_ratio": 0.0,
                    "total_training_steps": -1,
                    "lr_warmup_steps": -1,
                    "min_lr_ratio": 0.0,
                    "num_cycles": 0.5,
                    "warmup_style": "constant"
                },
                "ppo_mini_batch_size": prompt_mini_bsz,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": micro_bsz_per_gpu,
                "use_kl_loss": True,
                "kl_loss_coef": float(kl_coef),
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": 0.0,
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",  # 处理长度偏置
                "use_dynamic_bsz": True,
                "ulysses_sequence_parallel_size": 1,
                "ppo_max_token_len_per_gpu": safe_max_token_len,
                "use_remove_padding": True,
                "use_fused_kernels": False,
                "use_torch_compile": False,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,
                "shuffle": False,
                "policy_loss": {
                    "_target_": "verl.workers.config.PolicyLossConfig",
                    "loss_mode": "vanilla",
                    "clip_cov_ratio": 0.0002,
                    "clip_cov_lb": 1.0,
                    "clip_cov_ub": 5.0,
                    "kl_cov_ratio": 0.0002,
                    "ppo_kl_coef": 0.1
                },
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "fsdp_size": gpus,
                    "param_offload": True,
                    "optimizer_offload": True,
                    "forward_prefetch": False,
                    "offload_policy": False,
                    "reshard_after_forward": True,
                    "wrap_policy": {"min_num_params": 0},
                },
                "checkpoint": {
                    "_target_": "verl.trainer.config.CheckpointConfig",
                    "save_contents": ["model", "optimizer", "extra"], 
                    "load_contents": ["model", "optimizer", "extra"],
                    "async_save": False
                },
                "profiler": {
                    "_target_": "verl.utils.profiler.ProfilerConfig",
                    "tool": None,
                    "enable": False,
                    "all_ranks": False,
                    "ranks": [],
                    "save_path": None,
                    "tool_config": {
                        "nsys": {
                            "_target_": "verl.utils.profiler.config.NsightToolConfig",
                            "discrete": False
                        },
                        "torch": {
                            "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                            "step_start": -1,
                            "step_end": -1
                        },
                        "torch_memory": {
                            "_target_": "verl.utils.profiler.config.TorchMemoryToolConfig",
                            "trace_alloc_max_entries": 100000,
                            "stack_depth": 32
                        },
                        "npu": {
                            "_target_": "verl.utils.profiler.config.NPUToolConfig",
                            "contents": [],
                            "level": "level1",
                            "analysis": False,
                            "discrete": False
                        }
                    },
                    "global_tool_config": None
                },
            },
            "rollout": {
                "_target_": "verl.workers.config.RolloutConfig",
                "name": "vllm",
                "mode": "sync",
                "n": rollout_n,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30,
                "do_sample": True,
                "over_sample_rate": 0,
                "prompt_length": max_prompt_length,
                "response_length": max_new_tokens,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.6,
                "ignore_eos": False,
                "enforce_eager": True,
                "cudagraph_capture_sizes": None,
                "free_cache_engine": True,
                "tensor_model_parallel_size": 1,
                "max_num_batched_tokens": 4096,
                "max_model_len": max_model_len,
                "max_num_seqs": 1024,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_bsz_per_gpu,
                "log_prob_use_dynamic_bsz": True,
                "log_prob_max_token_len_per_gpu": safe_max_token_len,
                "disable_log_stats": False,
                "multi_stage_wake_up": False,
                "engine_kwargs": {
                    "vllm": {},
                    "sglang": {}
                },
                "calculate_log_probs": False,
                "agent": {
                    "_target_": "verl.workers.config.AgentLoopConfig",
                    "num_workers": 8,
                    "agent_loop_config_path": None,
                    "custom_async_server": {
                        "_target_": "verl.workers.config.CustomAsyncServerConfig",
                        "path": None,
                        "name": None
                    }
                },
                "trace": {
                    "_target_": "verl.workers.config.TraceConfig",
                    "backend": None,
                    "token2text": False
                },
                "update_weights_bucket_megabytes": 512,
                "skip_rollout": False,
                "skip_dump_dir": "/tmp/rollout_dump",
                "profiler": {
                    "_target_": "verl.utils.profiler.ProfilerConfig",
                    "tool": None,
                    "enable": False,
                    "all_ranks": False,
                    "ranks": [],
                    "save_path": None,
                    "tool_config": {
                        "nsys": {
                            "_target_": "verl.utils.profiler.config.NsightToolConfig",
                            "discrete": False
                        }
                    },
                    "global_tool_config": None
                },
                "enable_chunked_prefill": False,
                "load_format": "auto",
                "layered_summon": False,
                "layer_name_map": {},
                "val_kwargs": {
                    "_target_": "verl.workers.config.SamplingConfig",
                    "do_sample": True, 
                    "temperature": 0.8, 
                    "top_p": 0.9, 
                    "top_k": 30, 
                    "n": 1
                },
                "calculate_log_probs": False,
                "free_cache_engine": True,
                "ignore_eos": False,
                "over_sample_rate": 0,
                "multi_stage_wake_up": False,
                "engine_kwargs": {
                    "vllm": {},
                    "sglang": {}
                },
                "update_weights_bucket_megabytes": 512,
                "trace": {
                    "_target_": "verl.workers.config.TraceConfig",
                    "backend": None,
                    "token2text": False
                },
                "skip_rollout": False,
                "skip_dump_dir": "/tmp/rollout_dump",
                "profiler": {
                    "_target_": "verl.utils.profiler.ProfilerConfig",
                    "tool": None,
                    "enable": False,
                    "all_ranks": False,
                    "ranks": [],
                    "save_path": None,
                    "tool_config": {
                        "nsys": {
                            "_target_": "verl.utils.profiler.config.NsightToolConfig",
                            "discrete": False
                        }
                    },
                    "global_tool_config": None
                },
                "multi_turn": {
                    "_target_": "verl.workers.config.MultiTurnConfig",
                    "enable": False, 
                    "max_assistant_turns": None, 
                    "tool_config_path": None, 
                    "max_user_turns": None,
                    "max_parallel_calls": 1,
                    "max_tool_response_length": 256,
                    "tool_response_truncate_side": "middle",
                    "interaction_config_path": None,
                    "use_inference_chat_template": False, 
                    "tokenization_sanity_check_mode": "strict", 
                    "format": "hermes"
                },
            },
            "ref": {
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": micro_bsz_per_gpu,
                "ulysses_sequence_parallel_size": 1,
                "log_prob_use_dynamic_bsz": True,
                "log_prob_max_token_len_per_gpu": safe_max_token_len,
                "use_remove_padding": True,
                "use_fused_kernels": False,
                "entropy_from_logits_with_chunking": False,
                "use_torch_compile": False,
                "entropy_checkpointing": False,
                "grad_clip": 1.0,
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "fsdp_size": gpus, 
                    "param_offload": True, 
                    "optimizer_offload": True,
                    "forward_prefetch": False, 
                    "offload_policy": False,
                    "reshard_after_forward": True,
                    "wrap_policy": {"min_num_params": 0}
                },
            },
            "hybrid_engine": True,
        },
        "critic": {
            "_target_": "verl.workers.config.FSDPCriticConfig",
            "enable": True,  # 🔧 修复 Missing key critic.enable
            "strategy": "fsdp",
            "rollout_n": rollout_n,
            "ppo_mini_batch_size": prompt_mini_bsz,
            "ppo_micro_batch_size": None,
            "ppo_micro_batch_size_per_gpu": micro_bsz_per_gpu,
            "use_dynamic_bsz": True,
            "forward_micro_batch_size": None,
            "forward_micro_batch_size_per_gpu": micro_bsz_per_gpu,
            "ulysses_sequence_parallel_size": 1,
            "grad_clip": 1.0,
            "optim": {
                "_target_": "verl.workers.config.FSDPOptimizerConfig",
                "lr": learning_rate,
                "weight_decay": 0.01,
                "lr_warmup_steps_ratio": 0.0,
                "total_training_steps": -1,
                "lr_warmup_steps": -1,
                "min_lr_ratio": None,
                "warmup_style": "constant"
            },
            "model": {
                "_target_": "verl.workers.config.FSDPCriticModelCfg",
                "path": model_path,
                "tokenizer_path": model_path,
                "override_config": {},
                "external_lib": None,
                "trust_remote_code": False,
                "use_shm": False,
                "enable_gradient_checkpointing": True,
                "enable_activation_offload": False,
                "use_remove_padding": True,
                "lora_rank": 0,
                "lora_alpha": 16,
                "target_modules": "all-linear",
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "param_offload": True,
                    "optimizer_offload": True,
                    "offload_policy": False,
                    "reshard_after_forward": True,
                    "wrap_policy": {"min_num_params": 0},
                    "fsdp_size": gpus,
                    "forward_prefetch": False
                }
            },
            "ppo_epochs": 1,
            "shuffle": False,
            "cliprange_value": 0.5,
            "loss_agg_mode": "token-mean",
            "ppo_max_token_len_per_gpu": safe_max_token_len,
            "forward_max_token_len_per_gpu": safe_max_token_len,
            "checkpoint": {
                "_target_": "verl.trainer.config.CheckpointConfig",
                "save_contents": ["model", "optimizer", "extra"],
                "load_contents": ["model", "optimizer", "extra"],
                "async_save": False
            },
            "profiler": {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": None,
                "enable": False,
                "all_ranks": False,
                "ranks": [],
                "save_path": None,
                "tool_config": {
                    "nsys": {
                        "_target_": "verl.utils.profiler.config.NsightToolConfig",
                        "discrete": False
                    },
                    "torch": {
                        "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                        "step_start": -1,
                        "step_end": -1
                    },
                    "torch_memory": {
                        "_target_": "verl.utils.profiler.config.TorchMemoryToolConfig",
                        "trace_alloc_max_entries": 100000,
                        "stack_depth": 32
                    },
                    "npu": {
                        "_target_": "verl.utils.profiler.config.NPUToolConfig",
                        "contents": [],
                        "level": "level1",
                        "analysis": False,
                        "discrete": False
                    }
                },
                "global_tool_config": None
            },

        },
        "reward_model": {"enable": False, "launch_reward_fn_async": False},
        "custom_reward_function": {"path": reward_file_path, "name": "compute_score"},
        "trainer": {
            "total_epochs": epochs,
            "total_training_steps": None,
            "project_name": "llm_sr_grpo_v2",
            "experiment_name": "direct_weight_tuning_v2",
            "logger": ["console"],
            "n_gpus_per_node": gpus,
            "nnodes": 1,
            "save_freq": 2,
            "test_freq": 5,
            "val_before_train": True,
            "default_local_dir": output_dir,
            "device": "cuda",
            "resume_mode": "disable",
            "critic_warmup": 0,
            "balance_batch": True,
            "log_val_generations": 10,
            "log_train_generations": 5,
            "profile_steps": None,
            "default_hdfs_dir": None,
            "log_prob_max_token_len_per_gpu": safe_max_token_len,
        },
        "ray_kwargs": {
            "ray_init": {
                "num_cpus": None,
                "runtime_env": {
                    "env_vars": {
                        "PYTHONPATH": "/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/verl:/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR"
                    }
                }
            }
        },
        "global_profiler": {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": None,
            "steps": None,
            "profile_continuous_steps": False,
            "save_path": "outputs/profile",
            "global_tool_config": {
                "nsys": {
                    "_target_": "verl.utils.profiler.config.NsightToolConfig",
                    "discrete": False
                },
                "torch_memory": {
                    "trace_alloc_max_entries": 100000,
                    "stack_depth": 32,
                    "context": "all",
                    "stacks": "all",
                    "kw_args": {}
                }
            }
        },
    }
    return OmegaConf.create(cfg)


def train_llmsr_grpo_v2(
    template_path: str,
    data_path: str,
    model_path: str,
    output_dir: str,
    *,
    grid_train_data: bool = False,
    num_grid_groups: int = 10,
    gpus: int = 8,
    rollout_n: int = 4,
    kl_coef: float = 1e-3,
    max_prompt_length: int = 2048,
    max_new_tokens: int = 1024,
    max_model_len: int = 8192,
    learning_rate: float = 1e-6,
    epochs: int = 5,
    few_shot_k: int = 3,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    memory_dir = os.path.join(output_dir, "memory_v2")
    os.makedirs(memory_dir, exist_ok=True)

    # 1) 数据集
    dataset_path = create_llmsr_dataset_v2(
        template_path=template_path,
        data_path=data_path,
        output_dir=output_dir,
        memory_dir=memory_dir,
        grid_train_data=grid_train_data,
        num_grid_groups=num_grid_groups,
        few_shot_k=few_shot_k,
    )

    # 2) 奖励文件
    reward_file_path = create_llmsr_reward_file_v2(
        template_path=template_path,
        data_path=data_path,
        output_dir=output_dir,
        memory_dir=memory_dir,
        grid_train_data=grid_train_data,
    )

    # 3) 配置
    config = create_grpo_config_v2(
        model_path=model_path,
        dataset_path=dataset_path,
        reward_file_path=reward_file_path,
        output_dir=output_dir,
        gpus=gpus,
        rollout_n=rollout_n,
        kl_coef=kl_coef,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # 4) 训练
    cfg_path = os.path.join(output_dir, "grpo_config_v2.yaml")
    OmegaConf.save(config, cfg_path)
    # 让奖励函数能写入样本文件
    os.environ["LLMSR_OUTPUT_DIR"] = output_dir
    run_ppo(config)

    # 5) 训练结束后，从 sample.jsonl 中选出最优
    try:
        import json
        best_reward = None
        best_mse = None
        best_reward_rec = None
        best_mse_rec = None
        jsonl_path = os.path.join(output_dir, "sample.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    r = rec.get("reward")
                    m = rec.get("nmse")
                    if isinstance(r, (int, float)) and (best_reward is None or r > best_reward):
                        best_reward = r
                        best_reward_rec = rec
                    if isinstance(m, (int, float)) and (best_mse is None or m < best_mse):
                        best_mse = m
                        best_mse_rec = rec
        if best_reward_rec:
            with open(os.path.join(output_dir, "best_reward.json"), "w", encoding="utf-8") as f:
                json.dump(best_reward_rec, f, ensure_ascii=False, indent=2)
        if best_mse_rec:
            with open(os.path.join(output_dir, "best_mse.json"), "w", encoding="utf-8") as f:
                json.dump(best_mse_rec, f, ensure_ascii=False, indent=2)
    except Exception:
        pass



