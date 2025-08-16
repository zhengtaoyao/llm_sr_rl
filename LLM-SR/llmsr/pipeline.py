# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Implementation of the LLMSR pipeline. """
from __future__ import annotations

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence

from llmsr import code_manipulation
from llmsr import config as config_lib
from llmsr import evaluator
from llmsr import buffer
from llmsr import sampler
from llmsr import profile


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """ Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        use_rl: bool = False,
        **kwargs
):
    """ Launch a LLMSR experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
        use_rl       : if True, use GRPO instead of evolutionary search.
        **kwargs     : additional parameters for RL training.
    """
    
    # 🔥 NEW: GRPO Training Branch
    if use_rl:
        print("🚀 Switching to GRPO training mode...")
        
        # Import GRPO runner
        try:
            from llmsr.rl.grpo_runner import train_llmsr_grpo
        except ImportError as e:
            raise ImportError(
                f"Failed to import GRPO runner. Make sure VERL is installed and properly configured. Error: {e}"
            )
        
        # Extract required paths from kwargs
        template_path = kwargs.get('template_path')
        data_path = kwargs.get('data_path')
        model_path = kwargs.get('model_path', 'Qwen/Qwen2.5-1.5B-Instruct')
        output_dir = kwargs.get('output_dir', './llmsr_grpo_outputs')
        
        if not template_path:
            raise ValueError("template_path must be provided for GRPO training")
        if not data_path:
            raise ValueError("data_path must be provided for GRPO training")
        
        # Extract GRPO-specific configuration
        grpo_config = {
            'epochs': kwargs.get('epochs', 10),
            'batch_size': kwargs.get('batch_size', 64),
            'learning_rate': kwargs.get('learning_rate', 1e-6),
            'rollout_n': kwargs.get('rollout_n', 5),  # Group size for GRPO
            'experiment_name': kwargs.get('experiment_name', 'llmsr_grpo'),
            'gpus': kwargs.get('gpus', 1)
        }
        
        print(f"📋 Template: {template_path}")
        print(f"📊 Data: {data_path}")
        print(f"🤖 Model: {model_path}")
        print(f"⚙️ GRPO Config: {grpo_config}")
        
        # Start GRPO training
        train_llmsr_grpo(
            template_path=template_path,
            data_path=data_path,
            model_path=model_path,
            output_dir=output_dir,
            **grpo_config
        )
        
        print("✅ GRPO training completed!")
        return
    
    # ⚡ ORIGINAL: Evolutionary Search Branch
    print("🧬 Using original evolutionary search mode...")
    
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)

    # Set global max sample nums.
    samplers = [sampler.Sampler(database, evaluators, 
                                config.samples_per_prompt, 
                                max_sample_nums=max_sample_nums, 
                                llm_class=class_config.llm_class,
                                config = config) 
                                for _ in range(config.num_samplers)]

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler)
