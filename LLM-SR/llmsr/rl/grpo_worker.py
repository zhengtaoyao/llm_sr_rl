"""
GRPO Reward Worker for LLM-SR Integration

This module provides a custom Ray remote worker that evaluates program skeletons
using LLM-SR's BFGS-based optimization and returns rewards for GRPO training.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import tempfile
import os

import ray
from verl.single_controller.base import Worker
from verl import DataProto

from llmsr import evaluator
from llmsr import code_manipulation
from llmsr.config import Config


@ray.remote(num_cpus=1)
class LLMSRRewardWorker(Worker):
    """
    Ray remote worker for evaluating program skeletons using LLM-SR's evaluation logic.
    
    This worker takes generated code completions and evaluates them using the same
    BFGS optimization and MSE calculation as the original LLM-SR system.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize the reward worker with LLM-SR configuration.
        
        Args:
            config_dict: Dictionary containing LLM-SR task configuration
        """
        super().__init__(config_dict)
        
        # Extract task configuration
        self.template_path = config_dict['task']['template_path']
        self.data_path = config_dict['task']['data_path'] 
        self.timeout_seconds = config_dict['task'].get('timeout_seconds', 30)
        
        # Load the problem template
        with open(self.template_path, 'r') as f:
            self.specification = f.read()
            
        # Parse template to get function names
        self.function_to_evolve, self.function_to_run = self._extract_function_names()
        self.template = code_manipulation.text_to_program(self.specification)
        
        # Load dataset 
        self.dataset = self._load_dataset()
        
        # Initialize evaluator (without database since we're not using buffer)
        self.evaluator_core = evaluator.Evaluator(
            database=None,
            template=self.template, 
            function_to_evolve=self.function_to_evolve,
            function_to_run=self.function_to_run,
            inputs=self.dataset,
            timeout_seconds=self.timeout_seconds,
            sandbox_class=evaluator.LocalSandbox
        )
        
        print(f"✅ Initialized LLMSRRewardWorker for {self.function_to_evolve}")
    
    def _extract_function_names(self):
        """Extract function names from specification template."""
        run_functions = list(code_manipulation.yield_decorated(
            self.specification, 'evaluate', 'run'))
        evolve_functions = list(code_manipulation.yield_decorated(
            self.specification, 'equation', 'evolve'))
            
        if len(run_functions) != 1 or len(evolve_functions) != 1:
            raise ValueError("Template must have exactly one @evaluate.run and one @equation.evolve function")
            
        return evolve_functions[0], run_functions[0]
    
    def _load_dataset(self):
        """Load the training dataset."""
        df = pd.read_csv(self.data_path)
        data = np.array(df)
        
        # Assume last column is output, others are inputs
        X = data[:, :-1] 
        y = data[:, -1].reshape(-1)
        
        data_dict = {'inputs': X, 'outputs': y}
        return {'data': data_dict}
    
    def forward(self, data_proto: DataProto) -> Dict[str, np.ndarray]:
        """
        Evaluate a batch of generated program completions.
        
        Args:
            data_proto: DataProto containing prompts and responses
            
        Returns:
            Dictionary with 'reward' key containing reward values
        """
        batch_size = len(data_proto.batch['responses'])
        rewards = []
        
        for i in range(batch_size):
            # Get the generated response (program completion)
            response_tokens = data_proto.batch['responses'][i]
            
            # Convert tokens back to text
            # Note: This assumes we have access to tokenizer, might need to adjust
            if hasattr(data_proto, 'tokenizer'):
                generated_code = data_proto.tokenizer.decode(response_tokens, skip_special_tokens=True)
            else:
                # Fallback: assume responses are already strings
                generated_code = response_tokens if isinstance(response_tokens, str) else str(response_tokens)
            
            # Evaluate the generated code
            reward = self._evaluate_single_program(generated_code)
            rewards.append(reward)
        
        return {"reward": np.array(rewards, dtype=np.float32)}
    
    def _evaluate_single_program(self, generated_code: str) -> float:
        """
        Evaluate a single generated program using LLM-SR's evaluation logic.
        
        Args:
            generated_code: The generated program skeleton code
            
        Returns:
            Reward value (negative MSE, higher is better)
        """
        try:
            # Convert generated code to proper function format
            new_function, program = self._sample_to_program(generated_code)
            
            # Evaluate on all test inputs
            scores_per_test = {}
            
            for test_input in self.dataset:
                test_output, runs_ok = self.evaluator_core._sandbox.run(
                    program,
                    self.function_to_run, 
                    self.function_to_evolve,
                    self.dataset,
                    test_input,
                    self.timeout_seconds
                )
                
                if runs_ok and test_output is not None:
                    if isinstance(test_output, (int, float)):
                        scores_per_test[test_input] = test_output
                    else:
                        # Invalid output format
                        return -1e6
                else:
                    # Execution failed
                    return -1e6
            
            if not scores_per_test:
                return -1e6
                
            # Calculate average score (negative MSE)
            avg_score = np.mean(list(scores_per_test.values()))
            
            # Return reward (higher is better for GRPO)
            return float(avg_score) if not (np.isnan(avg_score) or np.isinf(avg_score)) else -1e6
            
        except Exception as e:
            print(f"⚠️ Error evaluating program: {e}")
            return -1e6
    
    def _sample_to_program(self, generated_code: str, version_generated: int = None):
        """
        Convert generated code sample to executable program.
        
        This is adapted from LLM-SR's evaluator._sample_to_program function.
        """
        # Clean the generated code body
        body = self._trim_function_body(generated_code)
        
        if version_generated is not None:
            body = code_manipulation.rename_function_calls(
                code=body,
                source_name=f'{self.function_to_evolve}_v{version_generated}',
                target_name=self.function_to_evolve
            )
        
        # Create new program with evolved function
        import copy
        program = copy.deepcopy(self.template)
        evolved_function = program.get_function(self.function_to_evolve)
        evolved_function.body = body
        
        return evolved_function, str(program)
    
    def _trim_function_body(self, generated_code: str) -> str:
        """
        Extract and clean the function body from generated code.
        
        This is adapted from LLM-SR's evaluator._trim_function_body function.
        """
        if not generated_code:
            return ''
        
        # Add proper indentation if missing
        lines = generated_code.strip().split('\n')
        indented_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                if not line.startswith('    '):  # Add indentation if missing
                    indented_lines.append('    ' + line.lstrip())
                else:
                    indented_lines.append(line)
            else:
                indented_lines.append(line)
        
        return '\n'.join(indented_lines) + '\n\n'


# Factory function for creating workers
def create_llmsr_reward_worker(config_dict: Dict[str, Any]):
    """Factory function to create LLMSRRewardWorker instances."""
    return LLMSRRewardWorker.remote(config_dict)


# Custom reward function for VERL integration  
def compute_llmsr_reward(data_proto: DataProto, **kwargs) -> Dict[str, Any]:
    """
    Custom reward function that can be used directly with VERL's reward system.
    
    This function creates reward workers on-demand and evaluates the batch.
    For better performance, it's recommended to use the pre-created remote workers.
    """
    config_dict = kwargs.get('config_dict', {})
    
    # Create a temporary worker for this batch
    worker = LLMSRRewardWorker.remote(config_dict)
    
    # Get rewards
    result = ray.get(worker.forward.remote(data_proto))
    
    return {
        "reward_tensor": result["reward"],
        "reward_extra_info": {}
    } 