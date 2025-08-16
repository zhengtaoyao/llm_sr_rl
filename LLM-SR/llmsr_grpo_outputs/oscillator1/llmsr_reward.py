
"""
Custom reward function for LLM-SR integration with VERL GRPO.
"""

import sys
import os
from pathlib import Path

# Add LLM-SR to path
llmsr_path = str(Path(__file__).parent.parent)
if llmsr_path not in sys.path:
    sys.path.append(llmsr_path)

from llmsr.rl.grpo_worker import compute_llmsr_reward

def compute_score(data_proto, **kwargs):
    """
    Compute rewards for LLM-SR symbolic regression task.
    
    This function evaluates generated program skeletons using LLM-SR's
    BFGS optimization and returns negative MSE as reward.
    """
    
    # Configuration for LLM-SR task
    config_dict = {
        'task': {
            'template_path': './specs/specification_oscillator1_numpy.txt',
            'data_path': './data/oscillator1/train.csv',
            'timeout_seconds': 30
        }
    }
    
    # Use the LLM-SR reward computation
    return compute_llmsr_reward(data_proto, config_dict=config_dict)
