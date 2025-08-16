import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [CSV_REWARD] - %(levelname)s - %(message)s'
)

# Cache for loaded dataframes to avoid repeated file I/O for the same file
DATA_CACHE: Dict[str, pd.DataFrame] = {}

def _get_single_reward(solution_str: str, ground_truth: Dict[str, Any]) -> float:
    """
    Calculates the reward for a single solution string.
    This is the core logic, now refactored to handle one item.
    """
    try:
        csv_path = ground_truth.get("data_path")
        target_var = ground_truth.get("target_variable")

        if not csv_path or not target_var:
            logging.error(f"'data_path' or 'target_variable' missing. Received: {ground_truth}")
            return -100.0

        if csv_path in DATA_CACHE:
            df = DATA_CACHE[csv_path]
        else:
            logging.info(f"Loading data from new path: {csv_path}")
            df = pd.read_csv(csv_path)
            DATA_CACHE[csv_path] = df
        
        if target_var not in df.columns:
            logging.error(f"Target '{target_var}' not in columns of {csv_path}. Available: {df.columns.tolist()}")
            return -100.0

        eval_context = {col: df[col].to_numpy() for col in df.columns}
        eval_context.update({
            'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt
        })

        y_true = df[target_var].to_numpy()
        y_pred = eval(solution_str, {"__builtins__": {}}, eval_context)

        if not isinstance(y_pred, np.ndarray):
            y_pred = np.full_like(y_true, fill_value=y_pred)

        mse = np.mean((y_true - y_pred)**2)
        
        if not np.isfinite(mse):
            logging.warning(f"MSE is not finite for '{solution_str}'. Assigning max penalty.")
            return -100.0

        reward = max(-mse, -100.0)
        return float(reward)

    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        logging.warning(f"Failed to evaluate '{solution_str}'. Error: {type(e).__name__} - {e}")
        return -100.0
    except Exception as e:
        logging.error(f"Unexpected error for '{solution_str}'. Error: {e}", exc_info=True)
        return -100.0

def get_reward(data_sources: List[str], solution_strs: List[str], ground_truths: List[Dict[str, Any]], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    🔥 Batch-processing reward function called by VERL.

    This function iterates over the batch of solutions and calculates the
    reward for each one individually.

    Args:
        data_sources (List[str]): List of data source identifiers.
        solution_strs (List[str]): List of LLM-generated equations.
        ground_truths (List[Dict[str, Any]]): List of dictionaries, each containing
            the 'data_path' and 'target_variable' for a sample.
        extra_infos (List[Dict[str, Any]]): List of additional info dicts.

    Returns:
        List[float]: A list of rewards corresponding to each solution string.
    """
    rewards = []
    num_solutions = len(solution_strs)
    
    if num_solutions == 0:
        return []

    logging.info(f"Received batch of {num_solutions} solutions to evaluate.")

    for i in range(num_solutions):
        solution_str = solution_strs[i]
        # VERL passes the 'reward_model' dict from the dataset as the 'ground_truth' argument
        ground_truth_info = ground_truths[i] 
        
        reward = _get_single_reward(solution_str, ground_truth_info)
        rewards.append(reward)

    logging.info(f"Finished evaluating batch. Average reward: {np.mean(rewards):.4f}")
    return rewards