#!/usr/bin/env python3
"""
🔥 LLM-SR dataset processor for VERL GRPO

Processes datasets under ./data into VERL-compatible parquet files placed in ./verl_datasets.
Note: ground_truth text is a non-supervisory placeholder to satisfy certain schema users; the real
supervision is the CSV mapping (inputs -> outputs). Reward should compute errors against CSV values.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import re


def extract_function_names(specification: str) -> Tuple[str, str]:
    """Extract function names from specification template."""
    run_pattern = r'@evaluate\.run\s+def\s+(\w+)'
    run_match = re.search(run_pattern, specification)
    run_function = run_match.group(1) if run_match else "evaluate_function"

    evolve_pattern = r'@equation\.evolve\s+def\s+(\w+)'
    evolve_match = re.search(evolve_pattern, specification)
    evolve_function = evolve_match.group(1) if evolve_match else "symbolic_regression"
    return evolve_function, run_function


def build_problem_prompt(template_content: str, problem_name: str) -> str:
    """Build prompt from template (header before @equation.evolve target function)."""
    lines = template_content.split('\n')
    prompt_lines: List[str] = []
    in_evolve_function = False

    for line in lines:
        if '@equation.evolve' in line:
            in_evolve_function = True
            continue
        if in_evolve_function and line.strip().startswith('def '):
            prompt_lines.append(line.rstrip())
            break
        if not in_evolve_function:
            prompt_lines.append(line.rstrip())

    base_prompt = '\n'.join(prompt_lines).strip()

    # English task hint (concise; non-binding)
    if 'oscillator' in problem_name:
        task_description = """
You are discovering a physical equation for a (possibly damped) harmonic oscillator.
Given x (position) and v (velocity), predict a (acceleration). Prefer simple, dimensionally
consistent expressions aligned with Hooke's law and basic physics.
"""
    elif 'bactgrow' in problem_name:
        task_description = """
You are discovering a biology equation for bacterial growth rate.
Given b (biomass), s (substrate), temp, pH, predict db. Consider Monod-like effects
and temperature/pH influences. Prefer simple, plausible expressions.
"""
    elif 'stressstrain' in problem_name:
        task_description = """
You are discovering a materials equation for stress-strain relations.
Given strain and temp, predict stress. Prefer simple, physically plausible forms
(e.g., elastic regimes) unless data suggests otherwise.
"""
    else:
        task_description = """
You are discovering a symbolic equation from data. Propose a concise mathematical
expression mapping inputs to the output with good fit and simplicity.
"""

    return base_prompt + task_description


def get_ground_truth_equations() -> Dict[str, str]:
    """Return placeholder textual hints (NOT used as supervision)."""
    # These are non-supervisory placeholders to satisfy pipelines expecting a string.
    return {
        'oscillator1': 'placeholder: see CSV labels',
        'oscillator2': 'placeholder: see CSV labels',
        'bactgrow': 'placeholder: see CSV labels',
        'stressstrain': 'placeholder: see CSV labels'
    }


def create_verl_dataset_entry(
    prompt: str,
    problem_name: str,
    data_sample: Dict[str, float],
    ground_truth: str,
    data_source: str = "llm_sr_train"
) -> Dict[str, Any]:
    """Create one VERL dataset entry."""
    chat_messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # ground_truth is a placeholder; supervision uses CSV numeric labels.
    entry = {
        "prompt": chat_messages,
        "data_source": data_source,
        "ability": f"symbolic_regression_{problem_name}",
        "extra_info": {
            "problem_type": problem_name,
            "task": "symbolic_regression",
            "domain": "scientific_equation_discovery",
            "data_sample": data_sample
        },
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
            "evaluation_type": "mse_based",
            "target_variables": list(data_sample.keys())
        }
    }
    return entry


def process_single_dataset(
    problem_name: str,
    data_dir: Path,
    spec_dir: Path,
    output_dir: Path,
    max_samples: int = 1000
) -> bool:
    """Process one dataset into VERL parquet."""
    print(f"\n🔄 Processing dataset: {problem_name}")

    train_file = data_dir / problem_name / "train.csv"
    spec_file = spec_dir / f"specification_{problem_name}_numpy.txt"

    if not train_file.exists():
        print(f"❌ Missing train data: {train_file}")
        return False
    if not spec_file.exists():
        print(f"❌ Missing spec file: {spec_file}")
        return False

    try:
        df = pd.read_csv(train_file)
        with open(spec_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print(f"📊 Data shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ IO error: {e}")
        return False

    evolve_function, run_function = extract_function_names(template_content)
    prompt = build_problem_prompt(template_content, problem_name)

    # Non-supervisory placeholder string
    ground_truths = get_ground_truth_equations()
    ground_truth = ground_truths.get(problem_name, "placeholder: see CSV labels")

    print(f"🎯 Target function: {evolve_function}")
    print(f"🔍 Eval function: {run_function}")

    num_samples = min(max_samples, len(df))
    df_sampled = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    print(f"📊 Selected samples: {num_samples}")

    dataset_entries: List[Dict[str, Any]] = []
    for _, row in df_sampled.iterrows():
        data_sample = row.to_dict()
        entry = create_verl_dataset_entry(
            prompt=prompt,
            problem_name=problem_name,
            data_sample=data_sample,
            ground_truth=ground_truth,
            data_source=f"llm_sr_{problem_name}_train"
        )
        dataset_entries.append(entry)

    output_file = output_dir / f"{problem_name}_train_verl.parquet"
    try:
        table = pa.Table.from_pylist(dataset_entries)
        pq.write_table(table, output_file)
        print(f"✅ Saved: {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"❌ Save error: {e}")
        return False


def main():
    print("🔥 LLM-SR dataset processor for VERL GRPO")
    print("=" * 70)

    base_dir = Path(".")
    data_dir = base_dir / "data"
    spec_dir = base_dir / "specs"
    output_dir = base_dir / "verl_datasets"

    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output dir: {output_dir.absolute()}")

    if not data_dir.exists():
        print(f"❌ Data dir not found: {data_dir}")
        return
    if not spec_dir.exists():
        print(f"❌ Specs dir not found: {spec_dir}")
        return

    available_problems: List[str] = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "train.csv").exists():
            available_problems.append(item.name)
    print(f"🔍 Found datasets: {available_problems}")
    if not available_problems:
        print("❌ No dataset found")
        return

    success_count = 0
    for problem_name in available_problems:
        try:
            ok = process_single_dataset(
                problem_name=problem_name,
                data_dir=data_dir,
                spec_dir=spec_dir,
                output_dir=output_dir,
                max_samples=1000,
            )
            if ok:
                success_count += 1
        except Exception as e:
            print(f"❌ Error processing {problem_name}: {e}")
            continue

    print(f"\n{'='*70}")
    print("🎉 Completed!")
    print(f"✅ Success: {success_count}/{len(available_problems)}")
    print(f"📁 Output dir: {output_dir.absolute()}")
    if success_count:
        print("\n📋 Generated parquet files:")
        for file in output_dir.glob("*_train_verl.parquet"):
            print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
        print("\n🚀 Next: point VERL-based scripts to the parquet if needed.")


if __name__ == "__main__":
    main() 