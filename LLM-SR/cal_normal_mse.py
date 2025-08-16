#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List the TOP-5 equations found by LLM-SR and print their positive NMSE
on test_id / test_ood.  Completely hard-coded paths; only the math
inside nmse() and the X-unpacking have changed.
"""

import json, re, textwrap, types
from pathlib import Path

import numpy as np
import pandas as pd
from  scipy.optimize import minimize


# ─────────────── paths (DO NOT EDIT) ───────────────────────────────────────
ROOT = "/storage/home/westlakeLab/zhangjunlei/llm_rl_pde/LLM-SR/"

SAMPLES_DIR   = "/storage/home/westlakeLab/zhangjunlei/llm_rl_pde/LLM-SR/logs/oscillator1_local_saved/samples"
DATA_DIR      = "/storage/home/westlakeLab/zhangjunlei/llm_rl_pde/LLM-SR/data/oscillator1"
# SAMPLES_DIR   = "/storage/home/westlakeLab/zhangjunlei/llm_rl_pde/LLM-SR/logs/stresstrain_local/samples"
# DATA_DIR      = "/storage/home/westlakeLab/zhangjunlei/llm_rl_pde/LLM-SR/data/stressstrain"

TRAIN_CSV     = DATA_DIR + "/train.csv"
TEST_ID_CSV   = DATA_DIR + "/test_id.csv"
TEST_OOD_CSV  = DATA_DIR + "/test_ood.csv"
# ───────────────────────────────────────────────────────────────────────────


def nmse(pred, true):
    """Normalized MSE – identical to evaluator.py (+1e-12 for safety)."""
    return np.mean((pred - true) ** 2) / (np.mean(true ** 2) + 1e-12)


def load_csv(csv_path):
    arr = pd.read_csv(csv_path).to_numpy(float)
    return arr[:, :-1], arr[:, -1]                    # X matrix, y vector


def compile_equation(src):
    """Return (callable, #params) from source string."""
    mod = types.ModuleType("mod")
    mod.__dict__["np"] = np
    exec(textwrap.dedent(src), mod.__dict__)
    eq = mod.equation
    n_params = (
        max(int(m.group(1)) for m in re.finditer(r"params\[(\d+)\]", src)) + 1
        if "params[" in src else 0
    )
    return eq, n_params


def fit_theta(eq, n_p, X, y):
    def obj(p): return nmse(eq(*X.T, p), y)
    return minimize(obj, x0=np.ones(n_p), method="BFGS",
                    options=dict(maxiter=10_000)).x


# ─────────────────────────── main ─────────────────────────────────────────
def main():
    sd = Path(SAMPLES_DIR)
    scored = [(json.load(open(p))["score"], p)
              for p in sd.glob("*.json") if json.load(open(p))["score"] is not None]
    if not scored:
        raise RuntimeError("no JSON samples")

    top5 = sorted(scored, key=lambda t: t[0], reverse=True)[:5]  # highest scores
    X_tr, y_tr = load_csv(TRAIN_CSV)

    print(f"Found {len(scored)} samples.  Showing top-5 (score = −NMSE):\n")
    for k, (score, path) in enumerate(top5, 1):
        obj  = json.load(open(path))
        eq, n_p = compile_equation(obj["function"])
        theta   = fit_theta(eq, n_p, X_tr, y_tr)

        res = {}
        for tag, csv in (("test_id", TEST_ID_CSV), ("test_ood", TEST_OOD_CSV)):
            if not Path(csv).exists():
                continue
            X, y   = load_csv(csv)
            res[tag] = nmse(eq(*X.T, theta), y)

        print(f"#{k}  {path.name}")
        print(f"    stored score  : {score:.6g}")
        print(f"    θ (refit)     : {theta}")
        print(f"    test_id NMSE  : {res.get('test_id', float('nan')):.6e}")
        print(f"    test_ood NMSE : {res.get('test_ood', float('nan')):.6e}\n")


if __name__ == "__main__":
    main()
