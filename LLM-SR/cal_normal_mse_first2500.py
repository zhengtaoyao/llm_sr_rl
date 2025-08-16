#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List the TOP-5 equations found by LLM-SR (only samples_1–2500.json) and
print their positive NMSE on test_id / test_ood.  All paths are hard-coded;
only nmse(), the X-unpacking, and the file-filtering logic have changed.
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
    """
    Return (callable, n_params).

    Handles both
        params[3]        access style  and
        a, b, c = params unpack style.
    """
    # 1) import numpy and exec the code
    mod = types.ModuleType("mod")
    mod.__dict__["np"] = np
    exec(textwrap.dedent(src), mod.__dict__)
    eq = mod.equation                # the callable we care about

    # ------------------------------------------------------------------
    # 2) First, look for explicit index access  params[  k  ]
    param_idx = re.findall(r"params\s*\[\s*(\d+)\s*\]", src)
    if param_idx:
        n_params = max(map(int, param_idx)) + 1
        return eq, n_params
    # ------------------------------------------------------------------
    # 3) Next, look for tuple-unpack assignments  ‘a, b, … = params’
    unpack_rgx = re.compile(r"([A-Za-z0-9_,\s]+?)\s*=\s*params\b")
    sizes = []
    for m in unpack_rgx.finditer(src):
        lhs = m.group(1)
        # count non-empty variable names
        sizes.append(len([v for v in lhs.split(",") if v.strip()]))
    if sizes:
        return eq, max(sizes)
    # ------------------------------------------------------------------
    # 4) Fallback: try to *discover* how many numbers the function wants
    #    by calling it on the first training row with growing vectors.
    def discover_n(eq_func):
        dummy_x = [0.0] * (eq_func.__code__.co_argcount - 1)  # all X vars
        for n in range(1, 41):                                # cap at 40
            try:
                eq_func(*dummy_x, np.zeros(n))
                return n          # success → n works
            except ValueError as e:
                # look for “expected N, got n”
                m = re.search(r"expected\s+(\d+)", str(e))
                if m:
                    return int(m.group(1))
                continue
        return 0                   # give up
    return eq, discover_n(eq)



def fit_theta(eq, n_p, X, y):
    def obj(p): return nmse(eq(*X.T, p), y)
    return minimize(obj, x0=np.ones(n_p), method="BFGS",
                    options=dict(maxiter=10_000)).x


# ─────────────────────────── main ─────────────────────────────────────────
def main():
    sd      = Path(SAMPLES_DIR)
    pattern = re.compile(r"samples_(\d+)\.json$")

    scored = []
    for p in sd.glob("samples_*.json"):
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if not (1 <= idx <= 2500):
            continue                      # keep only 1–2500
        obj = json.load(open(p))
        if obj["score"] is not None:
            scored.append((obj["score"], p))

    if not scored:
        raise RuntimeError("No JSON samples with numeric id 1–2500 found.")

    # show the 5 highest-scoring entries
    top5       = sorted(scored, key=lambda t: t[0], reverse=True)[:5]
    X_tr, y_tr = load_csv(TRAIN_CSV)

    print(f"Found {len(scored)} eligible samples (id 1–2500).  "
          f"Showing top-5 (score = −NMSE):\n")

    for k, (score, path) in enumerate(top5, 1):
        obj      = json.load(open(path))
        eq, n_p  = compile_equation(obj["function"])
        theta    = fit_theta(eq, n_p, X_tr, y_tr)

        res = {}
        for tag, csv in (("test_id", TEST_ID_CSV), ("test_ood", TEST_OOD_CSV)):
            if Path(csv).exists():
                X, y   = load_csv(csv)
                res[tag] = nmse(eq(*X.T, theta), y)

        print(f"#{k}  {path.name}")
        print(f"    stored score  : {score:.6g}")
        print(f"    θ (refit)     : {theta}")
        print(f"    test_id NMSE  : {res.get('test_id', float('nan')):.6e}")
        print(f"    test_ood NMSE : {res.get('test_ood', float('nan')):.6e}\n")


if __name__ == "__main__":
    main()
