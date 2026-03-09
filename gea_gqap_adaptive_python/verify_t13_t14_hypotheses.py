#!/usr/bin/env python3
"""
Verify hypotheses for T13/T14 worse results:
1. Cost formula: cost_function(X) vs cost_function_perm(perm) match when X = create_xij(perm)
2. Dimensions: permutation length J, X shape (I,J), no indexing errors for I=20/40, J=1600
3. Heuristic2: produces feasible solution, finite cost
4. Optional: iterations per 1000s (run short and extrapolate)
"""
import math
import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gea_gqap_adaptive_python.model_loader import load_model
from gea_gqap_adaptive_python.utils import (
    create_xij,
    cost_function,
    cost_function_perm,
    evaluate_permutation,
)
from gea_gqap_adaptive_python.heuristics import heuristic2
from gea_gqap_adaptive_python.algorithm import run_ga
from gea_gqap_adaptive_python.models import AlgorithmConfig
import numpy as np

def test_cost_consistency(model_name: str):
    """Hypothesis 2 & 3: cost formula and dimensions."""
    print(f"\n=== {model_name}: cost consistency and dimensions ===")
    model = load_model(model_name)
    I, J = model.I, model.J
    print(f"  I={I}, J={J}")
    assert model.cij.shape == (I, J), f"cij {model.cij.shape}"
    assert model.DIS.shape == (I, I), f"DIS {model.DIS.shape}"
    assert model.F.shape == (J, J), f"F {model.F.shape}"

    # Use Heuristic2 solution so we have a feasible permutation
    ind0 = heuristic2(model)
    perm = ind0.permutation
    X = create_xij(perm, model)
    assert X.shape == (I, J), f"X shape {X.shape}"
    assert np.allclose(X.sum(axis=0), 1), "each job assigned once"
    assert np.allclose(X.sum(axis=1), np.bincount(perm, minlength=I)), "row sums"
    assert np.allclose(X, ind0.xij), "create_xij matches heuristic2 xij"

    cost_perm, cvar_perm = cost_function_perm(perm, model)
    cost_x, cvar_x = cost_function(X.astype(float), model)
    ok = math.isfinite(cost_perm) and math.isfinite(cost_x) and np.isclose(cost_perm, cost_x)
    print(f"  cost_function_perm = {cost_perm}, cost_function(X) = {cost_x}, match = {ok}")
    if not ok:
        print("  FAIL: cost mismatch")
        return False
    print("  OK: cost formula consistent")
    return True


def test_heuristic2(model_name: str):
    """Heuristic2 produces feasible solution."""
    print(f"\n=== {model_name}: Heuristic2 ===")
    model = load_model(model_name)
    ind = heuristic2(model)
    print(f"  cost = {ind.cost}, finite = {math.isfinite(ind.cost)}")
    print(f"  permutation shape = {ind.permutation.shape}, xij shape = {ind.xij.shape}")
    assert ind.permutation.shape == (model.J,), ind.permutation.shape
    assert ind.xij.shape == (model.I, model.J), ind.xij.shape
    cost2, _ = cost_function_perm(ind.permutation, model)
    assert np.isclose(ind.cost, cost2), f"stored cost {ind.cost} vs recomputed {cost2}"
    if not math.isfinite(ind.cost):
        print("  FAIL: Heuristic2 returned inf")
        return False
    print("  OK: Heuristic2 feasible")
    return True


def test_iterations_per_time(model_name: str, time_limit_sec: float = 5.0, population_size: int = 50):
    """Hypothesis 1: how many generations in time_limit (short run)."""
    print(f"\n=== {model_name}: generations in {time_limit_sec}s (pop={population_size}) ===")
    model = load_model(model_name)
    cfg = AlgorithmConfig(time_limit=time_limit_sec, iterations=10000, population_size=population_size)
    t0 = time.perf_counter()
    result = run_ga(model, config=cfg)
    elapsed = time.perf_counter() - t0
    n_iter = len(result.stats.best_cost_trace)
    print(f"  elapsed = {elapsed:.2f}s, iterations = {n_iter}")
    if time_limit_sec > 0:
        rate = n_iter / elapsed
        extrapolated_1000s = rate * 1000
        print(f"  rate = {rate:.1f} iter/s, extrapolated for 1000s = {extrapolated_1000s:.0f} iter")
    return True


def main():
    for name in ["T13", "T14"]:
        test_cost_consistency(name)
        test_heuristic2(name)
    # Short run: small pop to get iteration rate
    for name in ["T13", "T14"]:
        test_iterations_per_time(name, time_limit_sec=8.0, population_size=50)
    # With full population (350) to estimate real generations in 1000s
    print("\n--- With population_size=350 (paper config) ---")
    for name in ["T13", "T14"]:
        test_iterations_per_time(name, time_limit_sec=15.0, population_size=350)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
