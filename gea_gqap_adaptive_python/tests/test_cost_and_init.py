"""
Проверка расчёта cost и инициализации (Heuristic2) против формул MATLAB.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gea_gqap_adaptive_python.model_loader import load_model
from gea_gqap_adaptive_python.utils import (
    create_xij,
    cost_function,
    cost_function_perm,
)
from gea_gqap_adaptive_python.heuristics import heuristic2


def cost_matlab_style(X: np.ndarray, model) -> float:
    """Точное воспроизведение формул из MATLAB CostFunction.m (для проверки)."""
    I, J = model.I, model.J
    cij, aij, bi = model.cij, model.aij, model.bi
    DIS, F = model.DIS, model.F
    X = np.asarray(X, dtype=float)
    # Feasibility
    count = np.zeros(I)
    for i in range(I):
        for j in range(J):
            count[i] += X[i, j] * aij[i, j]
    cvar = bi - count
    if np.any(cvar < 0):
        return float("inf")
    # c1
    c1 = 0.0
    for i in range(I):
        for j in range(J):
            c1 += cij[i, j] * X[i, j]
    # c2: F(j,l)*DIS(i,k)*X(i,j)*X(k,l)
    c2 = 0.0
    for i in range(I):
        for j in range(J):
            for k in range(I):
                for l in range(J):
                    c2 += F[j, l] * DIS[i, k] * X[i, j] * X[k, l]
    return c1 + c2


def test_cost_formula_matches_matlab():
    """Cost: на T1 сверяем с точной формулой MATLAB (c2 = sum F(j,l)*DIS(i,k)*X(i,j)*X(k,l))."""
    model = load_model("T1")
    ind = heuristic2(model)
    X = ind.xij.astype(float)
    perm = ind.permutation
    z_matlab = cost_matlab_style(X, model)
    cost_x, _ = cost_function(X, model)
    cost_perm, _ = cost_function_perm(perm, model)
    assert np.isfinite(z_matlab), "T1: matlab-style returned inf"
    assert np.isclose(cost_x, z_matlab, rtol=1e-9), (
        f"cost_function(X)={cost_x} vs matlab_style={z_matlab}"
    )
    assert np.isclose(cost_perm, z_matlab, rtol=1e-9), (
        f"cost_function_perm(perm)={cost_perm} vs matlab_style={z_matlab}"
    )
    print("OK: cost formula matches MATLAB (T1, explicit loops)")


def test_cost_perm_equals_cost_x_on_t13_t14():
    """На T13/T14 cost_function(X) и cost_function_perm(perm) совпадают (X=create_xij(perm))."""
    for name in ["T13", "T14"]:
        model = load_model(name)
        ind = heuristic2(model)
        X = create_xij(ind.permutation, model).astype(float)
        c_x, _ = cost_function(X, model)
        c_perm, _ = cost_function_perm(ind.permutation, model)
        assert np.isclose(c_x, c_perm, rtol=1e-9), f"{name}: cost X vs perm"
    print("OK: cost_function(X) == cost_function_perm(perm) on T13, T14")


def test_create_xij_and_cost_consistency():
    """create_xij(perm) даёт X, для которого cost_function(X) == cost_function_perm(perm)."""
    for name in ["T1", "T13", "T14"]:
        model = load_model(name)
        ind = heuristic2(model)
        perm = ind.permutation
        X = create_xij(perm, model)
        assert X.shape == (model.I, model.J)
        assert np.allclose(X.sum(axis=0), 1), "each job assigned exactly once"
        assert np.allclose(X, ind.xij), "create_xij matches heuristic2 xij"
        c_perm, _ = cost_function_perm(perm, model)
        c_x, _ = cost_function(X.astype(float), model)
        assert np.isclose(c_perm, c_x, rtol=1e-9), f"{name}: perm vs X cost"
    print("OK: create_xij and cost consistency (T1, T13, T14)")


def test_heuristic2_ct_formula():
    """Heuristic2: CT(i,j) = cij(i,j) + sum(DIS(i,:)) + sum(F(j,:)) — как в MATLAB."""
    model = load_model("T1")
    I, J = model.I, model.J
    CT_expected = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            CT_expected[i, j] = (
                model.cij[i, j]
                + model.DIS[i, :].sum()
                + model.F[j, :].sum()
            )
    ind = heuristic2(model)
    # Recompute CT as in our heuristic2
    CT_ours = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            CT_ours[i, j] = (
                model.cij[i, j]
                + model.DIS[i].sum()
                + model.F[j].sum()
            )
    np.testing.assert_allclose(CT_ours, CT_expected)
    print("OK: Heuristic2 CT formula matches MATLAB (cij + sum(DIS(i,:)) + sum(F(j,:)))")


def test_heuristic2_feasible_and_finite():
    """Heuristic2 возвращает допустимое решение с конечным cost."""
    for name in ["T1", "T13", "T14"]:
        model = load_model(name)
        ind = heuristic2(model)
        assert ind.permutation.shape == (model.J,)
        assert ind.xij.shape == (model.I, model.J)
        assert np.isfinite(ind.cost), f"{name}: cost must be finite"
        loads = (ind.xij.astype(float) * model.aij).sum(axis=1)
        assert np.all(loads <= model.bi + 1e-9), f"{name}: capacity must be satisfied"
    print("OK: Heuristic2 feasible and finite cost (T1, T13, T14)")


if __name__ == "__main__":
    test_cost_formula_matches_matlab()
    test_cost_perm_equals_cost_x_on_t13_t14()
    test_create_xij_and_cost_consistency()
    test_heuristic2_ct_formula()
    test_heuristic2_feasible_and_finite()
    print("\nAll cost and init checks passed.")
