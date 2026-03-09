from __future__ import annotations

import numpy as np

from .models import Individual, Model
from .utils import evaluate_permutation


def heuristic2(model: Model) -> Individual:
    """
    Initial solution + repair. Matches MATLAB Heuristic2.m:
    CT(i,j) = cij(i,j) + sum(DIS(i,:)) + sum(F(j,:)); assign job j to facility with min CT(:,j)
    respecting capacity (try all in order); then cascading repair while min(cvar)<0.
    """
    I, J = model.I, model.J
    X = np.zeros((I, J), dtype=int)
    count = np.zeros(I, dtype=float)

    CT = np.zeros((I, J), dtype=float)
    for i in range(I):
        for j in range(J):
            CT[i, j] = model.cij[i, j] + model.DIS[i].sum() + model.F[j].sum()

    for j in range(J):
        candidates = CT[:, j]
        order = np.argsort(candidates)
        assigned = False
        for idx in order:
            if count[idx] + model.aij[idx, j] <= model.bi[idx]:
                X[idx, j] = 1
                count[idx] += model.aij[idx, j]
                assigned = True
                break
        if not assigned:
            idx = order[0]
            X[idx, j] = 1
            count[idx] += model.aij[idx, j]

    # Repair feasibility (MATLAB-style cascading repair)
    cvar = model.bi - count
    Wij = X * model.aij
    max_repair_passes = I * J
    repair_pass = 0
    while np.any(cvar < -1e-9) and repair_pass < max_repair_passes:
        repair_pass += 1
        for i in range(I):
            while cvar[i] < -1e-9:
                assigned_jobs = np.where(X[i] == 1)[0]
                if assigned_jobs.size == 0:
                    break
                b = assigned_jobs[np.argmax(Wij[i, assigned_jobs])]
                count[i] -= model.aij[i, b]
                cvar[i] = model.bi[i] - count[i]
                X[i, b] = 0
                Wij[i, b] = 0
                d = int(np.argmin(model.aij[:, b]))
                if d == i:
                    d = int(np.argmax(cvar))
                count[d] += model.aij[d, b]
                cvar[d] = model.bi[d] - count[d]
                X[d, b] = 1
                Wij[d, b] = model.aij[d, b]

    permutation = np.argmax(X, axis=0)
    return evaluate_permutation(permutation, model)

