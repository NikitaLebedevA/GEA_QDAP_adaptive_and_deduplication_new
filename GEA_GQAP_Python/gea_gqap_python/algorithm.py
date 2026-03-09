from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .heuristics import heuristic2
from .models import AlgorithmConfig, AlgorithmResult, AlgorithmStats, Individual, Model
from .operators import (
    analyze_perm,
    combine_q,
    crossover,
    mask_mutation,
    mutation,
    roulette_wheel_selection,
)
from .utils import evaluate_permutation


DEFAULT_INSTRUCTION = (True, True, True)


def _ensure_instruction_tuple(instruction: Sequence[bool] | None) -> Tuple[bool, bool, bool]:
    if instruction is None:
        return DEFAULT_INSTRUCTION
    if len(instruction) != 3:
        raise ValueError("instruction must contain exactly three boolean flags")
    return tuple(bool(x) for x in instruction)  # type: ignore[return-value]


def _select_population_dedupe(
    pool: List[Tuple[Individual, str]],
    population_size: int,
    model: Model,
    rng: np.random.Generator,
) -> Tuple[List[Individual], List[str]]:
    """
    Из пула отбирает уникальных по генотипу (permutation), при нехватке дозаполняет
    случайными (мутация от лучшего). Возвращает (population, top_origins).
    Лимит времени не применяется — дозаполнение всегда выполняется до конца.
    """
    seen: set = set()
    unique_list: List[Tuple[Individual, str]] = []
    for ind, origin in pool:
        key = tuple(ind.permutation.tolist())
        if key not in seen:
            seen.add(key)
            unique_list.append((ind, origin))
    if len(unique_list) >= population_size:
        unique_list.sort(key=lambda x: x[0].cost)
        pop = [ind for ind, _ in unique_list[:population_size]]
        origins = [o for _, o in unique_list[:population_size]]
        return pop, origins
    best_ind = min(unique_list, key=lambda x: x[0].cost)[0]
    while len(unique_list) < population_size:
        new_perm = mutation(best_ind.permutation, model, rng)
        new_ind = evaluate_permutation(new_perm, model)
        if math.isfinite(new_ind.cost):
            unique_list.append((new_ind, "fill"))
    unique_list.sort(key=lambda x: x[0].cost)
    pop = [ind for ind, _ in unique_list[:population_size]]
    origins = [o for _, o in unique_list[:population_size]]
    return pop, origins


def run_ga(
    model: Model,
    config: AlgorithmConfig | None = None,
    instruction: Sequence[bool] | None = None,
) -> AlgorithmResult:
    cfg = config or AlgorithmConfig()
    instruction_tuple = _ensure_instruction_tuple(instruction or cfg.enable_scenario)
    rng = np.random.default_rng(cfg.random_seed)

    start_time = time.perf_counter()

    ncrossover = int(2 * round((cfg.crossover_rate * cfg.population_size) / 2))
    nmutation = int(math.floor(cfg.mutation_rate * cfg.population_size))

    ncrossover_scenario = int(
        math.floor(cfg.scenario_crossover_rate * (cfg.p_scenario3 * cfg.population_size))
    )
    nmutate_scenario = int(
        math.floor(cfg.scenario_mutation_rate * (cfg.p_scenario3 * cfg.population_size))
    )

    population: List[Individual] = []
    best_solution = heuristic2(model)
    population.append(best_solution)

    while len(population) < cfg.population_size:
        mutated = mutation(population[0].permutation, model, rng)
        individual = evaluate_permutation(mutated, model)
        if math.isfinite(individual.cost):
            population.append(individual)

    population.sort(key=lambda ind: ind.cost)
    best_solution = population[0]
    worst_cost = population[-1].cost
    beta = 10.0

    stats = AlgorithmStats()

    for iteration in range(cfg.iterations):
        n_pop = len(population)
        if n_pop == 0:
            break
        costs = np.array([ind.cost for ind in population], dtype=float)
        probabilities = np.exp(-beta * costs / worst_cost)
        probabilities /= probabilities.sum()

        offspring: List[Individual] = []
        crossover_origins: List[str] = []

        for _ in range(0, ncrossover, 2):
            i1 = roulette_wheel_selection(probabilities, rng)
            i2 = roulette_wheel_selection(probabilities, rng)
            parents = (population[i1], population[i2])
            child_perm1, child_perm2 = crossover(parents, rng)
            child1 = evaluate_permutation(child_perm1, model)
            child2 = evaluate_permutation(child_perm2, model)
            if math.isfinite(child1.cost):
                offspring.append(child1)
                crossover_origins.append("crossover")
            if math.isfinite(child2.cost):
                offspring.append(child2)
                crossover_origins.append("crossover")

        mutations: List[Individual] = []
        mutation_origins: List[str] = []
        for _ in range(nmutation):
            idx = rng.integers(0, n_pop)
            mutated_perm = mutation(population[idx].permutation, model, rng)
            mutated_individual = evaluate_permutation(mutated_perm, model)
            if math.isfinite(mutated_individual.cost):
                mutations.append(mutated_individual)
                mutation_origins.append("mutation")

        scenario_candidates: List[Individual] = []
        scenario_origins: List[str] = []

        if any(instruction_tuple):
            p_scenario1_count = min(max(1, int(cfg.p_scenario1 * cfg.population_size)), n_pop)
            p_scenario2_count = min(max(1, int(cfg.p_scenario2 * cfg.population_size)), n_pop)
            p_scenario3_count = min(max(1, int(cfg.p_scenario3 * cfg.population_size)), n_pop)

            if instruction_tuple[0] and p_scenario1_count >= 2 and ncrossover_scenario > 0:
                _, _, dominant_individual, _ = analyze_perm(
                    population[:p_scenario1_count], cfg, model, rng
                )
                for _ in range(ncrossover_scenario):
                    idx = roulette_wheel_selection(probabilities, rng)
                    parents = (dominant_individual, population[idx])
                    child_perm1, child_perm2 = crossover(parents, rng)
                    for perm in (child_perm1, child_perm2):
                        child = evaluate_permutation(perm, model)
                        if math.isfinite(child.cost):
                            scenario_candidates.append(child)
                            scenario_origins.append("scenario")

            if instruction_tuple[1] and p_scenario2_count >= 1 and nmutate_scenario > 0:
                _, mask_matrix, _, _ = analyze_perm(
                    population[:p_scenario2_count], cfg, model, rng
                )
                mask_slice = mask_matrix[:p_scenario2_count]
                for _ in range(nmutate_scenario):
                    ii = int(rng.integers(0, p_scenario2_count))
                    mutated_perm = mask_mutation(
                        cfg.mask_mutation_index,
                        population[ii].permutation,
                        mask_slice[ii],
                        model,
                        rng,
                    )
                    child = evaluate_permutation(mutated_perm, model)
                    if math.isfinite(child.cost):
                        scenario_candidates.append(child)
                        scenario_origins.append("scenario")

            if instruction_tuple[2] and p_scenario3_count >= 1 and nmutate_scenario > 0:
                _, _, dominant_individual, dominant_mask = analyze_perm(
                    population[:p_scenario3_count], cfg, model, rng
                )
                tail_indices = np.arange(max(0, n_pop - p_scenario3_count), n_pop)
                for _ in range(nmutate_scenario):
                    jj = int(rng.choice(tail_indices))
                    combined_perm = combine_q(
                        dominant_individual.permutation, population[jj].permutation, dominant_mask
                    )
                    child = evaluate_permutation(combined_perm, model)
                    if math.isfinite(child.cost):
                        scenario_candidates.append(child)
                        scenario_origins.append("scenario")

        pool = list(zip(population, ["previous"] * len(population)))
        pool.extend(zip(offspring, crossover_origins))
        pool.extend(zip(mutations, mutation_origins))
        pool.extend(zip(scenario_candidates, scenario_origins))

        pool.sort(key=lambda item: item[0].cost)
        if getattr(cfg, "deduplicate", False):
            population, top_origins = _select_population_dedupe(
                pool, cfg.population_size, model, rng,
            )
        else:
            population = [ind for ind, _ in pool[: cfg.population_size]]
            top_origins = [origin for _, origin in pool[: cfg.population_size]]

        total = len(top_origins)
        n_mut = top_origins.count("mutation") + top_origins.count("fill")
        stats.contribution_rate.append(
            (
                top_origins.count("previous") / total if total else 0.0,
                top_origins.count("crossover") / total if total else 0.0,
                n_mut / total if total else 0.0,
                top_origins.count("scenario") / total if total else 0.0,
            )
        )

        population.sort(key=lambda ind: ind.cost)
        worst_cost = max(worst_cost, population[-1].cost)

        if population[0].cost < best_solution.cost:
            best_solution = population[0]

        stats.best_cost_trace.append(best_solution.cost)

        if cfg.time_limit is not None and (time.perf_counter() - start_time) >= cfg.time_limit:
            break

    elapsed = time.perf_counter() - start_time
    return AlgorithmResult(
        best_cost=best_solution.cost,
        best_individual=best_solution,
        population=population,
        stats=stats,
        elapsed_time=elapsed,
    )

