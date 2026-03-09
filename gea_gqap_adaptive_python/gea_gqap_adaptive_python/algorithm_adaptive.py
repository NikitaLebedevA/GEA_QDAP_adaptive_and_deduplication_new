from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

from .heuristics import heuristic2
from .models import (
    AdaptiveAlgorithmConfig,
    AdaptiveAlgorithmResult,
    AdaptiveAlgorithmStats,
    Individual,
    Model,
)
from .operators import (
    analyze_perm,
    combine_q,
    crossover,
    mask_mutation,
    mutation,
    roulette_wheel_selection,
)
from .utils import evaluate_permutation


def _update_lambda(
    current_lambda: float,
    delta_sum: float,
    alpha: float,
    lambda_min: float,
    lambda_max: float,
) -> float:
    """Обновляет значение лямбды на основе суммы дельта (согласно формуле 2 из статьи)."""
    new_lambda = current_lambda + alpha * delta_sum
    return max(lambda_min, min(lambda_max, new_lambda))


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


def run_adaptive_ga(
    model: Model,
    config: AdaptiveAlgorithmConfig | None = None,
) -> AdaptiveAlgorithmResult:
    """
    Запускает адаптивный генетический алгоритм для GQAP.
    
    Адаптивность реализована через динамическое обновление лямбд для каждого
    оператора на основе их эффективности (дельта улучшения).
    """
    cfg = config or AdaptiveAlgorithmConfig()
    rng = np.random.default_rng()

    start_time = time.perf_counter()

    # Параметры адаптивности
    epsilon = cfg.adaptive_epsilon
    alpha = cfg.adaptive_alpha
    lambda_min = cfg.adaptive_lambda_min
    lambda_max = cfg.adaptive_lambda_max

    # Инициализация адаптивных лямбд
    lambda_crossover = 1.0
    lambda_mutation = 1.0
    lambda_scenario1 = 1.0
    lambda_scenario2 = 1.0
    lambda_scenario3 = 1.0

    # Базовые количества операций
    base_ncrossover = int(2 * round((cfg.crossover_rate * cfg.population_size) / 2))
    base_nmutation = int(math.floor(cfg.mutation_rate * cfg.population_size))
    base_ncrossover_scenario = int(
        math.floor(cfg.scenario_crossover_rate * (cfg.p_scenario3 * cfg.population_size))
    )
    base_nmutate_scenario = int(
        math.floor(cfg.scenario_mutation_rate * (cfg.p_scenario3 * cfg.population_size))
    )
    
    # Общее базовое количество операций (для сохранения постоянной суммы)
    base_total_operations = base_ncrossover + base_nmutation + base_ncrossover_scenario + base_nmutate_scenario

    # Инициализация популяции
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

    stats = AdaptiveAlgorithmStats()

    for iteration in range(cfg.iterations):
        n_pop = len(population)
        if n_pop == 0:
            break
        costs = np.array([ind.cost for ind in population], dtype=float)
        probabilities = np.exp(-beta * costs / worst_cost)
        probabilities /= probabilities.sum()

        # Адаптивные количества операций (пропорциональное распределение)
        # Вычисляем взвешенные базовые значения
        weighted_crossover = base_ncrossover * lambda_crossover
        weighted_mutation = base_nmutation * lambda_mutation
        weighted_scenario1 = base_ncrossover_scenario * lambda_scenario1
        weighted_scenario2 = base_nmutate_scenario * lambda_scenario2
        
        # Вычисляем сумму взвешенных значений
        total_weighted = weighted_crossover + weighted_mutation + weighted_scenario1 + weighted_scenario2
        
        # Нормализуем так, чтобы сумма оставалась равной базовой сумме
        # Это сохраняет общее количество операций постоянным, меняя только пропорции
        if total_weighted > 0:
            normalization_factor = base_total_operations / total_weighted
            ncrossover = int(weighted_crossover * normalization_factor)
            nmutation = int(weighted_mutation * normalization_factor)
            ncrossover_scenario = int(weighted_scenario1 * normalization_factor)
            nmutate_scenario = int(weighted_scenario2 * normalization_factor)
        else:
            # Fallback на базовые значения, если все веса нулевые
            ncrossover = base_ncrossover
            nmutation = base_nmutation
            ncrossover_scenario = base_ncrossover_scenario
            nmutate_scenario = base_nmutate_scenario

        # Счетчики дельта для каждого оператора
        crossover_delta_sum = 0.0
        crossover_count = 0
        mutation_delta_sum = 0.0
        mutation_count = 0
        scenario1_delta_sum = 0.0
        scenario1_count = 0
        scenario2_delta_sum = 0.0
        scenario2_count = 0
        scenario3_delta_sum = 0.0
        scenario3_count = 0

        offspring: List[Individual] = []
        crossover_origins: List[str] = []

        # Кроссовер
        for _ in range(0, ncrossover, 2):
            i1 = roulette_wheel_selection(probabilities, rng)
            i2 = roulette_wheel_selection(probabilities, rng)
            parents = (population[i1], population[i2])
            better_parent_cost = min(population[i1].cost, population[i2].cost)

            child_perm1, child_perm2 = crossover(parents, rng)
            child1 = evaluate_permutation(child_perm1, model)
            child2 = evaluate_permutation(child_perm2, model)

            # Согласно статье: для кроссовера используем минимум потомков как f(Snew)
            valid_children = []
            if math.isfinite(child1.cost):
                offspring.append(child1)
                crossover_origins.append("crossover")
                valid_children.append(child1.cost)
            if math.isfinite(child2.cost):
                offspring.append(child2)
                crossover_origins.append("crossover")
                valid_children.append(child2.cost)

            # Вычисляем один delta для пары потомков (минимум потомков vs минимум родителей)
            if valid_children:
                min_offspring_cost = min(valid_children)
                delta = (better_parent_cost - min_offspring_cost) / (better_parent_cost + epsilon)
                crossover_delta_sum += delta
                crossover_count += 1

        # Мутация
        mutations: List[Individual] = []
        mutation_origins: List[str] = []
        for _ in range(nmutation):
            idx = rng.integers(0, n_pop)
            parent = population[idx]
            parent_cost = parent.cost

            mutated_perm = mutation(population[idx].permutation, model, rng)
            mutated_individual = evaluate_permutation(mutated_perm, model)

            if math.isfinite(mutated_individual.cost):
                mutations.append(mutated_individual)
                mutation_origins.append("mutation")
                delta = (parent_cost - mutated_individual.cost) / (parent_cost + epsilon)
                mutation_delta_sum += delta
                mutation_count += 1

        # Сценарии
        scenario_candidates: List[Individual] = []
        scenario_origins: List[str] = []

        if any(cfg.enable_scenario):
            p_scenario1_count = min(max(1, int(cfg.p_scenario1 * cfg.population_size)), n_pop)
            p_scenario2_count = min(max(1, int(cfg.p_scenario2 * cfg.population_size)), n_pop)
            p_scenario3_count = min(max(1, int(cfg.p_scenario3 * cfg.population_size)), n_pop)

            # Сценарий 1: Кроссовер с доминантной хромосомой
            if cfg.enable_scenario[0] and p_scenario1_count >= 2 and ncrossover_scenario > 0:
                _, _, dominant_individual, _ = analyze_perm(
                    population[:p_scenario1_count], cfg, model, rng
                )
                dominant_cost = dominant_individual.cost

                for _ in range(ncrossover_scenario):
                    idx = roulette_wheel_selection(probabilities, rng)
                    parents = (dominant_individual, population[idx])
                    child_perm1, child_perm2 = crossover(parents, rng)

                    # Согласно статье: для кроссовера используем минимум потомков как f(Snew)
                    valid_children = []
                    for perm in (child_perm1, child_perm2):
                        child = evaluate_permutation(perm, model)
                        if math.isfinite(child.cost):
                            scenario_candidates.append(child)
                            scenario_origins.append("scenario")
                            valid_children.append(child.cost)

                    # Вычисляем один delta для пары потомков (минимум потомков vs доминантный родитель)
                    if valid_children:
                        min_offspring_cost = min(valid_children)
                        delta = (dominant_cost - min_offspring_cost) / (dominant_cost + epsilon)
                        scenario1_delta_sum += delta
                        scenario1_count += 1

            # Сценарий 2: Направленная мутация
            if cfg.enable_scenario[1] and p_scenario2_count >= 1 and nmutate_scenario > 0:
                _, mask_matrix, _, _ = analyze_perm(
                    population[:p_scenario2_count], cfg, model, rng
                )
                mask_slice = mask_matrix[:p_scenario2_count]

                for _ in range(nmutate_scenario):
                    ii = int(rng.integers(0, p_scenario2_count))
                    parent = population[ii]
                    parent_cost = parent.cost

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
                        delta = (parent_cost - child.cost) / (parent_cost + epsilon)
                        scenario2_delta_sum += delta
                        scenario2_count += 1

            # Сценарий 3: Инъекция генов
            if cfg.enable_scenario[2] and p_scenario3_count >= 1 and nmutate_scenario > 0:
                _, _, dominant_individual, dominant_mask = analyze_perm(
                    population[:p_scenario3_count], cfg, model, rng
                )
                tail_indices = np.arange(
                    max(0, n_pop - p_scenario3_count), n_pop
                )

                for _ in range(nmutate_scenario):
                    jj = int(rng.choice(tail_indices))
                    parent = population[jj]
                    parent_cost = parent.cost

                    combined_perm = combine_q(
                        dominant_individual.permutation, population[jj].permutation, dominant_mask
                    )
                    child = evaluate_permutation(combined_perm, model)

                    if math.isfinite(child.cost):
                        scenario_candidates.append(child)
                        scenario_origins.append("scenario")
                        delta = (parent_cost - child.cost) / (parent_cost + epsilon)
                        scenario3_delta_sum += delta
                        scenario3_count += 1

        # Объединение и селекция
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

        # Статистика вклада (fill считаем как mutation)
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

        # Вычисление средних дельта для статистики и обновление лямбд
        # Согласно формуле 2 из статьи: λt+1_i = λt_i + α · (Σj ∆t_j) - используется СУММА
        crossover_delta_avg = crossover_delta_sum / max(crossover_count, 1)
        mutation_delta_avg = mutation_delta_sum / max(mutation_count, 1)
        scenario1_delta_avg = scenario1_delta_sum / max(scenario1_count, 1)
        scenario2_delta_avg = scenario2_delta_sum / max(scenario2_count, 1)
        scenario3_delta_avg = scenario3_delta_sum / max(scenario3_count, 1)

        # Обновление лямбд использует СУММУ дельта (согласно формуле 2)
        lambda_crossover = _update_lambda(
            lambda_crossover, crossover_delta_sum, alpha, lambda_min, lambda_max
        )
        lambda_mutation = _update_lambda(
            lambda_mutation, mutation_delta_sum, alpha, lambda_min, lambda_max
        )
        lambda_scenario1 = _update_lambda(
            lambda_scenario1, scenario1_delta_sum, alpha, lambda_min, lambda_max
        )
        lambda_scenario2 = _update_lambda(
            lambda_scenario2, scenario2_delta_sum, alpha, lambda_min, lambda_max
        )
        lambda_scenario3 = _update_lambda(
            lambda_scenario3, scenario3_delta_sum, alpha, lambda_min, lambda_max
        )

        # Сохранение истории лямбд и дельта
        stats.lambda_history.append(
            (lambda_crossover, lambda_mutation, lambda_scenario1, lambda_scenario2, lambda_scenario3)
        )
        stats.delta_history.append(
            (crossover_delta_avg, mutation_delta_avg, scenario1_delta_avg, scenario2_delta_avg, scenario3_delta_avg)
        )

        if cfg.time_limit is not None and (time.perf_counter() - start_time) >= cfg.time_limit:
            break

    elapsed = time.perf_counter() - start_time
    return AdaptiveAlgorithmResult(
        best_cost=best_solution.cost,
        best_individual=best_solution,
        population=population,
        stats=stats,
        elapsed_time=elapsed,
        adaptive_stats=stats,
    )


def save_results_to_json(
    result: AdaptiveAlgorithmResult,
    model_name: str,
    output_dir: Union[Path, str],
    config: AdaptiveAlgorithmConfig,
) -> Path:
    """
    Сохраняет результаты адаптивного алгоритма в JSON с метриками по каждой итерации.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{model_name}_adaptive_{timestamp}.json"

    data = {
        "model_name": model_name,
        "config": {
            "iterations": config.iterations,
            "population_size": config.population_size,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
            "adaptive_epsilon": config.adaptive_epsilon,
            "adaptive_alpha": config.adaptive_alpha,
            "adaptive_lambda_min": config.adaptive_lambda_min,
            "adaptive_lambda_max": config.adaptive_lambda_max,
        },
        "summary": {
            "best_cost": float(result.best_cost),
            "elapsed_time": float(result.elapsed_time),
            "iterations_completed": len(result.stats.best_cost_trace),
        },
        "iterations": [
            {
                "iteration": i + 1,
                "best_cost": float(cost),
                "contribution_rate": {
                    "previous": float(contrib[0]),
                    "crossover": float(contrib[1]),
                    "mutation": float(contrib[2]),
                    "scenario": float(contrib[3]),
                },
                "lambda_values": {
                    "crossover": float(lambdas[0]),
                    "mutation": float(lambdas[1]),
                    "scenario1": float(lambdas[2]),
                    "scenario2": float(lambdas[3]),
                    "scenario3": float(lambdas[4]),
                },
                "delta_values": {
                    "crossover": float(deltas[0]),
                    "mutation": float(deltas[1]),
                    "scenario1": float(deltas[2]),
                    "scenario2": float(deltas[3]),
                    "scenario3": float(deltas[4]),
                },
            }
            for i, (cost, contrib, lambdas, deltas) in enumerate(
                zip(
                    result.stats.best_cost_trace,
                    result.stats.contribution_rate,
                    result.adaptive_stats.lambda_history,
                    result.adaptive_stats.delta_history,
                )
            )
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path

