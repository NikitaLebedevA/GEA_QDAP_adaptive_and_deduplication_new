#!/usr/bin/env python3
"""
Тест с 4 типами алгоритмов на каждую модель (GA, GEA_1, GEA_2, GEA_3, GEA):
  adaptive, non_adaptive, adaptive_wo_duplicates, non_adaptive_wo_duplicates.
Параметры по статье: 1000 итераций, популяция 350, лимит времени 1000 с.
Датасеты: все доступные (c и T). По 30 запусков на тип, на каждый датасет.
Параллельное выполнение: запуски распределяются по NUM_WORKERS процессам.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics

import numpy as np

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent.parent))
sys.path.insert(0, str(TEST_DIR.parent.parent.parent / "GEA_GQAP_Python"))

from gea_gqap_adaptive_python import (
    AdaptiveAlgorithmConfig,
    run_adaptive_ga,
    load_model as load_model_adaptive,
    list_available_models,
)
from gea_gqap_python import load_model as load_model_na
from gea_gqap_python.algorithm import run_ga, AlgorithmConfig

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s} с"
    if s < 3600:
        return f"{s // 60} мин {s % 60} с"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} ч {m} мин {sec} с"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            if hasattr(s, "flush"):
                s.flush()
    def flush(self):
        for s in self.streams:
            if hasattr(s, "flush"):
                s.flush()


def load_test_config() -> Dict[str, Any]:
    config_path = TEST_DIR / "test_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Нужен test_config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    variants = cfg.get("model_variants", {})
    cfg["model_variants_tuple"] = {
        name: tuple(bool(x) for x in m.get("enable_scenario", [True, True, True]))
        for name, m in variants.items()
    }
    cfg.setdefault("algorithm_types", [
        "adaptive", "non_adaptive", "adaptive_wo_duplicates", "non_adaptive_wo_duplicates"
    ])
    return cfg


def _get_config():
    if not hasattr(_get_config, "_cfg"):
        _get_config._cfg = load_test_config()
    return _get_config._cfg

CONFIG = _get_config()
MODEL_VARIANTS = CONFIG["model_variants_tuple"]
ALGORITHM_TYPES = CONFIG["algorithm_types"]
ALGORITHM = CONFIG.get("algorithm", {})
NUM_RUNS = CONFIG.get("num_runs", 30)
ITERATIONS = CONFIG.get("iterations", 1000)
POPULATION_SIZE = CONFIG.get("population_size", 350)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _make_adaptive_config(enable_scenario: Tuple[bool, bool, bool], deduplicate: bool):
    alg = ALGORITHM
    return AdaptiveAlgorithmConfig(
        iterations=ITERATIONS,
        population_size=POPULATION_SIZE,
        crossover_rate=alg.get("crossover_rate", 0.7),
        mutation_rate=alg.get("mutation_rate", 0.3),
        scenario_crossover_rate=alg.get("scenario_crossover_rate", 0.5),
        scenario_mutation_rate=alg.get("scenario_mutation_rate", 0.2),
        p_fixed_x=alg.get("p_fixed_x", 0.9),
        p_scenario1=alg.get("p_scenario1", 0.3),
        p_scenario2=alg.get("p_scenario2", 0.3),
        p_scenario3=alg.get("p_scenario3", 0.5),
        mask_mutation_index=alg.get("mask_mutation_index", 2),
        time_limit=float(alg.get("time_limit", 1000.0)),
        enable_scenario=enable_scenario,
        deduplicate=deduplicate,
        adaptive_epsilon=float(alg.get("adaptive_epsilon", 1e-5)),
        adaptive_alpha=float(alg.get("adaptive_alpha", 0.1)),
        adaptive_lambda_min=float(alg.get("adaptive_lambda_min", 0.4)),
        adaptive_lambda_max=float(alg.get("adaptive_lambda_max", 1.5)),
    )


def _make_non_adaptive_config(enable_scenario: Tuple[bool, bool, bool], deduplicate: bool):
    alg = ALGORITHM
    return AlgorithmConfig(
        iterations=ITERATIONS,
        population_size=POPULATION_SIZE,
        crossover_rate=alg.get("crossover_rate", 0.7),
        mutation_rate=alg.get("mutation_rate", 0.3),
        scenario_crossover_rate=alg.get("scenario_crossover_rate", 0.5),
        scenario_mutation_rate=alg.get("scenario_mutation_rate", 0.2),
        p_fixed_x=alg.get("p_fixed_x", 0.9),
        p_scenario1=alg.get("p_scenario1", 0.3),
        p_scenario2=alg.get("p_scenario2", 0.3),
        p_scenario3=alg.get("p_scenario3", 0.5),
        mask_mutation_index=alg.get("mask_mutation_index", 2),
        time_limit=float(alg.get("time_limit", 1000.0)),
        enable_scenario=enable_scenario,
        deduplicate=deduplicate,
    )


def _run_one_adaptive(
    dataset_name: str,
    run_number: int,
    enable_scenario: Tuple[bool, bool, bool],
    deduplicate: bool,
) -> Dict[str, Any]:
    model = load_model_adaptive(dataset_name)
    config = _make_adaptive_config(enable_scenario, deduplicate)
    result = run_adaptive_ga(model, config=config)
    iterations_data = []
    if result.adaptive_stats.lambda_history:
        for i, (cost, contrib, lambdas, deltas) in enumerate(zip(
            result.stats.best_cost_trace,
            result.stats.contribution_rate,
            result.adaptive_stats.lambda_history,
            result.adaptive_stats.delta_history,
        )):
            iterations_data.append({
                "iteration": i + 1,
                "best_cost": float(cost),
                "contribution_rate": {
                    "previous": float(contrib[0]), "crossover": float(contrib[1]),
                    "mutation": float(contrib[2]), "scenario": float(contrib[3]),
                },
                "lambda_values": {
                    "crossover": float(lambdas[0]), "mutation": float(lambdas[1]),
                    "scenario1": float(lambdas[2]), "scenario2": float(lambdas[3]),
                    "scenario3": float(lambdas[4]),
                },
                "delta_values": {
                    "crossover": float(deltas[0]), "mutation": float(deltas[1]),
                    "scenario1": float(deltas[2]), "scenario2": float(deltas[3]),
                    "scenario3": float(deltas[4]),
                },
            })
    return {
        "run_number": run_number,
        "random_seed": int(time.time() * 1000) % (2**31) + run_number * 1000,
        "best_cost": float(result.best_cost),
        "elapsed_time": float(result.elapsed_time),
        "iterations_completed": len(result.stats.best_cost_trace),
        "iterations": iterations_data,
    }


def _run_one_non_adaptive(
    dataset_name: str,
    run_number: int,
    enable_scenario: Tuple[bool, bool, bool],
    deduplicate: bool,
) -> Dict[str, Any]:
    model = load_model_na(dataset_name)
    config = _make_non_adaptive_config(enable_scenario, deduplicate)
    result = run_ga(model, config=config)
    iterations_data = [
        {"iteration": i + 1, "best_cost": float(c)}
        for i, c in enumerate(result.stats.best_cost_trace)
    ]
    return {
        "run_number": run_number,
        "random_seed": int(time.time() * 1000) % (2**31) + run_number * 2000,
        "best_cost": float(result.best_cost),
        "elapsed_time": float(result.elapsed_time),
        "iterations_completed": len(result.stats.best_cost_trace),
        "iterations": iterations_data,
    }


def _worker(task: Tuple) -> Tuple[str, str, int, Dict[str, Any] | None, str | None]:
    """Execute a single algorithm run in a separate process."""
    dataset_name, model_key, algo_type, run_num, enable_scenario = task

    seed = hash((dataset_name, model_key, algo_type, run_num)) & 0x7FFFFFFF
    np.random.seed(seed)

    dedupe = "wo_duplicates" in algo_type
    is_adaptive = "adaptive" in algo_type and "non_adaptive" not in algo_type

    try:
        if is_adaptive:
            result = _run_one_adaptive(dataset_name, run_num, enable_scenario, deduplicate=dedupe)
        else:
            result = _run_one_non_adaptive(dataset_name, run_num, enable_scenario, deduplicate=dedupe)
        return (model_key, algo_type, run_num, result, None)
    except Exception as e:
        return (model_key, algo_type, run_num, None, str(e))


def run_dataset_tests(dataset_name: str, output_dir: Path) -> Dict[str, Any]:
    tasks = []
    for model_key, enable_scenario in MODEL_VARIANTS.items():
        for algo_type in ALGORITHM_TYPES:
            for run_num in range(1, NUM_RUNS + 1):
                tasks.append((dataset_name, model_key, algo_type, run_num, enable_scenario))

    total = len(tasks)
    print(f"  Задач: {total}, воркеров: {NUM_WORKERS}", flush=True)

    results_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    errors = 0
    t_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            model_key, algo_type, run_num, result, error = future.result()
            if error:
                errors += 1
                print(f"  [{_ts()}] ERR [{model_key}] {algo_type} run {run_num}: {error}", flush=True)
            else:
                key = (model_key, algo_type)
                results_map.setdefault(key, []).append(result)
            if done_count % NUM_WORKERS == 0 or done_count == total:
                elapsed = time.monotonic() - t_start
                pct = done_count / total * 100
                print(f"  [{_ts()}] {done_count}/{total} ({pct:.0f}%) | {_fmt_duration(elapsed)}", flush=True)

    ds_elapsed = time.monotonic() - t_start
    print(f"  --- Результаты ({dataset_name}, {_fmt_duration(ds_elapsed)}) ---", flush=True)

    models_results: Dict[str, Dict[str, Any]] = {}
    for model_key in MODEL_VARIANTS:
        models_results[model_key] = {}
        for algo_type in ALGORITHM_TYPES:
            runs = results_map.get((model_key, algo_type), [])
            runs.sort(key=lambda r: r["run_number"])
            best_costs = [r["best_cost"] for r in runs]
            elapsed_times = [r["elapsed_time"] for r in runs]
            stats_bc = calculate_statistics(best_costs)
            stats_et = calculate_statistics(elapsed_times)
            models_results[model_key][algo_type] = {
                "best_cost": stats_bc,
                "elapsed_time": stats_et,
                "runs": runs,
            }
            print(
                f"  [{model_key:5s}] {algo_type:30s} "
                f"{len(runs):2d}/{NUM_RUNS} runs | "
                f"cost: mean={stats_bc['mean']:.1f}, min={stats_bc['min']:.1f} | "
                f"time: mean={stats_et['mean']:.1f} с",
                flush=True,
            )

    if errors:
        print(f"  Ошибок: {errors}", flush=True)

    result = {
        "dataset": dataset_name,
        "num_runs_per_algorithm": NUM_RUNS,
        "iterations": ITERATIONS,
        "population_size": POPULATION_SIZE,
        "time_limit_seconds": float(ALGORITHM.get("time_limit", 1000)),
        "algorithm_types": ALGORITHM_TYPES,
        "models": models_results,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{dataset_name}_results.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Сохранено: {out_file}")
    return result


def create_excel_report(results_dir: Path, output_file: Path):
    import pandas as pd
    results = []
    for json_file in sorted(results_dir.glob("*_results.json")):
        if json_file.name == "summary_all_datasets.json":
            continue
        try:
            with json_file.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Ошибка загрузки {json_file.name}: {e}")
    if not results:
        print("Нет результатов для Excel.")
        return

    model_keys = list(MODEL_VARIANTS.keys())
    type_keys = list(ALGORITHM_TYPES)
    best_cost_rows = []
    elapsed_time_rows = []
    for result in results:
        dataset = result["dataset"]
        models = result.get("models", {})
        row_bc = {"Dataset": dataset}
        row_et = {"Dataset": dataset}
        for mk in model_keys:
            m = models.get(mk, {})
            for tk in type_keys:
                t = m.get(tk, {})
                bc = t.get("best_cost", {})
                et = t.get("elapsed_time", {})
                prefix = f"{mk}_{tk}_"
                row_bc[f"{prefix}min"] = bc.get("min", 0)
                row_bc[f"{prefix}max"] = bc.get("max", 0)
                row_bc[f"{prefix}std"] = bc.get("std", 0)
                row_bc[f"{prefix}median"] = bc.get("median", 0)
                row_bc[f"{prefix}mean"] = bc.get("mean", 0)
                row_et[f"{prefix}min"] = et.get("min", 0)
                row_et[f"{prefix}max"] = et.get("max", 0)
                row_et[f"{prefix}std"] = et.get("std", 0)
                row_et[f"{prefix}median"] = et.get("median", 0)
                row_et[f"{prefix}mean"] = et.get("mean", 0)
        best_cost_rows.append(row_bc)
        elapsed_time_rows.append(row_et)
    df_bc = pd.DataFrame(best_cost_rows)
    df_et = pd.DataFrame(elapsed_time_rows)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_bc.to_excel(writer, sheet_name="best_cost", index=False)
        df_et.to_excel(writer, sheet_name="elapsed_time", index=False)
    print(f"Excel создан: {output_file}")


def main():
    test_dir = TEST_DIR
    results_dir = test_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = test_dir / "test_output.log"
    log_file = open(log_path, "w", encoding="utf-8")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, log_file)
    sys.stderr = _Tee(_orig_stderr, log_file)
    try:
        _main_impl(test_dir, results_dir, log_path)
    finally:
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        log_file.close()


def _main_impl(test_dir: Path, results_dir: Path, log_path: Path):
    all_models = list_available_models()
    c_datasets = sorted(d for d in all_models if d.startswith("c"))
    t_datasets = sorted(d for d in all_models if d.startswith("T"))
    datasets = c_datasets + t_datasets
    time_limit = float(ALGORITHM.get("time_limit", 1000))
    total_tasks = len(datasets) * len(MODEL_VARIANTS) * len(ALGORITHM_TYPES) * NUM_RUNS

    print(f"[{_ts()}] === Старт тестирования ===")
    print(f"[{_ts()}] Лог: {log_path}")
    print(f"[{_ts()}] Типы: {', '.join(ALGORITHM_TYPES)}")
    print(f"[{_ts()}] Модели: {', '.join(MODEL_VARIANTS.keys())}")
    print(f"[{_ts()}] Датасетов: {len(datasets)} (c + T), по {NUM_RUNS} запусков на тип")
    print(f"[{_ts()}] Итераций: {ITERATIONS}, популяция: {POPULATION_SIZE}, лимит времени: {time_limit} с")
    print(f"[{_ts()}] Параллельных воркеров: {NUM_WORKERS}")
    print(f"[{_ts()}] Всего задач: {total_tasks}")

    all_results = []
    t_total_start = time.monotonic()

    for idx, dataset in enumerate(datasets, 1):
        print(f"\n{'#'*70}")
        print(f"[{_ts()}] Датасет {idx}/{len(datasets)}: {dataset}")
        print(f"{'#'*70}")
        try:
            stats = run_dataset_tests(dataset, results_dir)
            all_results.append(stats)
        except Exception as e:
            print(f"[{_ts()}] ОШИБКА на {dataset}: {e}")

    total_elapsed = time.monotonic() - t_total_start

    summary = {
        "test_date": datetime.now().strftime("%Y%m%d"),
        "test_timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "parameters": {
            "num_runs_per_algorithm": NUM_RUNS,
            "iterations": ITERATIONS,
            "population_size": POPULATION_SIZE,
            "time_limit_seconds": time_limit,
            "algorithm_types": ALGORITHM_TYPES,
            "model_variants": list(MODEL_VARIANTS.keys()),
            "num_workers": NUM_WORKERS,
        },
        "datasets": [r["dataset"] for r in all_results],
        "results": all_results,
    }
    summary_file = results_dir / "summary_all_datasets.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[{_ts()}] === Тестирование завершено ===")
    print(f"[{_ts()}] Общее время: {_fmt_duration(total_elapsed)}")
    print(f"[{_ts()}] Сводка: {summary_file}")
    excel_file = test_dir / "comparison_results.xlsx"
    create_excel_report(results_dir, excel_file)
    print(f"[{_ts()}] Лог записан: {log_path}")


if __name__ == "__main__":
    main()
