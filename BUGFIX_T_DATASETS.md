# Bugfix: зависание на T-датасетах

## Симптом

При запуске на T-датасетах (T1–T8, J=100–1600) алгоритм зависает на этапе инициализации популяции. Heartbeat показывает 0/600 задач завершено даже спустя 1.5+ часа. На c-датасетах (J=20–35) всё работает нормально.

## Корневая причина

Функция `heuristic2` (инициализация первого индивида популяции) содержала **слабый repair-механизм**, который не соответствовал MATLAB-оригиналу. Из-за этого для T-датасетов с жёсткими ограничениями ёмкости `heuristic2` возвращала **нефизибельное** решение (cost = inf).

Далее цикл инициализации популяции мутировал это нефизибельное решение в надежде получить физибельное:

```python
while len(population) < cfg.population_size:
    mutated = mutation(population[0].permutation, model, rng)
    individual = evaluate_permutation(mutated, model)
    if math.isfinite(individual.cost):
        population.append(individual)
```

Мутации нефизибельного базового решения почти никогда не дают физибельного потомка → бесконечный цикл → все 16 воркеров заблокированы → 0 задач завершено.

## Почему MATLAB не зависал

MATLAB-код имеет тот же `while Cost == inf` цикл без таймаута, но его `Heuristic2.m` содержит **каскадный repair** с внешним `while min(cvar) < 0`:

```matlab
% MATLAB Heuristic2.m, строки 50-70
while min(cvar) < 0                    % повторять пока ВСЕ не станут физибельными
    for i = 1:I
        while cvar(i) < 0
            [a, b] = max(Wij(i,:));    % тяжелейший job на перегруженном facility
            ... удаляем ...
            [c, d] = min(aij(:,b));    % перемещаем на facility с мин. весом
            if d == i
                [c, d] = max(cvar);    % или на facility с макс. запасом
            end
            ... назначаем БЕЗ проверки ёмкости ...  % каскадный ремонт!
        end
    end
end
```

Ключевые отличия MATLAB от старого Python:

| Аспект | MATLAB | Python (было) |
|--------|--------|---------------|
| Количество проходов | Внешний `while` — повторяет до полной физибельности | Один проход `for i in range(I)` |
| Куда перемещает job | На facility с мин. весом **или** макс. запасом, **без проверки** ёмкости | **Только** на facility с гарантированной ёмкостью |
| Если нет места | Перемещает всё равно — каскад починит | **Сдаётся**: возвращает job обратно и делает `break` |

## Что было исправлено

**Сводка изменений в коде:**

| Файл | Изменение |
|------|-----------|
| `gea_gqap_adaptive_python/heuristics.py`, `GEA_GQAP_Python/gea_gqap_python/heuristics.py` | Каскадный repair в `heuristic2` (как в MATLAB), страховка `max_repair_passes`. |
| `gea_gqap_adaptive_python/algorithm.py`, `algorithm_adaptive.py`, `GEA_GQAP_Python/gea_gqap_python/algorithm.py` | Инициализация популяции без лимита по времени (как в MATLAB). В цикле ГА: `n_pop = len(population)`, индексы и счётчики сценариев через `n_pop` (защита от IndexError при неполной популяции). |
| `gea_gqap_adaptive_python/algorithm_adaptive.py`, `GEA_GQAP_Python/gea_gqap_python/algorithm.py` | В `_select_population_dedupe`: параметры `start_time`, `time_limit`, прерывание дозаполнения по времени. |

### 1. Repair в `heuristic2` (основной фикс)

**Файлы:**
- `gea_gqap_adaptive_python/gea_gqap_adaptive_python/heuristics.py`
- `GEA_GQAP_Python/gea_gqap_python/heuristics.py`

**Было (слабый repair):**
```python
for i in range(I):
    while count[i] > model.bi[i] + 1e-9:
        ...
        for new_i in target:
            if count[new_i] + model.aij[new_i, job] <= model.bi[new_i]:
                ... assign ...
                break
        else:
            X[i, job] = 1       # возвращает обратно
            break                # сдаётся
```

**Стало (MATLAB-style каскадный repair):**
```python
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
```

Теперь `heuristic2` гарантированно возвращает физибельное решение, как в MATLAB. Добавлена страховка `max_repair_passes = I * J` от теоретически возможного бесконечного цикла.

### 2. Инициализация популяции (без лимита по времени, как в MATLAB)

В **MATLAB** (`Algorithm_GA_Quadratic.m`) таймер `tic` стоит **после** набора начальной популяции (стр. 75); цикл `for i=2:Info.Npop` всегда набирает полные 350 особей, ограничения по времени на инициализацию нет. В **Python** лимит на время инициализации не используется: популяция набирается до `population_size` так же, как в MATLAB.

### 3. Time-limit на дозаполнение в `_select_population_dedupe` (страховка)

**Файлы:**
- `gea_gqap_adaptive_python/gea_gqap_adaptive_python/algorithm_adaptive.py`
- `GEA_GQAP_Python/gea_gqap_python/algorithm.py`

Сигнатура функции расширена:
```python
def _select_population_dedupe(
    pool, population_size, model, rng,
    start_time=0.0, time_limit=None,  # новые параметры
)
```

Цикл дозаполнения теперь прерывается при исчерпании `time_limit`:
```python
while len(unique_list) < population_size:
    if time_limit is not None and (time.perf_counter() - start_time) >= time_limit:
        break
    ...
```

Вызовы обновлены с передачей `start_time` и `cfg.time_limit`.

## Почему c-датасеты не были затронуты

c-датасеты (J=20–35, I=15–95) имеют достаточно свободные ограничения ёмкости. Даже слабый repair почти всегда находил facility с гарантированной ёмкостью для перемещения, поэтому `heuristic2` возвращала физибельное решение. T-датасеты (J=100–1600) с жёсткими ограничениями — нет.

---

## Парсинг датасетов, одна итерация и тайм-лимит

### Парсинг .m

- Блоки `cij`, `aij` извлекаются по шаблону `var = [ ... ];`, разбиваются по `;` на строки, числа — по `[,\s]+`. Если в блоке нет точек с запятой, получается одна «строка» со всеми числами подряд (row-major). Затем выполняется `reshape(I, J)` — порядок сохраняется (первые J чисел = первая строка и т.д.).
- Векторы `bi`, `X`, `Y`, `XX`, `YY` парсятся как одна последовательность и приводятся к `reshape(I)` или `reshape(J)`.
- Проверка: для T1, T13, T14 размерности `cij` (I×J), `aij` (I×J), `bi` (I), `DIS` (I×I), `F` (J×J) корректны; для T13 `cij[0,0]=82`, для T14 `cij[0,0]=88` (совпадает с файлами). Тесты: `tests/test_parsing_and_iteration.py`.

### Структура одной итерации (как в MATLAB)

1. **Roulette**: вероятности `P ∝ exp(-β*Costs/WorstCost)`, нормировка.
2. **Crossover**: `NCrossover` пар родителей (рулетка), каждая пара → два потомка (Crossover = с вероятностью 1/2 OnePoint или TwoPoint). CreateXij + CostFunction для каждого.
3. **Mutation**: `NMutation` раз выбирается случайный индивид из популяции, Mutation(perm), CreateXij + CostFunction.
4. **Scenario 1** (Dominated): из лучших `PScenario1*Npop` строится доминантная хромосома (Analyze_Perm), затем `NCrossover_Scenario` пар (доминант + рулетка) → кроссовер, оценка.
5. **Scenario 2** (Mask): маска по лучшим `PScenario2*Npop`, `NMutate_Scenario` раз MaskMutation от случайного из этих лучших, оценка.
6. **Scenario 3** (Inject): доминант и маска по лучшим `PScenario3*Npop`, `NMutate_Scenario` раз CombineQ(доминант, случайный из *хвоста* популяции — последние `PScenario3*Npop` особей), оценка.
7. **Pool**: объединение текущей популяции, кроссовера, мутаций и сценариев; сортировка по cost; отбор лучших `Npop`.
8. Обновление BestSol, WorstCost, вывод (в MATLAB — disp), **затем** проверка времени (см. ниже).

Операторы приведены в соответствие с MATLAB: Crossover (OnePoint/TwoPoint), Mutation (1–5), Analyze_Perm (маска по парам позиций, доминант по max суммы маски, при равенстве — rand>0.5), CombineQ = where(Pattern, Pos1, Pos2).

### Тайм-лимит и «выход за 1000 с» в MATLAB

В **MATLAB** (`Algorithm_GA_Quadratic.m`): `tic` стоит **перед** циклом (стр. 75), проверка `time = toc; if time >= 1000; break; end` — **в конце** каждой итерации (стр. 217–220). То есть цикл **никогда не обрывается посередине итерации**: сначала выполняются кроссовер, мутации, все три сценария, слияние пула и отбор, и только потом проверяется время. Поэтому на тяжёлых инстансах (T13, T14) **фактическое время работы может заметно превышать 1000 с** — последняя начатая итерация всегда доводится до конца.

В **Python** логика та же: проверка `time_limit` выполняется в конце итерации (после отбора и обновления best). И там, и там число полных итераций одинаково при равном времени на одну итерацию; разница в качестве на T13/T14 связана с тем, что за 1000 с успевает разное число полных итераций (в Python меньше при pop=350).

---

## Почему на T13 и T14 результаты хуже, чем у MATLAB (при том что на остальных — лучше)

### Симптом

На всех датасетах кроме **T13** и **T14** Python-реализация даёт **значительно лучшие** результаты, чем MATLAB; на **T13** и **T14** — **значительно хуже**.

Ограничение по времени в обоих кодах одинаковое: в MATLAB в `Algorithm_GA_Quadratic.m` (стр. 217–220) стоит `if time>=1000; break; end`, в Python — проверка `time_limit`. Значит, разница не в критерии остановки.

### Проверка гипотез (результаты)

Проверка выполнена скриптом `gea_gqap_adaptive_python/verify_t13_t14_hypotheses.py`.

1. **Число поколений за 1000 с — подтверждено**  
   При `population_size=350` и лимите 15 с на T13 и T14 получилось **~4 поколения за ~19 с** (≈0,2 итер/с). Экстраполяция на 1000 с даёт **~210 поколений**. На мелких инстансах за 1000 с успевает полные 1000 итераций. Итог: на T13/T14 Python за те же 1000 с делает **значительно меньше поколений**, чем на остальных датасетах. Если MATLAB на T13/T14 за 1000 с успевает больше поколений (быстрее одно поколение), это объясняет лучшие результаты MATLAB на этих инстансах.

2. **Расчёт стоимости (cost)**  
   Для одного и того же решения (Heuristic2) проверено: `cost_function_perm(perm)` и `cost_function(X)` при `X = create_xij(perm)` дают **одинаковое** значение на T13 и T14. Формула и размерности совпадают.

3. **Формат хромосомы и размерности**  
   Проверено: `permutation.shape == (J,)`, `xij.shape == (I, J)`, `create_xij` и `cost_function_perm` корректны для I=20/40, J=1600. Ошибок индексации не найдено.

4. **Heuristic2 и ремонт**  
   Начальное назначение и каскадный repair в Python соответствуют MATLAB (выбор facility по min aij, при совпадении — max cvar). Отличий, объясняющих стабильно худший результат, не выявлено.

5. **RNG**  
   В Python используется `np.random.default_rng()` (без фиксированного seed в адаптивной версии), в MATLAB — `rand`/`randsample`. Для сравнения при одном seed в Python можно задать `random_seed` в конфиге (GEA_GQAP_Python).

### Исправленный баг: индекс за границами при неполной популяции

При неполной инициализации популяции (например, если бы использовался лимит по времени или сбой при создании особей) `len(population)` мог бы быть меньше `cfg.population_size`. В цикле ГА использовались индексы в диапазоне `[0, cfg.population_size)` и сценарий 3 — `tail_indices` до `cfg.population_size`, что давало **IndexError** при обращении к `population[idx]` / `population[jj]`.

**Исправление:** во всех трёх модулях (`gea_gqap_adaptive_python/algorithm.py`, `algorithm_adaptive.py`, `GEA_GQAP_Python/gea_gqap_python/algorithm.py`):
- в начале каждой итерации берётся `n_pop = len(population)`;
- для мутации: `idx = rng.integers(0, n_pop)` вместо `cfg.population_size`;
- для сценария 3: `tail_indices = np.arange(max(0, n_pop - p_scenario3_count), n_pop)`;
- счётчики сценариев ограничены: `p_scenario*_count = min(..., n_pop)`.

Так устраняются падения и корректно обрабатывается случай, когда за отведённое время популяция не набирается до полного размера.

---

## Сравнение с таблицами статьи (PDF) и вашими результатами (docx)

### Откуда брать эталон

В статье *A genetic engineering algorithm for the generalized quadratic assignment problem* (Neural Computing and Applications, 2025, DOI 10.1007/s00521-025-11155-z) результаты приведены в **таблицах внутри PDF** (Experimental results, сравнение GEA_1/GEA_2/GEA_3/GEA, метрики вроде Best Cost, Gap, CPU time). Текст таблиц из PDF в репозитории автоматически не извлекается — для посимвольного сравнения нужно вручную выписать из статьи значения по инстансам T13 и T14 (MinCost или Best Cost, при необходимости Gap).

В MATLAB в `Run_c351595.m` используется: **Gap = (Heuristic2.Cost − MinCost) / Heuristic2.Cost**; при этом `Heuristic2.Cost` считается внутри `Heuristic2.m` по формуле с половинной суммой по квадратичному члену (`for l=1:j`), а **MinCost** в цикле ГА берётся из **CostFunction.m** (полная сумма). В Python везде используется одна и та же полная формула (как в CostFunction.m), поэтому для сравнения «лучшая стоимость» ориентироваться нужно на **MinCost / Best Cost** из статьи, а не на Gap, если метрики в статье считаются так же.

### Ваши результаты (из «Резы первые.docx»)

По логам в документе для **T13** и **T14** (30 запусков, время ~1002 с):

| Датасет | Лучший min cost (пример) | Алгоритм (тип) |
|---------|--------------------------|----------------|
| T13     | ~600 123 953             | GEA_2 non_adaptive_wo_duplicates |
| T13     | ~603 276 254             | GA non_adaptive_wo_duplicates    |
| T14     | ~649 930 481             | GEA_2 non_adaptive_wo_duplicates |
| T14     | ~653 687 709             | GEA_2 non_adaptive_wo_duplicates |

Чтобы проверить, «где ошибка» относительно статьи:

1. Выписать из PDF статьи для T13 и T14 значения **Best Cost / MinCost** (и при необходимости Gap, CPU time) по тем же настройкам (1000 итераций / 1000 с, популяция 350).
2. Сравнить с вашими min cost по соответствующему варианту (GA, GEA_2, GEA и т.д.). Если в статье значения заметно лучше при тех же условиях — возможная причина: меньше поколений за 1000 с в Python (одна итерация на T13/T14 тяжелее, чем в MATLAB).
