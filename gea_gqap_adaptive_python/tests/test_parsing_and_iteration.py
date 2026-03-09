"""
Проверка парсинга датасетов и соответствия одной итерации + операторов MATLAB.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gea_gqap_adaptive_python.model_loader import (
    load_model,
    _extract_block,
    _parse_matrix,
    _extract_scalar,
    DATA_DIR,
)


def test_parsing_shapes_and_layout():
    """Парсинг: после reshape(I,J) матрицы имеют правильную форму и порядок (row-major)."""
    for name in ["T1", "T13", "T14"]:
        model = load_model(name)
        I, J = model.I, model.J
        assert model.cij.shape == (I, J), f"{name} cij shape"
        assert model.aij.shape == (I, J), f"{name} aij shape"
        assert model.bi.shape == (I,), f"{name} bi shape"
        assert model.DIS.shape == (I, I), f"{name} DIS shape"
        assert model.F.shape == (J, J), f"{name} F shape"
        assert np.isfinite(model.cij).all() and np.isfinite(model.aij).all()
        # T13/T14: в файле первая строка cij начинается с 82 (T13) или 88 (T14)
        if name == "T13":
            assert model.cij[0, 0] == 82.0, "T13 cij[0,0] from file"
        if name == "T14":
            assert model.cij[0, 0] == 88.0, "T14 cij[0,0] from file"
    print("OK: parsing shapes and sample values (T1, T13, T14)")


def test_parsing_cij_element_count():
    """Блок cij после разбора даёт ровно I*J чисел; reshape(I,J) — row-major."""
    path = DATA_DIR / "T1.m"
    content = path.read_text(encoding="utf-8")
    I = _extract_scalar(content, "I")
    J = _extract_scalar(content, "J")
    block = _extract_block(content, "cij")
    rows = [r.strip() for r in block.strip().split(";") if r.strip()]
    data = []
    for row in rows:
        parts = re.split(r"[,\s]+", row.strip())
        data.append([float(x) for x in parts if x])
    arr = np.array(data)
    flat = arr.flatten()
    assert flat.size == I * J, f"cij has I*J elements: {flat.size} vs {I*J}"
    model = load_model("T1")
    cij_loaded = model.cij
    expected = flat.reshape(I, J)
    np.testing.assert_allclose(cij_loaded, expected)
    print("OK: parsing cij element count and reshape (T1)")


def test_dis_f_formulas():
    """DIS и F в Python: симметрия и неотрицательность (формула как в MATLAB)."""
    model = load_model("T1")
    assert np.allclose(model.DIS, model.DIS.T), "DIS symmetric"
    assert np.all(model.DIS >= -1e-9), "DIS nonnegative"
    assert np.allclose(model.F, model.F.T), "F symmetric"
    assert np.all(model.F >= -1e-9), "F nonnegative"
    print("OK: DIS and F symmetric and nonnegative")


if __name__ == "__main__":
    test_parsing_shapes_and_layout()
    test_parsing_cij_element_count()
    test_dis_f_formulas()
    print("\nParsing and iteration-structure checks done.")
