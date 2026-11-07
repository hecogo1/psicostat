"""Mediación simple X → M → Y (#1): sendas OLS y efecto indirecto con bootstrap percentil.

Reference values: independent OLS fits (equivalent to PROCESS model 4 / lavaan with observed
variables and estimator ML for continuous Y) on a fixed simulated sample (seed 2024, n = 200).
Bootstrap CI: 5000 percentile resamples, seed 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from psicostat import Design, analyze, build_report, recommend, sample_size
from psicostat.analysis import bootstrap_indirect
from psicostat.examples import example
from psicostat.template import columns, make_template


def _ref_df(n=200, seed=2024):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    m = 0.5 * x + rng.normal(0, 1, n)
    y = 0.4 * m + 0.2 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"id": range(n), "x": x, "m": m, "y": y})


def _design(**kw):
    base = dict(analysis="mediation", iv_type="continuous", iv_name="x", mediator="m", dv_name="y")
    return Design(**{**base, **kw})


def _ols(y, *preds):
    X = sm.add_constant(np.column_stack([np.asarray(p, float) for p in preds]))
    return sm.OLS(np.asarray(y, float), X).fit()


def test_recommend_and_template():
    d = _design()
    assert recommend(d).test == "mediation"
    assert [c[0] for c in columns(d)] == ["id", "x", "m", "y"]
    tpl, dic = make_template(d, 10)
    assert list(tpl.columns) == ["id", "x", "m", "y"]
    assert len(dic) == 4


def test_paths_match_independent_ols(tmp_path):
    """Path a, b, c, c′ and a×b match independent OLS (PROCESS model 4 / lavaan observed)."""
    df = _ref_df()
    fa, fc, fymx = _ols(df.m, df.x), _ols(df.y, df.x), _ols(df.y, df.x, df.m)
    a, c = float(fa.params[1]), float(fc.params[1])
    cp, b = float(fymx.params[1]), float(fymx.params[2])
    res = analyze(_design(), df, str(tmp_path))
    got = {r["es_symbol"]: r["es"] for r in res.rows}
    assert got["a"] == pytest.approx(a, abs=1e-12)
    assert got["b"] == pytest.approx(b, abs=1e-12)
    assert got["c"] == pytest.approx(c, abs=1e-12)
    assert got["c′"] == pytest.approx(cp, abs=1e-12)
    assert got["a×b"] == pytest.approx(a * b, abs=1e-12)
    assert c == pytest.approx(cp + a * b, abs=1e-12)  # OLS decomposition identity


def test_bootstrap_ci_matches_reference(tmp_path):
    """Percentile bootstrap 95% CI with seed 0 and 5000 resamples (locked reference)."""
    df = _ref_df()
    ab, lo, hi, samples = bootstrap_indirect(df.x, df.m, df.y, n_boot=5000, seed=0)
    assert ab == pytest.approx(0.14632631415245737, abs=1e-12)
    assert lo == pytest.approx(0.08050076399885157, abs=1e-12)
    assert hi == pytest.approx(0.2246741088160927, abs=1e-12)
    assert lo > 0  # CI excludes 0 in this sample
    res = analyze(_design(), df, str(tmp_path))
    ind = [r for r in res.rows if r["es_symbol"] == "a×b"][0]
    assert ind["es"] == pytest.approx(ab, abs=1e-12)
    assert ind["ci"][0] == pytest.approx(lo, abs=1e-12)
    assert ind["ci"][1] == pytest.approx(hi, abs=1e-12)
    assert any("no incluye el 0" in n[0] or "does not include 0" in n[1] for n in res.notes)


def test_binary_x_and_report(tmp_path):
    rng = np.random.default_rng(7)
    n = 100
    x = rng.integers(0, 2, n)
    m = 0.6 * x + rng.normal(0, 1, n)
    y = 0.5 * m + rng.normal(0, 1, n)
    df = pd.DataFrame({"id": range(n), "grupo": np.where(x == 1, "tratamiento", "control"), "m": m, "y": y})
    d = Design(analysis="mediation", iv_type="nominal", iv_name="grupo", levels=["control", "tratamiento"],
               mediator="m", dv_name="y")
    res = analyze(d, df, str(tmp_path))
    assert res.test == "mediation" and any(r["es_symbol"] == "a×b" for r in res.rows)
    for lang in ("es", "en"):
        rep = build_report(res, lang)
        text = " ".join(rep["results"])
        assert ("indirecto" in text.lower()) or ("indirect" in text.lower())


def test_example_end_to_end(tmp_path):
    d, df = example("mediation")
    assert recommend(d).test == "mediation"
    res = analyze(d, df, str(tmp_path))
    assert res.rows and any(f.endswith("fig_mediacion.png") for f in res.figures)
    assert sample_size(d)["n_total"] == 78
