"""Fiabilidad de escalas (#30): α de Cronbach, ω de McDonald, estadísticos por ítem y N de Bonett (2002).

Reference values were computed with independent software, not with psicostat code:
- α and its Feldt et al. (1987) 95% CI: pingouin 0.7.0 ``pingouin.cronbach_alpha(df, ci=.95)``
  (0.705712, [0.569, 0.809]; pingouin rounds the CI to 3 decimals);
- one-factor ML loadings/uniquenesses: factor_analyzer 0.5.1 ``FactorAnalyzer(n_factors=1, rotation=None,
  method="ml")`` (identical to scikit-learn ``FactorAnalysis`` to 1e-6); ω computed from those loadings;
- sample size: Bonett (2002, Table 1), α = .10 two-sided, power = .95.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psicostat import Design, analyze, build_report, recommend, sample_size
from psicostat.analysis import DataError, cronbach_alpha, feldt_ci, feldt_test, mcdonald_omega, one_factor_ml
from psicostat.examples import example
from psicostat.power import bonett_alpha_n
from psicostat.template import columns, make_template

# 60 participants × 5 Likert items (1–5), simulated once from a one-factor model (seed 2026)
X_REF = np.array([
    [2, 1, 1, 1, 2], [4, 3, 3, 2, 3], [1, 1, 4, 1, 3], [5, 4, 3, 5, 3], [5, 3, 4, 4, 4], [2, 2, 3, 2, 4],
    [3, 1, 1, 1, 3], [5, 3, 3, 2, 2], [1, 3, 4, 2, 4], [3, 3, 3, 3, 4], [3, 2, 4, 3, 3], [5, 2, 4, 4, 4],
    [4, 3, 2, 4, 4], [4, 3, 4, 3, 2], [2, 2, 4, 2, 3], [4, 3, 4, 2, 1], [4, 1, 3, 3, 2], [5, 5, 4, 3, 3],
    [2, 3, 3, 2, 4], [1, 1, 1, 3, 3], [3, 5, 4, 2, 5], [3, 4, 4, 2, 5], [3, 3, 2, 4, 2], [2, 4, 2, 3, 5],
    [4, 4, 4, 3, 4], [5, 5, 4, 1, 4], [2, 2, 3, 3, 3], [3, 3, 4, 4, 4], [2, 2, 2, 2, 3], [4, 3, 4, 4, 3],
    [2, 2, 3, 1, 3], [4, 4, 3, 2, 2], [3, 3, 3, 3, 4], [4, 3, 3, 3, 3], [2, 4, 4, 4, 5], [3, 4, 4, 4, 5],
    [4, 3, 3, 2, 2], [3, 3, 1, 3, 3], [4, 3, 3, 3, 3], [3, 4, 5, 4, 2], [3, 2, 1, 4, 3], [2, 1, 1, 3, 3],
    [3, 4, 2, 3, 2], [4, 4, 5, 3, 5], [5, 4, 4, 2, 5], [4, 4, 4, 4, 3], [5, 4, 5, 4, 5], [2, 3, 5, 3, 3],
    [5, 4, 5, 5, 4], [5, 5, 4, 4, 3], [4, 5, 3, 4, 4], [3, 3, 4, 4, 3], [2, 2, 3, 2, 2], [4, 3, 4, 3, 5],
    [2, 3, 3, 2, 1], [3, 4, 2, 2, 1], [2, 3, 3, 3, 3], [3, 4, 3, 1, 4], [3, 5, 5, 5, 5], [4, 2, 3, 3, 2],
], float)
ITEMS = ["a", "b", "c", "d", "e"]


def _df(X=X_REF, items=ITEMS):
    df = pd.DataFrame(np.asarray(X), columns=items)
    df.insert(0, "id", [f"P{i + 1:03d}" for i in range(len(df))])
    return df


def _design(**kw):
    return Design(**{**dict(analysis="reliability", dv_name="escala", items=ITEMS), **kw})


def test_alpha_and_feldt_ci_match_pingouin():
    a = cronbach_alpha(X_REF)
    assert a == pytest.approx(0.7057115879138541, abs=1e-12)
    lo, hi = feldt_ci(a, 60, 5)
    assert lo == pytest.approx(0.569, abs=5e-4) and hi == pytest.approx(0.809, abs=5e-4)


def test_feldt_test_is_dual_to_the_ci():
    a = cronbach_alpha(X_REF)
    lo, hi = feldt_ci(a, 60, 5)
    for c in (0.40, 0.55, lo - 1e-3, lo + 1e-3, 0.70, hi - 1e-3, hi + 1e-3, 0.90):
        f, df1, df2, p = feldt_test(a, 60, 5, c)
        assert (df1, df2) == (59, 236)
        assert f == pytest.approx((1 - c) / (1 - a))
        assert (p < .05) == (not lo <= c <= hi)


def test_one_factor_loadings_and_omega_match_factor_analyzer():
    lam, psi = one_factor_ml(X_REF)
    ref_lam = np.array([0.55824009, 0.78695524, 0.63175185, 0.44890099, 0.43663580])
    ref_psi = np.array([0.68836801, 0.38070146, 0.60088960, 0.79848790, 0.80934918])
    np.testing.assert_allclose(lam, ref_lam, atol=2e-5)
    np.testing.assert_allclose(psi, ref_psi, atol=2e-5)
    assert mcdonald_omega(X_REF) == pytest.approx(0.7158829831496799, abs=2e-6)


@pytest.mark.parametrize("k, table", [(2, (266, 93, 38, 25)), (5, (167, 59, 25, 17)), (10, (149, 53, 22, 15)), (100, (136, 48, 21, 14))])
def test_bonett_sample_size_table1(k, table):
    # δ = (1 − α0)/(1 − α1) = 1.5, 2, 3, 4 with α0 = .50; Bonett printed Eq. 5 with z = 1.645 (k = 100, δ = 1.5: 136; exact z gives 135)
    for delta, n_ref in zip((1.5, 2, 3, 4), table):
        n = bonett_alpha_n(k, 0.5, 1 - 0.5 / delta, a=.10, power=.95)
        assert abs(n - n_ref) <= 1
        if (k, delta) != (100, 1.5):
            assert n == n_ref


def test_sample_size_from_design():
    ss = sample_size(_design(effect_label="medium"))         # α0 = .70 vs expected α = .85, k = 5
    assert ss["family"] == "rel" and ss["effect"] == pytest.approx(.85)
    assert ss["n_total"] == bonett_alpha_n(5, .70, .85) == 43
    assert sample_size(_design(effect_label="large"))["n_total"] < ss["n_total"]
    with pytest.raises(ValueError):
        sample_size(_design(effect_size=0.70))


def test_recommend_and_template():
    d = _design(reverse_items=["c"], scale_min=1, scale_max=7)
    assert recommend(d).test == "reliability" and recommend(d).family == "rel"
    cols = columns(d)
    assert [c[0] for c in cols] == ["id"] + ITEMS
    assert cols[3][4] == "1–7" and "inverso" in cols[3][1]
    tpl, dic = make_template(d, 10)
    assert list(tpl.columns) == ["id"] + ITEMS and len(tpl) == 10


@pytest.mark.parametrize("kw", [dict(items=["a", "b"]), dict(reverse_items=["z"]), dict(scale_min=5, scale_max=1),
                                dict(alpha_null=1.0), dict(items=["a", "a", "b"])])
def test_design_validation(kw):
    with pytest.raises(ValueError):
        _design(**kw)


def test_analysis_rows_tables_and_report():
    res = analyze(_design(), _df())
    r_a, r_o = res.rows
    assert r_a["es"] == pytest.approx(0.7057115879138541) and r_a["stat_symbol"] == "F" and r_a["df"] == "59, 236"
    assert r_o["es"] == pytest.approx(0.71588, abs=1e-4)
    lo, hi = r_o["ci"]
    assert lo < r_o["es"] < hi
    items = res.extra_tables["items"]
    # item-rest correlation and α-if-deleted recomputed directly
    tot = X_REF.sum(axis=1)
    assert items["r ítem-resto/item-rest"].iloc[1] == pytest.approx(np.corrcoef(X_REF[:, 1], tot - X_REF[:, 1])[0, 1])
    assert items["α si se elimina/if deleted"].iloc[0] == pytest.approx(cronbach_alpha(X_REF[:, 1:]))
    rel = res.extra_tables["reliability"].iloc[0]
    rbar = np.corrcoef(X_REF, rowvar=False)[np.triu_indices(5, 1)].mean()
    assert rel["α estandarizado/standardized"] == pytest.approx(5 * rbar / (1 + 4 * rbar))
    es = " ".join(build_report(res, "es")["results"])
    en = " ".join(build_report(res, "en")["results"])
    assert "α de Cronbach = .71, IC 95% [.57, .81]" in es and "ω de McDonald = .72" in es and "aceptable" in es
    assert "Cronbach's α = .71, 95% CI [.57, .81]" in en and "did not differ significantly from .70" in en
    assert "estadísticamente significativ" not in es       # no generic p-value boilerplate for reliability
    comma = " ".join(build_report(res, "es", decimal_comma=True)["results"])
    assert "IC 95% [,57; ,81]" in comma
    methods = " ".join(p for _, p in build_report(res, "es")["methods"])
    assert "Bonett (2002)" in methods and "Feldt et al. (1987)" in methods


def test_reverse_keyed_items():
    raw = X_REF.copy()
    raw[:, 2] = 6 - raw[:, 2]                     # item c answered in the opposite direction
    res_rev = analyze(_design(reverse_items=["c"]), _df(raw))
    assert res_rev.rows[0]["es"] == pytest.approx(cronbach_alpha(X_REF))
    res_raw = analyze(_design(), _df(raw))       # forgetting to mark it → lower α and a warning
    assert res_raw.rows[0]["es"] < res_rev.rows[0]["es"]
    assert any("correlacionan negativamente" in n[0] and "c" in n[0] for n in res_raw.notes)
    bad = raw.copy()
    bad[0, 2] = 9
    with pytest.raises(DataError):
        analyze(_design(reverse_items=["c"]), _df(bad))


def test_unidimensionality_check():
    rng = np.random.default_rng(7)
    n = 300
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    lam = .8
    two = np.column_stack([lam * f1 + rng.normal(0, .6, n) for _ in range(3)] + [lam * f2 + rng.normal(0, .6, n) for _ in range(3)])
    one = np.column_stack([lam * f1 + rng.normal(0, .6, n) for _ in range(6)])
    items = [f"i{j}" for j in range(6)]
    ok = analyze(_design(items=items), _df(one, items)).assumptions[0]
    bad = analyze(_design(items=items), _df(two, items)).assumptions[0]
    assert ok["key"] == bad["key"] == "a_unidimensionality"
    assert ok["ok"] and not bad["ok"] and "RMSEA" in bad["detail"]
    three = analyze(_design(items=items[:3]), _df(one[:, :3], items[:3])).assumptions[0]
    assert three["p"] is None and "saturado" in three["detail"]


def test_too_few_cases_and_constant_item():
    with pytest.raises(DataError):
        analyze(_design(), _df(X_REF[:8]))
    X = X_REF.copy()
    X[:, 4] = 3
    with pytest.raises(DataError):
        analyze(_design(), _df(X))


def test_example_dataset():
    d, df = example("reliability")
    res = analyze(d, df)
    assert d.reverse_items == ["bien3", "bien6"]
    assert res.rows[0]["es"] > .75 and res.rows[1]["es"] > .75
    assert all(res.extra_tables["items"]["r ítem-resto/item-rest"] > .3)
