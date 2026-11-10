"""Moderación X × W → Y (#27): interacción, ΔR², pendientes simples y gráfico.

Reference values come from independent statsmodels fits (formula API), not from psicostat code:
- coefficients, SE, t, p and 95% CI: ``smf.ols("y ~ xc * wc")`` with mean-centred X and W
  (equivalent to PROCESS model 1 with ``center = 1`` and to R ``lm(y ~ xc * wc)``);
- ΔR² and its F test: ``anova_lm(reduced, full)`` (nested-model F test);
- simple slopes: the "re-centring" trick (Aiken & West, 1991): refit the model with W − w₀; the
  coefficient of X (and its SE/t/p) is the conditional effect of X at W = w₀. This also holds with
  HC3 standard errors, so it validates the robust path as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from psicostat import Design, analyze, build_report, recommend, sample_size
from psicostat.examples import example
from psicostat.template import columns, make_template


def _ref_df(n=200, seed=2027, hetero=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    w = rng.normal(3, 1.2, n)
    e = rng.normal(0, 1, n) * (np.exp(.04 * (x - 50)) * 6 if hetero else 6)
    y = 20 + 0.4 * x + 1.5 * w + 0.12 * (x - 50) * (w - 3) + e
    return pd.DataFrame({"id": range(n), "x": x, "w": w, "y": y})


def _design(**kw):
    base = dict(analysis="moderation", iv_type="continuous", iv_name="x", moderator="w", dv_name="y")
    return Design(**{**base, **kw})


def _slope_rows(res):
    return [r for r in res.rows if " | " in r["label"]]


def test_recommend_template_and_sample_size():
    d = _design()
    assert recommend(d).test == "moderation"
    assert recommend(d).family == "f2"
    assert [c[0] for c in columns(d)] == ["id", "x", "w", "y"]
    tpl, dic = make_template(d, 10)
    assert list(tpl.columns) == ["id", "x", "w", "y"] and len(dic) == 4
    # G*Power 3.1, linear multiple regression: fixed model, R² increase; 1 tested / 3 total predictors, α = .05, 1 − β = .80
    assert sample_size(_design(effect_label="small"))["n_total"] == 395
    assert sample_size(_design(effect_label="medium"))["n_total"] == 55
    assert sample_size(_design(effect_label="large"))["n_total"] == 25


def test_nominal_template_prefills_both_factors():
    d = _design(iv_type="nominal", iv_name="grupo", levels=["control", "tto"], moderator="sexo",
                moderator_type="nominal", moderator_levels=["h", "m"])
    tpl, _ = make_template(d, 8)
    cells = list(zip(tpl["grupo"], tpl["sexo"]))
    assert {c: cells.count(c) for c in set(cells)} == {("control", "h"): 2, ("tto", "h"): 2, ("control", "m"): 2, ("tto", "m"): 2}


def test_coefficients_and_r2_change_match_statsmodels(tmp_path):
    df = _ref_df()
    res = analyze(_design(), df, str(tmp_path))
    ref = df.assign(xc=df.x - df.x.mean(), wc=df.w - df.w.mean())
    full = smf.ols("y ~ xc * wc", ref).fit()
    reduced = smf.ols("y ~ xc + wc", ref).fit()
    coef = res.extra_tables["coefficients"]
    ci = full.conf_int()
    for i, term in enumerate(["Intercept", "xc", "wc", "xc:wc"]):
        assert coef["b"][i] == pytest.approx(full.params[term], rel=1e-10)
        assert coef["EE/SE"][i] == pytest.approx(full.bse[term], rel=1e-10)
        assert coef["t"][i] == pytest.approx(full.tvalues[term], rel=1e-10)
        assert coef["p"][i] == pytest.approx(full.pvalues[term], rel=1e-8)
        assert coef["IC95_inf/CI95_low"][i] == pytest.approx(ci.loc[term, 0], rel=1e-10)
        assert coef["IC95_sup/CI95_high"][i] == pytest.approx(ci.loc[term, 1], rel=1e-10)
    aov = anova_lm(reduced, full)
    ch = res.extra_tables["r2_change"].iloc[0]
    assert ch["ΔR²"] == pytest.approx(full.rsquared - reduced.rsquared, rel=1e-10)
    assert ch["F"] == pytest.approx(aov["F"][1], rel=1e-10)
    assert ch["p"] == pytest.approx(aov["Pr(>F)"][1], rel=1e-8)
    assert ch["F"] == pytest.approx(full.tvalues["xc:wc"] ** 2, rel=1e-10)   # 1-df F = t²
    assert ch["f²"] == pytest.approx((full.rsquared - reduced.rsquared) / (1 - full.rsquared), rel=1e-10)
    top = res.rows[0]
    assert top["label"] == "x × w" and top["es_symbol"] == "f²"
    assert top["stat"] == pytest.approx(full.tvalues["xc:wc"], rel=1e-10)
    assert top["p"] < .001  # strong simulated interaction


def test_simple_slopes_match_recentred_models(tmp_path):
    df = _ref_df()
    res = analyze(_design(), df, str(tmp_path))
    m, sd = df.w.mean(), df.w.std(ddof=1)
    slopes = _slope_rows(res)
    assert len(slopes) == 3
    for row, w0 in zip(slopes, (m - sd, m, m + sd)):
        ref = smf.ols("y ~ xc * wc", df.assign(xc=df.x - df.x.mean(), wc=df.w - w0)).fit()
        assert row["es"] == pytest.approx(ref.params["xc"], rel=1e-10)
        assert row["stat"] == pytest.approx(ref.tvalues["xc"], rel=1e-9)
        assert row["p"] == pytest.approx(ref.pvalues["xc"], rel=1e-7)
        assert row["ci"][0] == pytest.approx(ref.conf_int().loc["xc", 0], rel=1e-9)
        assert row["ci"][1] == pytest.approx(ref.conf_int().loc["xc", 1], rel=1e-9)
    # positive interaction → the effect of X grows with W
    assert slopes[0]["es"] < slopes[1]["es"] < slopes[2]["es"]
    tab = res.extra_tables["simple_slopes"]
    assert tab["efecto de X / effect of X"].tolist() == pytest.approx([r["es"] for r in slopes], rel=1e-12)


def test_uncentred_matches_raw_formula_and_interaction_is_invariant(tmp_path):
    df = _ref_df()
    raw = smf.ols("y ~ x * w", df).fit()
    res_c = analyze(_design(), df, str(tmp_path / "c"))
    res_u = analyze(_design(center=False), df, str(tmp_path / "u"))
    cu = res_u.extra_tables["coefficients"]
    for i, term in enumerate(["Intercept", "x", "w", "x:w"]):
        assert cu["b"][i] == pytest.approx(raw.params[term], rel=1e-9)
        assert cu["EE/SE"][i] == pytest.approx(raw.bse[term], rel=1e-9)
    # centring changes neither the interaction nor the simple slopes (Hayes, 2022, ch. 9)
    assert res_c.rows[0]["stat"] == pytest.approx(res_u.rows[0]["stat"], rel=1e-9)
    for a, b in zip(_slope_rows(res_c), _slope_rows(res_u)):
        assert a["es"] == pytest.approx(b["es"], rel=1e-9) and a["p"] == pytest.approx(b["p"], rel=1e-7)
    assert any("Sin centrar" in n[0] for n in res_u.notes)


def test_hc3_when_heteroscedastic(tmp_path):
    df = _ref_df(n=300, seed=11, hetero=True)
    res = analyze(_design(), df, str(tmp_path))
    assert any("HC3" in n[0] for n in res.notes), "heteroscedastic sample should trigger HC3"
    ref_full = smf.ols("y ~ xc * wc", df.assign(xc=df.x - df.x.mean(), wc=df.w - df.w.mean())).fit(cov_type="HC3")
    assert res.extra_tables["coefficients"]["EE/SE"][3] == pytest.approx(ref_full.bse["xc:wc"], rel=1e-9)
    m, sd = df.w.mean(), df.w.std(ddof=1)
    for row, w0 in zip(_slope_rows(res), (m - sd, m, m + sd)):
        ref = smf.ols("y ~ xc * wc", df.assign(xc=df.x - df.x.mean(), wc=df.w - w0)).fit(cov_type="HC3")
        assert row["es"] == pytest.approx(ref.params["xc"], rel=1e-10)
        assert row["stat"] == pytest.approx(ref.tvalues["xc"], rel=1e-9)


def test_dichotomous_moderator_and_x(tmp_path):
    rng = np.random.default_rng(5)
    n = 160
    x = rng.normal(0, 1, n)
    g = rng.integers(0, 2, n)
    y = 1 + 0.2 * x + 0.3 * g + 0.6 * x * g + rng.normal(0, 1, n)
    df = pd.DataFrame({"id": range(n), "x": x, "sexo": np.where(g == 1, "mujer", "hombre"), "y": y})
    d = _design(moderator="sexo", moderator_type="nominal", moderator_levels=["hombre", "mujer"])
    res = analyze(d, df, str(tmp_path))
    slopes = _slope_rows(res)
    assert [r["label"].split("= ")[-1] for r in slopes] == ["hombre", "mujer"]
    ref = df.assign(xc=df.x - df.x.mean(), g=g)
    for row, lvl in zip(slopes, (0, 1)):
        fit = smf.ols("y ~ xc * gg", ref.assign(gg=ref.g - lvl)).fit()
        assert row["es"] == pytest.approx(fit.params["xc"], rel=1e-10)
        assert row["p"] == pytest.approx(fit.pvalues["xc"], rel=1e-7)
    # dichotomous X as well (2 × 2 design coded 0/1)
    df2 = df.assign(grupo=np.where(x > 0, "tto", "control"))
    d2 = _design(iv_type="nominal", iv_name="grupo", levels=["control", "tto"], moderator="sexo",
                 moderator_type="nominal", moderator_levels=["hombre", "mujer"])
    res2 = analyze(d2, df2, str(tmp_path / "b"))
    fit2 = smf.ols("y ~ t * m", df2.assign(t=(df2.grupo == "tto") * 1.0, m=g * 1.0)).fit()
    assert res2.rows[0]["stat"] == pytest.approx(fit2.tvalues["t:m"], rel=1e-10)


def test_report_text_and_example_end_to_end(tmp_path):
    d, df = example("moderation")
    assert recommend(d).test == "moderation"
    res = analyze(d, df, str(tmp_path))
    assert any(f.endswith("fig_moderacion.png") for f in res.figures)
    es = " ".join(build_report(res, "es")["results"])
    en = " ".join(build_report(res, "en")["results"])
    assert "La interacción estres × apoyo fue estadísticamente significativa" in es
    assert "moderó la relación" in es and "disminuyó" in es and "ΔR²" in es and "pendiente simple" in es
    assert "moderated the relationship" in en and "simple slope" in en
    methods = build_report(res, "es")["methods"]
    assert any("moderador (W)" in p for _, p in methods)
    assert sample_size(d)["n_total"] == 395


def test_non_significant_interaction_is_worded_cautiously(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    x, w = rng.normal(0, 1, n), rng.normal(0, 1, n)
    df = pd.DataFrame({"id": range(n), "x": x, "w": w, "y": 0.5 * x + 0.3 * w + rng.normal(0, 1, n)})
    res = analyze(_design(), df, str(tmp_path))
    assert res.rows[0]["p"] > .05
    assert any("no se respalda" in n[0] and "not supported" in n[1] for n in res.notes)


def test_errors():
    df = _ref_df(n=20)
    with pytest.raises(ValueError):
        analyze(_design(moderator_type="nominal", moderator_levels=["a", "b"]), df)   # W has > 2 categories
