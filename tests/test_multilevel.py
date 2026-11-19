"""Multilevel (linear mixed-effects) models, validated against lme4 (Bates et al., 2015) reference values.

Reference fits (R, lme4::lmer) for two classic datasets distributed with lme4:
- Dyestuff (Davies, 1947; 6 batches × 5 yields), random-intercept null model (Bates, 2010, ch. 1):
  REML: intercept 1527.50 (SE 19.38), batch variance 1764.0, residual 2451.3, REML criterion 319.7;
  ML: batch variance 1388.3, residual 2451.3, intercept SE 17.69, logLik −163.66.
- sleepstudy (Belenky et al., 2003; 18 subjects × 10 days; tests/data/sleepstudy.csv):
  Reaction ~ Days + (1 | Subject), REML: 251.4051 (SE 9.7467), 10.4673 (SE 0.8042); variances 1378.2 and 960.5.
  Reaction ~ Days + (Days | Subject), REML: SE 6.8246 and 1.5458; variances 612.10 (intercept), 35.07 (Days),
  correlation .066, residual 654.94; REML criterion 1743.6. ML: variances 565.52 and 32.68, logLik −875.97.
"""
import os

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from psicostat import Design, analyze, build_report, make_template, sample_size
from psicostat.analysis import DataError, fit_mixed, icc_null, lrt_mixture, nakagawa_r2
from psicostat.design import recommend
from psicostat.examples import example
from psicostat.report import methods_text

DYESTUFF = {"A": [1545, 1440, 1440, 1520, 1580], "B": [1540, 1555, 1490, 1560, 1495], "C": [1595, 1550, 1605, 1510, 1560],
            "D": [1445, 1440, 1595, 1465, 1545], "E": [1595, 1630, 1515, 1635, 1625], "F": [1520, 1455, 1450, 1480, 1445]}


def _dyestuff():
    return pd.DataFrame([(b, y) for b, ys in DYESTUFF.items() for y in ys], columns=["batch", "yield_"])


def _sleep():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "sleepstudy.csv"))
    df["Subject"] = df["Subject"].astype(str)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def _sleep_design(**kw):
    return Design(analysis="multilevel", dv_name="Reaction", cluster="Subject", predictors=["Days"], center=False, **kw)


def test_dyestuff_null_model_matches_lme4():
    df = _dyestuff()
    X = np.ones((len(df), 1))
    reml, _ = fit_mixed(df["yield_"], X, df["batch"])
    assert reml.fe_params.iloc[0] == pytest.approx(1527.5, abs=1e-6)
    assert reml.bse_fe.iloc[0] == pytest.approx(19.38, abs=0.01)
    assert reml.cov_re.iloc[0, 0] == pytest.approx(1764.0, rel=1e-3)
    assert reml.scale == pytest.approx(2451.3, rel=1e-3)
    assert -2 * reml.llf == pytest.approx(319.7, abs=0.05)
    assert icc_null(reml) == pytest.approx(1764.05 / (1764.05 + 2451.25), abs=1e-3)   # .418
    ml, _ = fit_mixed(df["yield_"], X, df["batch"], reml=False)
    assert ml.cov_re.iloc[0, 0] == pytest.approx(1388.3, rel=1e-3)
    assert ml.bse_fe.iloc[0] == pytest.approx(17.69, abs=0.01)
    assert ml.llf == pytest.approx(-163.66, abs=0.01)


def test_sleepstudy_random_intercept_matches_lme4():
    res = analyze(_sleep_design(), _sleep())
    fx = res.extra_tables["fixed"].set_index("término/term")
    assert fx.loc["intercepto/intercept", "B"] == pytest.approx(251.4051, abs=1e-3)
    assert fx.loc["intercepto/intercept", "EE/SE"] == pytest.approx(9.7467, abs=2e-3)
    assert fx.loc["Days", "B"] == pytest.approx(10.4673, abs=1e-3)
    assert fx.loc["Days", "EE/SE"] == pytest.approx(0.8042, abs=1e-3)
    assert fx.loc["Days", "nivel/level"] == "1"
    rnd = res.extra_tables["random"]["varianza/variance"].to_numpy()
    assert rnd[0] == pytest.approx(1378.2, rel=1e-3) and rnd[-1] == pytest.approx(960.5, rel=1e-3)
    # Nakagawa & Schielzeth (2013) R² from the lme4 variance components
    var_f = np.var(10.4673 * _sleep()["Days"], ddof=1)
    fit = res.extra_tables["model"].iloc[0]
    assert fit["R²m (marginal)"] == pytest.approx(var_f / (var_f + 1378.18 + 960.46), abs=1e-3)
    assert fit["R²c (condicional/conditional)"] == pytest.approx((var_f + 1378.18) / (var_f + 1378.18 + 960.46), abs=1e-3)
    # Wald z test and CI of the fixed effect
    z = 10.4673 / 0.8042
    assert fx.loc["Days", "z"] == pytest.approx(z, rel=2e-3)
    assert fx.loc["Days", "IC95_inf/CI95_low"] == pytest.approx(10.4673 - 1.959964 * 0.8042, abs=3e-3)
    assert res.n == 180 and res.extra_tables["ml_null"].iloc[0]["J"] == 18


def test_sleepstudy_random_slope_matches_lme4():
    res = analyze(_sleep_design(random_slope="Days"), _sleep())
    fx = res.extra_tables["fixed"].set_index("término/term")
    assert fx.loc["intercepto/intercept", "EE/SE"] == pytest.approx(6.8246, abs=2e-3)
    assert fx.loc["Days", "EE/SE"] == pytest.approx(1.5458, abs=1e-3)
    rnd = res.extra_tables["random"].set_index("componente/component")
    assert rnd.iloc[0]["varianza/variance"] == pytest.approx(612.10, rel=2e-3)
    assert rnd.iloc[1]["varianza/variance"] == pytest.approx(35.07, rel=2e-3)
    assert rnd.iloc[2]["DT/SD"] == pytest.approx(0.066, abs=2e-3)
    assert rnd.iloc[3]["varianza/variance"] == pytest.approx(654.94, rel=1e-3)
    assert res.extra_tables["model"].iloc[0]["−2LL (REML)"] == pytest.approx(1743.6, abs=0.05)
    # ML fit (used for the omnibus test of fixed effects)
    df = _sleep()
    X = sm.add_constant(df[["Days"]])
    ml, _ = fit_mixed(df["Reaction"], X, df["Subject"], X, reml=False)
    assert ml.cov_re.iloc[0, 0] == pytest.approx(565.52, rel=2e-3)
    assert ml.cov_re.iloc[1, 1] == pytest.approx(32.68, rel=2e-3)
    assert ml.llf == pytest.approx(-875.97, abs=0.01)
    # random-slope LRT: REML deviance 1786.47 (random intercept) − 1743.63 (random slope), χ² mixture of 1 and 2 df
    fit = res.extra_tables["model"].iloc[0]
    assert fit["χ² (pendiente aleatoria/random slope)"] == pytest.approx(1786.47 - 1743.63, abs=0.05)
    assert fit["p (pendiente/slope)"] < 1e-8
    assert "La pendiente de Days varió significativamente" in " ".join(n[0] for n in res.notes)


def test_lrt_mixture():
    assert lrt_mixture(0.0, 0) == 1.0
    assert lrt_mixture(3.8415, 0) == pytest.approx(0.025, abs=1e-4)          # ½·χ²(1)
    assert lrt_mixture(5.1385, 1) == pytest.approx(0.05, abs=1e-4)           # ½·χ²(1) + ½·χ²(2) critical value at .05
    assert lrt_mixture(-1e-9, 1) == 1.0


def test_nakagawa_r2_random_slope_uses_johnson_extension():
    df = _sleep()
    X = sm.add_constant(df[["Days"]])
    fit, _ = fit_mixed(df["Reaction"], X, df["Subject"], X)
    r2m, r2c = nakagawa_r2(fit, X, X)
    S = np.asarray(fit.cov_re)
    d = df["Days"].to_numpy()
    var_r = np.mean(S[0, 0] + 2 * S[0, 1] * d + S[1, 1] * d ** 2)
    var_f = np.var(fit.fe_params.iloc[1] * d, ddof=1)
    tot = var_f + var_r + fit.scale
    assert r2m == pytest.approx(var_f / tot) and r2c == pytest.approx((var_f + var_r) / tot)


def test_null_model_lrt_against_ols():
    df = _dyestuff()
    X = np.ones((len(df), 1))
    ml, _ = fit_mixed(df["yield_"], X, df["batch"], reml=False)
    lr = 2 * (ml.llf - sm.OLS(df["yield_"], X).fit().llf)
    assert lr > 0
    assert lrt_mixture(lr, 0) == pytest.approx(0.5 * stats.chi2.sf(lr, 1))


def test_sample_size_design_effect():
    # f² = .15, one tested predictor: G*Power (linear multiple regression, R² increase) gives N = 55
    d = Design(analysis="multilevel", predictors=["x"], cluster="clase", cluster_size=20, icc_plan=0.10)
    ss = sample_size(d)
    assert "N efectivo = 55" in ss["method"][0]
    # DE = 1 + 19 × .10 = 2.9 → 55 × 2.9 = 159.5 → 8 clusters of 20
    assert (ss["n_total"], ss["groups"], ss["n_per_group"]) == (160, 8, 20)
    assert "30 grupos" in ss["notes"][0][0]
    no_icc = sample_size(Design(analysis="multilevel", predictors=["x"], cluster="clase", icc_plan=0.0))
    assert no_icc["n_total"] == 60        # 55 rounded up to 3 whole clusters of 20
    more = sample_size(Design(analysis="multilevel", predictors=["x"], cluster="clase", icc_plan=0.30))
    assert more["n_total"] > ss["n_total"]


def test_design_validation():
    with pytest.raises(ValueError):
        Design(analysis="multilevel", predictors=["x"], random_slope="z")
    with pytest.raises(ValueError):
        Design(analysis="multilevel", predictors=["clase"], cluster="clase")
    with pytest.raises(ValueError):
        Design(analysis="multilevel", predictors=["x"], icc_plan=1.2)
    rec = recommend(Design(analysis="multilevel", predictors=["x"], random_slope="x"))
    assert rec.test == "mixedlm" and rec.family == "f2" and "a_n_clusters" in rec.assumptions
    assert "pendiente aleatoria de x" in rec.notes_text("es")[0]


def test_template_has_cluster_column():
    d = Design(analysis="multilevel", dv_name="bienestar", cluster="centro", predictors=["edad", "apoyo"], random_slope="apoyo", cluster_size=5)
    tpl, dic = make_template(d, 12)
    assert list(tpl.columns) == ["id", "centro", "edad", "apoyo", "bienestar"]
    assert tpl["centro"].tolist()[:6] == ["G01"] * 5 + ["G02"]
    assert "pendiente aleatoria" in dic.loc[dic["columna/column"] == "apoyo", "descripcion"].iloc[0]


def test_example_end_to_end_and_report():
    d, df = example("multilevel")
    res = analyze(d, df)
    fx = res.extra_tables["fixed"].set_index("término/term")
    assert fx.loc["horas_estudio", "nivel/level"] == "1" and fx.loc["tamano_clase", "nivel/level"] == "2"
    assert fx.loc["horas_estudio", "B"] == pytest.approx(2.0, abs=0.5)     # simulated slope
    assert 0.05 < res.extra_tables["ml_null"].iloc[0]["CCI/ICC"] < 0.5
    assert len(res.figures) == 2
    es = " ".join(build_report(res, "es")["results"])
    en = " ".join(build_report(res, "en")["results"])
    assert "CCI = ." in es and "R² marginal" in es and "El predictor horas_estudio" in es
    assert "ICC = ." in en and "conditional R²" in en
    meth = " ".join(p for _, p in methods_text(d, "es"))
    assert "multinivel" in meth and "clase" in meth and "grupos de 20" in meth


def test_conflated_level1_predictor_note_and_errors():
    rng = np.random.default_rng(3)
    J, m = 25, 8
    g = np.repeat(np.arange(J), m)
    xbar = rng.normal(0, 2, J)[g]
    x = xbar + rng.normal(0, 1, J * m)          # most of the variance of x is between clusters
    y = 1 + 0.5 * x + rng.normal(0, 1, J)[g] + rng.normal(0, 1, J * m)
    df = pd.DataFrame({"id": range(J * m), "grupo": [f"g{i}" for i in g], "x": x, "y": y})
    d = Design(analysis="multilevel", dv_name="y", cluster="grupo", predictors=["x"])
    notes = " ".join(n[0] for n in analyze(d, df).notes)
    assert "Enders y Tofighi" in notes and "Solo hay 25 grupos" in notes
    df["w"] = xbar                               # level-2 predictor cannot have a random slope
    with pytest.raises(DataError):
        analyze(Design(analysis="multilevel", dv_name="y", cluster="grupo", predictors=["x", "w"], random_slope="w"), df)
    with pytest.raises(DataError):
        analyze(d, df[df["grupo"].isin(["g0", "g1"])])
