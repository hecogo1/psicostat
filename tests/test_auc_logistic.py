"""AUC (área bajo la curva ROC) con IC de DeLong para la regresión logística."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from psicostat import Design, analyze, build_report
from psicostat import effects as E
from psicostat.examples import example

# aSAH (pROC): resultado (1 = Poor), sexo (1 = Female) y NDKA de 113 pacientes con hemorragia subaracnoidea.
POOR = "00001101001011001011101000000000100110110000000001100010001010011101101111000110100000010000000000001111110001000"
FEMALE = "11111000110100110110111011011011111011011110100001101111010110111000011101010001111101001011111101010111011100110"
NDKA = [3.01, 8.54, 8.09, 10.42, 17.4, 12.75, 6.0, 13.2, 15.54, 6.01, 15.96, 17.86, 5.18, 8.9, 13.41, 20.75, 11.6, 16.11,
        32.37, 54.82, 32.41, 49.94, 40.34, 9.47, 6.29, 12.53, 6.54, 6.3, 80.3, 12.8, 9.8, 9.81, 9.85, 18.21, 5.03, 14.04, 21.93,
        8.02, 7.42, 8.38, 6.59, 9.63, 13.12, 7.96, 14.34, 41.43, 7.63, 7.06, 12.59, 13.56, 3.87, 9.44, 7.66, 12.98, 419.19,
        27.19, 22.27, 9.95, 21.22, 11.73, 10.4, 58.83, 6.39, 11.09, 12.22, 13.67, 17.21, 22.63, 12.9, 9.7, 21.57, 8.23, 72.57,
        15.54, 10.51, 10.6, 14.57, 5.19, 9.63, 4.61, 21.48, 17.3, 12.71, 9.44, 11.07, 19.46, 10.83, 5.37, 11.97, 7.75, 9.83,
        10.55, 28.49, 8.53, 12.57, 12.9, 46.83, 11.68, 12.67, 9.01, 9.57, 34.06, 11.72, 14.26, 47.61, 11.67, 24.58, 10.33,
        13.87, 15.89, 22.43, 6.79, 13.45]


def _asah():
    return np.array([int(c) for c in POOR]), np.array([int(c) for c in FEMALE]), np.array(NDKA)


def test_auc_delong_matches_proc():
    # valores esperados de los tests de R pROC 1.18.5: ci.auc(roc(outcome, ndka)) (método DeLong)
    y, fem, x = _asah()
    auc, (lo, hi) = E.auc_delong(y, x)
    assert (lo, auc, hi) == pytest.approx((0.501244999271703, 0.611957994579946, 0.722670989888189), abs=1e-10)
    f = fem == 1
    auc, (lo, hi) = E.auc_delong(y[f], x[f])
    assert (lo, auc, hi) == pytest.approx((0.5261398281, 0.6671428571, 0.8081458862), abs=1e-9)


def _delong_bruteforce(y, s):
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    return psi.mean(), np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg))


def test_auc_delong_equals_mann_whitney_and_pairwise_definition():
    rng = np.random.default_rng(3)
    y = (rng.random(80) < .4).astype(int)
    s = np.round(rng.normal(y * .8, 1), 1)  # redondeo para forzar empates
    auc, (lo, hi) = E.auc_delong(y, s)
    u = stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic
    assert auc == pytest.approx(u / ((y == 1).sum() * (y == 0).sum()))
    a2, se = _delong_bruteforce(y, s)
    z = stats.norm.ppf(.975)
    assert auc == pytest.approx(a2)
    assert (lo, hi) == pytest.approx((a2 - z * se, a2 + z * se))


def test_auc_delong_degenerate_and_bounds():
    auc, ci = E.auc_delong([0, 0, 1, 1], [.1, .2, .8, .9])
    assert auc == 1 and ci[1] <= 1
    auc, ci = E.auc_delong([0, 1, 1, 1], [.1, .2, .8, .9])  # un solo no-evento: sin IC
    assert np.isnan(auc) and np.isnan(ci[0])


def test_logistic_report_includes_auc():
    d, df = example("logistic")
    res = analyze(d, df)
    assert res.test == "logreg"
    m = res.extra_tables["model"].iloc[0]
    assert 0.5 < m["AUC_IC95_inf"] < m["AUC"] < m["AUC_IC95_sup"] <= 1
    y = df["abandono"].to_numpy()
    assert m["AUC"] == pytest.approx(E.auc_delong(y, -df["alianza"].to_numpy())[0], abs=.05)
    es = build_report(res, "es")
    assert any("AUC = " in p and "DeLong" in p and "IC 95% [" in p for p in es["results"])
    en = build_report(res, "en")
    assert any("area under the ROC curve" in p for p in en["results"])
    comma = build_report(res, "es", decimal_comma=True)
    p = next(p for p in comma["results"] if "AUC = " in p)
    assert "AUC = ," in p and "; " in p.split("IC 95% [")[1].split("]")[0]


def test_auc_label_thresholds():
    from psicostat.analysis import _auc_label
    assert [_auc_label(v)[1] for v in (.5, .65, .75, .85, .95)] == ["none (no better than chance)", "poor", "acceptable",
                                                                     "excellent", "outstanding"]


def test_logistic_single_predictor_auc_equals_predictor_auc():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 60)
    y = (rng.random(60) < 1 / (1 + np.exp(-1.5 * x))).astype(int)
    d = Design(analysis="regression", dv_type="nominal", iv_type="continuous", predictors=["x"], dv_name="y")
    res = analyze(d, pd.DataFrame({"id": range(60), "x": x, "y": y}))
    # con un único predictor, la AUC del modelo es la del propio predictor (transformación monótona)
    assert res.extra_tables["model"].iloc[0]["AUC"] == pytest.approx(E.auc_delong(y, x)[0])
