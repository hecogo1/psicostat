"""Acuerdo entre evaluadores (#32): κ de Cohen, κ de Fleiss, CCI y tamaño muestral.

Reference values come from independent sources, not from psicostat code:
- Cohen's κ, its asymptotic SE/95% CI and the SE under H₀: κ = 0: SAS PROC FREQ printout reproduced in the statsmodels
  test-suite (``TestUnweightedCohens``: κ = 0.4842, ASE = 0.1380, CI [0.2137, 0.7547], ASE₀ = 0.1484, z = 3.2626, p = .0011);
- quadratic-weighted κ for the same table: scikit-learn ``cohen_kappa_score(weights="quadratic")`` = 0.450161;
- Fleiss' κ: Fleiss (1971) psychiatric diagnoses, 30 patients × 6 raters, κ = .430; R ``irr::kappam.fleiss`` reports z = 17.7;
- ICCs: Shrout & Fleiss (1979, Table 4) data and published values ICC(2,1) = .29, ICC(3,1) = .71, ICC(2,4) = .62, ICC(3,4) = .91;
  95% CIs from pingouin 0.7.0 ``intraclass_corr`` (rounded by pingouin to 2 decimals) and F(5, 15) = 11.03.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.inter_rater import cohens_kappa as sm_kappa

from psicostat import Design, analyze, build_report, recommend, sample_size
from psicostat.analysis import DataError, cohen_kappa, fleiss_kappa_se0, icc_label, icc_test, icc_two_way, kappa_label
from psicostat.examples import example
from psicostat.power import icc_n, icc_power, kappa_avar, kappa_n, kappa_table
from psicostat.template import columns, make_template

TABLE10 = [[0, 4, 1], [0, 8, 0], [0, 1, 5]]          # SAS example (rows = rater 1, columns = rater 2)
SHROUT_FLEISS = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
# Fleiss (1971): diagnoses of 30 patients by 6 psychiatrists (1 = depression, 2 = personality disorder, 3 = schizophrenia,
# 4 = neurosis, 5 = other); as in R irr::diagnoses
DIAGNOSES = np.array([
    [4, 4, 4, 4, 4, 4], [2, 2, 2, 5, 5, 5], [2, 3, 3, 3, 3, 5], [5, 5, 5, 5, 5, 5], [2, 2, 2, 4, 4, 4], [1, 1, 3, 3, 3, 3],
    [3, 3, 3, 3, 5, 5], [1, 1, 3, 3, 3, 4], [1, 1, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [1, 4, 4, 4, 4, 4], [1, 2, 4, 4, 4, 4],
    [2, 2, 2, 3, 3, 3], [1, 4, 4, 4, 4, 4], [2, 2, 4, 4, 4, 5], [3, 3, 3, 3, 3, 5], [1, 1, 1, 4, 5, 5], [1, 1, 1, 1, 1, 2],
    [2, 2, 4, 4, 4, 4], [1, 3, 3, 5, 5, 5], [5, 5, 5, 5, 5, 5], [2, 4, 4, 4, 4, 4], [2, 2, 4, 5, 5, 5], [1, 1, 4, 4, 4, 4],
    [1, 4, 4, 4, 4, 5], [2, 2, 2, 2, 2, 4], [1, 1, 1, 1, 5, 5], [2, 2, 4, 4, 4, 4], [1, 3, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5]])


def _pairs(table, cats=("a", "b", "c")):
    r1, r2 = [], []
    for i, row in enumerate(table):
        for j, cnt in enumerate(row):
            r1 += [cats[i]] * cnt
            r2 += [cats[j]] * cnt
    return pd.DataFrame({"id": [f"C{i:03d}" for i in range(len(r1))], "r1": r1, "r2": r2})


def _kdesign(**kw):
    base = dict(analysis="agreement", dv_type="nominal", dv_name="clase", dv_levels=["a", "b", "c"], raters=["r1", "r2"], agree_null=0)
    return Design(**{**base, **kw})


# ---------------------------------------------------------------- Cohen's κ
def test_cohen_kappa_matches_sas(tmp_path):
    res = analyze(_kdesign(), _pairs(TABLE10), outdir=tmp_path)
    r = res.rows[0]
    assert res.test == "kappa" and r["es_symbol"] == "κ"
    assert r["es"] == pytest.approx(0.4842, abs=5e-5)
    assert r["ci"] == pytest.approx((0.2137, 0.7547), abs=5e-5)
    assert r["stat"] == pytest.approx(3.2626, abs=5e-4)          # z with the SE under H₀ (SAS)
    assert r["p"] == pytest.approx(0.0011, abs=5e-5)
    assert res.extra_tables["agreement"]["EE/SE"].iloc[0] == pytest.approx(0.1380, abs=5e-5)
    assert res.extra_tables["agreement"]["% acuerdo/agreement"].iloc[0] == pytest.approx(100 * 13 / 19)
    assert res.extra_tables["crosstab"].iloc[:, 1:].to_numpy().tolist() == TABLE10
    assert res.figures and res.figures[0].endswith("fig_acuerdo.png")


def test_weighted_kappa_ordinal(tmp_path):
    d = _kdesign(dv_type="ordinal", dv_levels=["a", "b", "c"])
    res = analyze(d, _pairs(TABLE10), outdir=tmp_path)
    assert res.rows[0]["es_symbol"] == "κ_w"
    assert res.rows[0]["es"] == pytest.approx(0.450161, abs=1e-6)       # scikit-learn, quadratic weights
    assert res.rows[1]["es"] == pytest.approx(0.4842, abs=5e-5)         # unweighted κ also reported
    es = " ".join(build_report(res, "es")["results"])
    assert "ponderada cuadrática" in es and "κ sin ponderar" in es


def test_kappa_wald_test_against_minimum(tmp_path):
    res = analyze(_kdesign(agree_null=0.40), _pairs(TABLE10), outdir=tmp_path)
    r = res.rows[0]
    assert r["stat"] == pytest.approx((0.484163 - 0.40) / 0.1380, abs=5e-3)   # Wald z with the non-null SE
    es, en = build_report(res, "es")["results"], build_report(res, "en")["results"]
    assert any("no difirió significativamente de .40" in p for p in es)
    assert any("did not differ significantly from .40" in p for p in en)


def test_kappa_numeric_categories_and_missing(tmp_path):
    rng = np.random.default_rng(5)
    a = rng.integers(1, 4, 60)
    b = np.where(rng.random(60) < .7, a, rng.integers(1, 4, 60)).astype(float)
    b[3] = np.nan                                         # float column with a missing value: 2.0 must match category "2"
    df = pd.DataFrame({"id": range(60), "r1": a, "r2": b})
    res = analyze(_kdesign(dv_levels=["1", "2", "3"]), df, outdir=tmp_path)
    assert res.n == 59 and res.n_excluded == 1
    tab = np.zeros((3, 3))
    for x, y in zip(a[np.isfinite(b)], b[np.isfinite(b)]):
        tab[x - 1, int(y) - 1] += 1
    assert res.rows[0]["es"] == pytest.approx(sm_kappa(tab).kappa)


def test_kappa_unknown_category():
    df = _pairs(TABLE10)
    df.loc[0, "r2"] = "z"
    with pytest.raises(DataError, match="categor"):
        analyze(_kdesign(), df)


def test_kappa_paradox_note(tmp_path):
    # 92% raw agreement but a rare category → low κ
    df = _pairs([[45, 2], [2, 1]], cats=("no", "si"))
    res = analyze(_kdesign(dv_levels=["no", "si"]), df, outdir=tmp_path)
    assert res.rows[0]["es"] < .60
    assert any("paradojas de κ" in n[0] for n in res.notes)


def test_cohen_kappa_helper_matches_statsmodels():
    r = cohen_kappa(["a", "a", "b", "b", "c", "a"], ["a", "b", "b", "b", "c", "c"], ["a", "b", "c"])
    tab = np.array([[1, 1, 1], [0, 2, 0], [0, 0, 1]])
    assert r["kappa"] == pytest.approx(sm_kappa(tab).kappa)
    assert r["se"] == pytest.approx(sm_kappa(tab).std_kappa)


# ---------------------------------------------------------------- Fleiss' κ
def test_fleiss_kappa_reference():
    counts = np.array([[int((row == c).sum()) for c in range(1, 6)] for row in DIAGNOSES])
    kap, se0 = fleiss_kappa_se0(counts)
    assert kap == pytest.approx(0.430, abs=5e-4)           # Fleiss (1971)
    assert kap / se0 == pytest.approx(17.7, abs=0.05)      # irr::kappam.fleiss z


def test_fleiss_analysis(tmp_path):
    raters = [f"psiq{i}" for i in range(1, 7)]
    df = pd.DataFrame(DIAGNOSES, columns=raters)
    df.insert(0, "id", range(30))
    d = Design(analysis="agreement", dv_type="nominal", dv_name="diagnostico", dv_levels=["1", "2", "3", "4", "5"], raters=raters, agree_null=0)
    assert recommend(d).test == "fleiss"
    res = analyze(d, df, outdir=tmp_path)
    r = res.rows[0]
    assert r["es"] == pytest.approx(0.430, abs=5e-4)
    assert r["stat"] == pytest.approx(17.65, abs=0.01) and r["p"] < .001
    assert r["ci"][0] < 0.43 < r["ci"][1] and 0.25 < r["ci"][0] and r["ci"][1] < 0.60
    tab = res.extra_tables["agreement"].iloc[0]
    assert tab["% unanimidad/unanimous"] == pytest.approx(100 * 5 / 30)      # patients 1, 4, 10, 21 and 30
    es = " ".join(build_report(res, "es")["results"])
    assert "κ de Fleiss = .43" in es and "acuerdo esperado por azar" in es


# ---------------------------------------------------------------- ICC
def test_icc_shrout_fleiss():
    out = icc_two_way(SHROUT_FLEISS)
    assert out["A,1"][0] == pytest.approx(0.29, abs=.005)
    assert out["C,1"][0] == pytest.approx(0.71, abs=.005)
    assert out["A,k"][0] == pytest.approx(0.62, abs=.005)
    assert out["C,k"][0] == pytest.approx(0.91, abs=.005)
    # pingouin 0.7.0 95% CIs (2 decimals)
    for form, ci in (("A,1", (0.02, 0.76)), ("C,1", (0.34, 0.95)), ("A,k", (0.07, 0.93)), ("C,k", (0.68, 0.99))):
        assert out[form][1:] == pytest.approx(ci, abs=.006), form
    f, df1, df2, p = icc_test("A,1", out["ms"], 6, 4, 0.0)
    assert (round(f, 2), df1, df2) == (11.03, 5, 15)
    assert p == pytest.approx(2 * 0.000135, abs=2e-6)         # two-sided (pingouin reports the one-sided p)


def test_icc_consistency_test_is_dual_to_ci():
    out = icc_two_way(SHROUT_FLEISS)
    lo, hi = out["C,1"][1:]
    assert icc_test("C,1", out["ms"], 6, 4, lo)[3] == pytest.approx(0.05, abs=1e-6)
    assert icc_test("C,1", out["ms"], 6, 4, hi)[3] == pytest.approx(0.05, abs=1e-6)


def test_icc_absolute_agreement_test_type_i_error():
    # true ICC(A,1) = 1 / (1 + .25 + .75) = .50 with random rater biases; the McGraw-Wong F test should reject ≈ 5%
    rng = np.random.default_rng(3)
    rej = 0
    for _ in range(1500):
        Y = rng.normal(0, 1, (30, 1)) + rng.normal(0, .5, (1, 3)) + rng.normal(0, np.sqrt(.75), (30, 3))
        out = icc_two_way(Y)
        rej += icc_test("A,1", out["ms"], 30, 3, 0.5)[3] < .05
    assert 0.035 <= rej / 1500 <= 0.07


def test_icc_analysis(tmp_path):
    raters = ["j1", "j2", "j3", "j4"]
    df = pd.DataFrame(np.tile(SHROUT_FLEISS, (2, 1)), columns=raters)
    df.insert(0, "id", range(12))
    d = Design(analysis="agreement", dv_type="continuous", dv_name="puntuacion", raters=raters)
    assert recommend(d).test == "icc"
    res = analyze(d, df, outdir=tmp_path)
    assert res.rows[0]["es_symbol"] == "ICC(A,1)"
    assert set(res.extra_tables["icc"]["forma/form"]) == {"(A,1)", "(C,1)", "(A,k)", "(C,k)"}
    assert [a["key"] for a in res.assumptions] == ["a_normality"]
    es, en = build_report(res, "es")["results"], build_report(res, "en")["results"]
    assert any("CCI(A,1) =" in p and "Koo y Li" in p for p in es)
    assert any("ICC(A,1) =" in p and "Koo & Li" in p for p in en)
    assert any("difirieron sistemáticamente" in p for p in es)      # judge 1 scores much higher than judge 2
    comma = build_report(res, "es", decimal_comma=True)["results"]
    assert any("CCI(A,1) = ," in p for p in comma)


def test_icc_example(tmp_path):
    d, df = example("icc")
    res = analyze(d, df, outdir=tmp_path)
    assert res.test == "icc" and 0.6 < res.rows[0]["es"] < 0.95
    m = " ".join(p for _, p in build_report(res, "es")["methods"])
    assert "3 evaluadores" in m and "coeficientes de acuerdo" in m


def test_labels():
    assert [kappa_label(v) for v in (-.1, .1, .3, .5, .7, .9)] == [
        "kap_poor", "kap_slight", "kap_fair", "kap_moderate", "kap_substantial", "kap_perfect"]
    assert [icc_label(v) for v in (.4, .6, .8, .95)] == ["icc_poor", "icc_moderate", "icc_good", "icc_excellent"]


# ---------------------------------------------------------------- sample size
def test_kappa_variance_formula():
    rng = np.random.default_rng(1)
    T = rng.integers(1, 30, (4, 4)) + np.diag([20, 30, 10, 25])
    assert kappa_avar(T) == pytest.approx(sm_kappa(T).var_kappa * T.sum())
    assert sm_kappa(kappa_table(3, .7) * 1000).kappa == pytest.approx(0.7)


@pytest.mark.parametrize("c,k0,k1", [(3, .4, .7), (2, .4, .6), (5, .4, .8), (3, 0, .5)])
def test_kappa_n_power_by_simulation(c, k0, k1):
    n = kappa_n(c, k0, k1)
    P = kappa_table(c, k1).ravel()
    rng = np.random.default_rng(11)
    rej = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(1000):
            r = sm_kappa(rng.multinomial(n, P).reshape(c, c))
            se = r.std_kappa0 if k0 == 0 else r.std_kappa
            rej += bool(se > 0 and abs((r.kappa - k0) / se) > 1.96)
    assert rej / 1000 >= 0.75


def test_icc_n_exact_and_walter():
    for k, r0, r1 in ((2, .4, .6), (3, .4, .7), (4, .5, .8)):
        n = icc_n(k, r0, r1)
        assert icc_power(n, k, r0, r1) >= .80 > icc_power(n - 1, k, r0, r1)
        # Walter, Eliasziw & Donner (1998) closed-form approximation (two-sided α)
        c0 = (1 + k * r0 / (1 - r0)) / (1 + k * r1 / (1 - r1))
        walter = 1 + 2 * (1.959964 + 0.841621) ** 2 * k / (np.log(c0) ** 2 * (k - 1))
        assert abs(n - walter) <= 2.5


def test_sample_size_and_explain():
    ss = sample_size(Design(analysis="agreement", dv_type="continuous", raters=["a", "b", "c"]))
    assert ss["family"] == "agr" and ss["n_total"] == icc_n(3, .4, .7)
    ss = sample_size(Design(analysis="agreement", dv_type="nominal", dv_levels=["x", "y", "z"]))
    assert ss["n_total"] == kappa_n(3, .4, .7)
    ss = sample_size(Design(analysis="agreement", dv_type="nominal", dv_levels=["x", "y"], raters=["a", "b", "c"]))
    assert any("conservador" in n[0] for n in ss["notes"])


# ---------------------------------------------------------------- design, template
def test_design_validation():
    with pytest.raises(ValueError):
        Design(analysis="agreement", raters=["solo"])
    with pytest.raises(ValueError):
        Design(analysis="agreement", raters=["a", "a"])
    with pytest.raises(ValueError):
        Design(analysis="agreement", agree_null=1.0)
    with pytest.raises(ValueError):
        Design(analysis="agreement", dv_type="nominal", dv_levels=["x"])
    d = Design(analysis="agreement", dv_type="ordinal")
    assert d.raters == ["evaluador1", "evaluador2"] and d.dv_levels == ["1", "2", "3", "4", "5"]


def test_template():
    d = Design(analysis="agreement", dv_type="nominal", dv_name="dx", dv_levels=["a", "b"], raters=["e1", "e2", "e3"])
    cols = columns(d)
    assert [c[0] for c in cols] == ["id", "e1", "e2", "e3"]
    assert cols[1][4] == "a, b"
    tpl, dic = make_template(d, 5)
    assert list(tpl.columns) == ["id", "e1", "e2", "e3"] and len(dic) == 4


def test_kappa_example(tmp_path):
    d, df = example("kappa")
    res = analyze(d, df, outdir=tmp_path)
    assert res.test == "kappa" and 0.4 < res.rows[0]["es"] < 0.9
    md = build_report(res, "en")
    assert any("Cohen's κ =" in p for p in md["results"])
