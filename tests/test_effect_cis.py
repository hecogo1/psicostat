"""Effect sizes and CIs for Welch's ANOVA (ω²), McNemar (Cohen's g) and Cochran's Q (η²_Q) (#7)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.proportion import proportion_confint

from psicostat import Design, analyze, build_report
from psicostat import effects as E
from psicostat.examples import example


def _welch_df():
    rng = np.random.default_rng(5)
    return pd.DataFrame({"id": range(150), "grupo": np.repeat(["a", "b", "c"], 50),
                         "puntuacion": np.r_[rng.normal(0, 1, 50), rng.normal(.5, 4, 50), rng.normal(2, 8, 50)]})


def test_welch_anova_omega2_ci():
    res = analyze(Design(n_groups=3, levels=["a", "b", "c"]), _welch_df())
    r = res.rows[0]
    assert res.test == "welch_anova"
    lo, hi = r["ci"]
    assert 0 <= lo <= r["es"] <= hi < 1
    # the noncentral-F limits must reproduce the 2.5% / 97.5% quantiles at the observed F
    df1, df2 = 2, float(r["df"].split(", ")[1])
    n = 150
    lam_lo, lam_hi = lo * n / (1 - lo), hi * n / (1 - hi)
    assert stats.ncf.cdf(r["stat"], df1, df2, lam_lo) == pytest.approx(.975, abs=1e-4)
    assert stats.ncf.cdf(r["stat"], df1, df2, lam_hi) == pytest.approx(.025, abs=1e-4)
    rep = build_report(res, "es")
    assert any("IC 95% [" in p and "ω²" in p for p in rep["results"])
    assert any("Steiger, 2004" in p for p in rep["results"])


def test_omega2_welch_ci_lower_bound_zero_when_f_small():
    lo, hi = E.omega2_welch_ci(0.5, 2, 40.0, 60)
    assert lo == 0 and 0 < hi < 1


def test_omega2_welch_ci_coverage_equal_variances():
    # with homogeneous variances ω² is well defined: the 95% CI should cover it ≈ 95% of the time
    rng = np.random.default_rng(1)
    mus, ns, k = np.array([0, .4, .8]), np.array([30, 30, 30]), 3
    N = ns.sum()
    f2 = np.sum(ns / N * (mus - mus.mean()) ** 2)
    om = f2 / (1 + f2)
    hits = 0
    for _ in range(300):
        g = [rng.normal(m, 1, n) for m, n in zip(mus, ns)]
        w = ns / np.array([np.var(x, ddof=1) for x in g])
        ms = np.array([x.mean() for x in g])
        mw = (w * ms).sum() / w.sum()
        tmp = (((1 - w / w.sum()) ** 2) / (ns - 1)).sum()
        F = (w * (ms - mw) ** 2).sum() / (k - 1) / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
        lo, hi = E.omega2_welch_ci(F, k - 1, (k ** 2 - 1) / (3 * tmp), N)
        hits += lo <= om <= hi
    assert .90 <= hits / 300 <= .99


@pytest.mark.parametrize("b,c", [(3, 12), (10, 10), (0, 7), (25, 5), (6, 0), (1, 1)])
def test_cohen_g_ci_matches_clopper_pearson(b, c):
    g, (lo, hi) = E.cohen_g_mcnemar(b, c)
    p_lo, p_hi = proportion_confint(b, b + c, method="beta")
    assert g == pytest.approx(abs(b / (b + c) - .5))
    if p_lo <= .5 <= p_hi:
        assert lo == 0 and hi == pytest.approx(max(.5 - p_lo, p_hi - .5))
    else:
        assert (lo, hi) == pytest.approx(tuple(sorted((abs(p_lo - .5), abs(p_hi - .5)))))
    assert 0 <= lo <= g <= hi <= .5


def test_cohen_g_no_discordant_pairs():
    g, ci = E.cohen_g_mcnemar(0, 0)
    assert g == 0 and all(np.isnan(ci))


def test_mcnemar_example_reports_ci():
    d, df = example("mcnemar")
    res = analyze(d, df)
    r = res.rows[0]
    assert r["es_symbol"] == "g_C" and r["ci"][0] <= r["es"] <= r["ci"][1]
    es_txt = " ".join(build_report(res, "es")["results"])
    assert "g_C = " in es_txt and "IC 95% [" in es_txt and "Clopper" in es_txt


def _cochran_data(n=80, ps=(.3, .5, .7), seed=2):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    mat = np.column_stack([(rng.random(n) < stats.norm.cdf(base + stats.norm.ppf(p))).astype(int) for p in ps])
    df = pd.DataFrame({"id": range(n), **{f"acierto_{lv}": mat[:, j] for j, lv in enumerate(["t1", "t2", "t3"])}})
    d = Design(analysis="compare", design="within", dv_type="nominal", n_groups=3, levels=["t1", "t2", "t3"],
               dv_name="acierto", iv_name="tiempo")
    return d, df, mat


def test_eta2_q_textbook_value_and_statsmodels():
    # Serlin et al. (1982): η²_Q = Q / [N(k − 1)]; e.g. Q = 19.385, N = 150, k = 4 → .043
    assert 19.385 / (150 * 3) == pytest.approx(.043, abs=5e-4)
    _, _, mat = _cochran_data()
    q = cochrans_q(mat).statistic
    assert E.eta2_q(mat) == pytest.approx(q / (len(mat) * 2))
    assert E.eta2_q(np.ones((10, 3), int)) == 0


def test_cochran_reports_eta2_q_with_bootstrap_ci():
    d, df, mat = _cochran_data()
    res = analyze(d, df)
    r = res.rows[0]
    assert res.test == "cochran" and r["es_symbol"] == "η²_Q"
    assert r["stat"] == pytest.approx(cochrans_q(mat).statistic)
    assert r["es"] == pytest.approx(r["stat"] / (len(mat) * 2))
    assert 0 <= r["ci"][0] <= r["es"] <= r["ci"][1] <= 1
    es_txt = " ".join(build_report(res, "es")["results"])
    assert "η²_Q = " in es_txt and "IC 95% [" in es_txt and "Serlin" in es_txt
    assert "Serlin" in " ".join(build_report(res, "en")["results"])
