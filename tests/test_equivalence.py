import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.weightstats import ttost_ind, ttost_paired

from psicostat import Design, analyze, build_report, sample_size
from psicostat.cli import main
from psicostat.effects import tost_paired, tost_welch
from psicostat.examples import example
from psicostat.power import tost_power, tost_sample_size

rng = np.random.default_rng(7)
X, Y = rng.normal(10, 2, 40), rng.normal(10.3, 3, 35)


def test_tost_welch_matches_statsmodels():
    r = tost_welch(X, Y, 1.0, units="raw")
    p, (t1, p1, df1), (t2, p2, df2) = ttost_ind(X, Y, -1.0, 1.0, usevar="unequal")
    assert r["p"] == pytest.approx(p)
    assert (r["t_low"], r["p_low"], r["t_upp"], r["p_upp"]) == pytest.approx((t1, p1, t2, p2))
    assert r["df"] == pytest.approx(df1)


def test_tost_paired_matches_statsmodels():
    r = tost_paired(X[:35] - Y, 1.0, units="raw")
    p, (t1, p1, _), (t2, p2, _) = ttost_paired(X[:35], Y, -1.0, 1.0)
    assert r["p"] == pytest.approx(p)
    assert (r["t_low"], r["p_low"], r["t_upp"], r["p_upp"]) == pytest.approx((t1, p1, t2, p2))
    assert r["df"] == 34


def test_d_bound_uses_average_sd_and_ci_rule():
    r = tost_welch(X, Y, 0.5, units="d", alpha=0.05)
    sd = np.sqrt((np.var(X, ddof=1) + np.var(Y, ddof=1)) / 2)
    assert r["upp"] == pytest.approx(0.5 * sd) and r["low"] == pytest.approx(-0.5 * sd)
    assert r["conf"] == pytest.approx(0.90)
    # equivalence at α  <=>  the (1 − 2α) CI lies within the bounds
    inside = r["low"] < r["ci"][0] and r["ci"][1] < r["upp"]
    assert r["equivalent"] == inside
    rp = tost_paired(X[:35] - Y, 0.5, units="d")
    assert rp["upp"] == pytest.approx(0.5 * np.std(X[:35] - Y, ddof=1))


@pytest.mark.parametrize("name", ["ttest", "paired", "equivalence"])
def test_analysis_adds_tost_matching_statsmodels(name):
    d, df = example(name)
    d = Design.from_dict({**d.to_dict(), "equivalence_bound": 0.5, "equivalence_units": "raw"})
    res = analyze(d, df)
    eq = res.equivalence
    assert eq is not None and "equivalence" in res.extra_tables
    if name == "paired":
        p = ttost_paired(df.estres_pre, df.estres_post, -0.5, 0.5)[0]
    else:
        g = d.iv_name
        a, b = (df[df[g] == lv][d.dv_name] for lv in d.levels)
        p = ttost_ind(a, b, -0.5, 0.5, usevar="unequal")[0]
    assert eq["p"] == pytest.approx(p)


def test_equivalence_example_concludes_equivalence():
    d, df = example("equivalence")
    res = analyze(d, df)
    assert res.rows[0]["p"] > .05 and res.equivalence["equivalent"]
    es = " ".join(build_report(res, "es")["results"])
    en = " ".join(build_report(res, "en")["results"])
    assert "fue estadísticamente equivalente a cero" in es and "apoyan la ausencia de un efecto relevante" in es
    assert "IC 90%" in es and "Lakens et al., 2018" in es
    assert "was statistically equivalent to zero" in en
    assert "no demuestra ausencia de efecto" not in es       # the generic warning is replaced by the TOST conclusion
    meth = " ".join(p for _, p in build_report(res, "es")["methods"])
    assert "TOST" in meth and "±0.40 (d de Cohen)" in meth
    assert "equivalencia (TOST" in meth


def _two(n, m1, m2, sd=1.0, seed=3):
    r = np.random.default_rng(seed)
    a, b = r.normal(0, sd, n), r.normal(0, sd, n)
    a, b = (a - a.mean()) / a.std(ddof=1) * sd + m1, (b - b.mean()) / b.std(ddof=1) * sd + m2
    return pd.DataFrame({"id": range(2 * n), "grupo": ["a"] * n + ["b"] * n, "y": np.r_[a, b]})


@pytest.mark.parametrize("n,m2,frag", [
    (400, 0.25, "es real, aunque probablemente irrelevante"),            # significant and equivalent
    (200, 0.80, "no puede descartarse que sea de una magnitud relevante"),  # significant, not equivalent
    (300, 0.05, "apoyan la ausencia de un efecto relevante"),           # not significant, equivalent
    (20, 0.10, "resultado no concluyente"),                             # neither
])
def test_four_conclusions(n, m2, frag):
    d = Design(levels=["a", "b"], iv_name="grupo", dv_name="y", equivalence_bound=0.5, auto_fallback=False)
    res = analyze(d, _two(n, 0.0, m2))
    assert frag in " ".join(build_report(res, "es")["results"])


def test_decimal_comma_report():
    d, df = example("equivalence")
    rep = build_report(analyze(d, df), "es", decimal_comma=True)
    txt = " ".join(rep["results"])
    assert "±0,40 (d de Cohen; ±0,55 en unidades de nota)" in txt and "IC 90% [-0,39; 0,13]" in txt


def test_fallback_skips_tost_with_note():
    r = np.random.default_rng(1)
    df = pd.DataFrame({"id": range(24), "grupo": ["a"] * 12 + ["b"] * 12, "y": np.r_[r.exponential(1, 12) ** 3, r.exponential(1, 12) ** 3]})
    d = Design(levels=["a", "b"], iv_name="grupo", dv_name="y", equivalence_bound=0.5)
    res = analyze(d, df)
    assert res.test == "mannwhitney" and res.equivalence is None
    assert any("No se realizó la prueba de equivalencia" in n[0] for n in res.notes)


@pytest.mark.parametrize("kw", [
    {"n_groups": 3},                                                   # one-way ANOVA
    {"dv_type": "ordinal"},                                            # Mann-Whitney planned
    {"analysis": "correlation", "iv_type": "continuous"},
    {"alternative": "greater"},                                        # TOST needs the two-sided hypothesis
])
def test_equivalence_rejected_for_other_designs(kw):
    with pytest.raises(ValueError):
        Design(equivalence_bound=0.5, **kw)


def test_invalid_bounds_and_units():
    with pytest.raises(ValueError):
        Design(equivalence_bound=-0.3)
    with pytest.raises(ValueError):
        Design(equivalence_bound=0.3, equivalence_units="sd")
    assert not Design().equivalence and Design(equivalence_bound=0.3).equivalence


def test_tost_sample_size_and_power():
    # Monte Carlo (statsmodels ttost_ind, 4000 replications, true difference 0): power ≈ .803 with n = 70 per group, Δ = 0.5
    assert tost_power(70, 0.5) == pytest.approx(0.80, abs=0.01)
    assert tost_sample_size(0.5) == 70 and tost_sample_size(0.5, paired=True) == 36
    # Chow et al. normal approximation 2(z₁₋α + z₁₋β/₂)² / Δ² = 68.5 per group: the t version is slightly larger
    assert 69 <= tost_sample_size(0.5) <= 71
    ss = sample_size(Design(equivalence_bound=0.5))
    assert ss["n_total"] == 128                                      # the difference-test N is unchanged
    assert any("N = 140 (70 por grupo)" in n[0] for n in ss["notes"])
    ss = sample_size(Design(equivalence_bound=2.0, equivalence_units="raw"))
    assert not any("TOST" in n[0] for n in ss["notes"])              # raw bounds need an SD: no note


def test_cli_equivalence(tmp_path, capsys):
    ex = tmp_path / "ex"
    assert main(["example", "ttest", "--out", str(ex)]) == 0
    assert main(["analyze", str(ex / "diseno.json"), str(ex / "datos.csv"), "--equivalence", "0.5", "--out", str(tmp_path / "r")]) == 0
    out = capsys.readouterr().out
    assert "Prueba de equivalencia (TOST)" in out
    assert main(["plan", str(ex / "diseno.json"), "--equivalence", "0.5"]) == 0
    assert "TOST" in capsys.readouterr().out
