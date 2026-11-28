"""One-sample comparison with a reference value μ₀ (#41)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.weightstats import ttost_paired

from psicostat import Design, analyze, build_report, make_template
from psicostat.cli import main
from psicostat.design import recommend
from psicostat.examples import example
from psicostat.power import sample_size
from psicostat.template import columns


def _design(**kw):
    return Design(**{"analysis": "onesample", "dv_name": "ci", "test_value": 100, **kw})


def test_recommendation_and_validation():
    assert recommend(_design()).test == "ttest_one" and recommend(_design()).fallback == "wilcoxon_one"
    assert recommend(_design(dv_type="ordinal")).test == "wilcoxon_one"
    with pytest.raises(ValueError):
        _design(dv_type="nominal")
    with pytest.raises(ValueError):
        _design(test_value="abc")
    assert _design(test_value="3,5").test_value == 3.5     # decimal comma accepted
    # one-sided and TOST are allowed for the one-sample t test
    assert _design(alternative="less").one_sided
    assert _design(equivalence_bound=0.3).equivalence
    with pytest.raises(ValueError):
        _design(dv_type="ordinal", equivalence_bound=0.3)


def test_t_test_matches_scipy_and_cohen_d():
    d, df = example("onesample")
    res = analyze(d, df)
    x = df[d.dv_name].to_numpy(float)
    ref = stats.ttest_1samp(x, d.test_value)
    r = res.rows[0]
    assert res.test == "ttest_one" and r["df"] == str(len(x) - 1)
    assert r["stat"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(ref.pvalue)
    dval = (x.mean() - d.test_value) / x.std(ddof=1)
    assert r["es"] == pytest.approx(dval)
    assert r["ci"][0] < dval < r["ci"][1]
    tab = res.extra_tables["one_sample"].iloc[0]
    lo, hi = stats.ttest_1samp(x, d.test_value).confidence_interval()
    assert tab["M − μ₀"] == pytest.approx(x.mean() - d.test_value)
    assert tab["IC95_inf/CI95_low"] == pytest.approx(lo - d.test_value) and tab["IC95_sup/CI95_high"] == pytest.approx(hi - d.test_value)


@pytest.mark.parametrize("alt", ["greater", "less"])
def test_one_sided(alt):
    d, df = example("onesample")
    d = Design.from_dict({**d.to_dict(), "alternative": alt})
    res = analyze(d, df)
    ref = stats.ttest_1samp(df[d.dv_name], d.test_value, alternative=alt)
    assert res.rows[0]["p"] == pytest.approx(ref.pvalue)
    txt = " ".join(build_report(res, "es")["results"])
    assert "(unilateral)" in txt and f"μ₀ = {d.test_value:g}" in txt
    if alt == "less":
        assert "dirección contraria" in txt


def test_wilcoxon_ordinal_and_fallback():
    rng = np.random.default_rng(3)
    likert = pd.DataFrame({"id": range(40), "actitud": rng.choice([1, 2, 3, 4, 5], 40, p=[.05, .1, .25, .35, .25])})
    d = Design(analysis="onesample", dv_name="actitud", dv_type="ordinal", test_value=3)
    res = analyze(d, likert)
    diff = likert.actitud.to_numpy(float) - 3
    ref = stats.wilcoxon(diff)
    assert res.test == "wilcoxon_one" and res.rows[0]["p"] == pytest.approx(ref.pvalue)
    assert res.rows[0]["stat"] == pytest.approx(ref.statistic) and res.rows[0]["es_symbol"] == "r_rb"
    assert "Mdn" in " ".join(build_report(res, "es")["results"])
    # skewed small sample: automatic fallback from the t test
    skew = pd.DataFrame({"id": range(20), "tr": np.r_[rng.exponential(1, 19) + 5, 30]})
    res2 = analyze(Design(analysis="onesample", dv_name="tr", test_value=5), skew)
    assert res2.planned_test == "ttest_one" and res2.test == "wilcoxon_one" and res2.fallback_reason == "a_normality"
    assert [a["key"] for a in res2.assumptions].count("a_outliers") == 1


def test_tost_matches_statsmodels():
    d, df = example("onesample")
    d = Design.from_dict({**d.to_dict(), "equivalence_bound": 5, "equivalence_units": "raw"})
    res = analyze(d, df)
    x = df[d.dv_name].to_numpy(float)
    p = ttost_paired(x, np.full_like(x, d.test_value), -5, 5)[0]
    assert res.equivalence["p"] == pytest.approx(p) and res.equivalence["df"] == len(x) - 1
    txt = " ".join(build_report(res, "es")["results"])
    assert "Prueba de equivalencia (TOST)" in txt and f"t({len(x) - 1}) =" in txt


def test_sample_size_matches_ttestpower():
    d = _design(effect_size=0.5)
    assert sample_size(d)["n_total"] == 34                     # G*Power: one-sample t, d = 0.5, α = .05, 1 − β = .80
    one = _design(effect_size=0.3, alternative="greater", power=0.9)
    ref = TTestPower().solve_power(effect_size=0.3, alpha=0.05, power=0.9, alternative="larger")
    assert sample_size(one)["n_total"] == int(np.ceil(ref))
    assert sample_size(_design(dv_type="ordinal", effect_size=0.5))["n_total"] == int(np.ceil(34 / 0.864))
    eq = sample_size(_design(equivalence_bound=0.5))
    assert any("TOST" in n[0] for n in eq["notes"])


def test_template_report_and_decimal_comma():
    d = _design(test_value=3.5)
    assert [c[0] for c in columns(d)] == ["id", "ci"]
    tpl, dic = make_template(d, 10)
    assert list(tpl.columns) == ["id", "ci"] and "μ₀ = 3.5" in dic.iloc[1]["descripcion"]
    rng = np.random.default_rng(0)
    res = analyze(d, pd.DataFrame({"id": range(30), "ci": rng.normal(4, 1, 30).round(2)}))
    es, en = build_report(res, "es", decimal_comma=True), build_report(res, "en")
    assert any("μ₀ = 3,5" in p for _, p in es["methods"])
    assert any("single-sample design" in p for _, p in en["methods"])
    assert "the reference value μ₀ = 3.5" in " ".join(en["results"])
    assert any(name.startswith("Comparación con el valor de referencia") for name, _ in es["tables"])


def test_cli_wizard_and_example(tmp_path, monkeypatch, capsys):
    # analysis 10 = onesample; continuous DV, μ₀ = 100; medium effect (d = 0.5)
    answers = iter(["T", "Q", "10", "1", "ci", "100", "2", "", "", "", "", "1", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["wizard", "--out", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "prueba t para una muestra" in out and "N = 34" in out
    assert open(tmp_path / "w" / "plantilla.csv").readline().strip() == "id,ci"
    ex = tmp_path / "ex"
    assert main(["example", "onesample", "--out", str(ex)]) == 0
    assert main(["analyze", str(ex / "diseno.json"), str(ex / "datos.csv"), "--out", str(tmp_path / "res")]) == 0
    assert "μ₀ = 50" in capsys.readouterr().out
    assert (tmp_path / "res" / "figuras" / "fig_una_muestra.png").exists()
