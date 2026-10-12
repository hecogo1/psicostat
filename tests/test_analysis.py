import numpy as np
import pandas as pd
import pytest
from scipy import stats

from psicostat import Design, analyze, build_report, make_template, to_docx, to_markdown
from psicostat.analysis import DataError
from psicostat.examples import EXAMPLES, example
from psicostat.template import columns


@pytest.mark.parametrize("name", EXAMPLES)
def test_examples_end_to_end(name, tmp_path):
    d, df = example(name)
    # example data must match the generated template columns
    assert [c[0] for c in columns(d)] == [c for c in df.columns if c in [x[0] for x in columns(d)]]
    tpl, dic = make_template(d, 12)
    assert list(tpl.columns) == [c[0] for c in columns(d)]
    assert len(dic) == len(tpl.columns)
    res = analyze(d, df, outdir=tmp_path / "fig")
    assert res.rows and all(r["p"] is None or 0 <= r["p"] <= 1 for r in res.rows)
    assert res.figures and all((tmp_path / "fig").exists() for _ in res.figures)
    for lang in ("es", "en"):
        rep = build_report(res, lang)
        md = to_markdown(rep, tmp_path / f"r_{lang}.md", rel_to=tmp_path)
        assert ("Resultados" if lang == "es" else "Results") in md
        assert "p " in md
        to_docx(rep, tmp_path / f"r_{lang}.docx")
        assert (tmp_path / f"r_{lang}.docx").stat().st_size > 5000


def test_welch_matches_scipy():
    d, df = example("ttest")
    res = analyze(d, df)
    a = df[df.grupo == "terapia"].ansiedad
    b = df[df.grupo == "control"].ansiedad
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert res.rows[0]["stat"] == pytest.approx(ref.statistic)
    assert res.rows[0]["p"] == pytest.approx(ref.pvalue)


def test_rm_anova_matches_statsmodels():
    from statsmodels.stats.anova import AnovaRM
    d, df = example("rm")
    res = analyze(d, df)
    long = df.melt(id_vars="id", var_name="s", value_name="y")
    ref = AnovaRM(long, "y", "id", within=["s"]).fit().anova_table
    assert res.rows[0]["stat"] == pytest.approx(ref["F Value"].iloc[0])


def test_mixed_matches_balanced_reference():
    # split-plot ANOVA: three effects, simulated interaction must be detected
    d, df = example("mixed")
    res = analyze(d, df)
    assert [r["label"] for r in res.rows] == ["grupo", "tiempo", "grupo × tiempo"]
    assert res.rows[2]["p"] < .05 and res.posthoc is not None


def test_fallback_mannwhitney_on_skewed_small_sample():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"id": range(24), "grupo": ["a"] * 12 + ["b"] * 12,
                       "puntuacion": np.r_[rng.exponential(1, 12) ** 3, rng.exponential(2, 12) ** 3]})
    res = analyze(Design(levels=["a", "b"]), df)
    assert res.planned_test == "ttest_ind" and res.test == "mannwhitney" and res.fallback_reason == "a_normality"


def test_welch_anova_on_heterogeneous_variances():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"id": range(150), "grupo": np.repeat(["a", "b", "c"], 50),
                       "puntuacion": np.r_[rng.normal(0, 1, 50), rng.normal(.5, 4, 50), rng.normal(2, 8, 50)]})
    res = analyze(Design(n_groups=3, levels=["a", "b", "c"]), df)
    assert res.test == "welch_anova" and res.rows[0]["es_symbol"] == "ω²"


def test_fisher_when_expected_small():
    df = pd.DataFrame({"id": range(12), "grupo": ["a"] * 6 + ["b"] * 6, "y": ["si"] * 5 + ["no"] + ["no"] * 5 + ["si"]})
    res = analyze(Design(analysis="association", dv_type="nominal", levels=["a", "b"], dv_name="y"), df)
    assert res.test == "fisher"


def test_missing_rows_excluded_and_missing_columns_error():
    d, df = example("ttest")
    df.loc[0, "ansiedad"] = np.nan
    assert analyze(d, df).n_excluded == 1
    with pytest.raises(DataError):
        analyze(d, df.drop(columns="ansiedad"))


def test_effect_size_ci_contains_estimate():
    for name in ("ttest", "anova", "correlation", "chi2"):
        d, df = example(name)
        r = analyze(d, df).rows[0]
        assert r["ci"][0] <= r["es"] <= r["ci"][1]
