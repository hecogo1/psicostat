import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from psicostat import Design, analyze, build_report, recommend, sample_size, to_markdown
from psicostat.cli import main
from psicostat.effects import partial_corr, r_ci
from psicostat.examples import example
from psicostat.template import make_template


def _data(n=150, seed=1):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    x = 0.6 * z1 + 0.3 * z2 + rng.normal(size=n)
    y = 0.5 * z1 - 0.4 * z2 + 0.25 * x + rng.normal(size=n)
    return pd.DataFrame({"id": range(n), "x": x, "y": y, "z1": z1, "z2": z2})


def test_partial_r_matches_recursive_formula_and_ols_t():
    df = _data()
    r, t, dfree, p = partial_corr(df.x, df.y, df[["z1"]])
    R = df[["x", "y", "z1"]].corr().to_numpy()
    rxy, rxz, ryz = R[0, 1], R[0, 2], R[1, 2]
    assert r == pytest.approx((rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)), abs=1e-12)
    # t of x in OLS y ~ x + Z equals the t of the partial correlation (df = n − 2 − k)
    r, t, dfree, p = partial_corr(df.x, df.y, df[["z1", "z2"]])
    fit = sm.OLS(df.y, sm.add_constant(df[["x", "z1", "z2"]])).fit()
    assert t == pytest.approx(fit.tvalues["x"], rel=1e-9)
    assert p == pytest.approx(fit.pvalues["x"], rel=1e-9)
    assert dfree == len(df) - 4 == fit.df_resid


def test_partial_one_sided_and_too_many_covariates():
    df = _data()
    r, t, dfree, p2 = partial_corr(df.x, df.y, df[["z1", "z2"]])
    _, _, _, pg = partial_corr(df.x, df.y, df[["z1", "z2"]], alternative="greater")
    _, _, _, pl = partial_corr(df.x, df.y, df[["z1", "z2"]], alternative="less")
    assert pg == pytest.approx(stats.t.sf(t, dfree)) and pg + pl == pytest.approx(1)
    assert min(pg, pl) == pytest.approx(p2 / 2)
    with pytest.raises(ValueError):
        partial_corr([1, 2, 3], [2, 1, 3], [[1], [2], [4]])


def test_partial_ci_uses_n_minus_k():
    lo, hi = r_ci(0.3, 50, k=2)
    se = 1 / math.sqrt(50 - 3 - 2)
    assert lo == pytest.approx(math.tanh(math.atanh(0.3) - 1.959964 * se), abs=1e-6)
    assert hi == pytest.approx(math.tanh(math.atanh(0.3) + 1.959964 * se), abs=1e-6)
    assert r_ci(0.3, 50) != (lo, hi)


def test_analyze_partial_pearson():
    df = _data()
    d = Design(analysis="correlation", iv_type="continuous", iv_name="x", dv_name="y", covariates=["z1", "z2"])
    res = analyze(d, df)
    row = res.rows[0]
    r, _, dfree, p = partial_corr(df.x, df.y, df[["z1", "z2"]])
    assert res.test == "pearson" and row["df"] == str(dfree)
    assert row["es"] == pytest.approx(r) and row["p"] == pytest.approx(p)
    assert row["ci"] == pytest.approx(r_ci(r, len(df), k=2))
    # differs from the zero-order correlation
    assert abs(row["es"] - stats.pearsonr(df.x, df.y)[0]) > 0.05


def test_analyze_partial_spearman_on_ranks():
    df = _data()
    df["x"] = pd.qcut(df.x, 5, labels=False) + 1
    d = Design(analysis="correlation", iv_type="ordinal", iv_name="x", dv_name="y", covariates=["z1"])
    res = analyze(d, df)
    rk = df[["x", "y", "z1"]].rank()
    r, _, dfree, p = partial_corr(rk.x, rk.y, rk[["z1"]])
    row = res.rows[0]
    assert res.test == "spearman" and row["stat_symbol"] == "r_s" and row["df"] == str(dfree)
    assert row["es"] == pytest.approx(r) and row["p"] == pytest.approx(p)


def test_without_covariates_unchanged():
    d, df = example("correlation")
    row = analyze(d, df).rows[0]
    assert row["es"] == pytest.approx(stats.pearsonr(df.sueno, df.bienestar)[0])


def test_sample_size_adds_k():
    base = sample_size(Design(analysis="correlation", iv_type="continuous"))["n_total"]
    assert sample_size(Design(analysis="correlation", iv_type="continuous", covariates=["a", "b"]))["n_total"] == base + 2
    one = sample_size(Design(analysis="correlation", iv_type="continuous", alternative="greater"))["n_total"]
    assert sample_size(Design(analysis="correlation", iv_type="continuous", alternative="greater", covariates=["a"]))["n_total"] == one + 1


def test_design_validation_and_template():
    with pytest.raises(ValueError):
        Design(analysis="correlation", iv_name="x", dv_name="y", covariates=["x"])
    with pytest.raises(ValueError):
        Design(analysis="correlation", covariates=["a", "a"])
    d = Design(analysis="correlation", iv_type="continuous", iv_name="x", dv_name="y", covariates=[" edad ", ""])
    assert d.covariates == ["edad"]
    assert any("parcial" in n[0] for n in recommend(d).notes)
    tpl, dic = make_template(d)
    assert list(tpl.columns) == ["id", "x", "y", "edad"]
    assert "edad" in dic["columna/column"].tolist()


def test_report_partial_es_en(tmp_path):
    d, df = example("partial")
    res = analyze(d, df, outdir=str(tmp_path))
    es = to_markdown(build_report(res, lang="es"))
    en = to_markdown(build_report(res, lang="en"))
    assert "correlación parcial entre uso_movil y calidad_sueno (controlando estres, edad)" in es
    assert "controlando las covariables estres, edad" in es
    assert "partial correlation between uso_movil and calidad_sueno (controlling for estres, edad)" in en
    assert f"r({res.n - 4})" in en


def test_cli_analyze_partial(tmp_path):
    d, df = example("partial")
    (tmp_path / "diseno.json").write_text(d.to_json(), encoding="utf-8")
    df.to_csv(tmp_path / "datos.csv", index=False)
    main(["analyze", str(tmp_path / "diseno.json"), str(tmp_path / "datos.csv"), "--out", str(tmp_path / "res"), "--formats", "md"])
    txt = (tmp_path / "res" / "informe_es.md").read_text(encoding="utf-8")
    assert "correlación parcial" in txt


def test_decimal_comma_partial():
    d, df = example("partial")
    es = to_markdown(build_report(analyze(d, df), lang="es", decimal_comma=True))
    assert "r(116) = −,26" in es or "r(116) = -,26" in es
