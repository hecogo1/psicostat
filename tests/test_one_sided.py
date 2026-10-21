import numpy as np
import pandas as pd
import pytest
from scipy import stats

from psicostat import Design, analyze, build_report, recommend, sample_size
from psicostat.cli import main
from psicostat.design import DIRECTIONAL_TESTS
from psicostat.examples import example


@pytest.mark.parametrize("kw,two,one", [
    ({}, 128, 102),                                                       # d = .5: 64 -> 51 per group (G*Power)
    ({"design": "within"}, 34, 27),                                       # dz = .5: 34 -> 27 (G*Power)
])
def test_sample_size_one_sided_matches_gpower(kw, two, one):
    assert sample_size(Design(**kw))["n_total"] == two
    ss = sample_size(Design(alternative="greater", **kw))
    assert ss["n_total"] == one and ss["one_sided"]
    assert sample_size(Design(alternative="less", **kw))["n_total"] == one


def test_correlation_one_sided_sample_size():
    # r = .3, one-sided: G*Power exact 67; Fisher z approximation 68
    assert sample_size(Design(analysis="correlation", iv_type="continuous", alternative="greater"))["n_total"] in (67, 68)


def test_nonparametric_one_sided_inflated_by_are():
    assert sample_size(Design(dv_type="ordinal", alternative="greater"))["n_total"] == 2 * int(np.ceil(51 / 0.864))


@pytest.mark.parametrize("kw", [{"n_groups": 3}, {"analysis": "regression", "iv_type": "continuous"},
                                {"analysis": "association", "dv_type": "nominal"}, {"design": "mixed"}])
def test_one_sided_rejected_for_other_tests(kw):
    with pytest.raises(ValueError):
        Design(alternative="greater", **kw)
    assert recommend(Design(**kw)).test not in DIRECTIONAL_TESTS


def test_invalid_alternative():
    with pytest.raises(ValueError):
        Design(alternative="bigger")


@pytest.mark.parametrize("name,alt,fn", [
    ("ttest", "less", lambda df: stats.ttest_ind(df[df.grupo == "terapia"].ansiedad, df[df.grupo == "control"].ansiedad,
                                                   equal_var=False, alternative="less").pvalue),
    ("paired", "greater", lambda df: stats.ttest_rel(df.estres_pre, df.estres_post, alternative="greater").pvalue),
    ("correlation", "greater", lambda df: stats.pearsonr(df.sueno, df.bienestar, alternative="greater").pvalue),
    ("mannwhitney", "less", lambda df: stats.mannwhitneyu(df[df.modalidad == "online"].satisfaccion,
                                                          df[df.modalidad == "presencial"].satisfaccion, alternative="less").pvalue),
])
def test_one_sided_p_matches_scipy_and_is_half(name, alt, fn):
    d, df = example(name)
    two = analyze(d, df).rows[0]["p"]
    d.alternative = alt
    res = analyze(d, df)
    assert res.rows[0]["p"] == pytest.approx(fn(df))
    assert res.rows[0]["p"] == pytest.approx(two / 2, rel=0.05)
    assert res.rows[0]["alternative"] == alt


def test_wilcoxon_one_sided():
    rng = np.random.default_rng(0)
    pre = rng.normal(5, 1, 20)
    df = pd.DataFrame({"id": range(20), "y_pre": pre.round(0), "y_post": (pre - 1 + rng.normal(0, 1, 20)).round(0)})
    d = Design(design="within", dv_type="ordinal", levels=["pre", "post"], dv_name="y", alternative="greater")
    res = analyze(d, df)
    assert res.test == "wilcoxon"
    assert res.rows[0]["p"] == pytest.approx(stats.wilcoxon(df.y_pre, df.y_post, alternative="greater").pvalue)


def test_apa_wording_one_tailed():
    d, df = example("ttest")
    d.alternative = "less"
    res = analyze(d, df)
    es, en = build_report(res, "es"), build_report(res, "en")
    assert "(unilateral)" in " ".join(es["results"])
    assert "(one-tailed)" in " ".join(en["results"])
    assert "unilateral planificado a priori; H₁: ansiedad: terapia < control" in es["methods"][-1][1]
    assert "one-tailed test specified a priori" in en["methods"][-1][1]
    assert "unilateral" in es["methods"][0][1]          # power analysis paragraph
    assert "dirección contraria" not in " ".join(es["results"])
    # decimal comma keeps working with the one-tailed tag
    es_c = " ".join(build_report(res, "es", decimal_comma=True)["results"])
    assert "(unilateral); g = " in es_c


def test_warning_when_effect_in_opposite_direction():
    d, df = example("ttest")
    d.alternative = "greater"          # data show terapia < control
    rep = build_report(analyze(d, df), "es")
    assert "dirección contraria" in " ".join(rep["results"])


def test_two_sided_reports_unchanged():
    d, df = example("ttest")
    rep = build_report(analyze(d, df), "en")
    assert "one-tailed" not in " ".join(rep["results"]) and "(two-sided)" in rep["methods"][-1][1]


def test_cli_alternative(tmp_path, capsys):
    ex = tmp_path / "ex"
    main(["example", "paired", "--out", str(ex)])
    main(["plan", str(ex / "diseno.json"), "--alternative", "greater"])
    assert "N = 27" in capsys.readouterr().out
    main(["--lang", "en", "analyze", str(ex / "diseno.json"), str(ex / "datos.csv"), "--alternative", "greater",
          "--out", str(tmp_path / "r"), "--formats", "md"])
    assert "(one-tailed)" in (tmp_path / "r" / "informe_en.md").read_text(encoding="utf-8")


def test_wizard_one_sided(tmp_path, monkeypatch, capsys):
    answers = iter(["", "", "1", "1", "1", "ansiedad", "grupo", "terapia,control", "1", "", "2", "", "", "", "", "3"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    main(["wizard", "--out", str(tmp_path / "w")])
    assert "N = 102" in capsys.readouterr().out
