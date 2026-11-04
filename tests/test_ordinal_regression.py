"""Regresión logística ordinal de odds proporcionales (#5), validada con Stata `ologit` y `brant`."""
import math

import numpy as np
import pandas as pd
import pytest

from psicostat import Design, analyze, build_report
from psicostat.analysis import DataError, _fit_ordinal, _ordinal_codes, brant_test
from psicostat.design import recommend
from psicostat.examples import example
from psicostat.power import sample_size
from psicostat.template import columns, make_template

# UCLA OARC, ologit.dta (N = 400): apply (0 = unlikely, 1 = somewhat likely, 2 = very likely), pared, public, gpa (× 100).
APPLY = (
    "21011011010121000021210001112010000001100101100001002011100010002100000002000210000002001121020210110001111111"
    "10011000001000111000000111020000002000002111000110111000000000101201120112220010101011100000120021002002100011"
    "11010021000110000110011002110011111002011100001200100021101000100101000111001120010010010000020111010000001010"
    "1010000000001021122011100010210120011000000000110001011012000121100112"
)
PARED = (
    "01100000011000100000010000100000010000000100100000000000000000101000000001000111001000000010000000000000101100"
    "00000001000000101000000001010000000000001000100000000000000000000000100001000000000000000010000101000000000000"
    "10000010000000000000000001100110001001010010100100010001000000100010000000000010000000000000000101000000000011"
    "0000000100000001000000000000111000000000000100000000010000000001100000"
)
PUBLIC = (
    "00100100001010001010000011001001010000000000000000001001000000000010010000110100000000000001100000000000010000"
    "00001000100000100100000000000000000000011000000000000000010000010000000000010000010000000010000110001000000000"
    "00011100010000100000000001000000001000000000000000000000000000000010100101000001000000000000000001000000000000"
    "1000000000000000000010000101010000000001000000010000000010001000000000"
)
GPA = (
    "326321394281253259256273300350365284390268357309350217336340275320244283300327314337279290338295298381274262285"
    "250275226203285272289247304310257209294345276296289297391277251324244278294322350357317323291328332362255297363"
    "302360295381268372249272225253326257290305285400340324351255306300313316247289274316292357349256312265251308265"
    "299263280346361308328331257339226242260344298309334284281279240302265255312298361298319351310355298299305299232"
    "250290328295354311325244213322316339270309316228291365286339371325314241308302315295222286288262337351365342241"
    "321322253264294256312334322305329271287329336285279369356352338311320283308297264247307320255248329264322280362"
    "262304249310315265304305288286300223314267311373245304240343257284267345288278322330286283370315308292289353310"
    "336274273272353345298288321306240357288327292270253338316260276356287299268299296277328274190305244333356215283"
    "279272304303288321274357243311276313275260367326341324305238261318334284280321263231229351287309288247372320257"
    "305327330273263277315285225356270294258297338296244375307272313339238287291325351365311271305293235249326377349"
    "339337198293336312323390270234330311314368222261348308279312267354298332354370263225326352"
)
LEVELS = ["unlikely", "somewhat likely", "very likely"]


def _ucla():
    return pd.DataFrame({"id": range(1, 401), "apply": [LEVELS[int(c)] for c in APPLY], "pared": [int(c) for c in PARED],
                         "public": [int(c) for c in PUBLIC], "gpa": [int(GPA[i:i + 3]) / 100 for i in range(0, len(GPA), 3)]})


def _design(**kw):
    base = dict(analysis="regression", dv_type="ordinal", iv_type="continuous", dv_name="apply",
                predictors=["pared", "public", "gpa"], dv_levels=LEVELS)
    return Design(**{**base, **kw})


def test_data_integrity():
    df = _ucla()
    assert len(df) == 400
    assert df["apply"].value_counts()[LEVELS].tolist() == [220, 140, 40]
    assert df.gpa.mean() == pytest.approx(2.998925, abs=1e-6)


def test_ologit_matches_stata(tmp_path):
    # Stata: ologit apply pared public gpa (UCLA OARC, "Ordered logistic regression | Stata data analysis examples")
    res = analyze(_design(), _ucla(), str(tmp_path))
    c = res.extra_tables["coefficients"].set_index("término/term")
    assert c.loc[["pared", "public", "gpa"], "B"].tolist() == pytest.approx([1.047664, -0.0586828, 0.6157458], abs=2e-6)
    assert c.loc[["pared", "public", "gpa"], "EE/SE"].tolist() == pytest.approx([0.2657891, 0.2978588, 0.2606311], abs=1e-5)
    t = res.extra_tables["thresholds"]
    assert t["τ"].tolist() == pytest.approx([2.203323, 4.298767], abs=2e-6)
    assert t["EE/SE"].tolist() == pytest.approx([0.7795353, 0.8043146], abs=1e-5)
    assert t["umbral/threshold"].tolist() == ["unlikely | somewhat likely", "somewhat likely | very likely"]
    m = res.extra_tables["model"].iloc[0]
    assert m["-2LL"] == pytest.approx(2 * 358.51244, abs=1e-4)
    assert m["χ²(LR)"] == pytest.approx(24.18, abs=5e-3)
    assert m["gl/df"] == 3
    assert m["R²_McFadden"] == pytest.approx(0.0326, abs=5e-5)
    assert c.loc["pared", "OR"] == pytest.approx(2.850982, abs=1e-5)


def test_brant_matches_stata():
    # Stata: brant, detail (Long & Freese spost13)
    df = _ucla()
    y = np.array([int(ch) for ch in APPLY])
    (chi, gl, p), per, betas = brant_test(y, df[["pared", "public", "gpa"]])
    assert (chi, gl, p) == pytest.approx((4.34, 3, 0.227), abs=5e-3)
    assert [v[0] for v in per] == pytest.approx([0.13, 3.44, 0.18], abs=5e-3)
    assert [v[2] for v in per] == pytest.approx([0.716, 0.064, 0.672], abs=5e-4)
    assert all(v[1] == 1 for v in per)
    assert betas[0].tolist() == pytest.approx([-1.9829709, 1.0596117, -0.20055709, 0.54824568], abs=1e-5)
    assert betas[1].tolist() == pytest.approx([-4.7544684, 0.915596, 0.53508208, 0.73632132], abs=1e-5)


def test_brant_detects_non_parallel_slopes():
    # x solo influye en el primer punto de corte (Y > 0), no en el segundo (Y > 1)
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    y = np.where(x + rng.logistic(size=2000) > -.5, 1, 0)
    y[(y == 1) & (rng.logistic(size=2000) > 1)] = 2
    (chi, gl, p), _, _ = brant_test(y, pd.DataFrame({"x": x}))
    assert gl == 1 and p < .001
    (chi, gl, p), _, _ = brant_test(np.array([int(ch) for ch in APPLY]), _ucla()[["gpa"]])
    assert p > .05


def test_ordinal_codes():
    assert _ordinal_codes(pd.Series(["poco", " mucho", "nada"]), ["nada", "poco", "mucho"]).tolist() == [1, 2, 0]
    assert _ordinal_codes(pd.Series([1.0, 5, 3]), ["1", "2", "3", "4", "5"]).tolist() == [0, 4, 2]
    with pytest.raises(DataError):
        _ordinal_codes(pd.Series(["poco", "regular"]), ["nada", "poco", "mucho"])


def test_needs_three_categories(tmp_path):
    df = _ucla()
    df["apply"] = df["apply"].replace("very likely", "somewhat likely")
    with pytest.raises(DataError):
        analyze(_design(), df, str(tmp_path))


def test_empty_category_is_dropped_with_note(tmp_path):
    df = _ucla()
    res = analyze(_design(dv_levels=["unlikely", "a bit", "somewhat likely", "very likely"]), df, str(tmp_path))
    assert len(res.extra_tables["thresholds"]) == 2
    assert any("a bit" in n[0] for n in res.notes)


def test_recommendation_and_defaults():
    d = Design(analysis="regression", dv_type="ordinal", predictors=["x"], dv_name="y")
    assert d.dv_levels == ["1", "2", "3", "4", "5"]
    rec = recommend(d)
    assert rec.test == "ordreg" and "a_proportional_odds" in rec.assumptions
    assert recommend(Design(analysis="regression", dv_type="continuous", predictors=["x"], dv_name="y")).test == "linreg"


def test_template_lists_ordered_categories():
    d = Design(analysis="regression", dv_type="ordinal", predictors=["x"], dv_name="satis", dv_levels=["nada", "poco", "mucho"])
    cols = {c[0]: c for c in columns(d)}
    assert cols["satis"][-1] == "nada, poco, mucho"
    tpl, _ = make_template(d, 12)
    assert "satis" in tpl.columns and "x" in tpl.columns


def test_power_whitehead():
    # con k = 2 la fórmula de Whitehead coincide con la de la regresión logística binaria con p = .5: 4(z + z)² / (ln OR)²
    za, zb = 1.959963984540054, 0.8416212335729143
    for orr in (1.5, 2.0, 3.0):
        n = sample_size(Design(analysis="regression", dv_type="ordinal", predictors=["x"], dv_name="y",
                               dv_levels=["a", "b"], effect_size=orr))["n_total"]
        assert n == max(math.ceil(4 * (za + zb) ** 2 / math.log(orr) ** 2), 10)
    ss = [sample_size(Design(analysis="regression", dv_type="ordinal", predictors=["x"], dv_name="y", effect_label=e))["n_total"]
          for e in ("small", "medium", "large")]
    assert ss[0] > ss[1] > ss[2]
    assert ss[0] == math.ceil(3 * (za + zb) ** 2 / (math.log(1.5) ** 2 * (1 - 1 / 25)))
    with pytest.raises(ValueError):
        sample_size(Design(analysis="regression", dv_type="ordinal", predictors=["x"], dv_name="y", effect_size=1.0))


def test_report_and_example(tmp_path):
    d, df = example("ordinal")
    res = analyze(d, df, str(tmp_path))
    assert res.test == "ordreg"
    for lang in ("es", "en"):
        rep = build_report(res, lang)
        names = [n for n, _ in rep["tables"]]
        assert any("Brant" in n for n in names)
        assert any("mbral" in n or "hreshold" in n for n in names)
        assert "OR = " in " ".join(rep["results"])
    assert any(f.endswith("fig_ordinal.png") for f in res.figures)


def test_fit_does_not_depend_on_column_scale():
    df = _ucla()
    y = np.array([int(ch) for ch in APPLY])
    a = _fit_ordinal(y, df[["pared", "public", "gpa"]])
    df["gpa"] = df["gpa"] * 100
    b = _fit_ordinal(y, df[["pared", "public", "gpa"]])
    assert a.params.iloc[2] == pytest.approx(100 * b.params.iloc[2], rel=1e-5)
