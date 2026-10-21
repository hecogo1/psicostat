import re

import pytest
from docx import Document

from psicostat import analyze, build_report, to_docx, to_markdown
from psicostat.cli import main
from psicostat.examples import example
from psicostat.report import stat_str, to_decimal_comma

DOT_DECIMAL = re.compile(r"\d\.\d|[\s(\[=<>-]\.\d")


@pytest.mark.parametrize("src,expected", [
    ("t(37.62) = -4.45, p < .001, g = -0.99, IC 95% [-1.45, -0.52]", "t(37,62) = -4,45; p < ,001; g = -0,99; IC 95% [-1,45; -0,52]"),
    ("F(2, 87) = 8.91, p = .032, η²p = .17", "F(2; 87) = 8,91; p = ,032; η²p = ,17"),
    ("χ²(1, N = 100) = 10.26", "χ²(1; N = 100) = 10,26"),
    ("lectura (M = 6.03, DT = 1.17, n = 30).", "lectura (M = 6,03; DT = 1,17; n = 30)."),
    ("con 3 grupos (a, b, c) de la VI.", "con 3 grupos (a, b, c) de la VI."),
    ("PsicoStat v0.1.0 (Python; SciPy).", "PsicoStat v0.1.0 (Python; SciPy)."),
    ("", ""),
])
def test_to_decimal_comma(src, expected):
    assert to_decimal_comma(src) == expected


@pytest.mark.parametrize("name", ["ttest", "anova", "chi2", "regression", "mixed"])
def test_spanish_report_uses_comma_and_english_keeps_dot(name, tmp_path):
    d, df = example(name)
    d.decimal_comma = True
    res = analyze(d, df, outdir=tmp_path)
    es = build_report(res, "es")
    en = build_report(res, "en")
    assert es["decimal_comma"] and not en["decimal_comma"]
    es_text = " ".join(es["results"]) + " ".join(p for _, p in es["methods"]).replace(f"v{__import__('psicostat').__version__}", "")
    assert not DOT_DECIMAL.search(es_text), DOT_DECIMAL.search(es_text)
    assert "," in es_text
    assert DOT_DECIMAL.search(" ".join(en["results"]))
    for _, tab in es["tables"]:
        for col in tab.columns:
            assert not any(DOT_DECIMAL.search(" " + str(v)) for v in tab[col] if re.fullmatch(r"[-<>= .\d]+", str(v)))
    md = to_markdown(es)
    assert "IC 95% [" in md and "; " in md
    path = to_docx(es, tmp_path / "es.docx")
    doc = Document(path)
    cells = " ".join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    assert re.search(r"\d,\d\d", cells)


def test_default_keeps_dot():
    d, df = example("ttest")
    rep = build_report(analyze(d, df), "es")
    assert not rep["decimal_comma"]
    assert re.search(r"\d\.\d", " ".join(rep["results"]))
    assert build_report(analyze(d, df), "es", decimal_comma=True)["decimal_comma"]


def test_stat_str_comma():
    d, df = example("ttest")
    row = analyze(d, df).rows[0]
    assert "," in stat_str(row, "es", True).split("=")[1]
    assert stat_str(row, "en", True) == stat_str(row, "en")


def test_cli_flag(tmp_path):
    ex = tmp_path / "ex"
    main(["example", "anova", "--out", str(ex)])
    main(["analyze", str(ex / "diseno.json"), str(ex / "datos.csv"), "--out", str(tmp_path / "r"), "--decimal-comma", "--formats", "md"])
    es = (tmp_path / "r" / "informe_es.md").read_text(encoding="utf-8")
    en = (tmp_path / "r" / "informe_en.md").read_text(encoding="utf-8")
    assert "F(2; 87) = 8,91" in es and "F(2, 87) = 8.91" in en
