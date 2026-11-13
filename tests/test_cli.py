import os

from psicostat.cli import main


def test_cli_example_plan_template_analyze(tmp_path, capsys):
    ex = tmp_path / "ex"
    assert main(["example", "anova", "--out", str(ex)]) == 0
    assert main(["plan", str(ex / "diseno.json")]) == 0
    assert "ANOVA" in capsys.readouterr().out
    assert main(["template", str(ex / "diseno.json"), "--out", str(tmp_path / "t.csv")]) == 0
    assert os.path.exists(tmp_path / "t_diccionario.csv")
    assert main(["--lang", "en", "analyze", str(ex / "diseno.json"), str(ex / "datos.csv"), "--out", str(tmp_path / "res")]) == 0
    out = capsys.readouterr().out
    assert "one-way ANOVA" in out
    for f in ("informe_es.md", "informe_en.md", "informe_es.docx", "informe_en.docx"):
        assert os.path.exists(tmp_path / "res" / f)


def test_wizard_with_defaults(tmp_path, monkeypatch, capsys):
    answers = iter(["Mi estudio", "¿Funciona?", "1", "1", "1", "ansiedad", "grupo", "terapia,control", "1", "", "2", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["wizard", "--out", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "N = 128" in out
    assert os.path.exists(tmp_path / "w" / "plantilla.csv")


def test_wizard_moderation_dichotomous_w(tmp_path, monkeypatch, capsys):
    # analysis 6 = moderation; X continuous; W dichotomous (h, m); effect medium (f² = .15)
    answers = iter(["T", "Q", "6", "1", "estres", "2", "sexo", "h,m", "ansiedad", "2", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["wizard", "--out", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "moderación" in out and "N = 55" in out
    assert open(tmp_path / "w" / "plantilla.csv").readline().strip() == "id,estres,sexo,ansiedad"


def test_wizard_reliability(tmp_path, monkeypatch, capsys):
    # analysis 7 = reliability; 4 items, one reverse-keyed, 1–5; α0 = .70, expected α medium (.85)
    answers = iter(["T", "Q", "7", "autoestima", "ae1,ae2,ae3,ae4", "ae3", "", "", "", "2", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["wizard", "--out", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "Bonett" in out and "N = 46" in out
    assert open(tmp_path / "w" / "plantilla.csv").readline().strip() == "id,ae1,ae2,ae3,ae4"


def test_cli_reliability_example(tmp_path, capsys):
    ex = tmp_path / "ex"
    assert main(["example", "reliability", "--out", str(ex)]) == 0
    assert main(["analyze", str(ex / "diseno.json"), str(ex / "datos.csv"), "--out", str(tmp_path / "res")]) == 0
    out = capsys.readouterr().out
    assert "α de Cronbach" in out and "ω de McDonald" in out
