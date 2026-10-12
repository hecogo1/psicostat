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
    answers = iter(["Mi estudio", "¿Funciona?", "1", "1", "1", "ansiedad", "grupo", "terapia,control", "1", "", "2", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["wizard", "--out", str(tmp_path / "w")]) == 0
    out = capsys.readouterr().out
    assert "N = 128" in out
    assert os.path.exists(tmp_path / "w" / "plantilla.csv")
