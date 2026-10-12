"""Simulated example datasets (reproducible) for demos, tutorials and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import Design


def _ids(n):
    return [f"P{i + 1:03d}" for i in range(n)]


def example(name: str, seed: int = 42):
    """Return (Design, DataFrame) for an example name. See EXAMPLES for the list."""
    rng = np.random.default_rng(seed)
    if name == "ttest":
        d = Design(title="Terapia breve y ansiedad", question="¿Reduce la terapia breve la ansiedad frente a un grupo control?",
                   analysis="compare", design="between", n_groups=2, levels=["terapia", "control"],
                   dv_name="ansiedad", iv_name="grupo")
        n = 40
        df = pd.DataFrame({"id": _ids(2 * n), "grupo": ["terapia"] * n + ["control"] * n,
                           "ansiedad": np.r_[rng.normal(20, 5, n), rng.normal(24, 5, n)].round(1)})
    elif name == "mannwhitney":
        d = Design(title="Satisfacción por modalidad", analysis="compare", design="between", n_groups=2,
                   levels=["online", "presencial"], dv_type="ordinal", dv_name="satisfaccion", iv_name="modalidad")
        n = 30
        df = pd.DataFrame({"id": _ids(2 * n), "modalidad": ["online"] * n + ["presencial"] * n,
                           "satisfaccion": np.r_[rng.integers(1, 6, n), np.clip(rng.integers(2, 7, n), 1, 5)]})
    elif name == "paired":
        d = Design(title="Pre-post mindfulness", analysis="compare", design="within", n_groups=2, levels=["pre", "post"],
                   dv_name="estres", iv_name="momento")
        n = 35
        pre = rng.normal(30, 6, n)
        df = pd.DataFrame({"id": _ids(n), "estres_pre": pre.round(1), "estres_post": (pre - 3 + rng.normal(0, 4, n)).round(1)})
    elif name == "anova":
        d = Design(title="Método de estudio y rendimiento", analysis="compare", design="between", n_groups=3,
                   levels=["lectura", "subrayado", "autoevaluacion"], dv_name="rendimiento", iv_name="metodo")
        n = 30
        df = pd.DataFrame({"id": _ids(3 * n), "metodo": np.repeat(d.levels, n),
                           "rendimiento": np.r_[rng.normal(6, 1.5, n), rng.normal(6.3, 1.5, n), rng.normal(7.4, 1.5, n)].round(2)})
    elif name == "factorial":
        d = Design(title="Feedback × dificultad", analysis="compare", design="between", n_factors=2, n_groups=2,
                   levels=["sin_feedback", "feedback"], levels_b=["facil", "dificil"], dv_name="aciertos",
                   iv_name="feedback", factor_b_name="dificultad")
        rows = []
        mu = {("sin_feedback", "facil"): 15, ("sin_feedback", "dificil"): 9, ("feedback", "facil"): 16, ("feedback", "dificil"): 13}
        for (a, b), m in mu.items():
            for v in rng.normal(m, 3, 25):
                rows.append({"feedback": a, "dificultad": b, "aciertos": round(v, 1)})
        df = pd.DataFrame(rows)
        df.insert(0, "id", _ids(len(df)))
    elif name == "ancova":
        d = Design(title="Intervención en depresión controlando línea base", analysis="compare", design="between",
                   n_groups=2, levels=["intervencion", "control"], covariates=["depresion_pre"],
                   dv_name="depresion_post", iv_name="grupo")
        n = 40
        pre = rng.normal(25, 6, 2 * n)
        grp = np.repeat(d.levels, n)
        post = 5 + .7 * pre - np.where(grp == "intervencion", 4, 0) + rng.normal(0, 4, 2 * n)
        df = pd.DataFrame({"id": _ids(2 * n), "grupo": grp, "depresion_pre": pre.round(1), "depresion_post": post.round(1)})
    elif name == "rm":
        d = Design(title="Memoria a lo largo de tres sesiones", analysis="compare", design="within", n_groups=3,
                   levels=["s1", "s2", "s3"], dv_name="recuerdo", iv_name="sesion")
        n = 30
        subj = rng.normal(0, 3, n)
        df = pd.DataFrame({"id": _ids(n), **{f"recuerdo_{lv}": (10 + m + subj + rng.normal(0, 2, n)).round(1)
                                              for lv, m in zip(d.levels, (0, 1.5, 2.5))}})
    elif name == "mixed":
        d = Design(title="Tratamiento × tiempo", analysis="compare", design="mixed", n_groups=2, levels=["tcc", "lista_espera"],
                   levels_b=["pre", "post", "seguimiento"], dv_name="sintomas", iv_name="grupo", factor_b_name="tiempo")
        n = 25
        rows = []
        for g, eff in (("tcc", (0, -6, -5)), ("lista_espera", (0, -1, -1))):
            s = rng.normal(0, 4, n)
            for i in range(n):
                rows.append({"grupo": g, **{f"sintomas_{t}": round(30 + e + s[i] + rng.normal(0, 3), 1)
                                             for t, e in zip(d.levels_b, eff)}})
        df = pd.DataFrame(rows)
        df.insert(0, "id", _ids(len(df)))
    elif name == "chi2":
        d = Design(title="Género y búsqueda de ayuda", analysis="association", n_groups=2, levels=["mujer", "hombre"],
                   dv_levels=["si", "no"], iv_name="genero", dv_name="busca_ayuda", dv_type="nominal")
        n = 100
        g = rng.choice(d.levels, n)
        y = np.where(rng.random(n) < np.where(g == "mujer", .6, .35), "si", "no")
        df = pd.DataFrame({"id": _ids(n), "genero": g, "busca_ayuda": y})
    elif name == "correlation":
        d = Design(title="Horas de sueño y bienestar", analysis="correlation", iv_name="sueno", dv_name="bienestar", iv_type="continuous")
        n = 80
        x = rng.normal(7, 1, n)
        df = pd.DataFrame({"id": _ids(n), "sueno": x.round(2), "bienestar": (20 + 2 * x + rng.normal(0, 3, n)).round(1)})
    elif name == "regression":
        d = Design(title="Predictores del burnout", analysis="regression", iv_type="continuous",
                   predictors=["carga", "apoyo", "autoeficacia"], dv_name="burnout")
        n = 120
        X = rng.normal(0, 1, (n, 3))
        y = 50 + 5 * X[:, 0] - 3 * X[:, 1] - 2 * X[:, 2] + rng.normal(0, 6, n)
        df = pd.DataFrame({"id": _ids(n), "carga": X[:, 0].round(2), "apoyo": X[:, 1].round(2), "autoeficacia": X[:, 2].round(2),
                           "burnout": y.round(1)})
    elif name == "logistic":
        d = Design(title="Predicción del abandono terapéutico", analysis="regression", dv_type="nominal", iv_type="continuous",
                   predictors=["alianza", "edad"], dv_name="abandono", event_rate=.3)
        n = 200
        al, age = rng.normal(0, 1, n), rng.normal(35, 10, n)
        p = 1 / (1 + np.exp(-(-1 - .9 * al + .02 * (age - 35))))
        df = pd.DataFrame({"id": _ids(n), "alianza": al.round(2), "edad": age.round(0), "abandono": (rng.random(n) < p).astype(int)})
    elif name == "mcnemar":
        d = Design(title="Síntoma presente antes y después", analysis="compare", design="within", dv_type="nominal",
                   n_groups=2, levels=["antes", "despues"], dv_name="insomnio", iv_name="momento")
        n = 60
        a = (rng.random(n) < .6).astype(int)
        b = np.where(a == 1, (rng.random(n) < .5).astype(int), (rng.random(n) < .1).astype(int))
        df = pd.DataFrame({"id": _ids(n), "insomnio_antes": a, "insomnio_despues": b})
    else:
        raise KeyError(f"Unknown example '{name}'. Available: {', '.join(EXAMPLES)}")
    return d, df


EXAMPLES = ["ttest", "mannwhitney", "paired", "anova", "factorial", "ancova", "rm", "mixed", "chi2",
            "correlation", "regression", "logistic", "mcnemar"]
