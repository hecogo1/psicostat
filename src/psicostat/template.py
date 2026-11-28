"""CSV template + data dictionary generation matching the design."""
from __future__ import annotations

import pandas as pd

from .design import Design, recommend

TYPE_LABEL = {
    "continuous": ("cuantitativa (intervalo/razón)", "quantitative (interval/ratio)"),
    "ordinal": ("ordinal (números enteros ordenados)", "ordinal (ordered integers)"),
    "nominal": ("nominal (categoría)", "nominal (category)"),
    "id": ("identificador", "identifier"),
}


def columns(d: Design):
    """Return list of (column, description_es, description_en, type, allowed) for the design."""
    rec = recommend(d)
    cols = [("id", "Identificador único del participante", "Unique participant identifier", "id", "")]
    dvt = d.dv_type
    if rec.test in ("pearson", "spearman"):
        cols += [(d.iv_name, "Variable X", "Variable X", d.iv_type, ""),
                 (d.dv_name, "Variable Y", "Variable Y", d.dv_type, "")]
        cols += [(c, "Covariable que se controla (correlación parcial)", "Covariate to control for (partial correlation)", "continuous", "")
                 for c in d.covariates]
    elif rec.test == "mediation":
        if d.iv_type == "nominal":
            cols.append((d.iv_name, "Predictor (X), dicotómico", "Predictor (X), dichotomous", "nominal", ", ".join(d.levels)))
        else:
            cols.append((d.iv_name, "Predictor (X)", "Predictor (X)", "continuous", ""))
        cols += [(d.mediator, "Mediador (M)", "Mediator (M)", "continuous", ""),
                 (d.dv_name, "Variable dependiente (Y)", "Dependent variable (Y)", "continuous", "")]
    elif rec.test == "moderation":
        if d.iv_type == "nominal":
            cols.append((d.iv_name, "Predictor (X), dicotómico", "Predictor (X), dichotomous", "nominal", ", ".join(d.levels)))
        else:
            cols.append((d.iv_name, "Predictor (X)", "Predictor (X)", "continuous", ""))
        if d.moderator_type == "nominal":
            cols.append((d.moderator, "Moderador (W), dicotómico", "Moderator (W), dichotomous", "nominal", ", ".join(d.moderator_levels)))
        else:
            cols.append((d.moderator, "Moderador (W)", "Moderator (W)", "continuous", ""))
        cols.append((d.dv_name, "Variable dependiente (Y)", "Dependent variable (Y)", "continuous", ""))
    elif rec.test == "reliability":
        rng = f"{d.scale_min:g}–{d.scale_max:g}"
        for i, it in enumerate(d.items, 1):
            inv = it in d.reverse_items
            cols.append((it, f"Ítem {i} de la escala {d.dv_name}" + (" (inverso: se recodifica automáticamente)" if inv else ""),
                         f"Item {i} of the {d.dv_name} scale" + (" (reverse-keyed: recoded automatically)" if inv else ""),
                         "ordinal", rng))
    elif rec.test in ("kappa", "fleiss", "icc"):
        cat = rec.test != "icc"
        for i, r in enumerate(d.raters, 1):
            cols.append((r, f"Evaluador {i}: {'categoría asignada' if cat else 'puntuación'} de {d.dv_name}",
                         f"Rater {i}: {'assigned category' if cat else 'score'} of {d.dv_name}",
                         d.dv_type, ", ".join(d.dv_levels) if cat else ""))
    elif rec.test == "mixedlm":
        cols[0] = ("id", "Identificador único de la observación (fila)", "Unique observation (row) identifier", "id", "")
        cols.append((d.cluster, "Grupo de nivel 2 al que pertenece la observación (p. ej., clase, centro, terapeuta o participante)",
                     "Level-2 cluster the observation belongs to (e.g., class, school, therapist or participant)", "nominal", ""))
        for p in d.predictors:
            slope = " — con pendiente aleatoria" if p == d.random_slope else ""
            slope_en = " — with random slope" if p == d.random_slope else ""
            cols.append((p, f"Predictor (nivel 1 o 2; dicotómico = 0/1){slope}", f"Predictor (level 1 or 2; dichotomous = 0/1){slope_en}", "continuous", ""))
        cols.append((d.dv_name, "Variable dependiente (nivel 1)", "Dependent variable (level 1)", "continuous", ""))
    elif rec.test in ("linreg", "logreg", "ordreg"):
        for p in d.predictors:
            cols.append((p, "Predictor", "Predictor", d.iv_type,
                         "" if d.iv_type != "nominal" else "categorías / categories"))
        if rec.test == "ordreg":
            cols.append((d.dv_name, "Variable criterio ordinal (VD), categorías de menor a mayor",
                         "Ordinal outcome (DV), categories from lowest to highest", d.dv_type, ", ".join(d.dv_levels)))
        else:
            allowed = "0, 1" if rec.test == "logreg" else ""
            cols.append((d.dv_name, "Variable criterio (VD)" + (" — 1 = evento" if allowed else ""),
                         "Outcome (DV)" + (" — 1 = event" if allowed else ""), d.dv_type, allowed))
    elif rec.test in ("chi2", "fisher"):
        cols += [(d.iv_name, "Variable categórica 1 (VI)", "Categorical variable 1 (IV)", "nominal", ", ".join(d.levels)),
                 (d.dv_name, "Variable categórica 2 (VD)", "Categorical variable 2 (DV)", "nominal", ", ".join(d.dv_levels))]
    elif rec.test in ("mcnemar", "cochran"):
        for lv in d.levels:
            cols.append((f"{d.dv_name}_{lv}", f"VD en la condición '{lv}'", f"DV in condition '{lv}'", "nominal", "0, 1"))
    elif rec.test in ("ttest_one", "wilcoxon_one"):
        cols.append((d.dv_name, f"Variable dependiente (se compara con el valor de referencia μ₀ = {d.test_value:g})",
                     f"Dependent variable (compared with the reference value μ₀ = {d.test_value:g})", dvt, ""))
    elif d.design == "within":
        if d.n_factors == 2:
            for a in d.levels:
                for b in d.levels_b:
                    cols.append((f"{d.dv_name}_{a}_{b}", f"VD en {d.iv_name}={a}, {d.factor_b_name}={b}",
                                 f"DV at {d.iv_name}={a}, {d.factor_b_name}={b}", dvt, ""))
        else:
            for lv in d.levels:
                cols.append((f"{d.dv_name}_{lv}", f"VD en la condición '{lv}'", f"DV in condition '{lv}'", dvt, ""))
    elif d.design == "mixed":
        cols.append((d.iv_name, "Grupo (factor entre sujetos)", "Group (between-subjects factor)", "nominal", ", ".join(d.levels)))
        for lv in d.levels_b:
            cols.append((f"{d.dv_name}_{lv}", f"VD en la medida '{lv}' ({d.factor_b_name})",
                         f"DV at measurement '{lv}' ({d.factor_b_name})", dvt, ""))
    else:
        cols.append((d.iv_name, "Grupo (factor entre sujetos)", "Group (between-subjects factor)", "nominal", ", ".join(d.levels)))
        if d.n_factors == 2:
            cols.append((d.factor_b_name, "Factor B (entre sujetos)", "Factor B (between-subjects)", "nominal", ", ".join(d.levels_b)))
        for c in d.covariates:
            cols.append((c, "Covariable (medida antes del tratamiento)", "Covariate (measured before treatment)", "continuous", ""))
        cols.append((d.dv_name, "Variable dependiente", "Dependent variable", dvt, ""))
    return cols


def make_template(d: Design, n: int = 20):
    """Return (template_df, dictionary_df). Group/factor columns are pre-filled in a balanced way."""
    cols = columns(d)
    names = [c[0] for c in cols]
    df = pd.DataFrame({c: [""] * n for c in names})
    df["id"] = [f"P{i + 1:03d}" for i in range(n)]
    rec = recommend(d)
    if d.design in ("between", "mixed") and rec.test not in ("chi2", "fisher", "pearson", "spearman", "linreg", "logreg", "ordreg", "mediation", "moderation", "reliability",
                                                            "kappa", "fleiss", "icc", "mixedlm",
                                                            "ttest_one", "wilcoxon_one"):
        if d.n_factors == 2 and d.design == "between":
            cells = [(a, b) for a in d.levels for b in d.levels_b]
            df[d.iv_name] = [cells[i % len(cells)][0] for i in range(n)]
            df[d.factor_b_name] = [cells[i % len(cells)][1] for i in range(n)]
        else:
            df[d.iv_name] = [d.levels[i % len(d.levels)] for i in range(n)]
    if rec.test == "mixedlm":
        df[d.cluster] = [f"G{i // d.cluster_size + 1:02d}" for i in range(n)]
    if rec.test in ("mediation", "moderation") and d.iv_type == "nominal" and d.levels:
        df[d.iv_name] = [d.levels[i % len(d.levels)] for i in range(n)]
    if rec.test == "moderation" and d.moderator_type == "nominal" and d.moderator_levels:
        df[d.moderator] = [d.moderator_levels[(i // max(len(d.levels), 1)) % len(d.moderator_levels)] for i in range(n)]
    dic = pd.DataFrame([{
        "columna/column": c[0], "descripcion": c[1], "description": c[2],
        "tipo": TYPE_LABEL[c[3]][0], "type": TYPE_LABEL[c[3]][1], "valores/values": c[4],
    } for c in cols])
    return df, dic
