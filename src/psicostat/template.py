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
    elif rec.test in ("linreg", "logreg"):
        for p in d.predictors:
            cols.append((p, "Predictor", "Predictor", d.iv_type,
                         "" if d.iv_type != "nominal" else "categorías / categories"))
        allowed = "0, 1" if rec.test == "logreg" else ""
        cols.append((d.dv_name, "Variable criterio (VD)" + (" — 1 = evento" if allowed else ""),
                     "Outcome (DV)" + (" — 1 = event" if allowed else ""), d.dv_type, allowed))
    elif rec.test in ("chi2", "fisher"):
        cols += [(d.iv_name, "Variable categórica 1 (VI)", "Categorical variable 1 (IV)", "nominal", ", ".join(d.levels)),
                 (d.dv_name, "Variable categórica 2 (VD)", "Categorical variable 2 (DV)", "nominal", ", ".join(d.dv_levels))]
    elif rec.test in ("mcnemar", "cochran"):
        for lv in d.levels:
            cols.append((f"{d.dv_name}_{lv}", f"VD en la condición '{lv}'", f"DV in condition '{lv}'", "nominal", "0, 1"))
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
    if d.design in ("between", "mixed") and rec.test not in ("chi2", "fisher", "pearson", "spearman", "linreg", "logreg"):
        if d.n_factors == 2 and d.design == "between":
            cells = [(a, b) for a in d.levels for b in d.levels_b]
            df[d.iv_name] = [cells[i % len(cells)][0] for i in range(n)]
            df[d.factor_b_name] = [cells[i % len(cells)][1] for i in range(n)]
        else:
            df[d.iv_name] = [d.levels[i % len(d.levels)] for i in range(n)]
    dic = pd.DataFrame([{
        "columna/column": c[0], "descripcion": c[1], "description": c[2],
        "tipo": TYPE_LABEL[c[3]][0], "type": TYPE_LABEL[c[3]][1], "valores/values": c[4],
    } for c in cols])
    return df, dic
