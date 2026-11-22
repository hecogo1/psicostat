"""APA 7 style methods and results text (es/en) + Markdown/DOCX export."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from . import __version__
from .design import DIRECTIONAL_TESTS, EQUIVALENCE_TESTS, assumption_label, hypothesis_text, recommend
from .effects import fmt_num, fmt_p, magnitude
from .i18n import t
from .power import explain, sample_size

AGREEMENT_TESTS = ("kappa", "fleiss", "icc")
DESIGN_TXT = {
    "between": ("entre sujetos", "between-subjects"),
    "within": ("intra sujetos (medidas repetidas)", "within-subjects (repeated measures)"),
    "mixed": ("mixto", "mixed"),
}


_VERSION_RE = re.compile(r"v\d+\.\d+\.\d+")
_DF_RE = re.compile(r"(?<=[A-Za-zχ²_])\((\d+(?:\.\d+)?), ((?:N = )?\d+(?:\.\d+)?)\)")
_ITEM_SEP_RE = re.compile(r"(\d|\]|\(unilateral\)), (?=(?:[^\s,;()]+ [=<>] )|IC )")
_CI_RE = re.compile(r"\[([^\[\]]*?), ([^\[\]]*?)\]")


def to_decimal_comma(text: str) -> str:
    """Convert APA numbers in a text to decimal comma (Spanish style).

    Intervals, degrees of freedom and statistic lists use ';' as separator to avoid ambiguity:
    t(37.62) -> t(37,62); F(2, 37.6) -> F(2; 37,6); [0.08, 1.36] -> [0,08; 1,36]; p = .032 -> p = ,032.
    Version strings (v0.2.0) are preserved.
    """
    if not text:
        return text
    keep = _VERSION_RE.findall(text)
    for i, v in enumerate(keep):
        text = text.replace(v, f"\x00{i}\x00", 1)
    text = _CI_RE.sub(lambda m: f"[{m.group(1)}; {m.group(2)}]", text)
    text = _DF_RE.sub(lambda m: f"({m.group(1)}; {m.group(2)})", text)
    text = _ITEM_SEP_RE.sub(lambda m: f"{m.group(1)}; ", text)
    text = re.sub(r"(\d)\.(\d)", r"\1,\2", text)
    text = re.sub(r"(^|[\s(\[=<>≈–-])\.(\d)", r"\1,\2", text)
    for i, v in enumerate(keep):
        text = text.replace(f"\x00{i}\x00", v)
    return text


def _L(lang):
    return 0 if lang == "es" else 1


def stat_str(row, lang="es", decimal_comma=False):
    """APA-formatted statistic string, e.g. t(37.62) = 2.31, p = .026, g = 0.72, 95% CI [0.08, 1.36]."""
    sym = row["stat_symbol"]
    if sym == "OR":
        s = f"OR = {fmt_num(row['stat'])}"
    elif sym == "binomial":
        s = f"{'binomial exacta' if lang == 'es' else 'exact binomial'}"
    else:
        df = f"({row['df']})" if row["df"] else ""
        bounded = sym in ("r", "r_s")
        s = f"{sym}{df} = {fmt_num(row['stat'], bounded)}"
    s += f", {fmt_p(row['p'])}"
    if row.get("alternative", "two-sided") != "two-sided":
        s += " (unilateral)" if lang == "es" else " (one-tailed)"
    if row["es_symbol"] and row["es"] is not None:
        s += f", {row['es_symbol']} = {fmt_num(row['es'], row['bounded'])}" if row["es_symbol"] != row["stat_symbol"] else ""
        if row["ci"] is not None and all(np.isfinite(row["ci"])):
            lab = "IC 95%" if lang == "es" else "95% CI"
            s += f", {lab} [{fmt_num(row['ci'][0], row['bounded'])}, {fmt_num(row['ci'][1], row['bounded'])}]"
    return to_decimal_comma(s) if decimal_comma and lang == "es" else s


def _sig(row, alpha, lang):
    if row["p"] is None:
        ci = row.get("ci")
        if ci is not None and len(ci) == 2 and all(x is not None and np.isfinite(x) for x in ci):
            excludes = ci[0] > 0 or ci[1] < 0
            if lang == "es":
                return ("fue estadísticamente significativo (el IC no incluye el 0)" if excludes else
                        "no fue estadísticamente significativo (el IC incluye el 0)")
            return ("was statistically significant (CI excludes 0)" if excludes else
                    "was not statistically significant (CI includes 0)")
        return ""
    if lang == "es":
        return "fue estadísticamente significativo" if row["p"] < alpha else "no fue estadísticamente significativo"
    return "was statistically significant" if row["p"] < alpha else "was not statistically significant"


def _subject(r, lang):
    lab = r["label"]
    es = lang == "es"
    if lab == "Modelo/Model":
        return "El modelo global" if es else "The overall model"
    if lab.startswith("a × b") or "indirecto" in lab or lab.endswith("(indirect)"):
        return "El efecto indirecto" if es else "The indirect effect"
    if lab.startswith("a ("):
        return "La senda a" if es else "Path a"
    if lab.startswith("b ("):
        return "La senda b" if es else "Path b"
    if lab.startswith("c ("):
        return "El efecto total" if es else "The total effect"
    if lab.startswith("c′") or lab.startswith("c'"):
        return "El efecto directo" if es else "The direct effect"
    if r["stat_symbol"] == "t" and " | " in lab:      # simple slope (moderation): "X | W = value"
        x, cond = lab.split(" | ", 1)
        return f"La pendiente simple de {x} cuando {cond}" if es else f"The simple slope of {x} when {cond}"
    if r["stat_symbol"] == "t" and " × " in lab:      # X × W interaction term (moderation)
        return f"La interacción {lab}" if es else f"The {lab} interaction"
    if " vs " in lab:
        a, b = lab.split(" vs ", 1)
        return f"La diferencia entre {a} y {b}" if es else f"The difference between {a} and {b}"
    if " – " in lab:
        a, b = lab.split(" – ", 1)
        return f"La correlación entre {a} y {b}" if es else f"The correlation between {a} and {b}"
    if " × " in lab and r["stat_symbol"] in ("χ²", "OR"):
        a, b = lab.split(" × ", 1)
        return f"La asociación entre {a} y {b}" if es else f"The association between {a} and {b}"
    if r["stat_symbol"] in ("t", "z"):
        return f"El predictor {lab}" if es else f"The predictor {lab}"
    if " × " in lab:
        return f"La interacción {lab}" if es else f"The {lab} interaction"
    return f"El efecto de {lab}" if es else f"The effect of {lab}"


def _desc_sentence(res, lang):
    dsc = res.descriptives
    if dsc is None or "M" not in dsc.columns:
        return ""
    parts = []
    ordinal = res.test in ("mannwhitney", "wilcoxon", "kruskal", "friedman")
    sd = "DT" if lang == "es" else "SD"
    for _, r in dsc.iterrows():
        if ordinal:
            iqr = "RIC" if lang == "es" else "IQR"
            parts.append(f"{r['grupo/group']} (Mdn = {r['Mdn']:.2f}, {iqr} = {r['RIC/IQR']:.2f}, n = {int(r['n'])})")
        else:
            parts.append(f"{r['grupo/group']} (M = {r['M']:.2f}, {sd} = {r['DT/SD']:.2f}, n = {int(r['n'])})")
    head = "Los estadísticos descriptivos fueron: " if lang == "es" else "Descriptive statistics were: "
    return head + "; ".join(parts) + "."


def _directional(d):
    return d.alternative != "two-sided" and recommend(d).test in DIRECTIONAL_TESTS


def _tails_es(d):
    if not _directional(d):
        return "bilateral"
    return f"contraste unilateral planificado a priori; H₁: {hypothesis_text(d, 'es')}"


def _tails_en(d):
    if not _directional(d):
        return "two-sided"
    return f"one-tailed test specified a priori; H₁: {hypothesis_text(d, 'en')}"


def _bound_txt(d, lang, raw=None):
    """'±0.50 (d de Cohen)' / '±2 puntos de <dv>' description of the equivalence bounds."""
    es = lang == "es"
    b = fmt_num(d.equivalence_bound)
    if d.equivalence_units == "raw":
        return f"±{b} {'en unidades de' if es else 'in units of'} {d.dv_name}"
    sym = "d_z" if d.design == "within" else "d"
    extra = f"; ±{fmt_num(raw)} {'en unidades de' if es else 'in units of'} {d.dv_name}" if raw is not None else ""
    return f"±{b} ({sym} de Cohen{extra})" if es else f"±{b} (Cohen's {sym}{extra})"


def _equivalence_methods(d, lang):
    if not getattr(d, "equivalence", False) or recommend(d).test not in EQUIVALENCE_TESTS:
        return ""
    if lang == "es":
        return (f"Para poder concluir la ausencia de un efecto relevante, se complementó la prueba con una prueba de equivalencia mediante "
                f"dos contrastes unilaterales (TOST; Schuirmann, 1987; Lakens, 2017) con límites de equivalencia de {_bound_txt(d, lang)}, "
                f"fijados como el menor tamaño del efecto de interés; se concluye equivalencia si ambos contrastes son significativos, "
                f"es decir, si el IC {round((1 - 2 * d.alpha) * 100):d}% de la diferencia queda dentro de los límites. "
                "El tamaño muestral se calculó para la prueba de diferencia, no para la de equivalencia. ")
    return (f"To be able to conclude the absence of a meaningful effect, the test was complemented with an equivalence test using "
            f"two one-sided tests (TOST; Schuirmann, 1987; Lakens, 2017) with equivalence bounds of {_bound_txt(d, lang)}, "
            f"set as the smallest effect size of interest; equivalence is concluded if both tests are significant, "
            f"that is, if the {round((1 - 2 * d.alpha) * 100):d}% CI of the difference lies within the bounds. "
            "Sample size was planned for the difference test, not for the equivalence test. ")


def equivalence_text(res, lang="es"):
    """APA paragraph for the TOST result plus the combined conclusion with the NHST (Lakens et al., 2018)."""
    eq, d = res.equivalence, res.design
    if not eq:
        return ""
    es = lang == "es"
    conf = f"{round(eq['conf'] * 100):d}%"
    ci = f"{'IC ' + conf if es else conf + ' CI'} [{fmt_num(eq['ci'][0])}, {fmt_num(eq['ci'][1])}]"
    t_max = eq["t_low"] if eq["p_low"] >= eq["p_upp"] else eq["t_upp"]
    df = f"{eq['df']:.2f}" if d.design == "between" else f"{eq['df']:.0f}"
    stat = f"t({df}) = {fmt_num(t_max)}, {fmt_p(eq['p'])}"
    bounds = _bound_txt(d, lang, eq["upp"] if eq["units"] == "d" else None)
    sig = res.rows[0]["p"] is not None and res.rows[0]["p"] < d.alpha
    if es:
        s = (f"Prueba de equivalencia (TOST): con límites de equivalencia de {bounds}, la diferencia observada ({eq['label']} = "
             f"{fmt_num(eq['diff'])}, {ci}) {'fue' if eq['equivalent'] else 'no fue'} estadísticamente equivalente a cero, {stat} "
             "(el mayor de los dos valores p unilaterales). ")
        s += ("Conclusión (Lakens et al., 2018): la diferencia es estadísticamente significativa pero menor que el efecto mínimo de interés; es real, "
              "aunque probablemente irrelevante en la práctica." if sig and eq["equivalent"] else
              "Conclusión (Lakens et al., 2018): hay una diferencia y no puede descartarse que sea de una magnitud relevante." if sig else
              "Conclusión (Lakens et al., 2018): no hubo una diferencia significativa y el efecto es menor que el mínimo de interés; los datos apoyan "
              "la ausencia de un efecto relevante." if eq["equivalent"] else
              "Conclusión (Lakens et al., 2018): resultado no concluyente; no hubo una diferencia significativa, pero tampoco puede afirmarse la equivalencia "
              "(haría falta una muestra mayor o un efecto mínimo de interés más amplio, justificado de antemano).")
        return s
    s = (f"Equivalence test (TOST): with equivalence bounds of {bounds}, the observed difference ({eq['label']} = "
         f"{fmt_num(eq['diff'])}, {ci}) was {'' if eq['equivalent'] else 'not '}statistically equivalent to zero, {stat} "
         "(the larger of the two one-sided p values). ")
    s += ("Conclusion (Lakens et al., 2018): the difference is statistically significant but smaller than the smallest effect of interest; it is real "
          "but probably negligible in practice." if sig and eq["equivalent"] else
          "Conclusion (Lakens et al., 2018): there is a difference and a meaningful effect size cannot be ruled out." if sig else
          "Conclusion (Lakens et al., 2018): there was no significant difference and the effect is smaller than the smallest effect of interest; the data "
          "support the absence of a meaningful effect." if eq["equivalent"] else
          "Conclusion (Lakens et al., 2018): inconclusive result; there was no significant difference, but equivalence cannot be claimed either "
          "(a larger sample or a wider, a priori justified, smallest effect of interest would be needed).")
    return s


def methods_text(design, lang="es"):
    d = design
    rec = recommend(d)
    ss = sample_size(d, rec)
    P = []
    if lang == "es":
        P.append(("Participantes", explain(ss, lang) +
                  " El tamaño muestral debe justificarse antes de recoger los datos (Lakens, 2022)."))
        if d.analysis == "compare":
            ds = f"Se empleó un diseño {DESIGN_TXT[d.design][0]}"
            if d.design == "mixed":
                ds += (f" {d.n_groups} × {d.n_levels_b}, con {d.iv_name} ({', '.join(d.levels)}) como factor entre sujetos "
                       f"y {d.factor_b_name} ({', '.join(d.levels_b)}) como factor intra sujetos")
            elif d.n_factors == 2:
                ds += (f" factorial {d.n_groups} × {d.n_levels_b}, con los factores {d.iv_name} ({', '.join(d.levels)}) "
                       f"y {d.factor_b_name} ({', '.join(d.levels_b)})")
            else:
                ds += f" con {d.n_groups} {'condiciones' if d.design == 'within' else 'grupos'} ({', '.join(d.levels)}) de la variable independiente {d.iv_name}"
            ds += f". La variable dependiente fue {d.dv_name}"
            if d.covariates:
                ds += f", controlando las covariables {', '.join(d.covariates)}"
            P.append(("Diseño", ds + "."))
        elif d.analysis == "correlation":
            P.append(("Diseño", f"Se empleó un diseño correlacional para examinar la relación entre {d.iv_name} y {d.dv_name}."))
        elif d.analysis == "regression":
            P.append(("Diseño", f"Se empleó un diseño correlacional-predictivo con {d.dv_name} como variable criterio y "
                                f"{', '.join(d.predictors)} como predictor(es)."))
        elif d.analysis == "mediation":
            P.append(("Diseño", f"Se examinó un modelo de mediación simple con {d.iv_name} como predictor (X), "
                                f"{d.mediator} como mediador (M) y {d.dv_name} como variable dependiente (Y)."))
        elif d.analysis == "reliability":
            ds = (f"Se evaluó la fiabilidad (consistencia interna) de la escala {d.dv_name}, formada por {len(d.items)} ítems "
                  f"({', '.join(d.items)}) con formato de respuesta de {d.scale_min:g} a {d.scale_max:g}")
            if d.reverse_items:
                ds += (f"; los ítems {', '.join(d.reverse_items)} se recodificaron de forma inversa "
                       f"({d.scale_min:g} + {d.scale_max:g} − x) antes del análisis")
            P.append(("Diseño", ds + "."))
        elif d.analysis == "agreement":
            kind = (f"clasificaron {d.dv_name} en {len(d.dv_levels)} categorías{' ordenadas' if d.dv_type == 'ordinal' else ''} "
                    f"({', '.join(d.dv_levels)})" if d.dv_type != "continuous" else f"puntuaron {d.dv_name}")
            P.append(("Diseño", f"Se evaluó el acuerdo entre evaluadores (fiabilidad interjueces): {len(d.raters)} evaluadores "
                                f"({', '.join(d.raters)}) {kind} de forma independiente para cada caso."))
        elif d.analysis == "multilevel":
            P.append(("Diseño", f"Se empleó un diseño multinivel (datos anidados) de dos niveles: las observaciones (nivel 1) estaban agrupadas en "
                                f"unidades de {d.cluster} (nivel 2). La variable dependiente fue {d.dv_name} y los predictores, {', '.join(d.predictors)}"
                                + (f"; se permitió que el efecto de {d.random_slope} variase entre grupos (pendiente aleatoria)" if d.random_slope else "") + "."))
        elif d.analysis == "moderation":
            P.append(("Diseño", f"Se examinó un modelo de moderación con {d.iv_name} como predictor (X), {d.moderator} como "
                                f"moderador (W) y {d.dv_name} como variable dependiente (Y), evaluando la interacción X × W."))
        else:
            P.append(("Diseño", f"Se examinó la asociación entre las variables categóricas {d.iv_name} y {d.dv_name}."))
        ana = (f"El análisis principal planificado fue: {t(rec.test, lang)}. "
               f"Antes del análisis se comprobaron los supuestos: {', '.join(assumption_label(a, lang)[0].lower() + assumption_label(a, lang)[1:] for a in rec.assumptions)}. ")
        if rec.fallback:
            ana += f"Si no se cumplían los supuestos, se planificó como alternativa: {t(rec.fallback, lang)}. "
        ana += " ".join(rec.notes_text(lang)) + " "
        ana += ("Se informan los coeficientes de fiabilidad con intervalos de confianza al 95%. " if rec.test == "reliability" else
                "Se informan los coeficientes de acuerdo con intervalos de confianza al 95%. " if rec.test in AGREEMENT_TESTS else
                "Se informan los coeficientes no estandarizados con intervalos de confianza al 95% y los componentes de varianza. " if rec.test == "mixedlm" else
                "Se informan tamaños del efecto con intervalos de confianza al 95%. ")
        ana += _equivalence_methods(d, "es")
        ana += (f"El nivel de significación fue α = {fmt_num(d.alpha, True)} "
                f"({_tails_es(d)}). Los análisis se realizaron con PsicoStat v{__version__} (Python; SciPy y statsmodels).")
        P.append(("Análisis de datos", ana))
    else:
        P.append(("Participants", explain(ss, lang) + " Sample size should be justified before data collection (Lakens, 2022)."))
        if d.analysis == "compare":
            ds = f"A {DESIGN_TXT[d.design][1]} design was used"
            if d.design == "mixed":
                ds = (f"A {d.n_groups} × {d.n_levels_b} mixed design was used, with {d.iv_name} ({', '.join(d.levels)}) as "
                      f"between-subjects factor and {d.factor_b_name} ({', '.join(d.levels_b)}) as within-subjects factor")
            elif d.n_factors == 2:
                ds = (f"A {d.n_groups} × {d.n_levels_b} {DESIGN_TXT[d.design][1]} factorial design was used, with factors "
                      f"{d.iv_name} ({', '.join(d.levels)}) and {d.factor_b_name} ({', '.join(d.levels_b)})")
            else:
                ds += f" with {d.n_groups} {'conditions' if d.design == 'within' else 'groups'} ({', '.join(d.levels)}) of the independent variable {d.iv_name}"
            ds += f". The dependent variable was {d.dv_name}"
            if d.covariates:
                ds += f", controlling for {', '.join(d.covariates)}"
            P.append(("Design", ds + "."))
        elif d.analysis == "correlation":
            P.append(("Design", f"A correlational design was used to examine the relationship between {d.iv_name} and {d.dv_name}."))
        elif d.analysis == "regression":
            P.append(("Design", f"A correlational-predictive design was used with {d.dv_name} as outcome and "
                                f"{', '.join(d.predictors)} as predictor(s)."))
        elif d.analysis == "mediation":
            P.append(("Design", f"A simple mediation model was examined with {d.iv_name} as predictor (X), "
                                f"{d.mediator} as mediator (M) and {d.dv_name} as dependent variable (Y)."))
        elif d.analysis == "reliability":
            ds = (f"The reliability (internal consistency) of the {d.dv_name} scale was assessed; it comprises {len(d.items)} items "
                  f"({', '.join(d.items)}) with a {d.scale_min:g}–{d.scale_max:g} response format")
            if d.reverse_items:
                ds += (f"; items {', '.join(d.reverse_items)} were reverse-scored ({d.scale_min:g} + {d.scale_max:g} − x) "
                       "before the analysis")
            P.append(("Design", ds + "."))
        elif d.analysis == "agreement":
            kind = (f"classified {d.dv_name} into {len(d.dv_levels)} {'ordered ' if d.dv_type == 'ordinal' else ''}categories "
                    f"({', '.join(d.dv_levels)})" if d.dv_type != "continuous" else f"scored {d.dv_name}")
            P.append(("Design", f"Inter-rater agreement (inter-rater reliability) was assessed: {len(d.raters)} raters "
                                f"({', '.join(d.raters)}) independently {kind} for each case."))
        elif d.analysis == "multilevel":
            P.append(("Design", f"A two-level multilevel (nested) design was used: observations (level 1) were nested in {d.cluster} units (level 2). "
                                f"The dependent variable was {d.dv_name} and the predictors were {', '.join(d.predictors)}"
                                + (f"; the effect of {d.random_slope} was allowed to vary across clusters (random slope)" if d.random_slope else "") + "."))
        elif d.analysis == "moderation":
            P.append(("Design", f"A moderation model was examined with {d.iv_name} as predictor (X), {d.moderator} as "
                                f"moderator (W) and {d.dv_name} as dependent variable (Y), testing the X × W interaction."))
        else:
            P.append(("Design", f"The association between the categorical variables {d.iv_name} and {d.dv_name} was examined."))
        ana = (f"The planned main analysis was: {t(rec.test, lang)}. Assumptions were checked beforehand: "
               f"{', '.join(assumption_label(a, lang)[0].lower() + assumption_label(a, lang)[1:] for a in rec.assumptions)}. ")
        if rec.fallback:
            ana += f"If assumptions were not met, the planned alternative was: {t(rec.fallback, lang)}. "
        ana += " ".join(rec.notes_text(lang)) + " "
        ana += ("Reliability coefficients are reported with 95% confidence intervals. " if rec.test == "reliability" else
                "Agreement coefficients are reported with 95% confidence intervals. " if rec.test in AGREEMENT_TESTS else
                "Unstandardized coefficients are reported with 95% confidence intervals, together with the variance components. " if rec.test == "mixedlm" else
                "Effect sizes are reported with 95% confidence intervals. ")
        ana += _equivalence_methods(d, "en")
        ana += (f"The significance level was α = {fmt_num(d.alpha, True)} "
                f"({_tails_en(d)}). Analyses were run with PsicoStat v{__version__} (Python; SciPy and statsmodels).")
        P.append(("Data analysis", ana))
    return P


def results_text(res, lang="es"):
    d = res.design
    es = lang == "es"
    out = []
    intro = []
    if res.n_excluded:
        intro.append(f"Se excluyeron {res.n_excluded} casos con datos incompletos; la muestra analizada fue N = {res.n}." if es else
                     f"{res.n_excluded} cases with incomplete data were excluded; the analysed sample was N = {res.n}.")
    else:
        intro.append(f"La muestra analizada fue N = {res.n}." if es else f"The analysed sample was N = {res.n}.")
    ds = _desc_sentence(res, lang)
    if ds:
        intro.append(ds)
    out.append(" ".join(intro))
    # assumptions
    if res.assumptions:
        bad = [a for a in res.assumptions if not a["ok"]]
        if bad:
            s = ("Se detectaron incumplimientos de supuestos: " if es else "Assumption violations were detected: ") + "; ".join(
                f"{assumption_label(a['key'], lang)} ({a['detail']})" for a in bad) + "."
        else:
            s = "Se cumplieron los supuestos evaluados." if es else "The assumptions checked were met."
        if res.fallback_reason:
            s += (f" Por ello se aplicó la alternativa planificada: {t(res.test, lang)} (en lugar de: {t(res.planned_test, lang)})." if es else
                  f" Therefore the planned alternative was used: {t(res.test, lang)} (instead of: {t(res.planned_test, lang)}).")
        elif res.test != res.planned_test:
            s += f" Prueba aplicada: {t(res.test, lang)}." if es else f" Test used: {t(res.test, lang)}."
        out.append(s)
    for note in res.notes:
        out.append(note[_L(lang)])
    if any(r.get("alternative", "two-sided") != "two-sided" for r in res.rows):
        r0 = res.rows[0]
        wrong = r0["es"] is not None and ((r0["alternative"] == "greater" and r0["es"] < 0) or (r0["alternative"] == "less" and r0["es"] > 0))
        hyp = hypothesis_text(d, lang)
        if es:
            s = f"Se aplicó un contraste unilateral (H₁: {hyp}); los valores p son unilaterales."
            if wrong:
                s += (" El efecto observado va en la dirección contraria a la hipótesis, por lo que el contraste unilateral "
                      "no permite concluir un efecto en esa dirección.")
        else:
            s = f"A one-tailed test was used (H₁: {hyp}); p values are one-tailed."
            if wrong:
                s += " The observed effect goes in the opposite direction to the hypothesis, so the one-tailed test cannot support an effect in that direction."
        out.append(s)
    if res.test == "reliability" or res.test in AGREEMENT_TESTS:      # the APA paragraph is built in the analysis (notes); no "significant" boilerplate
        return [p for p in out if p]
    # main results
    tl = t(res.test, lang)
    sents = []
    for r in res.rows:
        mag = magnitude(r["es_symbol"], r["es"]) if r["es_symbol"] else None
        magtxt = ""
        if mag:
            magtxt = (f"; tamaño del efecto {t(mag, lang)}" if es else f"; {t(mag, lang)} effect size")
        subj, sig = _subject(r, lang), _sig(r, d.alpha, lang)
        if es and subj.startswith("La "):
            sig = sig.replace("significativo", "significativa")
        sents.append(f"{subj} {sig}, {stat_str(r, lang)}{magtxt}.")
    head = (f"Resultados ({tl}): " if es else f"Results ({tl}): ")
    out.append(head + " ".join(sents))
    if getattr(res, "equivalence", None):
        out.append(equivalence_text(res, lang))
    if res.posthoc is not None and "p_ajust/p_adj" in res.posthoc.columns and "A" in res.posthoc.columns:
        sig = res.posthoc[res.posthoc["p_ajust/p_adj"] < d.alpha]
        if len(sig):
            pairs = "; ".join(f"{r['A']} vs {r['B']} ({fmt_p(r['p_ajust/p_adj'])})" for _, r in sig.iterrows())
            out.append(f"Las comparaciones post hoc ({res.posthoc_method}) mostraron diferencias significativas en: {pairs}." if es else
                        f"Post hoc comparisons ({res.posthoc_method}) showed significant differences for: {pairs}.")
        else:
            out.append(f"Ninguna comparación post hoc ({res.posthoc_method}) fue significativa." if es else
                       f"No post hoc comparison ({res.posthoc_method}) was significant.")
    if getattr(res, "equivalence", None):
        return [p for p in out if p]
    out.append("Recuerda: un resultado no significativo no demuestra ausencia de efecto." if es and any(r["p"] is not None and r["p"] >= d.alpha for r in res.rows)
               else ("Remember: a non-significant result does not show the absence of an effect." if (not es and any(r["p"] is not None and r["p"] >= d.alpha for r in res.rows)) else ""))
    return [p for p in out if p]


def _tables(res, lang):
    es = lang == "es"
    tabs = []
    if res.descriptives is not None:
        tabs.append(("Estadísticos descriptivos" if es else "Descriptive statistics", res.descriptives))
    if res.assumptions:
        tabs.append(("Comprobación de supuestos" if es else "Assumption checks",
                     pd.DataFrame([{("supuesto" if es else "assumption"): assumption_label(a["key"], lang), ("detalle" if es else "detail"): a["detail"],
                                    ("cumple" if es else "met"): ("sí" if es else "yes") if a["ok"] else "no"} for a in res.assumptions])))
    for k, v in res.extra_tables.items():
        name = {"coefficients": ("Coeficientes", "Coefficients"), "model": ("Ajuste del modelo", "Model fit"),
                "adjusted_means": ("Medias ajustadas", "Adjusted means"),
                "thresholds": ("Umbrales (puntos de corte)", "Thresholds (cut-points)"),
                "brant": ("Prueba de Brant de odds proporcionales", "Brant test of proportional odds"),
                "paths": ("Sendas de mediación", "Mediation paths"),
                "indirect": ("Efecto indirecto (bootstrap)", "Indirect effect (bootstrap)"),
                "r2_change": ("Cambio en R² por la interacción X × W", "R² change due to the X × W interaction"),
                "simple_slopes": ("Pendientes simples de X según W", "Simple slopes of X by W"),
                "reliability": ("Coeficientes de fiabilidad", "Reliability coefficients"),
                "items": ("Estadísticos de los ítems", "Item statistics"),
                "agreement": ("Acuerdo entre evaluadores", "Inter-rater agreement"),
                "crosstab": ("Tabla de contingencia entre evaluadores", "Rater contingency table"),
                "categories": ("Frecuencia de cada categoría por evaluador", "Category frequencies by rater"),
                "icc": ("Coeficientes de correlación intraclase (modelo de dos factores)", "Intraclass correlation coefficients (two-way model)"),
                "ml_null": ("Modelo nulo: varianza entre grupos y correlación intraclase", "Null model: between-cluster variance and intraclass correlation"),
                "fixed": ("Efectos fijos", "Fixed effects"),
                "random": ("Efectos aleatorios (componentes de varianza)", "Random effects (variance components)"),
                "equivalence": ("Prueba de equivalencia: dos contrastes unilaterales (TOST)", "Equivalence test: two one-sided tests (TOST)")}.get(k, (k, k))[_L(lang)]
        tabs.append((name, v))
    if res.posthoc is not None:
        tabs.append(((f"Post hoc: {res.posthoc_method}"), res.posthoc))
    return tabs


def _fmt_cell(col, x, comma):
    if isinstance(x, (float, np.floating)):
        if pd.isna(x):
            return ""
        out = fmt_p(x).replace("p = ", "").replace("p ", "") if str(col).startswith("p") else f"{x:.2f}"
    else:
        out = str(x)
    return to_decimal_comma(out) if comma else out


def format_table(df: pd.DataFrame, comma=False) -> pd.DataFrame:
    """Return an all-string copy of a table with APA-style numbers (optionally decimal comma)."""
    out = df.copy()
    for c in out.columns:
        out[c] = [_fmt_cell(c, x, comma) for x in out[c]]
    return out


def build_report(res, lang="es", decimal_comma=None):
    """Build the report dict. decimal_comma=None uses design.decimal_comma; it only applies to Spanish."""
    if decimal_comma is None:
        decimal_comma = getattr(res.design, "decimal_comma", False)
    comma = bool(decimal_comma) and lang == "es"
    conv = to_decimal_comma if comma else (lambda x: x)
    methods = [(h, conv(p)) for h, p in methods_text(res.design, lang)]
    results = [conv(p) for p in results_text(res, lang)]
    tables = [(name, format_table(df, comma)) for name, df in _tables(res, lang)]
    return {"title": res.design.title or ("Informe de análisis" if lang == "es" else "Analysis report"),
            "question": res.design.question, "methods": methods, "results": results, "tables": tables,
            "figures": [f for f in res.figures if f], "lang": lang, "decimal_comma": comma}


def _md_table(df: pd.DataFrame):
    df = format_table(df)
    cols = [str(c).replace("|", "\\|") for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in r.values) + " |")
    return "\n".join(lines)


def to_markdown(rep, path=None, rel_to=None):
    es = rep["lang"] == "es"
    L = [f"# {rep['title']}", ""]
    if rep["question"]:
        L += [f"**{'Pregunta de investigación' if es else 'Research question'}:** {rep['question']}", ""]
    L += [f"## {'Método' if es else 'Method'}", ""]
    for h, p in rep["methods"]:
        L += [f"### {h}", "", p, ""]
    L += [f"## {'Resultados' if es else 'Results'}", ""]
    L += [p + "\n" for p in rep["results"]]
    for i, (name, df) in enumerate(rep["tables"], 1):
        L += [f"**{'Tabla' if es else 'Table'} {i}**", "", f"*{name}*", "", _md_table(df), ""]
    for i, f in enumerate(rep["figures"], 1):
        src = os.path.relpath(f, rel_to) if rel_to else f
        L += [f"**{'Figura' if es else 'Figure'} {i}**", "", f"![]({src})", ""]
    L += ["---", "", f"*{t('disclaimer', rep['lang'])}*", ""]
    md = "\n".join(L)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(md)
    return md


def to_docx(rep, path):
    from docx import Document
    from docx.shared import Inches, Pt

    es = rep["lang"] == "es"
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    doc.add_heading(rep["title"], 0)
    if rep["question"]:
        doc.add_paragraph(rep["question"])
    doc.add_heading("Método" if es else "Method", 1)
    for h, p in rep["methods"]:
        doc.add_heading(h, 2)
        doc.add_paragraph(p)
    doc.add_heading("Resultados" if es else "Results", 1)
    for p in rep["results"]:
        doc.add_paragraph(p)
    for i, (name, df) in enumerate(rep["tables"], 1):
        doc.add_paragraph().add_run(f"{'Tabla' if es else 'Table'} {i}").bold = True
        doc.add_paragraph().add_run(name).italic = True
        tb = doc.add_table(rows=1, cols=len(df.columns))
        tb.style = "Table Grid"
        for j, c in enumerate(df.columns):
            tb.rows[0].cells[j].text = str(c)
        for _, r in df.iterrows():
            cells = tb.add_row().cells
            for j, v in enumerate(r.values):
                cells[j].text = _fmt_cell(df.columns[j], v, rep.get("decimal_comma", False))
        doc.add_paragraph()
    for i, f in enumerate(rep["figures"], 1):
        doc.add_paragraph().add_run(f"{'Figura' if es else 'Figure'} {i}").bold = True
        doc.add_picture(f, width=Inches(5.5))
    doc.add_paragraph().add_run(t("disclaimer", rep["lang"])).italic = True
    doc.save(path)
    return path
