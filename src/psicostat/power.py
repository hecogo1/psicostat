"""A priori sample size computation (statsmodels + noncentral distributions)."""
from __future__ import annotations

import math

from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

from .design import BENCHMARKS, Design, recommend

ARE_RANK = 0.864   # minimum asymptotic relative efficiency of rank tests (Hodges & Lehmann, 1956)
NONPARAM = {"mannwhitney", "wilcoxon", "kruskal", "friedman"}


def effect_value(d: Design, family: str) -> float:
    if d.effect_size is not None:
        return float(d.effect_size)
    return BENCHMARKS[family][d.effect_label]


def _min_n(power_fn, target, n_min, n_max=100000):
    """Smallest integer n >= n_min with power_fn(n) >= target (power is increasing in n)."""
    lo, hi = n_min, n_min
    while power_fn(hi) < target:
        lo, hi = hi, hi * 2
        if hi > n_max:
            raise ValueError("Sample size too large (> 100000); revise the effect size.")
    while lo < hi:
        mid = (lo + hi) // 2
        if power_fn(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _f_power(lam, df1, df2, alpha):
    if df2 <= 0:
        return 0.0
    crit = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(crit, df1, df2, lam))


def _chi2_power(lam, df, alpha):
    crit = stats.chi2.ppf(1 - alpha, df)
    return float(stats.ncx2.sf(crit, df, lam))


def sample_size(d: Design, rec=None) -> dict:
    """Return dict with n_total, n_per_group, effect, family, method (es/en text) and details."""
    rec = rec or recommend(d)
    test, a, pw = rec.test, d.alpha, d.power
    fam = rec.family
    es = effect_value(d, fam)
    groups = 1
    method = ("", "")
    if test in ("ttest_ind", "mannwhitney"):
        n1 = TTestIndPower().solve_power(effect_size=es, alpha=a, power=pw, ratio=1.0, alternative="two-sided")
        npg = math.ceil(n1)
        groups = 2
        n_total = 2 * npg
        method = ("prueba t para dos muestras independientes (bilateral)", "two-sample t test (two-sided)")
    elif test in ("ttest_paired", "wilcoxon"):
        n_total = math.ceil(TTestPower().solve_power(effect_size=es, alpha=a, power=pw, alternative="two-sided"))
        npg = n_total
        method = ("prueba t para muestras relacionadas (d_z, bilateral)", "paired t test (d_z, two-sided)")
    elif test in ("anova_oneway", "kruskal"):
        k = d.n_groups
        n_total = _min_n(lambda n: _f_power(es ** 2 * n, k - 1, n - k, a), pw, k + 1)
        n_total = math.ceil(n_total / k) * k
        groups = k
        npg = n_total // k
        method = (f"ANOVA de un factor con {k} grupos (F, efecto f)", f"one-way ANOVA with {k} groups (F, effect f)")
    elif test in ("anova_factorial", "ancova"):
        if test == "ancova":
            k, c = d.n_groups, len(d.covariates)
            effects = [(k - 1, k + c)]
            groups = k
        else:
            ka, kb = d.n_groups, d.n_levels_b
            groups = ka * kb
            effects = [(ka - 1, groups), (kb - 1, groups), ((ka - 1) * (kb - 1), groups)]
        n_total = max(_min_n(lambda n, df1=df1, p=p: _f_power(es ** 2 * n, df1, n - p, a), pw, p + 2)
                      for df1, p in effects)
        n_total = math.ceil(n_total / groups) * groups
        npg = n_total // groups
        method = (("ANCOVA (F, efecto f del factor ajustado por covariables)", "ANCOVA (F, effect f of the adjusted factor)")
                  if test == "ancova" else
                  ("ANOVA factorial (máximo N entre efectos principales e interacción)",
                   "factorial ANOVA (largest N across main effects and interaction)"))
    elif test in ("anova_rm", "friedman"):
        if d.n_factors == 2:
            m = d.n_groups * d.n_levels_b
            dfs = [d.n_groups - 1, d.n_levels_b - 1, (d.n_groups - 1) * (d.n_levels_b - 1)]
        else:
            m = d.n_groups
            dfs = [m - 1]
        rho = d.rho
        n_total = max(_min_n(lambda n, df1=df1: _f_power(es ** 2 * n * m / (1 - rho), df1, (n - 1) * df1, a), pw, 3)
                      for df1 in dfs)
        npg = n_total
        method = (f"ANOVA de medidas repetidas ({m} medidas, ρ = {rho}, ε = 1)",
                  f"repeated-measures ANOVA ({m} measurements, ρ = {rho}, ε = 1)")
    elif test == "anova_mixed":
        g, m, rho = d.n_groups, d.n_levels_b, d.rho
        groups = g
        cand = [
            _min_n(lambda n: _f_power(es ** 2 * n * m / (1 + (m - 1) * rho), g - 1, n - g, a), pw, g + 1),
            _min_n(lambda n: _f_power(es ** 2 * n * m / (1 - rho), m - 1, (n - g) * (m - 1), a), pw, g + 1),
            _min_n(lambda n: _f_power(es ** 2 * n * m / (1 - rho), (g - 1) * (m - 1), (n - g) * (m - 1), a), pw, g + 1),
        ]
        n_total = math.ceil(max(cand) / g) * g
        npg = n_total // g
        method = (f"ANOVA mixto {g}×{m} (máximo N entre efecto entre, intra e interacción; ρ = {rho})",
                  f"{g}×{m} mixed ANOVA (largest N across between, within and interaction effects; ρ = {rho})")
    elif test in ("chi2", "fisher", "mcnemar", "cochran"):
        if test == "chi2" or test == "fisher":
            df = (d.n_groups - 1) * (len(d.dv_levels) - 1)
        else:
            df = max(d.n_groups - 1, 1)
        df = max(df, 1)
        n_total = _min_n(lambda n: _chi2_power(es ** 2 * n, df, a), pw, 2)
        npg = None
        method = (f"prueba chi-cuadrado (gl = {df}, efecto w)", f"chi-square test (df = {df}, effect w)")
    elif test in ("pearson", "spearman"):
        z = stats.norm.ppf(1 - a / 2) + stats.norm.ppf(pw)
        n_total = math.ceil((z / math.atanh(es)) ** 2 + 3)
        npg = None
        method = ("correlación (transformación z de Fisher, bilateral)", "correlation (Fisher z transformation, two-sided)")
        if test == "spearman":
            n_total = math.ceil(n_total / 0.912)  # Bonett & Wright (2000) approximate efficiency
            method = (method[0] + "; ajuste por eficiencia de Spearman (÷ 0.912)", method[1] + "; Spearman efficiency adjustment (÷ 0.912)")
    elif test == "linreg":
        p = len(d.predictors)
        n_total = _min_n(lambda n: _f_power(es * n, p, n - p - 1, a), pw, p + 3)
        npg = None
        method = (f"regresión lineal, prueba F de R² con {p} predictor(es) (efecto f²)",
                  f"linear regression, F test of R² with {p} predictor(s) (effect f²)")
    elif test == "logreg":
        p = len(d.predictors)
        pmin = min(d.event_rate, 1 - d.event_rate)
        n_total = math.ceil(10 * p / pmin)
        n_total = max(n_total, 100)
        npg = None
        method = (f"regla de 10 eventos por predictor (Peduzzi et al., 1996) con proporción del evento = {d.event_rate} y mínimo de 100",
                  f"10 events-per-predictor rule (Peduzzi et al., 1996) with event proportion = {d.event_rate} and minimum of 100")
    else:  # pragma: no cover
        raise ValueError(test)

    notes = []
    if test in NONPARAM:
        n_total = math.ceil(n_total / ARE_RANK)
        if npg:
            npg = math.ceil(npg / ARE_RANK)
            n_total = npg * groups
        notes.append(("Prueba de rangos: N inflado dividiendo por la eficiencia relativa asintótica mínima (0.864).",
                      "Rank-based test: N inflated by dividing by the minimum asymptotic relative efficiency (0.864)."))
    n_before = n_total
    if d.dropout and d.dropout > 0:
        if npg:
            npg = math.ceil(npg / (1 - d.dropout))
            n_total = npg * groups
        else:
            n_total = math.ceil(n_total / (1 - d.dropout))
        notes.append((f"Incluye un {d.dropout:.0%} de pérdida esperada (N sin ajuste = {n_before}).",
                      f"Includes {d.dropout:.0%} expected dropout (unadjusted N = {n_before})."))
    return {"test": test, "family": fam, "effect": es, "effect_label": d.effect_label if d.effect_size is None else "custom",
            "alpha": a, "power": pw, "n_total": int(n_total), "n_per_group": int(npg) if npg else None,
            "groups": groups, "method": method, "notes": notes}


def _b(x):
    return f"{x:.2f}".lstrip("0")


def explain(ss: dict, lang="es") -> str:
    sym = {"d": "d", "f": "f", "r": "r", "w": "w", "f2": "f²", "or": "OR"}[ss["family"]]
    i = 0 if lang == "es" else 1
    if lang == "es":
        txt = (f"Análisis de potencia a priori ({ss['method'][i]}): para detectar un efecto {sym} = {ss['effect']:.2f} "
               f"con α = {_b(ss['alpha'])} y potencia = {_b(ss['power'])} se necesitan N = {ss['n_total']} participantes")
        if ss["n_per_group"] and ss["groups"] > 1:
            txt += f" ({ss['n_per_group']} por grupo)"
    else:
        txt = (f"A priori power analysis ({ss['method'][i]}): to detect an effect of {sym} = {ss['effect']:.2f} "
               f"with α = {_b(ss['alpha'])} and power = {_b(ss['power'])}, N = {ss['n_total']} participants are required")
        if ss["n_per_group"] and ss["groups"] > 1:
            txt += f" ({ss['n_per_group']} per group)"
    txt += "."
    for n in ss["notes"]:
        txt += " " + n[i]
    return txt
