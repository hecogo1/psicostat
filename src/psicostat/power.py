"""A priori sample size computation (statsmodels + noncentral distributions)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

from .design import BENCHMARKS, DIRECTIONAL_TESTS, Design, recommend

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
    one = d.alternative != "two-sided" and test in DIRECTIONAL_TESTS
    side_es, side_en = ("unilateral", "one-sided") if one else ("bilateral", "two-sided")
    sm_alt = "larger" if one else "two-sided"
    if test in ("ttest_ind", "mannwhitney"):
        n1 = TTestIndPower().solve_power(effect_size=abs(es), alpha=a, power=pw, ratio=1.0, alternative=sm_alt)
        npg = math.ceil(n1)
        groups = 2
        n_total = 2 * npg
        method = (f"prueba t para dos muestras independientes ({side_es})", f"two-sample t test ({side_en})")
    elif test in ("ttest_paired", "wilcoxon"):
        n_total = math.ceil(TTestPower().solve_power(effect_size=abs(es), alpha=a, power=pw, alternative=sm_alt))
        npg = n_total
        method = (f"prueba t para muestras relacionadas (d_z, {side_es})", f"paired t test (d_z, {side_en})")
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
        z = stats.norm.ppf(1 - (a if one else a / 2)) + stats.norm.ppf(pw)
        n_total = math.ceil((z / math.atanh(abs(es))) ** 2 + 3)
        npg = None
        method = (f"correlación (transformación z de Fisher, {side_es})", f"correlation (Fisher z transformation, {side_en})")
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
    elif test == "ordreg":
        k = max(len(d.dv_levels), 2)
        if es <= 0 or es == 1:
            raise ValueError("La OR esperada debe ser > 0 y distinta de 1 / the expected OR must be > 0 and different from 1")
        za = stats.norm.ppf(1 - a / 2)
        n_total = max(math.ceil(3 * (za + stats.norm.ppf(pw)) ** 2 / (math.log(es) ** 2 * (1 - 1 / k ** 2))), 10)
        npg = None
        method = (f"regresión ordinal de odds proporcionales, fórmula de Whitehead (1993) para un predictor estandarizado "
                  f"(OR por DT) con {k} categorías equiprobables",
                  f"proportional-odds ordinal regression, Whitehead (1993) formula for one standardized predictor "
                  f"(OR per SD) with {k} equally likely categories")
    elif test == "mediation":
        # Fritz & MacKinnon (2007), Table 3, percentile bootstrap, α=.05, power=.80, equal paths a = b
        # (empirical empirical N for detecting a×b; not an analytical power formula).
        table = ((0.14, 462), (0.26, 148), (0.39, 78), (0.59, 36))
        n_total = table[0][1]
        for thr, n in table:
            if abs(es) >= thr - 1e-12:
                n_total = n
        # scale roughly if power/alpha differ from .80/.05 (conservative: only support the published cell)
        npg = None
        method = (f"mediación simple, N orientativo de Fritz y MacKinnon (2007, tabla 3, bootstrap percentil) "
                  f"para sendas estandarizadas a = b = {es:.2f} (α = .05, potencia = .80)",
                  f"simple mediation, approximate N from Fritz and MacKinnon (2007, Table 3, percentile bootstrap) "
                  f"for standardized paths a = b = {es:.2f} (α = .05, power = .80)")
    elif test == "moderation":
        # F test of the R² increase due to X × W (1 df) in a model with 3 predictors (X, W, X × W); λ = f² · N
        # (G*Power: linear multiple regression, fixed model, R² increase).
        n_total = _min_n(lambda n: _f_power(es * n, 1, n - 4, a), pw, 6)
        npg = None
        method = ("moderación, prueba F del incremento de R² por la interacción X × W (1 gl, 3 predictores; efecto f²)",
                  "moderation, F test of the R² increase due to the X × W interaction (1 df, 3 predictors; effect f²)")
    elif test == "reliability":
        n_total = bonett_alpha_n(len(d.items), d.alpha_null, es, a, pw)
        npg = None
        method = (f"fiabilidad, fórmula de Bonett (2002) para contrastar H₀: α = {_b(d.alpha_null)} frente a un α de Cronbach esperado "
                  f"de {_b(es)} con {len(d.items)} ítems",
                  f"reliability, Bonett (2002) formula to test H₀: α = {_b(d.alpha_null)} against an expected Cronbach's α "
                  f"of {_b(es)} with {len(d.items)} items")
    elif test == "icc":
        k = len(d.raters)
        n_total = icc_n(k, d.agree_null, es, a, pw)
        npg = None
        method = (f"acuerdo entre evaluadores, potencia exacta de la prueba F del CCI para contrastar H₀: CCI = {_b(d.agree_null)} "
                  f"frente a un CCI esperado de {_b(es)} con {k} evaluadores",
                  f"inter-rater agreement, exact power of the ICC F test of H₀: ICC = {_b(d.agree_null)} "
                  f"against an expected ICC of {_b(es)} with {k} raters")
    elif test in ("kappa", "fleiss"):
        c = len(d.dv_levels)
        n_total = kappa_n(c, d.agree_null, es, a, pw)
        npg = None
        method = (f"acuerdo entre evaluadores, prueba z de la κ de Cohen (varianza asintótica de Fleiss, Cohen y Everitt, 1969) para "
                  f"contrastar H₀: κ = {_b(d.agree_null)} frente a una κ esperada de {_b(es)} con {c} categorías equiprobables",
                  f"inter-rater agreement, z test of Cohen's κ (Fleiss, Cohen & Everitt, 1969 asymptotic variance) of "
                  f"H₀: κ = {_b(d.agree_null)} against an expected κ of {_b(es)} with {c} equally likely categories")
    elif test == "mixedlm":
        # Effective N for the F (≡ t²) test of one fixed effect (1 df) among p predictors, λ = f² · N
        # (G*Power: linear multiple regression, R² increase), inflated by the design effect DE = 1 + (m − 1)·ICC
        # (Kish, 1965) and rounded up to whole clusters of m observations.
        p, m, icc = len(d.predictors), d.cluster_size, d.icc_plan
        n_eff = _min_n(lambda n: _f_power(es * n, 1, n - p - 1, a), pw, p + 3)
        deff = 1 + (m - 1) * icc
        groups = max(math.ceil(n_eff * deff / m), 2)
        npg = m
        n_total = groups * m
        method = (f"modelo multinivel, prueba de un efecto fijo (1 gl, {p} predictor(es); efecto f²) con N efectivo = {n_eff}, "
                  f"multiplicado por el efecto de diseño 1 + ({m} − 1) × {_b(icc)} = {deff:.2f} (Kish, 1965) para grupos de {m} observaciones",
                  f"multilevel model, test of one fixed effect (1 df, {p} predictor(s); effect f²) with effective N = {n_eff}, "
                  f"multiplied by the design effect 1 + ({m} − 1) × {_b(icc)} = {deff:.2f} (Kish, 1965) for clusters of {m} observations")
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
    if test == "ordreg":
        notes.append(("La fórmula de Whitehead es una aproximación: supone categorías igual de frecuentes y, en muestras pequeñas, la potencia real "
                      "puede quedar algo por debajo de la nominal. Si esperas categorías muy desequilibradas, aumenta N.",
                      "Whitehead's formula is an approximation: it assumes equally frequent categories and, in small samples, actual power may fall "
                      "slightly below nominal. If you expect very unbalanced categories, increase N."))
    if test == "ordreg" and len(d.predictors) > 1:
        notes.append(("Con varios predictores correlacionados, divide N por (1 − R²) del predictor de interés con los demás (Hsieh et al., 1998).",
                      "With several correlated predictors, divide N by (1 − R²) of the predictor of interest on the others (Hsieh et al., 1998)."))
    if test == "mediation":
        notes.append(("Valores tomados de la diagonal a = b de Fritz y MacKinnon (2007); si esperas sendas desiguales o potencia ≠ .80, "
                      "consulta la tabla original o un análisis de potencia por simulación.",
                      "Values taken from the a = b diagonal of Fritz and MacKinnon (2007); if you expect unequal paths or power ≠ .80, "
                      "see the original table or a simulation-based power analysis."))
    if test == "moderation":
        notes.append(("f² de la interacción = ΔR² / (1 − R² del modelo completo). En psicología las interacciones suelen ser pequeñas "
                      "(f² ≈ .01–.02; Aguinis et al., 2005), así que «mediano» puede subestimar mucho el N necesario.",
                      "Interaction f² = ΔR² / (1 − full-model R²). In psychology interactions are usually small "
                      "(f² ≈ .01–.02; Aguinis et al., 2005), so “medium” can badly underestimate the N needed."))
    if test == "reliability":
        notes.append(("La fórmula de Bonett es una aproximación al contraste F exacto de Feldt (difiere en 1 caso como mucho; supone ítems "
                      "paralelos y normalidad). Para validar la estructura de un cuestionario nuevo con análisis factorial se suelen necesitar más participantes.",
                      "Bonett's formula approximates Feldt's exact F test (off by at most 1 case; it assumes parallel items and normality). "
                      "Validating the structure of a new questionnaire with factor analysis usually needs more participants."))
    if test == "icc":
        notes.append(("Se supone que los evaluadores no difieren sistemáticamente (sin sesgo de evaluador) y normalidad; si esperas "
                      "diferencias entre evaluadores, el CCI de acuerdo absoluto será menor y conviene aumentar N.",
                      "Raters are assumed not to differ systematically (no rater bias) and normality; if you expect differences "
                      "between raters, the absolute-agreement ICC will be lower and N should be increased."))
    if test in ("kappa", "fleiss"):
        notes.append(("El cálculo supone categorías igual de frecuentes; si una categoría es muy rara, κ es menos precisa y se necesitan más casos."
                      + (" Se calcula para la κ sin ponderar." if d.dv_type == "ordinal" and test == "kappa" else "")
                      + (f" Se calcula para dos evaluadores, lo que es conservador con {len(d.raters)}." if test == "fleiss" else ""),
                      "The computation assumes equally frequent categories; if a category is very rare, κ is less precise and more cases are needed."
                      + (" It is computed for unweighted κ." if d.dv_type == "ordinal" and test == "kappa" else "")
                      + (f" It is computed for two raters, which is conservative with {len(d.raters)}." if test == "fleiss" else "")))
    if test == "mixedlm":
        notes.append(("El efecto de diseño es adecuado para predictores de nivel 2 (que solo varían entre grupos) y conservador para los de nivel 1, "
                      "cuya varianza es sobre todo intragrupo (Snijders y Bosker, 2012). "
                      "Con pocos grupos los errores típicos de nivel 2 y las varianzas se estiman mal: procura al menos 30 grupos (Maas y Hox, 2005)"
                      + (f"; con este cálculo solo habría {groups}." if groups < 30 else "."),
                      "The design effect suits level-2 predictors (varying only between clusters) and is conservative for level-1 ones, "
                      "whose variance is mostly within clusters (Snijders & Bosker, 2012). "
                      "With few clusters, level-2 standard errors and variances are poorly estimated: aim for at least 30 clusters (Maas & Hox, 2005)"
                      + (f"; this computation gives only {groups}." if groups < 30 else ".")))
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
            "groups": groups, "method": method, "notes": notes, "one_sided": one, "multilevel": test == "mixedlm"}


def _b(x):
    return f"{x:.2f}".lstrip("0")


def bonett_alpha_n(k, alpha0, alpha1, a=0.05, power=0.80, two_sided=True):
    """N to reject H₀: α = alpha0 when Cronbach's α = alpha1 (Bonett, 2002, eq. 5).

    n = [2k / (k − 1)] (z_{a/2} + z_β)² / ln(δ)² + 2, with δ = (1 − alpha0) / (1 − alpha1), rounded up.
    """
    if k < 2:
        raise ValueError("Se necesitan al menos 2 ítems / at least 2 items are needed")
    if not (0 <= alpha0 < 1 and 0 < alpha1 < 1) or alpha1 == alpha0:
        raise ValueError("El α esperado debe estar entre 0 y 1 y ser distinto del α mínimo (H₀) / "
                         "the expected α must be in (0, 1) and differ from the minimum α (H₀)")
    za = stats.norm.ppf(1 - a / 2) if two_sided else stats.norm.ppf(1 - a)
    zb = stats.norm.ppf(power)
    delta = (1 - alpha0) / (1 - alpha1)
    return math.ceil(2 * k / (k - 1) * (za + zb) ** 2 / math.log(delta) ** 2 + 2)


def icc_power(n, k, rho0, rho1, a=0.05):
    """Exact power of the two-sided F test of H₀: ICC = rho0 when ICC = rho1 (normal two-way model without rater bias).

    F₀ = (MSR / MSE) · (1 − ρ₀) / (1 + (k − 1)ρ₀) is distributed as c · F(n − 1, (n − 1)(k − 1)) with
    c = [(1 + (k − 1)ρ₁) / (1 − ρ₁)] · [(1 − ρ₀) / (1 + (k − 1)ρ₀)] (Shrout & Fleiss, 1979; Walter, Eliasziw & Donner, 1998).
    """
    df1, df2 = n - 1, (n - 1) * (k - 1)
    c = (1 + (k - 1) * rho1) / (1 - rho1) * (1 - rho0) / (1 + (k - 1) * rho0)
    hi, lo = stats.f.ppf(1 - a / 2, df1, df2), stats.f.ppf(a / 2, df1, df2)
    return float(stats.f.sf(hi / c, df1, df2) + stats.f.cdf(lo / c, df1, df2))


def icc_n(k, rho0, rho1, a=0.05, power=0.80):
    """Smallest number of rated subjects for the exact ICC F test (see `icc_power`)."""
    if k < 2:
        raise ValueError("Se necesitan al menos 2 evaluadores / at least 2 raters are needed")
    if not (0 <= rho0 < 1 and 0 < rho1 < 1) or rho1 == rho0:
        raise ValueError("El CCI esperado debe estar entre 0 y 1 y ser distinto del mínimo (H₀) / "
                         "the expected ICC must be in (0, 1) and differ from the minimum (H₀)")
    return _min_n(lambda n: icc_power(n, k, rho0, rho1, a), power, 3)


def kappa_table(c, kappa):
    """Expected c × c probability table for two raters with equal, uniform marginals and Cohen's κ = kappa."""
    p = 1 / c
    return (1 - kappa) * p * p * (1 - np.eye(c)) + np.eye(c) * (p * p + kappa * p * (1 - p))


def kappa_avar(P):
    """Per-observation asymptotic variance of Cohen's κ̂ for a population probability table P (Fleiss, Cohen & Everitt, 1969)."""
    P = np.asarray(P, float) / np.sum(P)
    r, s = P.sum(axis=1), P.sum(axis=0)
    pe = float(r @ s)
    kap = (np.trace(P) - pe) / (1 - pe)
    A = float(np.sum(np.diag(P) * (1 - (r + s) * (1 - kap)) ** 2))
    W = (s[:, None] + r[None, :]) ** 2          # (p.i + pj.)² for cell (i, j)
    B = float((1 - kap) ** 2 * np.sum((P * W)[~np.eye(len(P), dtype=bool)]))
    C = (kap - pe * (1 - kap)) ** 2
    return (A + B - C) / (1 - pe) ** 2


def kappa_n(c, kappa0, kappa1, a=0.05, power=0.80):
    """N to reject H₀: κ = kappa0 (two-sided z test) when κ = kappa1, with c equally likely categories.

    n = [(z_{a/2} σ₀ + z_β σ₁) / (κ₁ − κ₀)]² (Cantor, 1996), with σ₀² and σ₁² the per-observation asymptotic variances of κ̂
    at κ₀ and κ₁ (Fleiss, Cohen & Everitt, 1969). Using σ₀ for the critical value keeps the Wald test used in the analysis
    at or above the nominal power in small samples (checked by simulation in the tests), where the pure Wald formula falls short.
    """
    if c < 2:
        raise ValueError("Se necesitan al menos 2 categorías / at least 2 categories are needed")
    lo = -1 / (c - 1)
    if not (lo < kappa0 < 1 and lo < kappa1 < 1) or kappa1 == kappa0:
        raise ValueError("La κ esperada debe ser < 1 y distinta de la κ mínima (H₀) / the expected κ must be < 1 and differ from the minimum κ (H₀)")
    s0, s1 = math.sqrt(kappa_avar(kappa_table(c, kappa0))), math.sqrt(kappa_avar(kappa_table(c, kappa1)))
    za, zb = stats.norm.ppf(1 - a / 2), stats.norm.ppf(power)
    return max(math.ceil(((za * s0 + zb * s1) / (kappa1 - kappa0)) ** 2), 10)


def explain(ss: dict, lang="es") -> str:
    sym = {"d": "d", "f": "f", "r": "r", "w": "w", "f2": "f²", "or": "OR", "path": "a=b", "rel": "α", "agr": "κ/CCI"}[ss["family"]]
    i = 0 if lang == "es" else 1
    if ss["family"] in ("rel", "agr"):
        if lang == "es":
            txt = (f"Tamaño muestral a priori ({ss['method'][i]}): con un nivel de significación de {_b(ss['alpha'])} y potencia = "
                   f"{_b(ss['power'])} se necesitan N = {ss['n_total']} participantes")
        else:
            txt = (f"A priori sample size ({ss['method'][i]}): with a significance level of {_b(ss['alpha'])} and power = "
                   f"{_b(ss['power'])}, N = {ss['n_total']} participants are required")
    elif lang == "es":
        txt = (f"Análisis de potencia a priori ({ss['method'][i]}): para detectar un efecto {sym} = {ss['effect']:.2f} "
               f"con α = {_b(ss['alpha'])} y potencia = {_b(ss['power'])} se necesitan N = {ss['n_total']} participantes")
        if ss.get("multilevel"):
            txt = txt.replace(" participantes", f" observaciones ({ss['groups']} grupos de {ss['n_per_group']})")
        elif ss["n_per_group"] and ss["groups"] > 1:
            txt += f" ({ss['n_per_group']} por grupo)"
    else:
        txt = (f"A priori power analysis ({ss['method'][i]}): to detect an effect of {sym} = {ss['effect']:.2f} "
               f"with α = {_b(ss['alpha'])} and power = {_b(ss['power'])}, N = {ss['n_total']} participants are required")
        if ss.get("multilevel"):
            txt = txt.replace(" participants", f" observations ({ss['groups']} clusters of {ss['n_per_group']})")
        elif ss["n_per_group"] and ss["groups"] > 1:
            txt += f" ({ss['n_per_group']} per group)"
    txt += "."
    for n in ss["notes"]:
        txt += " " + n[i]
    return txt
