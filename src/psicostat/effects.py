"""Effect sizes, confidence intervals and small helpers."""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats


def _ncp_bounds(cdf, conf=0.95):
    """Invert a cdf(lam) that decreases with the noncentrality parameter lam."""
    lo_t, hi_t = 1 - (1 - conf) / 2, (1 - conf) / 2

    def solve(target):
        if cdf(0.0) < target:
            return 0.0
        hi = 10.0
        while cdf(hi) > target:
            hi *= 2
            if hi > 1e7:
                return hi
        return optimize.brentq(lambda lam: cdf(lam) - target, 0.0, hi)

    return solve(lo_t), solve(hi_t)


def partial_eta2_ci(F, df1, df2, conf=0.95):
    lo, hi = _ncp_bounds(lambda lam: stats.ncf.cdf(F, df1, df2, lam) if lam > 0 else stats.f.cdf(F, df1, df2), conf)
    tot = df1 + df2 + 1
    return lo / (lo + tot), hi / (hi + tot)


def cramers_v_ci(chi2, df, n, k_min, conf=0.95):
    lo, hi = _ncp_bounds(lambda lam: stats.ncx2.cdf(chi2, df, lam) if lam > 0 else stats.chi2.cdf(chi2, df), conf)
    f = lambda lam: math.sqrt(max(lam, 0) / (n * (k_min - 1)))
    return f(lo), min(f(hi), 1.0)


def omega2_welch_ci(F, df1, df2, n, conf=0.95):
    """CI for ω² after Welch's ANOVA by inverting the noncentral F distribution (Steiger, 2004).

    Welch's F is treated as approximately noncentral F(df1, df2_Welch, λ) and each limit of λ
    is converted with ω² = λ / (λ + N), the same mapping as the point estimate
    ω̂² = df1(F − 1) / (df1(F − 1) + N). The interval is approximate
    because Welch's statistic is only approximately F-distributed under heteroscedasticity.
    """
    lo, hi = _ncp_bounds(lambda lam: stats.ncf.cdf(F, df1, df2, lam) if lam > 0 else stats.f.cdf(F, df1, df2), conf)
    return lo / (lo + n), hi / (hi + n)


def cohen_g_mcnemar(b, c, conf=0.95):
    """Cohen's g = |b/(b+c) − .5| for McNemar with an exact Clopper-Pearson CI (Clopper & Pearson, 1934).

    The exact CI [π_L, π_U] for π = b/(b+c) (proportion of discordant pairs in cell b) is mapped
    to |π − .5|; if it contains .5 the lower limit of g is 0 and the upper one is the limit
    farthest from .5. Returns (g, (low, high)); the CI is NaN when there are no discordant pairs.
    """
    m = b + c
    if m == 0:
        return 0.0, (float("nan"), float("nan"))
    a = 1 - conf
    p_lo = stats.beta.ppf(a / 2, b, c + 1) if b > 0 else 0.0
    p_hi = stats.beta.ppf(1 - a / 2, b + 1, c) if c > 0 else 1.0
    g = abs(b / m - .5)
    if p_lo <= .5 <= p_hi:
        ci = (0.0, max(.5 - p_lo, p_hi - .5))
    else:
        ci = tuple(sorted((abs(p_lo - .5), abs(p_hi - .5))))
    return g, (float(ci[0]), float(ci[1]))


def eta2_q(mat):
    """Serlin, Carr & Marascuilo's (1982) η²_Q = Q / (N(k − 1)) for Cochran's Q (N subjects, k conditions)."""
    from statsmodels.stats.contingency_tables import cochrans_q
    mat = np.asarray(mat)
    n, k = mat.shape
    if np.all(mat == mat[:, :1]):  # every subject constant across conditions: no differences, Q is 0/0
        return 0.0
    return float(cochrans_q(mat).statistic) / (n * (k - 1))


def hedges_g(x, y, conf=0.95):
    n1, n2 = len(x), len(y)
    sp = math.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    d = (np.mean(x) - np.mean(y)) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * j
    se = math.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return g, (g - z * se, g + z * se)


def cohen_dz(diff, conf=0.95):
    n = len(diff)
    dz = np.mean(diff) / np.std(diff, ddof=1)
    se = math.sqrt(1 / n + dz ** 2 / (2 * n))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return dz, (dz - z * se, dz + z * se)


def tost(diff, se, df, low, upp, alpha=0.05):
    """Two one-sided t tests (TOST) of H₀: diff ≤ low or diff ≥ upp (Schuirmann, 1987; Lakens, 2017).

    diff: observed mean difference; se: its standard error; df: degrees of freedom (Welch or n − 1).
    Equivalence is declared when both one-sided tests reject at level α, i.e. when the
    (1 − 2α) CI of the difference lies entirely inside [low, upp]. Returns a dict with both
    t statistics and p values, the TOST p (the larger of the two) and the (1 − 2α) CI.
    Same results as statsmodels ttost_ind(usevar="unequal") / ttost_paired.
    """
    t_low, t_upp = (diff - low) / se, (diff - upp) / se
    p_low, p_upp = float(stats.t.sf(t_low, df)), float(stats.t.cdf(t_upp, df))
    q = stats.t.ppf(1 - alpha, df)
    return {"diff": float(diff), "se": float(se), "df": float(df), "low": float(low), "upp": float(upp),
            "t_low": float(t_low), "p_low": p_low, "t_upp": float(t_upp), "p_upp": p_upp,
            "p": max(p_low, p_upp), "ci": (float(diff - q * se), float(diff + q * se)),
            "conf": 1 - 2 * alpha, "equivalent": bool(max(p_low, p_upp) < alpha)}


def tost_welch(x, y, bound, units="d", alpha=0.05):
    """TOST for two independent means with Welch's SE and df.

    bound Δ in Cohen's d units is converted to raw units with the average SD
    √((s₁² + s₂²)/2), the standardizer TOSTER uses for Welch's test (Lakens, 2017).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    raw = bound * math.sqrt((v1 + v2) / 2) if units == "d" else bound
    out = tost(np.mean(x) - np.mean(y), se, df, -raw, raw, alpha)
    out.update(bound=float(bound), units=units, sd_std=math.sqrt((v1 + v2) / 2))
    return out


def tost_paired(diff, bound, units="d", alpha=0.05):
    """TOST for paired means (difference scores); Δ in d_z units is converted with the SD of the differences."""
    diff = np.asarray(diff, float)
    n = len(diff)
    sd = np.std(diff, ddof=1)
    raw = bound * sd if units == "d" else bound
    out = tost(np.mean(diff), sd / math.sqrt(n), n - 1, -raw, raw, alpha)
    out.update(bound=float(bound), units=units, sd_std=float(sd))
    return out


def r_ci(r, n, conf=0.95, spearman=False, k=0):
    """Fisher-z CI for r (Bonett & Wright, 2000, SE for Spearman); k = number of covariates of a partial r."""
    if n - k <= 3 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    se = math.sqrt((1 + r ** 2 / 2) / (n - 3 - k)) if spearman else 1 / math.sqrt(n - 3 - k)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return math.tanh(math.atanh(r) - z * se), math.tanh(math.atanh(r) + z * se)


def auc_delong(y, score, conf=0.95):
    """Area under the ROC curve (AUC) with DeLong et al.'s (1988) nonparametric CI.

    y: 0/1 outcome (1 = event); score: predicted probability or any continuous predictor
    (higher = more likely event). Ties count 1/2. Same estimator and interval as R pROC
    (ci.auc, method = "delong"): AUC ± z·SE, truncated to [0, 1].
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    pos, neg = score[y == 1], score[y == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return float("nan"), (float("nan"), float("nan"))
    r_all = stats.rankdata(np.r_[pos, neg])
    # structural components (placement values): V10_i = P(neg < pos_i), V01_j = P(pos > neg_j)
    v10 = (r_all[:m] - stats.rankdata(pos)) / n
    v01 = 1 - (r_all[m:] - stats.rankdata(neg)) / m
    auc = float(v10.mean())
    se = math.sqrt(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return auc, (max(0.0, auc - z * se), min(1.0, auc + z * se))


def bootstrap_ci(fn, arrays, n_boot=2000, conf=0.95, seed=42, paired=False):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        if paired:
            idx = rng.integers(0, len(arrays[0]), len(arrays[0]))
            res = [a[idx] for a in arrays]
        else:
            res = [a[rng.integers(0, len(a), len(a))] for a in arrays]
        try:
            v = fn(*res)
        except Exception:  # pragma: no cover
            continue
        if np.isfinite(v):
            vals.append(v)
    a = (1 - conf) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def rank_biserial_ind(x, y):
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def rank_biserial_paired(diff):
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    r = stats.rankdata(np.abs(diff))
    tp, tm = r[diff > 0].sum(), r[diff < 0].sum()
    return (tp - tm) / (tp + tm)


def kendall_w(mat):
    n, k = mat.shape
    chi2 = stats.friedmanchisquare(*[mat[:, j] for j in range(k)]).statistic
    return chi2 / (n * (k - 1))


def magnitude(symbol, value):
    """Cohen-style qualitative label for an effect size."""
    if value is None or not np.isfinite(value):
        return None
    v = abs(value)
    th = {"g": (.2, .5, .8), "d": (.2, .5, .8), "d_z": (.2, .5, .8),
          "r": (.1, .3, .5), "r_s": (.1, .3, .5), "r_rb": (.1, .3, .5), "V": (.1, .3, .5), "W": (.1, .3, .5),
          "η²p": (.01, .06, .14), "ω²": (.01, .06, .14), "ε²": (.01, .06, .14),
          "R²": (.02, .13, .26), "β": (.1, .3, .5), "g_C": (.05, .15, .25),
          "a": (.14, .39, .59), "b": (.14, .39, .59), "c": (.14, .39, .59), "c′": (.14, .39, .59), "a×b": (.01, .09, .25),
          "f²": (.02, .15, .35)}
    if symbol == "OR":
        v = v if v >= 1 else 1 / v
        th_ = (1.5, 2.5, 4.3)
    elif symbol in th:
        th_ = th[symbol]
    else:
        return None
    if v < th_[0]:
        return "negligible"
    if v < th_[1]:
        return "small"
    if v < th_[2]:
        return "medium"
    return "large"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "p = —"
    if p < .001:
        return "p < .001"
    return "p = " + f"{p:.3f}".lstrip("0")


def fmt_num(x, bounded=False, dec=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:.{dec}f}"
    if bounded:
        s = s.replace("0.", ".", 1) if abs(x) < 1 else s
    return s


def partial_corr(x, y, Z, alternative="two-sided"):
    """Partial correlation of x and y controlling for the columns of Z (residual method).

    Returns (r, t, df, p) with df = n − 2 − k. For a partial Spearman correlation pass the ranks
    of x, y and Z. The t statistic equals the t of x in the OLS regression of y on x + Z.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    A = np.column_stack([np.ones(len(x)), Z])
    ex = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ey = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = float(np.clip(np.sum(ex * ey) / math.sqrt(np.sum(ex ** 2) * np.sum(ey ** 2)), -1.0, 1.0))
    df = len(x) - 2 - Z.shape[1]
    if df < 1:
        raise ValueError("Demasiadas covariables para el número de casos / too many covariates for the number of cases")
    tval = r * math.sqrt(df / (1 - r ** 2)) if abs(r) < 1 else math.copysign(math.inf, r)
    if alternative == "greater":
        p = stats.t.sf(tval, df)
    elif alternative == "less":
        p = stats.t.cdf(tval, df)
    else:
        p = 2 * stats.t.sf(abs(tval), df)
    return r, tval, df, float(p)
