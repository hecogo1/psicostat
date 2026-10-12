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


def r_ci(r, n, conf=0.95, spearman=False):
    if n <= 3 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    se = math.sqrt((1 + r ** 2 / 2) / (n - 3)) if spearman else 1 / math.sqrt(n - 3)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return math.tanh(math.atanh(r) - z * se), math.tanh(math.atanh(r) + z * se)


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
          "R²": (.02, .13, .26), "β": (.1, .3, .5), "g_C": (.05, .15, .25)}
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
