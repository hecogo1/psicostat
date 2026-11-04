"""Analysis of a filled CSV according to the design."""
from __future__ import annotations

import itertools
import os
import warnings
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from cycler import cycler
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.anova import AnovaRM, anova_lm
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from . import effects as E
from .design import Design, recommend
from .template import columns

warnings.filterwarnings("ignore", category=FutureWarning)

PALETTE = ["#0F766E", "#3730A3", "#D97706", "#BE123C", "#0E7490", "#7C3AED"]
ACCENT = "#D97706"
plt.rcParams.update({
    "axes.prop_cycle": cycler(color=PALETTE), "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#94A3B8", "axes.labelcolor": "#1E293B", "axes.titleweight": "bold",
    "xtick.color": "#475569", "ytick.color": "#475569", "axes.grid": True, "grid.color": "#E7E3DA",
    "grid.linewidth": .8, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "#FFFDF8",
    "legend.frameon": False, "savefig.facecolor": "white",
})
SMALL_N = 30


class DataError(ValueError):
    pass


@dataclass
class Result:
    design: Design
    test: str
    planned_test: str
    fallback_reason: str | None = None          # assumption key that triggered fallback
    n: int = 0
    n_excluded: int = 0
    descriptives: pd.DataFrame | None = None
    assumptions: list = field(default_factory=list)   # dicts: key, detail, p, ok
    rows: list = field(default_factory=list)          # main inferential results
    posthoc: pd.DataFrame | None = None
    posthoc_method: str | None = None
    extra_tables: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)
    notes: list = field(default_factory=list)          # (es, en)


def _row(label, sym, df, stat, p, es_sym=None, es=None, ci=None, bounded=False, alternative="two-sided"):
    return {"alternative": alternative, "label": label, "stat_symbol": sym, "df": df, "stat": float(stat) if stat is not None else None,
            "p": float(p) if p is not None else None, "es_symbol": es_sym,
            "es": float(es) if es is not None else None,
            "ci": (float(ci[0]), float(ci[1])) if ci is not None else None, "bounded": bounded}


def _desc(series_by_group: dict, ordinal=False):
    out = []
    for g, x in series_by_group.items():
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        tcrit = stats.t.ppf(.975, len(x) - 1) if len(x) > 1 else np.nan
        out.append({"grupo/group": g, "n": len(x), "M": x.mean(), "DT/SD": x.std(ddof=1),
                    "IC95_inf/CI95_low": x.mean() - tcrit * se, "IC95_sup/CI95_high": x.mean() + tcrit * se,
                    "Mdn": x.median(), "RIC/IQR": x.quantile(.75) - x.quantile(.25), "Min": x.min(), "Max": x.max()})
    return pd.DataFrame(out)


def _shapiro(x):
    x = np.asarray(x, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    if len(x) > 5000:
        x = np.random.default_rng(0).choice(x, 5000, replace=False)
    r = stats.shapiro(x)
    return r.statistic, r.pvalue


def _normality_groups(groups: dict, key="a_normality"):
    res, ok_all = [], True
    for g, x in groups.items():
        w, p = _shapiro(x)
        ok = not (p < .05)
        ok_all &= ok
        res.append({"key": key, "detail": f"{g}: W = {E.fmt_num(w, True, 3)}, {E.fmt_p(p)}", "p": p, "ok": ok})
    return res, ok_all


def _levene(groups: list):
    groups = [np.asarray(g, float) for g in groups if len(g) > 1]
    r = stats.levene(*groups, center="median")
    df2 = sum(len(g) for g in groups) - len(groups)
    return {"key": "a_homogeneity", "detail": f"F({len(groups) - 1}, {df2}) = {r.statistic:.2f}, {E.fmt_p(r.pvalue)}",
            "p": r.pvalue, "ok": not r.pvalue < .05}


def _outliers(groups: dict):
    n_out = 0
    for x in groups.values():
        x = np.asarray(x, float)
        if len(x) > 2 and np.std(x) > 0:
            n_out += int((np.abs(stats.zscore(x, ddof=1)) > 3).sum())
    return {"key": "a_outliers", "detail": f"{n_out}", "p": None, "ok": n_out == 0}


def _orth_contrasts(k):
    """k x (k - 1) orthonormal contrasts (columns orthogonal to the unit vector)."""
    return np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :k - 1]]))[0][:, 1:]


def _sphericity(z, k=None, df=None):
    """Mauchly's W and p, and the Greenhouse-Geisser and Huynh-Feldt epsilons for one within-subjects effect.

    `z` is the n x p matrix of orthonormal-contrast scores of the effect (p = effect df) and `k` the total
    number of repeated-measures cells (ka·kb in a two-way design; default p + 1). `df` is the error df of
    the covariance matrix: n − 1 (default) for a single group, N − G for the pooled within-group matrix of a
    mixed design (pass scores already centred within each group). The p value includes the second-order
    term of R `mauchly.test` / `car::Anova` (hence afex and JASP); ε_HF is the Huynh-Feldt formula used by
    R `car`, ((df + 1)·p·ε_GG − 2) / [p·(df − p·ε_GG)], truncated to 1.
    """
    n, p = z.shape
    if p < 2:
        return 1.0, np.nan, 1.0, 1.0
    k = p + 1 if k is None else k
    df = n - 1 if df is None else df
    m = np.cov(z, rowvar=False, ddof=n - df)
    tr = np.trace(m)
    sign, logdet = np.linalg.slogdet(m)
    logw = logdet - p * np.log(tr / p) if sign > 0 else -np.inf
    rho = 1 - (2 * p ** 2 + p + 2) / (6 * p * df)
    w2 = (p + 2) * (p - 1) * (p - 2) * (2 * p ** 3 + 6 * p ** 2 + 3 * k + 2) / (288 * (df * p * rho) ** 2)
    chi = -df * rho * logw
    f = p * (p + 1) / 2 - 1
    p1, p2 = stats.chi2.sf(chi, f), stats.chi2.sf(chi, f + 4)
    pval = float(np.clip(p1 + w2 * (p2 - p1), 0, 1))
    gg = min(max(tr ** 2 / (p * np.trace(m @ m)), 1 / p), 1.0)
    hf = min(((df + 1) * p * gg - 2) / (p * (df - p * gg)), 1.0)
    return float(np.exp(logw)), pval, gg, hf


def _mauchly(mat):
    """Mauchly's test and Greenhouse-Geisser epsilon for an n x k matrix."""
    k = mat.shape[1]
    if k < 3:
        return 1.0, np.nan, 1.0
    w, pval, gg, _ = _sphericity(mat @ _orth_contrasts(k))
    return w, pval, gg


def _games_howell(groups: dict):
    k = len(groups)
    rows = []
    for (a, x), (b, y) in itertools.combinations(groups.items(), 2):
        n1, n2, v1, v2 = len(x), len(y), np.var(x, ddof=1), np.var(y, ddof=1)
        diff = np.mean(x) - np.mean(y)
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        q = abs(diff) / se * np.sqrt(2)
        p = stats.studentized_range.sf(q, k, df)
        g, _ = E.hedges_g(np.asarray(x), np.asarray(y))
        rows.append({"A": a, "B": b, "dif/diff": diff, "p_ajust/p_adj": min(p, 1.0), "g": g})
    return pd.DataFrame(rows)


def _tukey(groups: dict):
    names = list(groups)
    r = stats.tukey_hsd(*[np.asarray(groups[n]) for n in names])
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        x, y = np.asarray(groups[names[i]]), np.asarray(groups[names[j]])
        g, _ = E.hedges_g(x, y)
        rows.append({"A": names[i], "B": names[j], "dif/diff": x.mean() - y.mean(),
                     "p_ajust/p_adj": r.pvalue[i, j], "g": g})
    return pd.DataFrame(rows)


def _dunn(groups: dict):
    allv = np.concatenate([np.asarray(v) for v in groups.values()])
    ranks = stats.rankdata(allv)
    n = len(allv)
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum() / (12 * (n - 1))
    pos, mean_r, sizes = 0, {}, {}
    for g, v in groups.items():
        mean_r[g] = ranks[pos:pos + len(v)].mean()
        sizes[g] = len(v)
        pos += len(v)
    rows = []
    for a, b in itertools.combinations(groups, 2):
        se = np.sqrt((n * (n + 1) / 12 - tie) * (1 / sizes[a] + 1 / sizes[b]))
        z = (mean_r[a] - mean_r[b]) / se
        rows.append({"A": a, "B": b, "z": z, "p": 2 * stats.norm.sf(abs(z))})
    df = pd.DataFrame(rows)
    df["p_ajust/p_adj"] = multipletests(df["p"], method="holm")[1]
    return df


def _paired_posthoc(mat, names):
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        diff = mat[:, i] - mat[:, j]
        r = stats.ttest_rel(mat[:, i], mat[:, j])
        dz, _ = E.cohen_dz(diff)
        rows.append({"A": names[i], "B": names[j], "dif/diff": diff.mean(), "t": r.statistic, "p": r.pvalue, "d_z": dz})
    df = pd.DataFrame(rows)
    df["p_ajust/p_adj"] = multipletests(df["p"], method="holm")[1]
    return df


def _wilcoxon_posthoc(mat, names):
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        r = stats.wilcoxon(mat[:, i], mat[:, j])
        rows.append({"A": names[i], "B": names[j], "V": r.statistic, "p": r.pvalue,
                     "r_rb": E.rank_biserial_paired(mat[:, i] - mat[:, j])})
    df = pd.DataFrame(rows)
    df["p_ajust/p_adj"] = multipletests(df["p"], method="holm")[1]
    return df


# ------------------------------------------------------------------ plotting
def _save(fig, outdir, name):
    if outdir is None:
        plt.close(fig)
        return None
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_groups(groups: dict, ylabel, outdir, name="fig_grupos.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(groups)
    data = [np.asarray(groups[g], float) for g in labels]
    ax.boxplot(data, widths=.5, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#CCFBF1", "edgecolor": "#0F766E"}, medianprops={"color": "#0F766E", "lw": 2},
               whiskerprops={"color": "#64748B"}, capprops={"color": "#64748B"})
    ax.set_xticks(range(1, len(labels) + 1), labels)
    rng = np.random.default_rng(1)
    for i, x in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-.12, .12, len(x)), x, s=12, alpha=.55, color="#3730A3", edgecolor="none")
        ax.errorbar(i + .3, x.mean(), yerr=stats.sem(x) * 1.96, fmt="D", color=ACCENT, capsize=4)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, name)


def _plot_profiles(mat, names, ylabel, outdir, name="fig_medidas.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    for row in mat:
        ax.plot(x, row, color="#0F766E", alpha=.18, lw=.8)
    m = mat.mean(axis=0)
    ax.errorbar(x, m, yerr=1.96 * mat.std(axis=0, ddof=1) / np.sqrt(len(mat)), color=ACCENT, lw=2.5, marker="o", capsize=4)
    ax.set_xticks(x, names)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, name)


def _plot_interaction(means: pd.DataFrame, ses: pd.DataFrame, xlabel, ylabel, outdir, name="fig_interaccion.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(means.columns))
    for i, (g, row) in enumerate(means.iterrows()):
        ax.errorbar(x + (i - len(means) / 2) * .04, row.values, yerr=1.96 * ses.loc[g].values, marker="o", capsize=3, label=str(g))
    ax.set_xticks(x, [str(c) for c in means.columns])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, name)


def _plot_scatter(x, y, xlabel, ylabel, outdir, name="fig_dispersion.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=16, alpha=.65, color="#3730A3", edgecolor="none")
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(np.min(x), np.max(x), 50)
    ax.plot(xs, a + b * xs, color=ACCENT, lw=2.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, name)


def _plot_bars(tab: pd.DataFrame, outdir, name="fig_proporciones.png"):
    prop = tab.div(tab.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    prop.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("%")
    ax.legend(frameon=False, title=tab.columns.name)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, name)


# ------------------------------------------------------------------ helpers
def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _check_columns(d: Design, df: pd.DataFrame):
    need = [c[0] for c in columns(d)]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DataError(f"Faltan columnas / missing columns: {missing}")
    return need


def _groups_from(df, gcol, ycol, levels):
    out = {}
    present = [lv for lv in levels if lv in set(df[gcol].astype(str))]
    extra = sorted(set(df[gcol].astype(str)) - set(levels))
    for lv in present + extra:
        out[lv] = df.loc[df[gcol].astype(str) == lv, ycol].to_numpy(float)
    return out


# ------------------------------------------------------------------ main entry
def analyze(d: Design, data, outdir=None) -> Result:
    """Analyse a DataFrame or CSV path according to design `d`. Figures go to `outdir`."""
    df = pd.read_csv(data) if isinstance(data, (str, os.PathLike)) else data.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rec = recommend(d)
    need = _check_columns(d, df)
    n0 = len(df)
    numeric_cols = [c for c in need if c != "id" and c not in (d.iv_name, d.factor_b_name)]
    if rec.test in ("chi2", "fisher"):
        numeric_cols = []
    if rec.test in ("linreg", "logreg") and d.iv_type == "nominal":
        numeric_cols = [d.dv_name] if rec.test == "linreg" else []
    if rec.test == "ordreg":
        numeric_cols = [] if d.iv_type == "nominal" else [c for c in numeric_cols if c != d.dv_name]
    for c in numeric_cols:
        df[c] = _num(df, c)
    df = df.replace("", np.nan).dropna(subset=[c for c in need if c != "id"])
    if len(df) < 3:
        raise DataError("Muy pocos casos completos / too few complete cases (< 3)")
    res = Result(design=d, test=rec.test, planned_test=rec.test, n=len(df), n_excluded=n0 - len(df))
    fn = {
        "ttest_ind": _two_groups, "mannwhitney": _two_groups,
        "ttest_paired": _paired, "wilcoxon": _paired,
        "anova_oneway": _oneway, "kruskal": _oneway,
        "anova_factorial": _factorial, "ancova": _ancova,
        "anova_rm": _rm, "friedman": _rm, "anova_mixed": _mixed,
        "chi2": _chi2, "fisher": _chi2, "mcnemar": _mcnemar, "cochran": _cochran,
        "pearson": _corr, "spearman": _corr, "linreg": _linreg, "logreg": _logreg, "ordreg": _ordreg,
    }[rec.test]
    fn(d, df, res, outdir)
    return res


def _two_groups(d, df, res, outdir):
    g = _groups_from(df, d.iv_name, d.dv_name, d.levels)
    if len(g) != 2:
        raise DataError(f"Se esperaban 2 grupos / expected 2 groups, found {list(g)}")
    (a, x), (b, y) = g.items()
    res.descriptives = _desc(g)
    if res.test == "ttest_ind":
        norm, ok = _normality_groups(g)
        res.assumptions += norm + [_levene([x, y]), _outliers(g)]
        if not ok and min(len(x), len(y)) < SMALL_N and d.auto_fallback:
            res.test, res.fallback_reason = "mannwhitney", "a_normality"
    if res.test == "ttest_ind":
        r = stats.ttest_ind(x, y, equal_var=False, alternative=d.alternative)
        gval, ci = E.hedges_g(x, y)
        res.rows.append(_row(f"{a} vs {b}", "t", f"{r.df:.2f}", r.statistic, r.pvalue, "g", gval, ci, alternative=d.alternative))
    else:
        r = stats.mannwhitneyu(x, y, alternative=d.alternative)
        rb = E.rank_biserial_ind(x, y)
        ci = E.bootstrap_ci(E.rank_biserial_ind, [x, y])
        res.rows.append(_row(f"{a} vs {b}", "U", None, r.statistic, r.pvalue, "r_rb", rb, ci, bounded=True, alternative=d.alternative))
    res.figures.append(_plot_groups(g, d.dv_name, outdir))


def _wide(df, d, levels):
    cols = [f"{d.dv_name}_{lv}" for lv in levels]
    return df[cols].to_numpy(float), cols


def _paired(d, df, res, outdir):
    mat, _ = _wide(df, d, d.levels)
    x, y = mat[:, 0], mat[:, 1]
    diff = x - y
    res.descriptives = _desc({lv: mat[:, i] for i, lv in enumerate(d.levels)})
    if res.test == "ttest_paired":
        w, p = _shapiro(diff)
        res.assumptions.append({"key": "a_normality_diff", "detail": f"W = {E.fmt_num(w, True, 3)}, {E.fmt_p(p)}",
                                "p": p, "ok": not p < .05})
        res.assumptions.append(_outliers({"diff": diff}))
        if p < .05 and len(diff) < SMALL_N and d.auto_fallback:
            res.test, res.fallback_reason = "wilcoxon", "a_normality"
    lab = f"{d.levels[0]} vs {d.levels[1]}"
    if res.test == "ttest_paired":
        r = stats.ttest_rel(x, y, alternative=d.alternative)
        dz, ci = E.cohen_dz(diff)
        res.rows.append(_row(lab, "t", f"{len(diff) - 1}", r.statistic, r.pvalue, "d_z", dz, ci, alternative=d.alternative))
    else:
        r = stats.wilcoxon(x, y, alternative=d.alternative)
        rb = E.rank_biserial_paired(diff)
        ci = E.bootstrap_ci(lambda dd: E.rank_biserial_paired(dd), [diff])
        res.rows.append(_row(lab, "V", None, r.statistic, r.pvalue, "r_rb", rb, ci, bounded=True, alternative=d.alternative))
    res.figures.append(_plot_profiles(mat, d.levels, d.dv_name, outdir))


def _oneway(d, df, res, outdir):
    g = _groups_from(df, d.iv_name, d.dv_name, d.levels)
    k, n = len(g), sum(len(v) for v in g.values())
    res.descriptives = _desc(g)
    lev = None
    if res.test == "anova_oneway":
        norm, ok = _normality_groups(g)
        lev = _levene(list(g.values()))
        res.assumptions += norm + [lev, _outliers(g)]
        if not ok and min(len(v) for v in g.values()) < SMALL_N and d.auto_fallback:
            res.test, res.fallback_reason = "kruskal", "a_normality"
        elif not lev["ok"]:
            res.test, res.fallback_reason = "welch_anova", "a_homogeneity"
    vals = list(g.values())
    if res.test == "anova_oneway":
        r = stats.f_oneway(*vals)
        df1, df2 = k - 1, n - k
        eta = r.statistic * df1 / (r.statistic * df1 + df2)
        res.rows.append(_row(d.iv_name, "F", f"{df1}, {df2}", r.statistic, r.pvalue, "η²p", eta,
                             E.partial_eta2_ci(r.statistic, df1, df2), bounded=True))
        if r.pvalue < d.alpha:
            res.posthoc, res.posthoc_method = _tukey(g), "Tukey HSD"
    elif res.test == "welch_anova":
        ns = np.array([len(v) for v in vals])
        ms = np.array([np.mean(v) for v in vals])
        vs = np.array([np.var(v, ddof=1) for v in vals])
        w = ns / vs
        mw = (w * ms).sum() / w.sum()
        A = (w * (ms - mw) ** 2).sum() / (k - 1)
        tmp = (((1 - w / w.sum()) ** 2) / (ns - 1)).sum()
        F = A / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
        df2 = (k ** 2 - 1) / (3 * tmp)
        p = stats.f.sf(F, k - 1, df2)
        om = max(0.0, (k - 1) * (F - 1) / ((k - 1) * (F - 1) + n))
        res.rows.append(_row(d.iv_name, "F", f"{k - 1}, {df2:.2f}", F, p, "ω²", om,
                             E.omega2_welch_ci(F, k - 1, df2, n), bounded=True))
        res.notes.append(("El IC 95% de ω² se obtuvo invirtiendo la distribución F no central con los grados de libertad de Welch "
                          "(Steiger, 2004); es aproximado porque la F de Welch solo sigue aproximadamente esa distribución.",
                          "The 95% CI for ω² was obtained by inverting the noncentral F distribution with Welch's degrees of freedom "
                          "(Steiger, 2004); it is approximate because Welch's F only approximately follows that distribution."))
        if p < d.alpha:
            res.posthoc, res.posthoc_method = _games_howell(g), "Games-Howell"
    else:
        r = stats.kruskal(*vals)
        eps = r.statistic / (n - 1)

        def _eps(*arrs):
            return stats.kruskal(*arrs).statistic / (sum(len(a) for a in arrs) - 1)
        res.rows.append(_row(d.iv_name, "H", f"{k - 1}", r.statistic, r.pvalue, "ε²", eps,
                             E.bootstrap_ci(_eps, vals, n_boot=1000), bounded=True))
        if r.pvalue < d.alpha:
            res.posthoc, res.posthoc_method = _dunn(g), "Dunn (Holm)"
    res.figures.append(_plot_groups(g, d.dv_name, outdir))


def _anova_table_rows(model, labels, res):
    tab = anova_lm(model, typ=3)
    ss_res, df_res = tab.loc["Residual", "sum_sq"], tab.loc["Residual", "df"]
    for term, lab in labels.items():
        F, p, df1 = tab.loc[term, "F"], tab.loc[term, "PR(>F)"], tab.loc[term, "df"]
        eta = tab.loc[term, "sum_sq"] / (tab.loc[term, "sum_sq"] + ss_res)
        res.rows.append(_row(lab, "F", f"{int(df1)}, {int(df_res)}", F, p, "η²p", eta,
                             E.partial_eta2_ci(F, df1, df_res), bounded=True))
    return tab


def _factorial(d, df, res, outdir):
    a, b, y = d.iv_name, d.factor_b_name, d.dv_name
    df = df.copy()
    df[a], df[b] = df[a].astype(str), df[b].astype(str)
    cells = {f"{i}·{j}": grp[y].to_numpy(float) for (i, j), grp in df.groupby([a, b])}
    res.descriptives = _desc(cells)
    model = smf.ols(f"Q('{y}') ~ C(Q('{a}'), Sum) * C(Q('{b}'), Sum)", data=df).fit()
    resid_ok = _shapiro(model.resid)
    res.assumptions.append({"key": "a_normality", "detail": f"residuos/residuals: W = {E.fmt_num(resid_ok[0], True, 3)}, {E.fmt_p(resid_ok[1])}",
                            "p": resid_ok[1], "ok": not resid_ok[1] < .05})
    res.assumptions += [_levene(list(cells.values())), _outliers(cells)]
    ta, tb = f"C(Q('{a}'), Sum)", f"C(Q('{b}'), Sum)"
    _anova_table_rows(model, {ta: a, tb: b, f"{ta}:{tb}": f"{a} × {b}"}, res)
    for row, fac, lv in ((res.rows[0], a, d.levels), (res.rows[1], b, d.levels_b)):
        if row["p"] < d.alpha and df[fac].nunique() > 2:
            res.posthoc, res.posthoc_method = _tukey(_groups_from(df, fac, y, lv)), f"Tukey HSD ({fac})"
    means = df.pivot_table(index=a, columns=b, values=y, aggfunc="mean")
    ses = df.pivot_table(index=a, columns=b, values=y, aggfunc=lambda s: s.std(ddof=1) / np.sqrt(len(s)))
    res.figures.append(_plot_interaction(means, ses, b, y, outdir))


def _ancova(d, df, res, outdir):
    a, y, covs = d.iv_name, d.dv_name, d.covariates
    df = df.copy()
    df[a] = df[a].astype(str)
    g = _groups_from(df, a, y, d.levels)
    res.descriptives = _desc(g)
    cov_terms = " + ".join(f"Q('{c}')" for c in covs)
    model = smf.ols(f"Q('{y}') ~ C(Q('{a}'), Sum) + {cov_terms}", data=df).fit()
    inter = smf.ols(f"Q('{y}') ~ C(Q('{a}'), Sum) * ({cov_terms})", data=df).fit()
    cmp = anova_lm(model, inter)
    p_sl = cmp["Pr(>F)"].iloc[1]
    res.assumptions.append({"key": "a_slopes", "detail": f"F({int(cmp['df_diff'].iloc[1])}, {int(cmp['df_resid'].iloc[1])}) = "
                            f"{cmp['F'].iloc[1]:.2f}, {E.fmt_p(p_sl)}", "p": p_sl, "ok": not p_sl < .05})
    w, p = _shapiro(model.resid)
    res.assumptions.append({"key": "a_normality", "detail": f"residuos/residuals: W = {E.fmt_num(w, True, 3)}, {E.fmt_p(p)}", "p": p, "ok": not p < .05})
    res.assumptions.append(_levene(list(g.values())))
    labels = {f"C(Q('{a}'), Sum)": a}
    labels.update({f"Q('{c}')": c for c in covs})
    _anova_table_rows(model, labels, res)
    # adjusted (estimated marginal) means at covariate means
    grid = pd.DataFrame({a: list(g)})
    for c in covs:
        grid[c] = df[c].mean()
    pred = model.get_prediction(grid).summary_frame()
    res.extra_tables["adjusted_means"] = pd.DataFrame({"grupo/group": list(g), "M_ajust/M_adj": pred["mean"].values,
                                                       "EE/SE": pred["mean_se"].values})
    res.figures.append(_plot_groups(g, y, outdir))


def _rm_oneway_stats(mat):
    n, k = mat.shape
    M = mat.mean()
    ss_c = n * ((mat.mean(axis=0) - M) ** 2).sum()
    ss_s = k * ((mat.mean(axis=1) - M) ** 2).sum()
    ss_t = ((mat - M) ** 2).sum()
    ss_e = ss_t - ss_c - ss_s
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (ss_c / df1) / (ss_e / df2)
    return F, df1, df2, ss_c / (ss_c + ss_e)


def _rm(d, df, res, outdir):
    if d.n_factors == 2:
        return _rm_two(d, df, res, outdir)
    mat, _ = _wide(df, d, d.levels)
    n, k = mat.shape
    res.descriptives = _desc({lv: mat[:, i] for i, lv in enumerate(d.levels)})
    eps = 1.0
    if res.test == "anova_rm":
        resid = (mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0) + mat.mean()).ravel()
        w, p = _shapiro(resid)
        res.assumptions.append({"key": "a_normality", "detail": f"residuos/residuals: W = {E.fmt_num(w, True, 3)}, {E.fmt_p(p)}", "p": p, "ok": not p < .05})
        W, pm, eps = _mauchly(mat)
        res.assumptions.append({"key": "a_sphericity", "detail": f"W = {E.fmt_num(W, True, 3)}, {E.fmt_p(pm)}; ε_GG = {E.fmt_num(eps, True, 3)}",
                                "p": pm, "ok": not (pm < .05)})
        res.assumptions.append(_outliers({lv: mat[:, i] for i, lv in enumerate(d.levels)}))
        if p < .05 and n < SMALL_N and d.auto_fallback:
            res.test, res.fallback_reason = "friedman", "a_normality"
        elif not (pm < .05):
            eps = 1.0
    if res.test == "anova_rm":
        F, df1, df2, eta = _rm_oneway_stats(mat)
        df1e, df2e = df1 * eps, df2 * eps
        p = stats.f.sf(F, df1e, df2e)
        dfs = f"{df1}, {df2}" if eps == 1 else f"{df1e:.2f}, {df2e:.2f}"
        res.rows.append(_row(d.iv_name, "F", dfs, F, p, "η²p", eta, E.partial_eta2_ci(F, df1, df2), bounded=True))
        if eps < 1:
            res.notes.append((f"Se aplicó la corrección de Greenhouse-Geisser (ε = {eps:.3f}).",
                              f"The Greenhouse-Geisser correction was applied (ε = {eps:.3f})."))
        if p < d.alpha and k > 2:
            res.posthoc, res.posthoc_method = _paired_posthoc(mat, d.levels), "t pareadas / paired t (Holm)"
    else:
        r = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])
        W = E.kendall_w(mat)
        ci = E.bootstrap_ci(lambda m: E.kendall_w(m), [mat], n_boot=1000)
        res.rows.append(_row(d.iv_name, "χ²", f"{k - 1}", r.statistic, r.pvalue, "W", W, ci, bounded=True))
        if r.pvalue < d.alpha:
            res.posthoc, res.posthoc_method = _wilcoxon_posthoc(mat, d.levels), "Wilcoxon (Holm)"
    res.figures.append(_plot_profiles(mat, d.levels, d.dv_name, outdir))


def _rm_two(d, df, res, outdir):
    a, b, y = d.iv_name, d.factor_b_name, d.dv_name
    long = []
    for _, row in df.iterrows():
        for la in d.levels:
            for lb in d.levels_b:
                long.append({"id": row["id"], a: la, b: lb, y: float(row[f"{y}_{la}_{lb}"])})
    long = pd.DataFrame(long)
    res.descriptives = _desc({f"{la}·{lb}": long.loc[(long[a] == la) & (long[b] == lb), y] for la in d.levels for lb in d.levels_b})
    tab = AnovaRM(long, y, "id", within=[a, b]).fit().anova_table
    # n x (ka·kb) matrix (columns: a1b1, a1b2, ..., a2b1, ...) and orthonormal contrasts per effect
    wide = long.pivot_table(index="id", columns=[a, b], values=y).reindex(columns=pd.MultiIndex.from_product([d.levels, d.levels_b]))
    ka, kb = len(d.levels), len(d.levels_b)
    ca, cb = _orth_contrasts(ka), _orth_contrasts(kb)
    contrasts = {a: np.kron(ca, np.ones((kb, 1)) / np.sqrt(kb)), b: np.kron(np.ones((ka, 1)) / np.sqrt(ka), cb),
                 f"{a}:{b}": np.kron(ca, cb)}
    corrected = []
    for term, lab in ((a, a), (b, b), (f"{a}:{b}", f"{a} × {b}")):
        F, df1, df2 = tab.loc[term, "F Value"], tab.loc[term, "Num DF"], tab.loc[term, "Den DF"]
        eps = 1.0
        if df1 > 1:
            W, pm, gg, hf = _sphericity(wide.to_numpy() @ contrasts[term], ka * kb)
            res.assumptions.append({"key": "a_sphericity", "detail": f"{lab}: W = {E.fmt_num(W, True, 3)}, {E.fmt_p(pm)}; "
                                                                     f"ε_GG = {E.fmt_num(gg, True, 3)}, ε_HF = {E.fmt_num(hf, True, 3)}",
                                    "p": pm, "ok": not pm < .05})
            if pm < .05:
                eps = gg
                corrected.append(f"{lab}: ε = {eps:.3f}")
        p = stats.f.sf(F, df1 * eps, df2 * eps)
        dfs = f"{int(df1)}, {int(df2)}" if eps == 1 else f"{df1 * eps:.2f}, {df2 * eps:.2f}"
        eta = F * df1 / (F * df1 + df2)
        res.rows.append(_row(lab, "F", dfs, F, p, "η²p", eta, E.partial_eta2_ci(F, df1, df2), bounded=True))
    if corrected:
        res.notes.append((f"Se aplicó la corrección de Greenhouse-Geisser a los efectos que violaron la esfericidad ({'; '.join(corrected)}).",
                          f"The Greenhouse-Geisser correction was applied to the effects that violated sphericity ({'; '.join(corrected)})."))
    means = long.pivot_table(index=a, columns=b, values=y, aggfunc="mean")
    ses = long.pivot_table(index=a, columns=b, values=y, aggfunc=lambda s: s.std(ddof=1) / np.sqrt(len(s)))
    res.figures.append(_plot_interaction(means, ses, b, y, outdir))


def _mixed(d, df, res, outdir):
    a, y = d.iv_name, d.dv_name
    df = df.copy()
    df[a] = df[a].astype(str)
    mat, _ = _wide(df, d, d.levels_b)
    grp = df[a].to_numpy()
    glevels = [lv for lv in d.levels if lv in set(grp)]
    N, k, G = mat.shape[0], mat.shape[1], len(glevels)
    M = mat.mean()
    subj_m = mat.mean(axis=1)
    ss_bs = k * ((subj_m - M) ** 2).sum()
    ss_a = sum(k * (grp == g).sum() * (mat[grp == g].mean() - M) ** 2 for g in glevels)
    ss_sa = ss_bs - ss_a
    ss_t = ((mat - M) ** 2).sum()
    ss_ws = ss_t - ss_bs
    cm = mat.mean(axis=0)
    ss_b_weighted = N * ((cm - M) ** 2).sum()
    ss_ab = sum((grp == g).sum() * ((mat[grp == g].mean(axis=0) - mat[grp == g].mean() - cm + M) ** 2).sum() for g in glevels)
    ss_err = ss_ws - ss_b_weighted - ss_ab
    # Type III SS for the within factor (R car::Anova type 3 / afex / SPSS): it tests the unweighted mean of the
    # group profiles, so with unequal n it uses the harmonic mean of the group sizes; identical to the
    # weighted SS when groups are balanced. SS_A, SS_A×B and both error terms are the same for every SS type here.
    n_g = np.array([(grp == g).sum() for g in glevels])
    gm = np.vstack([mat[grp == g].mean(axis=0) for g in glevels])
    um = gm.mean(axis=0)
    ss_b = G ** 2 / (1 / n_g).sum() * ((um - um.mean()) ** 2).sum()
    df_a, df_sa = G - 1, N - G
    df_b, df_ab, df_err = k - 1, (G - 1) * (k - 1), (N - G) * (k - 1)
    # sphericity on the within factor: pooled within-group covariance with N − G error df (car::Anova, afex, SPSS)
    W, pm, gg, hf = 1.0, np.nan, 1.0, 1.0
    if k > 2:
        centered = np.vstack([mat[grp == g] - mat[grp == g].mean(axis=0) for g in glevels])
        W, pm, gg, hf = _sphericity(centered @ _orth_contrasts(k), df=N - G)
    eps = gg if pm < .05 else 1.0
    cells = {f"{g}·{lb}": mat[grp == g, j] for g in glevels for j, lb in enumerate(d.levels_b)}
    res.descriptives = _desc(cells)
    resid = np.concatenate([(mat[grp == g] - mat[grp == g].mean(axis=0)).ravel() for g in glevels])
    w, p = _shapiro(resid)
    res.assumptions.append({"key": "a_normality", "detail": f"residuos/residuals: W = {E.fmt_num(w, True, 3)}, {E.fmt_p(p)}", "p": p, "ok": not p < .05})
    for j, lb in enumerate(d.levels_b):
        lv = _levene([mat[grp == g, j] for g in glevels])
        lv["detail"] = f"{lb}: " + lv["detail"]
        res.assumptions.append(lv)
    res.assumptions.append({"key": "a_sphericity", "detail": f"W = {E.fmt_num(W, True, 3)}, {E.fmt_p(pm)}; ε_GG = {E.fmt_num(gg, True, 3)}, "
                                                             f"ε_HF = {E.fmt_num(hf, True, 3)}" if k > 2 else "k = 2",
                            "p": pm if k > 2 else None, "ok": not pm < .05})
    for lab, ss, df1, sse, df2, e in ((a, ss_a, df_a, ss_sa, df_sa, 1.0), (d.factor_b_name, ss_b, df_b, ss_err, df_err, eps),
                                      (f"{a} × {d.factor_b_name}", ss_ab, df_ab, ss_err, df_err, eps)):
        F = (ss / df1) / (sse / df2)
        pval = stats.f.sf(F, df1 * e, df2 * e)
        dfs = f"{df1}, {df2}" if e == 1 else f"{df1 * e:.2f}, {df2 * e:.2f}"
        res.rows.append(_row(lab, "F", dfs, F, pval, "η²p", ss / (ss + sse), E.partial_eta2_ci(F, df1, df2), bounded=True))
    if eps < 1:
        res.notes.append((f"Efectos intra sujetos con corrección de Greenhouse-Geisser (ε = {eps:.3f}).",
                          f"Within-subjects effects with Greenhouse-Geisser correction (ε = {eps:.3f})."))
    if len(set(n_g.tolist())) > 1:
        res.notes.append(("Grupos de distinto tamaño: se usaron sumas de cuadrados de tipo III (como SPSS y R afex), "
                          "que dan el mismo peso a cada grupo en el efecto intra sujetos.",
                          "Unequal group sizes: Type III sums of squares were used (as in SPSS and R afex), "
                          "which weight each group equally in the within-subjects effect."))
    if res.rows[2]["p"] < d.alpha:
        rows = []
        for j, lb in enumerate(d.levels_b):
            r = stats.ttest_ind(*[mat[grp == g, j] for g in glevels[:2]], equal_var=False) if G == 2 else stats.f_oneway(*[mat[grp == g, j] for g in glevels])
            rows.append({d.factor_b_name: lb, "estadístico/statistic": r.statistic, "p": r.pvalue})
        ph = pd.DataFrame(rows)
        ph["p_ajust/p_adj"] = multipletests(ph["p"], method="holm")[1]
        res.posthoc = ph
        res.posthoc_method = "efectos simples del grupo en cada medida / simple effects of group at each measurement (Holm)"
    means = pd.DataFrame({lb: [mat[grp == g, j].mean() for g in glevels] for j, lb in enumerate(d.levels_b)}, index=glevels)
    ses = pd.DataFrame({lb: [stats.sem(mat[grp == g, j]) for g in glevels] for j, lb in enumerate(d.levels_b)}, index=glevels)
    res.figures.append(_plot_interaction(means, ses, d.factor_b_name, y, outdir))


def _chi2(d, df, res, outdir):
    a, b = d.iv_name, d.dv_name
    tab = pd.crosstab(df[a].astype(str), df[b].astype(str))
    res.descriptives = tab.reset_index()
    chi, p, dof, exp = stats.chi2_contingency(tab, correction=False)
    low = int((exp < 5).sum())
    res.assumptions.append({"key": "a_expected", "detail": f"{low}/{exp.size} < 5 (min = {exp.min():.2f})", "p": None, "ok": low == 0})
    n = int(tab.values.sum())
    kmin = min(tab.shape)
    v = np.sqrt(chi / (n * (kmin - 1)))
    if low and tab.shape == (2, 2) and d.auto_fallback:
        res.test, res.fallback_reason = "fisher", "a_expected"
        orr, pf = stats.fisher_exact(tab.values)
        res.rows.append(_row(f"{a} × {b}", "OR", None, orr, pf, "V", v, E.cramers_v_ci(chi, dof, n, kmin), bounded=True))
    else:
        res.rows.append(_row(f"{a} × {b}", "χ²", f"{dof}, N = {n}", chi, p, "V", v, E.cramers_v_ci(chi, dof, n, kmin), bounded=True))
        if low:
            res.notes.append(("Hay frecuencias esperadas < 5: interpreta con cautela o agrupa categorías.",
                              "Some expected counts are < 5: interpret with caution or collapse categories."))
    if tab.shape[0] > 2 or tab.shape[1] > 2:
        rs, cs = tab.sum(axis=1).values[:, None] / n, tab.sum(axis=0).values[None, :] / n
        adj = (tab.values - exp) / np.sqrt(exp * (1 - rs) * (1 - cs))
        res.posthoc = pd.DataFrame(adj, index=tab.index, columns=tab.columns).round(2).reset_index()
        res.posthoc_method = "residuos tipificados corregidos / adjusted standardized residuals (|z| > 1.96)"
    tab.columns.name = b
    res.figures.append(_plot_bars(tab, outdir))


def _binary(series):
    s = series.astype(str).str.strip()
    m = {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "si": 1, "sí": 1, "yes": 1, "no": 0}
    out = s.str.lower().map(m)
    if out.isna().any():
        raise DataError("Las columnas de McNemar/Cochran deben ser 0/1 / must be 0/1")
    return out.to_numpy(int)


def _mcnemar(d, df, res, outdir):
    cols = [f"{d.dv_name}_{lv}" for lv in d.levels]
    x, y = _binary(df[cols[0]]), _binary(df[cols[1]])
    tab = pd.crosstab(pd.Series(x, name=cols[0]), pd.Series(y, name=cols[1])).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    res.descriptives = tab.reset_index()
    b, c = tab.loc[0, 1], tab.loc[1, 0]
    exact = (b + c) < 25
    r = mcnemar(tab.values, exact=exact, correction=True)
    gC, ci = E.cohen_g_mcnemar(int(b), int(c))
    res.rows.append(_row(f"{cols[0]} vs {cols[1]}", "binomial" if exact else "χ²", None if exact else "1",
                         r.statistic, r.pvalue, "g_C", gC, ci, bounded=True))
    if b + c:
        res.notes.append(("El IC 95% de la g de Cohen se calculó a partir del intervalo exacto de Clopper-Pearson para la proporción "
                          "de pares discordantes (Clopper y Pearson, 1934).",
                          "The 95% CI for Cohen's g was derived from the exact Clopper-Pearson interval for the proportion of "
                          "discordant pairs (Clopper & Pearson, 1934)."))
    tab.columns.name = cols[1]
    res.figures.append(_plot_bars(tab, outdir))


def _cochran(d, df, res, outdir):
    cols = [f"{d.dv_name}_{lv}" for lv in d.levels]
    mat = np.column_stack([_binary(df[c]) for c in cols])
    res.descriptives = pd.DataFrame({"condicion/condition": d.levels, "n": len(mat), "proporción/proportion": mat.mean(axis=0)})
    r = cochrans_q(mat)
    eq = E.eta2_q(mat)
    ci = E.bootstrap_ci(E.eta2_q, [mat], n_boot=1000)
    res.rows.append(_row(d.iv_name, "Q", f"{r.df}", r.statistic, r.pvalue, "η²_Q", eq, ci, bounded=True))
    res.notes.append(("Tamaño del efecto: η²_Q = Q / [N(k − 1)] (Serlin et al., 1982), con IC 95% bootstrap percentil "
                      "(1000 remuestreos de participantes).",
                      "Effect size: η²_Q = Q / [N(k − 1)] (Serlin et al., 1982), with a 95% percentile bootstrap CI "
                      "(1000 resamples of participants)."))
    res.figures.append(_plot_profiles(mat.astype(float), d.levels, "p(1)", outdir))


def _corr(d, df, res, outdir):
    x, y = df[d.iv_name].to_numpy(float), df[d.dv_name].to_numpy(float)
    n = len(x)
    res.descriptives = _desc({d.iv_name: x, d.dv_name: y})
    if res.test == "pearson":
        norm, ok = _normality_groups({d.iv_name: x, d.dv_name: y})
        res.assumptions += norm + [_outliers({"x": x, "y": y})]
        if not ok and n < SMALL_N and d.auto_fallback:
            res.test, res.fallback_reason = "spearman", "a_normality"
    if res.test == "pearson":
        r, p = stats.pearsonr(x, y, alternative=d.alternative)
        res.rows.append(_row(f"{d.iv_name} – {d.dv_name}", "r", f"{n - 2}", r, p, "r", r, E.r_ci(r, n), bounded=True, alternative=d.alternative))
    else:
        r, p = stats.spearmanr(x, y, alternative=d.alternative)
        res.rows.append(_row(f"{d.iv_name} – {d.dv_name}", "r_s", f"{n - 2}", r, p, "r_s", r, E.r_ci(r, n, spearman=True), bounded=True, alternative=d.alternative))
    res.figures.append(_plot_scatter(x, y, d.iv_name, d.dv_name, outdir))


def _design_matrix(d, df):
    X = df[d.predictors].copy()
    if d.iv_type == "nominal":
        X = pd.get_dummies(X.astype(str), drop_first=True, dtype=float)
    else:
        X = X.astype(float)
    return X


def _linreg(d, df, res, outdir):
    y = df[d.dv_name].to_numpy(float)
    X = _design_matrix(d, df)
    res.descriptives = _desc({c: df[c] for c in list(d.predictors) + [d.dv_name]} if d.iv_type != "nominal" else {d.dv_name: y})
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    bp = het_breuschpagan(model.resid, Xc)
    w, p = _shapiro(model.resid)
    dw = durbin_watson(model.resid)
    vifs = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])] if X.shape[1] > 1 else [1.0]
    res.assumptions += [
        {"key": "a_normality", "detail": f"residuos/residuals: W = {E.fmt_num(w, True, 3)}, {E.fmt_p(p)}", "p": p, "ok": not p < .05},
        {"key": "a_homoscedasticity", "detail": f"LM = {bp[0]:.2f}, {E.fmt_p(bp[1])}", "p": bp[1], "ok": not bp[1] < .05},
        {"key": "a_multicollinearity", "detail": "VIF máx./max = " + f"{max(vifs):.2f}", "p": None, "ok": max(vifs) < 5},
        {"key": "a_independence", "detail": f"DW = {dw:.2f}", "p": None, "ok": 1.5 <= dw <= 2.5},
    ]
    robust = bp[1] < .05
    fit = model.get_robustcov_results("HC3") if robust else model
    if robust:
        res.notes.append(("Heterocedasticidad detectada: errores típicos robustos HC3.", "Heteroscedasticity detected: HC3 robust standard errors."))
    df1, df2 = int(model.df_model), int(model.df_resid)
    res.rows.append(_row("Modelo/Model", "F", f"{df1}, {df2}", model.fvalue, model.f_pvalue, "R²", model.rsquared,
                         E.partial_eta2_ci(model.fvalue, df1, df2), bounded=True))
    Z = (X - X.mean()) / X.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    std = sm.OLS(zy, sm.add_constant(Z)).fit()
    std = std.get_robustcov_results("HC3") if robust else std
    ci_std = np.asarray(std.conf_int())
    coef = []
    names = list(Xc.columns)
    params, bse, tv, pv = map(np.asarray, (fit.params, fit.bse, fit.tvalues, fit.pvalues))
    for i, name in enumerate(names):
        row = {"término/term": name, "b": params[i], "EE/SE": bse[i], "t": tv[i], "p": pv[i]}
        if i > 0:
            row["β"] = np.asarray(std.params)[i]
            res.rows.append(_row(name, "t", f"{df2}", tv[i], pv[i], "β", row["β"], tuple(ci_std[i]), bounded=False))
        coef.append(row)
    res.extra_tables["coefficients"] = pd.DataFrame(coef)
    res.extra_tables["model"] = pd.DataFrame([{"R²": model.rsquared, "R²_ajust/adj": model.rsquared_adj, "F": model.fvalue,
                                               "gl1/df1": df1, "gl2/df2": df2, "p": model.f_pvalue}])
    if X.shape[1] >= 1 and d.iv_type != "nominal":
        res.figures.append(_plot_scatter(X.iloc[:, 0].to_numpy(), y, X.columns[0], d.dv_name, outdir))


def _auc_label(auc):
    """Qualitative label for AUC (Hosmer, Lemeshow & Sturdivant, 2013, p. 177)."""
    if auc >= .9:
        return "excepcional", "outstanding"
    if auc >= .8:
        return "excelente", "excellent"
    if auc >= .7:
        return "aceptable", "acceptable"
    if auc > .5:
        return "pobre", "poor"
    return "nula (no mejor que el azar)", "none (no better than chance)"


def _logreg(d, df, res, outdir):
    raw = df[d.dv_name].astype(str).str.strip()
    lv = sorted(raw.unique())
    if len(lv) != 2:
        raise DataError("La VD debe tener exactamente 2 valores / DV must have exactly 2 values")
    event = "1" if "1" in lv else ("1.0" if "1.0" in lv else lv[1])
    y = (raw == event).astype(int).to_numpy()
    X = _design_matrix(d, df)
    res.descriptives = pd.DataFrame({"VD/DV": [d.dv_name], "n": [len(y)], "eventos/events": [int(y.sum())], "p(evento/event)": [y.mean()]})
    Xc = sm.add_constant(X)
    model = sm.Logit(y, Xc).fit(disp=0)
    k = X.shape[1]
    epv = min(y.sum(), len(y) - y.sum()) / k
    vifs = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])] if k > 1 else [1.0]
    res.assumptions += [{"key": "a_epv", "detail": f"EPV = {epv:.1f}", "p": None, "ok": epv >= 10},
                        {"key": "a_multicollinearity", "detail": "VIF máx./max = " + f"{max(vifs):.2f}", "p": None, "ok": max(vifs) < 5}]
    n = len(y)
    r2_cs = 1 - np.exp(2 * (model.llnull - model.llf) / n)
    r2_n = r2_cs / (1 - np.exp(2 * model.llnull / n))
    res.rows.append(_row("Modelo/Model", "χ²", f"{k}", model.llr, model.llr_pvalue, "R²_N", r2_n, None, bounded=True))
    ci = np.asarray(model.conf_int())
    coef = []
    for i, name in enumerate(Xc.columns):
        b, se, z, p = np.asarray(model.params)[i], np.asarray(model.bse)[i], np.asarray(model.tvalues)[i], np.asarray(model.pvalues)[i]
        coef.append({"término/term": name, "B": b, "EE/SE": se, "z": z, "p": p, "OR": np.exp(b),
                     "OR_IC95_inf": np.exp(ci[i, 0]), "OR_IC95_sup": np.exp(ci[i, 1])})
        if i > 0:
            res.rows.append(_row(name, "z", None, z, p, "OR", np.exp(b), (np.exp(ci[i, 0]), np.exp(ci[i, 1]))))
    res.extra_tables["coefficients"] = pd.DataFrame(coef)
    prob = np.asarray(model.predict(Xc))
    acc = ((prob >= .5).astype(int) == y).mean()
    auc, (auc_lo, auc_hi) = E.auc_delong(y, prob)
    res.extra_tables["model"] = pd.DataFrame([{"χ²(LR)": model.llr, "gl/df": k, "p": model.llr_pvalue, "R²_McFadden": model.prsquared,
                                               "R²_Nagelkerke": r2_n, "% correcto/correct": acc * 100,
                                               "AUC": auc, "AUC_IC95_inf": auc_lo, "AUC_IC95_sup": auc_hi}])
    if np.isfinite(auc):
        lab_es, lab_en = _auc_label(auc)
        a, lo, hi = (E.fmt_num(v, True) for v in (auc, auc_lo, auc_hi))
        res.notes.append((f"Capacidad discriminativa del modelo: área bajo la curva ROC, AUC = {a}, IC 95% [{lo}, {hi}] "
                          f"(DeLong et al., 1988); discriminación {lab_es} (Hosmer et al., 2013). "
                          f"Una AUC de .50 equivale a clasificar al azar.",
                          f"Model discrimination: area under the ROC curve, AUC = {a}, 95% CI [{lo}, {hi}] "
                          f"(DeLong et al., 1988); {lab_en} discrimination (Hosmer et al., 2013). "
                          f"An AUC of .50 means chance-level classification."))
    if d.iv_type != "nominal":
        fig, ax = plt.subplots(figsize=(5, 4))
        x0 = X.iloc[:, 0].to_numpy()
        ax.scatter(x0, y + np.random.default_rng(2).uniform(-.04, .04, n), s=12, alpha=.5)
        o = np.argsort(x0)
        ax.plot(x0[o], np.asarray(model.predict(Xc))[o], ".", color=ACCENT, ms=3)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(f"P({d.dv_name} = 1)")
        res.figures.append(_save(fig, outdir, "fig_logistica.png"))


# ------------------------------------------------------------------ ordinal (proportional-odds) regression
def _ordinal_codes(raw: pd.Series, levels):
    """Map an ordinal DV to codes 0..J-1 following `levels` (lowest to highest); numeric values match numerically."""
    raw = raw.astype(str).str.strip()
    levels = [str(x).strip() for x in levels]
    lut = {lv: i for i, lv in enumerate(levels)}
    codes = raw.map(lut)
    if codes.isna().any():
        num_lv, num_raw = pd.to_numeric(pd.Series(levels), errors="coerce"), pd.to_numeric(raw, errors="coerce")
        if num_lv.notna().all() and num_raw.notna().all():
            codes = num_raw.map({float(v): i for i, v in enumerate(num_lv)})
    if codes.isna().any():
        bad = sorted(set(raw[codes.isna()]))[:10]
        raise DataError(f"Valores de la VD no incluidos en sus categorías {levels} / DV values not in its categories: {bad}")
    return codes.to_numpy(int)


def _fit_ordinal(y, X):
    """Proportional-odds (cumulative logit) model, parameterised as R MASS::polr / Stata ologit:
    logit P(Y <= j) = τ_j − xβ. Newton-Raphson with a BFGS fallback."""
    mod = OrderedModel(y, X, distr="logit")
    fit = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, kw in (("newton", {"maxiter": 200}), ("bfgs", {"maxiter": 5000, "gtol": 1e-8})):
            try:
                f = mod.fit(method=method, disp=0, **kw)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.all(np.isfinite(np.asarray(f.bse))) and (fit is None or f.llf > fit.llf + 1e-9):
                fit = f
            if f.mle_retvals.get("converged", False) and np.all(np.isfinite(np.asarray(f.bse))):
                break
    if fit is None:
        raise DataError("El modelo ordinal no convergió (¿separación o categorías vacías?) / the ordinal model did not converge")
    return fit


def brant_test(y, X):
    """Brant (1990) Wald test of the proportional-odds (parallel lines) assumption.

    Fits the J − 1 binary logits of Y > j and tests that their slopes are equal, overall (gl = K(J − 2)) and for each
    predictor (gl = J − 2), as Stata `brant` (Long & Freese) and R `brant::brant`. Returns (overall, per_variable, betas)
    with overall/per_variable entries (χ², df, p)."""
    y = np.asarray(y, int)
    Xc = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    K = Xc.shape[1] - 1
    M = int(y.max())                       # number of binary splits (J − 1)
    betas, pis = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(M):
            f = sm.Logit((y > j).astype(int), Xc).fit(disp=0, maxiter=200)
            betas.append(np.asarray(f.params))
            pis.append(np.asarray(f.predict(Xc)))
    inv = [np.linalg.inv(Xc.T @ (Xc * (p * (1 - p))[:, None])) for p in pis]
    V = np.zeros((M * K, M * K))
    for a in range(M):
        for b in range(a, M):
            w = pis[b] - pis[a] * pis[b]   # Cov of indicators Y > a and Y > b (b ≥ a)
            v = (inv[a] @ (Xc.T @ (Xc * w[:, None])) @ inv[b])[1:, 1:]
            V[a * K:(a + 1) * K, b * K:(b + 1) * K] = v
            V[b * K:(b + 1) * K, a * K:(a + 1) * K] = v.T
    bstar = np.concatenate([b[1:] for b in betas])
    D = np.zeros(((M - 1) * K, M * K))
    for r in range(M - 1):
        D[r * K:(r + 1) * K, :K] = np.eye(K)
        D[r * K:(r + 1) * K, (r + 1) * K:(r + 2) * K] = -np.eye(K)

    def wald(Dm):
        db = Dm @ bstar
        chi = float(db @ np.linalg.pinv(Dm @ V @ Dm.T) @ db)
        df = Dm.shape[0]
        return chi, df, float(stats.chi2.sf(chi, df))

    overall = wald(D)
    per = [wald(D[[r * K + k for r in range(M - 1)]]) for k in range(K)]
    return overall, per, betas


def _plot_ordinal(x, X, params, thresholds, k_col, levels, xlabel, outdir, name="fig_ordinal.png"):
    """Predicted probability of each DV category along the first predictor (others at their mean)."""
    xs = np.linspace(np.min(x), np.max(x), 100)
    base = np.asarray(X.mean(), float)
    eta = np.array([np.dot(np.where(np.arange(len(base)) == k_col, v, base), params) for v in xs])
    cum = 1 / (1 + np.exp(-(np.asarray(thresholds)[None, :] - eta[:, None])))
    cum = np.column_stack([np.zeros(len(xs)), cum, np.ones(len(xs))])
    probs = np.diff(cum, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.get_cmap("viridis")(np.linspace(.1, .85, len(levels)))
    ax.stackplot(xs, probs.T, labels=[str(lv) for lv in levels], colors=colors, alpha=.85)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(categoría / category)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, .5), title="VD/DV")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, name)


def _ordreg(d, df, res, outdir):
    levels = list(d.dv_levels)
    y_all = _ordinal_codes(df[d.dv_name], levels)
    present = sorted(set(y_all))
    if len(present) < 3:
        raise DataError("La VD ordinal necesita al menos 3 categorías observadas; con 2, usa la regresión logística binaria "
                        "(VD nominal) / the ordinal DV needs at least 3 observed categories; with 2, use binary logistic regression")
    used = [levels[i] for i in present]
    remap = {c: i for i, c in enumerate(present)}
    y = np.array([remap[c] for c in y_all])
    X = _design_matrix(d, df)
    if X.shape[1] == 0 or (X.std(ddof=0) == 0).any():
        raise DataError("Algún predictor no varía / some predictor has no variance")
    counts = np.bincount(y, minlength=len(used))
    res.descriptives = pd.DataFrame({"categoría/category": used, "n": counts, "%": 100 * counts / len(y),
                                     "% acumulado/cumulative": 100 * np.cumsum(counts) / len(y)})
    if len(used) < len(levels):
        miss = [lv for lv in levels if lv not in used]
        res.notes.append((f"Categorías de la VD sin casos (se omiten del modelo): {', '.join(miss)}.",
                          f"DV categories with no cases (omitted from the model): {', '.join(miss)}."))
    fit = _fit_ordinal(y, X)
    k, n, J = X.shape[1], len(y), len(used)
    params = np.asarray(fit.params)
    beta, bse = params[:k], np.asarray(fit.bse)[:k]
    thr = np.asarray(fit.model.transform_threshold_params(fit.params))[1:-1]
    # SE of thresholds by the delta method (statsmodels uses τ_1 and log-increments)
    Jac = np.zeros((J - 1, J - 1))
    Jac[:, 0] = 1
    for j in range(1, J - 1):
        Jac[j:, j] = np.exp(params[k + j])
    cov = np.asarray(fit.cov_params())[k:, k:]
    thr_se = np.sqrt(np.diag(Jac @ cov @ Jac.T))
    ll0 = float(np.sum(counts * np.log(counts / n)))
    llr = 2 * (fit.llf - ll0)
    llr_p = float(stats.chi2.sf(llr, k))
    r2_mf = 1 - fit.llf / ll0
    r2_cs = 1 - np.exp(2 * (ll0 - fit.llf) / n)
    r2_n = r2_cs / (1 - np.exp(2 * ll0 / n))
    # assumptions
    Xc = sm.add_constant(X)
    vifs = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])] if k > 1 else [1.0]
    res.assumptions.append({"key": "a_multicollinearity", "detail": "VIF máx./max = " + f"{max(vifs):.2f}", "p": None, "ok": max(vifs) < 5})
    try:
        (chi, bdf, bp), per, _ = brant_test(y, X)
    except np.linalg.LinAlgError:
        chi, bdf, bp, per = np.nan, 0, np.nan, []
    res.assumptions.append({"key": "a_proportional_odds", "detail": f"Brant: χ²({bdf}) = {chi:.2f}, {E.fmt_p(bp)}", "p": bp,
                            "ok": not bp < .05})
    if per:
        res.extra_tables["brant"] = pd.DataFrame([{"término/term": "Global/Omnibus", "χ²": chi, "gl/df": bdf, "p": bp}] +
                                                 [{"término/term": c, "χ²": v[0], "gl/df": v[1], "p": v[2]} for c, v in zip(X.columns, per)])
    if bp < .05:
        bad = [c for c, v in zip(X.columns, per) if v[2] < .05]
        who = f" (predictor(es): {', '.join(bad)})" if bad else ""
        res.notes.append((f"La prueba de Brant indica que no se cumple el supuesto de odds proporcionales{who}: el efecto podría "
                          "diferir según el punto de corte de la VD. Interpreta las OR como un resumen promedio y considera un modelo "
                          "de odds proporcionales parciales o una regresión logística multinomial.",
                          f"The Brant test indicates that the proportional-odds assumption is not met{who}: the effect may differ "
                          "across cut-points of the DV. Interpret the ORs as an average summary and consider a partial "
                          "proportional-odds model or multinomial logistic regression."))
    small = [used[i] for i in range(J) if counts[i] < 5]
    if small:
        res.notes.append((f"Categorías con menos de 5 casos ({', '.join(small)}): las estimaciones pueden ser inestables; "
                          "valora agrupar categorías adyacentes.",
                          f"Categories with fewer than 5 cases ({', '.join(small)}): estimates may be unstable; "
                          "consider merging adjacent categories."))
    # results
    res.rows.append(_row("Modelo/Model", "χ²", f"{k}", llr, llr_p, "R²_N", r2_n, None, bounded=True))
    z = stats.norm.ppf(.975)
    coef = []
    for i, name in enumerate(X.columns):
        b, se = beta[i], bse[i]
        zval = b / se
        p = 2 * stats.norm.sf(abs(zval))
        lo, hi = b - z * se, b + z * se
        coef.append({"término/term": name, "B": b, "EE/SE": se, "z": zval, "p": p, "OR": np.exp(b),
                     "OR_IC95_inf": np.exp(lo), "OR_IC95_sup": np.exp(hi)})
        res.rows.append(_row(name, "z", None, zval, p, "OR", np.exp(b), (np.exp(lo), np.exp(hi))))
    res.extra_tables["coefficients"] = pd.DataFrame(coef)
    res.extra_tables["thresholds"] = pd.DataFrame([{"umbral/threshold": f"{used[j]} | {used[j + 1]}", "τ": thr[j], "EE/SE": thr_se[j]}
                                                   for j in range(J - 1)])
    res.extra_tables["model"] = pd.DataFrame([{"χ²(LR)": llr, "gl/df": k, "p": llr_p, "-2LL": -2 * fit.llf,
                                               "R²_McFadden": r2_mf, "R²_Nagelkerke": r2_n}])
    res.notes.append(("Modelo de odds proporcionales (logit acumulado), logit P(Y ≤ j) = τ_j − xβ: una OR > 1 indica mayor probabilidad "
                      f"de puntuar en categorías más altas de {d.dv_name} por cada unidad del predictor (manteniendo constantes los demás), "
                      "y la misma OR se aplica a todos los puntos de corte de la VD.",
                      "Proportional-odds (cumulative logit) model, logit P(Y ≤ j) = τ_j − xβ: an OR > 1 indicates a higher probability "
                      f"of scoring in higher categories of {d.dv_name} per unit of the predictor (holding the others constant), "
                      "and the same OR applies to every cut-point of the DV."))
    if d.iv_type != "nominal":
        res.figures.append(_plot_ordinal(X.iloc[:, 0].to_numpy(float), X, beta, thr, 0, used, X.columns[0], outdir))
