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


def _row(label, sym, df, stat, p, es_sym=None, es=None, ci=None, bounded=False):
    return {"label": label, "stat_symbol": sym, "df": df, "stat": float(stat) if stat is not None else None,
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


def _mauchly(mat):
    """Mauchly's test and Greenhouse-Geisser epsilon for an n x k matrix."""
    n, k = mat.shape
    if k < 3:
        return 1.0, np.nan, 1.0
    c = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :k - 1]]))[0][:, 1:]
    s = np.cov(mat, rowvar=False)
    m = c.T @ s @ c
    p = k - 1
    tr = np.trace(m)
    w = np.linalg.det(m) / (tr / p) ** p
    d = 1 - (2 * p ** 2 + p + 2) / (6 * p * (n - 1))
    chi = -(n - 1) * d * np.log(max(w, 1e-300))
    df = p * (p + 1) / 2 - 1
    pval = stats.chi2.sf(chi, df)
    eps = tr ** 2 / (p * np.trace(m @ m))
    return w, pval, min(max(eps, 1 / p), 1.0)


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
        "pearson": _corr, "spearman": _corr, "linreg": _linreg, "logreg": _logreg,
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
        r = stats.ttest_ind(x, y, equal_var=False)
        gval, ci = E.hedges_g(x, y)
        res.rows.append(_row(f"{a} vs {b}", "t", f"{r.df:.2f}", r.statistic, r.pvalue, "g", gval, ci))
    else:
        r = stats.mannwhitneyu(x, y, alternative="two-sided")
        rb = E.rank_biserial_ind(x, y)
        ci = E.bootstrap_ci(E.rank_biserial_ind, [x, y])
        res.rows.append(_row(f"{a} vs {b}", "U", None, r.statistic, r.pvalue, "r_rb", rb, ci, bounded=True))
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
        r = stats.ttest_rel(x, y)
        dz, ci = E.cohen_dz(diff)
        res.rows.append(_row(lab, "t", f"{len(diff) - 1}", r.statistic, r.pvalue, "d_z", dz, ci))
    else:
        r = stats.wilcoxon(x, y)
        rb = E.rank_biserial_paired(diff)
        ci = E.bootstrap_ci(lambda dd: E.rank_biserial_paired(dd), [diff])
        res.rows.append(_row(lab, "V", None, r.statistic, r.pvalue, "r_rb", rb, ci, bounded=True))
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
        res.rows.append(_row(d.iv_name, "F", f"{k - 1}, {df2:.2f}", F, p, "ω²", om, None, bounded=True))
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
    for term, lab in ((a, a), (b, b), (f"{a}:{b}", f"{a} × {b}")):
        F, df1, df2, p = tab.loc[term, "F Value"], tab.loc[term, "Num DF"], tab.loc[term, "Den DF"], tab.loc[term, "Pr > F"]
        eta = F * df1 / (F * df1 + df2)
        res.rows.append(_row(lab, "F", f"{int(df1)}, {int(df2)}", F, p, "η²p", eta, E.partial_eta2_ci(F, df1, df2), bounded=True))
    for la, lb in ((a, d.levels), (b, d.levels_b)):
        if len(lb) > 2:
            mat = long.pivot_table(index="id", columns=la, values=y).to_numpy()
            W, pm, eps = _mauchly(mat)
            res.assumptions.append({"key": "a_sphericity", "detail": f"{la}: W = {E.fmt_num(W, True, 3)}, {E.fmt_p(pm)}",
                                    "p": pm, "ok": not pm < .05})
    res.notes.append(("ANOVA de dos factores intra sujetos sin corrección de esfericidad automática (v0.1).",
                      "Two-way within-subjects ANOVA without automatic sphericity correction (v0.1)."))
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
    ss_b = N * ((cm - M) ** 2).sum()
    ss_ab = sum((grp == g).sum() * ((mat[grp == g].mean(axis=0) - mat[grp == g].mean() - cm + M) ** 2).sum() for g in glevels)
    ss_err = ss_ws - ss_b - ss_ab
    df_a, df_sa = G - 1, N - G
    df_b, df_ab, df_err = k - 1, (G - 1) * (k - 1), (N - G) * (k - 1)
    # sphericity on within factor (pooled within groups)
    centered = np.vstack([mat[grp == g] - mat[grp == g].mean(axis=0) for g in glevels]) + cm
    W, pm, eps = _mauchly(centered)
    if not pm < .05:
        eps = 1.0
    cells = {f"{g}·{lb}": mat[grp == g, j] for g in glevels for j, lb in enumerate(d.levels_b)}
    res.descriptives = _desc(cells)
    resid = np.concatenate([(mat[grp == g] - mat[grp == g].mean(axis=0)).ravel() for g in glevels])
    w, p = _shapiro(resid)
    res.assumptions.append({"key": "a_normality", "detail": f"residuos/residuals: W = {E.fmt_num(w, True, 3)}, {E.fmt_p(p)}", "p": p, "ok": not p < .05})
    for j, lb in enumerate(d.levels_b):
        lv = _levene([mat[grp == g, j] for g in glevels])
        lv["detail"] = f"{lb}: " + lv["detail"]
        res.assumptions.append(lv)
    res.assumptions.append({"key": "a_sphericity", "detail": f"W = {E.fmt_num(W, True, 3)}, {E.fmt_p(pm)}" if k > 2 else "k = 2",
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
    if len(set(int((grp == g).sum()) for g in glevels)) > 1:
        res.notes.append(("Grupos no equilibrados: las sumas de cuadrados son aproximadas (v0.1).",
                          "Unbalanced groups: sums of squares are approximate (v0.1)."))
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
    gC = abs(b / (b + c) - .5) if b + c else 0.0
    res.rows.append(_row(f"{cols[0]} vs {cols[1]}", "binomial" if exact else "χ²", None if exact else "1",
                         r.statistic, r.pvalue, "g_C", gC, None, bounded=True))
    tab.columns.name = cols[1]
    res.figures.append(_plot_bars(tab, outdir))


def _cochran(d, df, res, outdir):
    cols = [f"{d.dv_name}_{lv}" for lv in d.levels]
    mat = np.column_stack([_binary(df[c]) for c in cols])
    res.descriptives = pd.DataFrame({"condicion/condition": d.levels, "n": len(mat), "proporción/proportion": mat.mean(axis=0)})
    r = cochrans_q(mat)
    res.rows.append(_row(d.iv_name, "Q", f"{r.df}", r.statistic, r.pvalue))
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
        r, p = stats.pearsonr(x, y)
        res.rows.append(_row(f"{d.iv_name} – {d.dv_name}", "r", f"{n - 2}", r, p, "r", r, E.r_ci(r, n), bounded=True))
    else:
        r, p = stats.spearmanr(x, y)
        res.rows.append(_row(f"{d.iv_name} – {d.dv_name}", "r_s", f"{n - 2}", r, p, "r_s", r, E.r_ci(r, n, spearman=True), bounded=True))
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
    acc = ((model.predict(Xc) >= .5).astype(int) == y).mean()
    res.extra_tables["model"] = pd.DataFrame([{"χ²(LR)": model.llr, "gl/df": k, "p": model.llr_pvalue, "R²_McFadden": model.prsquared,
                                               "R²_Nagelkerke": r2_n, "% correcto/correct": acc * 100}])
    if d.iv_type != "nominal":
        fig, ax = plt.subplots(figsize=(5, 4))
        x0 = X.iloc[:, 0].to_numpy()
        ax.scatter(x0, y + np.random.default_rng(2).uniform(-.04, .04, n), s=12, alpha=.5)
        o = np.argsort(x0)
        ax.plot(x0[o], np.asarray(model.predict(Xc))[o], ".", color=ACCENT, ms=3)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(f"P({d.dv_name} = 1)")
        res.figures.append(_save(fig, outdir, "fig_logistica.png"))
