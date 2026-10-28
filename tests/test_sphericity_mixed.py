"""Sphericity in the mixed (split-plot) ANOVA uses the pooled within-group covariance with N − G df (#18).

Reference values computed with R 4 / car 3.1.5: ``summary(Anova(lm(cbind(t1, t2, t3, t4) ~ grupo),
idata = data.frame(f = factor(1:4)), idesign = ~f), multivariate = FALSE)`` (same as afex::aov_ez and SPSS).
Before the fix, Mauchly's p used N − 1 df (p = .0015 instead of .0042 for the data below).
"""
import numpy as np
import pandas as pd
import pytest

from psicostat import Design, analyze
from psicostat.analysis import _orth_contrasts, _sphericity
from psicostat.examples import example

M = np.array([
    [6.5, 10.9, 11.9, 12.2], [9.1, 10.3, 13.0, 14.2], [10.0, 10.2, 11.8, 13.7], [9.3, 10.7, 10.1, 9.5],
    [10.6, 14.5, 15.1, 19.7], [8.5, 10.5, 11.2, 15.4],
    [9.5, 11.1, 12.0, 12.6], [8.9, 11.3, 16.1, 16.8], [16.7, 19.0, 22.4, 29.8], [8.8, 10.9, 13.9, 14.0],
    [7.1, 7.6, 9.2, 12.2], [7.6, 10.6, 10.9, 14.3],
    [15.6, 16.7, 18.0, 19.8], [14.1, 13.8, 16.9, 20.1], [10.1, 11.9, 12.6, 15.7], [9.4, 10.9, 13.0, 15.6],
    [8.2, 10.3, 12.1, 17.4], [13.3, 13.9, 15.9, 19.0]])
GRP = np.repeat(["g1", "g2", "g3"], 6)
T = ["t1", "t2", "t3", "t4"]


def _pooled(mat, grp):
    levels = list(dict.fromkeys(grp))
    centered = np.vstack([mat[grp == g] - mat[grp == g].mean(axis=0) for g in levels])
    return _sphericity(centered @ _orth_contrasts(mat.shape[1]), df=len(mat) - len(levels))


@pytest.mark.parametrize("drop, ref", [
    # (W, p_Mauchly, ε_GG, ε_HF) from R car::Anova
    ([], (0.28545896319027, 0.00423427171728, 0.549905733173, 0.609112569796)),        # balanced 6/6/6
    ([11, 17], (0.23088823722216, 0.00430910797142, 0.525226232167, 0.585286460879)),  # unbalanced 6/5/5
])
def test_pooled_sphericity_matches_car(drop, ref):
    keep = np.setdiff1d(np.arange(len(M)), drop)
    W, p, gg, hf = _pooled(M[keep], GRP[keep])
    assert (W, p, gg, hf) == pytest.approx(ref, rel=1e-9)


def test_example_mixed_sphericity_matches_car():
    d, df = example("mixed")
    W, p, gg, hf = _pooled(df[[f"{d.dv_name}_{lv}" for lv in d.levels_b]].to_numpy(float), df[d.iv_name].to_numpy())
    assert (W, p, gg) == pytest.approx((0.969989875771, 0.488685251172, 0.970864243445), rel=1e-9)
    assert hf == 1.0  # car: 1.0112 > 1, treated as 1


def test_mixed_anova_uses_n_minus_g_df():
    d = Design(design="mixed", n_factors=2, n_groups=3, n_levels_b=4, levels=["g1", "g2", "g3"], levels_b=T,
               dv_name="y", iv_name="grupo", factor_b_name="tiempo")
    df = pd.DataFrame({"id": range(len(M)), "grupo": GRP, **{f"y_{t}": M[:, j] for j, t in enumerate(T)}})
    res = analyze(d, df)
    sph = next(a for a in res.assumptions if a["key"] == "a_sphericity")
    assert sph["p"] == pytest.approx(0.00423427171728, rel=1e-9) and not sph["ok"]
    assert "ε_GG = .550" in sph["detail"] and "ε_HF = .609" in sph["detail"]
    rows = {r["label"]: r for r in res.rows}
    # Greenhouse-Geisser corrected effects (car: Pr(>F[GG]))
    assert rows["tiempo"]["p"] == pytest.approx(1.397864813e-09, rel=1e-8)
    assert rows["grupo × tiempo"]["p"] == pytest.approx(0.4395974423, rel=1e-8)
    assert rows["tiempo"]["df"] == "1.65, 24.75"
