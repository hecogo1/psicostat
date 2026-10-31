"""Mixed (split-plot) ANOVA with unequal group sizes uses Type III sums of squares (#4).

Reference values computed with R 4 / car 3.1 (``options(contrasts = c("contr.sum", "contr.poly"))``;
``summary(Anova(lm(cbind(t1, t2, t3, t4) ~ grupo), idata = data.frame(tiempo = factor(1:4)),
idesign = ~tiempo, type = 3), multivariate = FALSE)``), identical to afex::aov_ez and SPSS GLM.
Before the fix the within factor used weighted (Type I/II) SS: F(tiempo) = 23.49 instead of 21.01
for the 6/3/2 groups below.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from test_sphericity_mixed import GRP, M, T

from psicostat import Design, analyze
from psicostat.analysis import _orth_contrasts


def _analyze(drop):
    keep = np.setdiff1d(np.arange(len(M)), drop)
    d = Design(design="mixed", n_factors=2, n_groups=3, n_levels_b=4, levels=["g1", "g2", "g3"], levels_b=T,
               dv_name="y", iv_name="grupo", factor_b_name="tiempo")
    df = pd.DataFrame({"id": range(len(keep)), "grupo": GRP[keep], **{f"y_{t}": M[keep, j] for j, t in enumerate(T)}})
    return analyze(d, df), M[keep], GRP[keep]


@pytest.mark.parametrize("drop, ref", [
    # group sizes 6/5/4; Mauchly significant -> GG-corrected p (car: Pr(>F[GG]))
    ([11, 16, 17], {"grupo": (1.02843603564, 0.387039528304, "2, 12"),
                    "tiempo": (40.5236041289, 3.44284957231e-07, "1.62, 19.43"),
                    "grupo × tiempo": (0.796281972608, 0.519134033649, "3.24, 19.43")}),
    # group sizes 6/3/2; sphericity holds -> uncorrected p
    ([9, 10, 11, 14, 15, 16, 17], {"grupo": (2.75615120184, 0.12286905652, "2, 8"),
                                   "tiempo": (21.0060362329, 6.73956775024e-07, "3, 24"),
                                   "grupo × tiempo": (0.886967533414, 0.519654206784, "6, 24")}),
])
def test_unbalanced_mixed_matches_car_type3(drop, ref):
    res, _, _ = _analyze(drop)
    rows = {r["label"]: r for r in res.rows}
    for lab, (F, p, dfs) in ref.items():
        assert rows[lab]["stat"] == pytest.approx(F, rel=1e-9)
        assert rows[lab]["p"] == pytest.approx(p, rel=1e-8)
        assert rows[lab]["df"] == dfs
    assert any("tipo III" in es and "Type III" in en for es, en in res.notes)


def test_partial_eta2_matches_afex():
    res, _, _ = _analyze([11, 16, 17])
    # afex::aov_ez(..., anova_table = list(es = "pes"))
    assert [r["es"] for r in res.rows] == pytest.approx([0.146325018885, 0.771531291521, 0.117164351894], rel=1e-9)


def test_within_ss_matches_statsmodels_intercept_test():
    """Independent route: Type III intercept test of each orthonormal contrast score on sum-coded groups."""
    res, m, g = _analyze([9, 10, 11, 14, 15, 16, 17])
    z = m @ _orth_contrasts(m.shape[1])
    ss = sse = 0.0
    for j in range(z.shape[1]):
        tab = anova_lm(smf.ols("z ~ C(g, Sum)", pd.DataFrame({"z": z[:, j], "g": g})).fit(), typ=3)
        ss += tab.loc["Intercept", "sum_sq"]
        sse += tab.loc["Residual", "sum_sq"]
    df_err = (len(m) - 3) * (m.shape[1] - 1)
    assert res.rows[1]["stat"] == pytest.approx((ss / 3) / (sse / df_err), rel=1e-10)


def test_balanced_groups_unchanged_and_no_note():
    res, _, _ = _analyze([])
    assert res.rows[1]["p"] == pytest.approx(1.397864813e-09, rel=1e-8)  # same as before (Type I = II = III)
    assert not any("tipo III" in es for es, _ in res.notes)
