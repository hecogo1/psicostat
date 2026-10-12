import pytest

from psicostat import Design, recommend


@pytest.mark.parametrize("kw,test,fallback", [
    (dict(n_groups=2), "ttest_ind", "mannwhitney"),
    (dict(n_groups=2, dv_type="ordinal"), "mannwhitney", None),
    (dict(design="within", n_groups=2), "ttest_paired", "wilcoxon"),
    (dict(design="within", n_groups=2, dv_type="ordinal"), "wilcoxon", None),
    (dict(n_groups=3), "anova_oneway", "kruskal"),
    (dict(n_groups=3, dv_type="ordinal"), "kruskal", None),
    (dict(n_factors=2), "anova_factorial", None),
    (dict(covariates=["edad"]), "ancova", None),
    (dict(design="within", n_groups=3), "anova_rm", "friedman"),
    (dict(design="within", n_groups=3, dv_type="ordinal"), "friedman", None),
    (dict(design="mixed"), "anova_mixed", None),
    (dict(dv_type="nominal"), "chi2", "fisher"),
    (dict(analysis="association", dv_type="nominal"), "chi2", "fisher"),
    (dict(design="within", dv_type="nominal"), "mcnemar", None),
    (dict(design="within", dv_type="nominal", n_groups=3), "cochran", None),
    (dict(analysis="correlation", iv_type="continuous"), "pearson", "spearman"),
    (dict(analysis="correlation", iv_type="ordinal"), "spearman", None),
    (dict(analysis="regression", iv_type="continuous"), "linreg", None),
    (dict(analysis="regression", dv_type="nominal", iv_type="continuous"), "logreg", None),
])
def test_recommend(kw, test, fallback):
    r = recommend(Design(**kw))
    assert (r.test, r.fallback) == (test, fallback)
    assert r.assumptions
    assert r.label("es") != r.label("en")


def test_invalid():
    with pytest.raises(ValueError):
        Design(analysis="magic")
    with pytest.raises(ValueError):
        Design(alpha=1.5)


def test_json_roundtrip(tmp_path):
    d = Design(title="x", levels=["a", "b"], covariates=["c"])
    p = tmp_path / "d.json"
    d.to_json(p)
    assert Design.from_json(p) == d
