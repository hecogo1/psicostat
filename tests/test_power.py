"""Sample sizes checked against G*Power 3.1 / published tables."""
import pytest

from psicostat import Design, sample_size


@pytest.mark.parametrize("kw,expected", [
    (dict(analysis="compare", design="between", n_groups=2), 128),                     # d = .5 -> 64 per group
    (dict(analysis="compare", design="within", n_groups=2), 34),                       # dz = .5 -> 34
    (dict(analysis="compare", design="between", n_groups=3), 159),                     # f = .25, k = 3 -> 159 (G*Power)
    (dict(analysis="compare", design="within", n_groups=3), 28),                       # f = .25, m = 3, rho = .5 -> 28
    (dict(analysis="association", n_groups=2, dv_type="nominal", iv_type="nominal"), 88),  # w = .3, df = 1 -> 88
    (dict(analysis="regression", iv_type="continuous", predictors=["a", "b", "c"]), 77),   # f2 = .15, 3 predictors -> 77
])
def test_against_gpower(kw, expected):
    assert sample_size(Design(**kw))["n_total"] == expected


def test_correlation_fisher():
    # r = .3, alpha .05, power .80 -> 85 with Fisher z (G*Power exact: 84)
    assert sample_size(Design(analysis="correlation", iv_type="continuous"))["n_total"] in (84, 85)


def test_dropout_and_nonparametric_inflate():
    base = sample_size(Design())["n_total"]
    assert sample_size(Design(dropout=0.2))["n_total"] > base
    assert sample_size(Design(dv_type="ordinal"))["n_total"] > base


def test_larger_effect_needs_fewer():
    assert sample_size(Design(effect_label="large"))["n_total"] < sample_size(Design(effect_label="small"))["n_total"]
    assert sample_size(Design(effect_size=0.8))["n_total"] == sample_size(Design(effect_label="large"))["n_total"]


@pytest.mark.parametrize("kw", [
    dict(design="mixed", n_groups=2, n_levels_b=3), dict(n_factors=2, n_groups=2, n_levels_b=3),
    dict(covariates=["pre"]), dict(analysis="regression", dv_type="nominal", predictors=["a", "b"]),
    dict(design="within", dv_type="nominal", n_groups=2),
])
def test_all_designs_have_n(kw):
    ss = sample_size(Design(**kw))
    assert ss["n_total"] > 0
