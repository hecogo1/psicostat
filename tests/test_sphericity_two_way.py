"""Sphericity (Mauchly, ε_GG, ε_HF) and Greenhouse-Geisser correction per effect in two-way within-subjects ANOVA (#3).

Reference values computed with pingouin 0.5+ ``rm_anova(..., correction=True)`` and ``epsilon(..., correction="hf")``,
which reproduce R ``mauchly.test`` / ``car::Anova`` (afex) for each effect, including the A × B interaction.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from psicostat import Design, analyze, build_report
from psicostat.analysis import _mauchly, _orth_contrasts, _sphericity

A, B = ["a1", "a2", "a3"], ["b1", "b2", "b3"]
M = np.array([
    [11.9, 12.2, 7.2, 12.7, 13.3, 21.4, 13.3, 18.3, 24.1], [7.0, 8.5, 10.3, 8.6, 5.4, 0.2, 8.1, 4.3, 2.1],
    [9.8, 13.3, 9.3, 10.8, 10.6, 13.4, 10.9, 13.8, 17.2], [9.2, 9.5, 6.9, 9.4, 10.6, 7.3, 10.0, 9.7, 11.5],
    [9.8, 9.6, 10.9, 10.4, 7.3, 12.6, 11.4, 9.4, 5.1], [9.6, 11.5, 12.0, 10.6, 9.2, 10.5, 9.9, 11.0, 9.9],
    [8.6, 7.2, 7.9, 8.4, 7.4, 3.7, 8.6, 4.2, -1.1], [9.6, 9.8, 8.4, 9.8, 11.9, 7.1, 10.6, 11.7, 11.5],
    [9.1, 10.7, 10.0, 9.8, 6.2, 11.7, 10.3, 7.4, 2.2], [14.3, 14.5, 14.0, 12.3, 17.6, 24.1, 14.7, 22.9, 33.4],
    [10.6, 10.7, 13.8, 11.1, 11.8, 13.6, 10.5, 12.0, 15.2], [9.3, 10.9, 11.9, 10.4, 9.3, 12.4, 9.6, 10.9, 7.2]])

# effect: (F, p_GG, ε_GG, W, p_Mauchly, ε_HF)
REF = {"tarea": (0.4727899084, 0.5273801825, 0.5653106566, 0.2310611195, 0.0006586192, 0.5860275568),
       "bloque": (0.4073619328, 0.5547856549, 0.5508787525, 0.1847185150, 0.0002150563, 0.5668223458),
       "tarea × bloque": (0.3644675195, 0.6705470262, 0.4352545700, 0.0182265795, 0.0000263242, 0.5101052128)}


def _design():
    return Design(design="within", n_factors=2, n_groups=3, n_levels_b=3, levels=A, levels_b=B,
                  dv_name="rt", iv_name="tarea", factor_b_name="bloque")


def _df(mat=M):
    cols = {f"rt_{la}_{lb}": mat[:, 3 * i + j] for i, la in enumerate(A) for j, lb in enumerate(B)}
    return pd.DataFrame({"id": range(len(mat)), **cols})


def test_two_way_rm_gg_correction_matches_reference():
    res = analyze(_design(), _df())
    rows = {r["label"]: r for r in res.rows}
    sph = {a["detail"].split(":")[0]: a for a in res.assumptions if a["key"] == "a_sphericity"}
    assert set(sph) == set(REF)
    for lab, (F, p_gg, eps, W, pm, hf) in REF.items():
        r = rows[lab]
        assert r["stat"] == pytest.approx(F, rel=1e-8)
        assert r["p"] == pytest.approx(p_gg, rel=1e-7)
        df1, df2 = (float(x) for x in r["df"].split(", "))
        q = 4 if lab == "tarea × bloque" else 2
        assert (df1, df2) == pytest.approx((eps * q, eps * q * 11), abs=.006)
        assert sph[lab]["p"] == pytest.approx(pm, rel=1e-6) and not sph[lab]["ok"]
    note = " ".join(n[0] for n in res.notes)
    assert "Greenhouse-Geisser" in note and "tarea × bloque: ε = 0.435" in note
    assert "sin corrección" not in note


def test_sphericity_helper_matches_reference_and_box_formula():
    wide = M
    ca, cb = _orth_contrasts(3), _orth_contrasts(3)
    C = {"tarea": np.kron(ca, np.ones((3, 1)) / np.sqrt(3)), "bloque": np.kron(np.ones((3, 1)) / np.sqrt(3), cb),
         "tarea × bloque": np.kron(ca, cb)}
    for lab, (_, _, eps, W, pm, hf) in REF.items():
        w, p, gg, h = _sphericity(wide @ C[lab], 9)
        assert (w, p, gg, h) == pytest.approx((W, pm, eps, hf), rel=1e-6)
    # Box (1954) ε from the double-centred covariance matrix of the cell means averaged over B (main effect A)
    S = np.cov(M.reshape(-1, 3, 3).mean(axis=2), rowvar=False)
    k = 3
    Sc = S - S.mean(axis=0) - S.mean(axis=1)[:, None] + S.mean()
    box = np.trace(Sc) ** 2 / ((k - 1) * (Sc ** 2).sum())
    assert _sphericity(wide @ C["tarea"], 9)[2] == pytest.approx(box)


def test_no_correction_when_sphericity_holds():
    rng = np.random.default_rng(1)
    mat = rng.normal(0, 1, (40, 9)) + rng.normal(0, 1, (40, 1))
    mat[:, 4:] += .4
    res = analyze(_design(), _df(mat))
    sph = [a for a in res.assumptions if a["key"] == "a_sphericity"]
    assert len(sph) == 3 and all(a["ok"] for a in sph)
    for r in res.rows:
        F, (df1, df2) = r["stat"], (int(x) for x in r["df"].split(", "))
        assert r["p"] == pytest.approx(stats.f.sf(F, df1, df2))
    assert not any("Greenhouse" in n[0] for n in res.notes)


def test_effects_with_one_df_are_not_tested_for_sphericity():
    d = Design(design="within", n_factors=2, n_groups=2, n_levels_b=3, levels=["x", "y"], levels_b=B,
               dv_name="rt", iv_name="tarea", factor_b_name="bloque")
    mat = M[:, [0, 1, 2, 6, 7, 8]]
    df = pd.DataFrame({"id": range(len(mat)), **{f"rt_{la}_{lb}": mat[:, 3 * i + j] for i, la in enumerate(["x", "y"]) for j, lb in enumerate(B)}})
    res = analyze(d, df)
    labels = [a["detail"].split(":")[0] for a in res.assumptions if a["key"] == "a_sphericity"]
    assert labels == ["bloque", "tarea × bloque"]
    assert res.rows[0]["df"] == "1, 11"


def test_one_way_mauchly_unchanged_api():
    W, p, eps = _mauchly(M[:, :3])
    w2, p2, gg, _ = _sphericity(M[:, :3] @ _orth_contrasts(3))
    assert (W, p, eps) == pytest.approx((w2, p2, gg))


def test_report_mentions_correction():
    rep = build_report(analyze(_design(), _df()), "es")
    text = " ".join(rep["results"])
    assert "Greenhouse-Geisser" in text
