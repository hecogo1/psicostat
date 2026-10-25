"""Study design description and test recommendation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .i18n import t

ALTERNATIVES = ("two-sided", "greater", "less")
# tests for which a one-sided (directional) alternative is valid
DIRECTIONAL_TESTS = {"ttest_ind", "mannwhitney", "ttest_paired", "wilcoxon", "pearson", "spearman"}
ANALYSES = ("compare", "correlation", "regression", "association")
DESIGNS = ("between", "within", "mixed")
LEVELS = ("continuous", "ordinal", "nominal")

# Cohen (1988) benchmarks per effect-size family
BENCHMARKS = {
    "d": {"small": 0.2, "medium": 0.5, "large": 0.8},
    "f": {"small": 0.10, "medium": 0.25, "large": 0.40},
    "r": {"small": 0.10, "medium": 0.30, "large": 0.50},
    "w": {"small": 0.10, "medium": 0.30, "large": 0.50},
    "f2": {"small": 0.02, "medium": 0.15, "large": 0.35},
    "or": {"small": 1.5, "medium": 2.5, "large": 4.3},  # Chen, Cohen & Chen (2010)
}


@dataclass
class Design:
    title: str = ""
    question: str = ""
    analysis: str = "compare"          # compare / correlation / regression / association
    design: str = "between"            # between / within / mixed
    n_factors: int = 1                 # 1 or 2 (factorial)
    n_groups: int = 2                  # levels of factor A (groups or conditions)
    n_levels_b: int = 2                # levels of factor B
    dv_type: str = "continuous"
    iv_type: str = "nominal"
    dv_name: str = "puntuacion"
    iv_name: str = "grupo"
    factor_b_name: str = "factor_b"
    levels: list = field(default_factory=list)
    levels_b: list = field(default_factory=list)
    dv_levels: list = field(default_factory=list)   # categories of a nominal DV
    predictors: list = field(default_factory=list)
    covariates: list = field(default_factory=list)
    effect_label: str = "medium"
    effect_size: Optional[float] = None
    alpha: float = 0.05
    power: float = 0.80
    dropout: float = 0.0
    rho: float = 0.5                   # correlation among repeated measures
    event_rate: float = 0.3            # logistic regression
    auto_fallback: bool = True
    decimal_comma: bool = False        # Spanish output with decimal comma (APA in some countries)
    alternative: str = "two-sided"     # two-sided / greater / less (directional H1; see DIRECTIONAL_TESTS)

    def __post_init__(self):
        if self.analysis not in ANALYSES:
            raise ValueError(f"analysis must be one of {ANALYSES}")
        if self.design not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}")
        if self.dv_type not in LEVELS or self.iv_type not in LEVELS:
            raise ValueError(f"variable types must be one of {LEVELS}")
        if not 0 < self.alpha < 1 or not 0 < self.power < 1:
            raise ValueError("alpha and power must be in (0, 1)")
        if self.design == "mixed":
            self.n_factors = 2
        self.n_groups = int(self.n_groups)
        if self.n_groups < 2 and self.analysis == "compare":
            raise ValueError("n_groups must be >= 2")
        if not self.levels:
            pre = "c" if self.design == "within" else "g"
            self.levels = [f"{pre}{i + 1}" for i in range(self.n_groups)]
        self.levels = [str(x) for x in self.levels]
        self.n_groups = len(self.levels) if self.analysis == "compare" or self.analysis == "association" else self.n_groups
        if not self.levels_b:
            pre = "t" if self.design == "mixed" else "b"
            self.levels_b = [f"{pre}{i + 1}" for i in range(int(self.n_levels_b))]
        self.levels_b = [str(x) for x in self.levels_b]
        self.n_levels_b = len(self.levels_b)
        if not self.dv_levels:
            self.dv_levels = ["si", "no"]
        if not self.predictors:
            self.predictors = ["x1"]
        if self.alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {ALTERNATIVES}")
        if self.alternative != "two-sided" and recommend(self).test not in DIRECTIONAL_TESTS:
            raise ValueError("Los contrastes unilaterales solo están disponibles para pruebas t, Mann-Whitney, Wilcoxon y correlaciones / "
                             "one-sided tests are only available for t tests, Mann-Whitney, Wilcoxon and correlations")

    @property
    def one_sided(self):
        return self.alternative != "two-sided"

    @property
    def within_levels(self):
        """Levels of the repeated-measures factor."""
        if self.design == "within":
            return self.levels
        if self.design == "mixed":
            return self.levels_b
        return []

    def to_dict(self):
        return asdict(self)

    def to_json(self, path=None):
        s = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class Recommendation:
    test: str
    fallback: Optional[str]
    family: str                        # effect size family for power analysis
    assumptions: list
    notes: list = field(default_factory=list)   # (es, en) tuples

    def label(self, lang="es"):
        return t(self.test, lang)

    def fallback_label(self, lang="es"):
        return t(self.fallback, lang) if self.fallback else "—"

    def notes_text(self, lang="es"):
        return [n[0] if lang == "es" else n[1] for n in self.notes]


def recommend(d: Design) -> Recommendation:
    """Return the recommended test for a design, with fallback and assumptions."""
    N, H, S = "a_normality", "a_homogeneity", "a_sphericity"
    indep = "a_independent_obs"
    if d.analysis == "association" or (d.analysis == "compare" and d.dv_type == "nominal"):
        if d.analysis == "compare" and d.design == "within":
            if d.n_groups == 2:
                return Recommendation("mcnemar", None, "w", ["a_independent_obs"],
                                      [("VD dicotómica medida dos veces en los mismos participantes.",
                                        "Dichotomous DV measured twice in the same participants.")])
            return Recommendation("cochran", None, "w", ["a_independent_obs"],
                                  [("VD dicotómica medida en 3+ condiciones.", "Dichotomous DV measured in 3+ conditions.")])
        return Recommendation("chi2", "fisher", "w", [indep, "a_expected"],
                              [("Si alguna frecuencia esperada es < 5 en una tabla 2×2 se usa la prueba exacta de Fisher.",
                                "If any expected count is < 5 in a 2×2 table, Fisher's exact test is used.")])
    if d.analysis == "correlation":
        if "ordinal" in (d.dv_type, d.iv_type):
            return Recommendation("spearman", None, "r", [indep, "a_linearity"],
                                  [("Variables ordinales: se usa directamente Spearman (monotonía).",
                                    "Ordinal variables: Spearman (monotonic relation) is used directly.")])
        return Recommendation("pearson", "spearman", "r", [indep, N, "a_linearity", "a_outliers"])
    if d.analysis == "regression":
        if d.dv_type == "nominal":
            return Recommendation("logreg", None, "or", [indep, "a_multicollinearity", "a_epv"],
                                  [("VD dicotómica (0/1).", "Dichotomous DV (0/1).")])
        return Recommendation("linreg", None, "f2",
                              [indep, "a_linearity", N, "a_homoscedasticity", "a_multicollinearity", "a_independence"],
                              [("Si hay heterocedasticidad, se informan errores típicos robustos (HC3).",
                                "If heteroscedasticity is present, robust (HC3) standard errors are reported.")])
    # compare groups with quantitative/ordinal DV
    ordinal = d.dv_type == "ordinal"
    if d.design == "mixed":
        return Recommendation("anova_mixed", None, "f", [N, H, S, "a_outliers"],
                              [("Un factor entre sujetos y uno intra sujetos; si se viola la esfericidad se aplica Greenhouse-Geisser.",
                                "One between- and one within-subjects factor; Greenhouse-Geisser is applied if sphericity is violated.")])
    if d.design == "within":
        if d.n_factors == 2:
            return Recommendation("anova_rm", None, "f", [N, S, "a_outliers"],
                                  [("La esfericidad se evalúa para cada efecto (los dos factores y su interacción); "
                                    "si se viola, se aplica la corrección de Greenhouse-Geisser a ese efecto.",
                                    "Sphericity is assessed for each effect (both factors and their interaction); "
                                    "if violated, the Greenhouse-Geisser correction is applied to that effect.")])
        if d.n_groups == 2:
            if ordinal:
                return Recommendation("wilcoxon", None, "d", ["a_outliers"])
            return Recommendation("ttest_paired", "wilcoxon", "d", [N + "_diff", "a_outliers"],
                                  [("La normalidad se evalúa sobre las diferencias pareadas.",
                                    "Normality is assessed on the paired differences.")])
        if ordinal:
            return Recommendation("friedman", None, "f", ["a_outliers"])
        return Recommendation("anova_rm", "friedman", "f", [N, S, "a_outliers"],
                              [("Si se viola la esfericidad se aplica la corrección de Greenhouse-Geisser.",
                                "If sphericity is violated, the Greenhouse-Geisser correction is applied.")])
    # between
    if d.covariates:
        return Recommendation("ancova", None, "f", [indep, N, H, "a_slopes"])
    if d.n_factors == 2:
        return Recommendation("anova_factorial", None, "f", [indep, N, H, "a_outliers"])
    if d.n_groups == 2:
        if ordinal:
            return Recommendation("mannwhitney", None, "d", [indep])
        return Recommendation("ttest_ind", "mannwhitney", "d", [indep, N, "a_outliers"],
                              [("Se usa la t de Welch por defecto (no asume varianzas iguales; Delacre et al., 2017).",
                                "Welch's t is used by default (does not assume equal variances; Delacre et al., 2017).")])
    if ordinal:
        return Recommendation("kruskal", None, "f", [indep])
    return Recommendation("anova_oneway", "kruskal", "f", [indep, N, H, "a_outliers"],
                          [("Si las varianzas no son homogéneas se usa el ANOVA de Welch con post hoc de Games-Howell; "
                            "si son homogéneas, post hoc de Tukey.",
                            "If variances are heterogeneous, Welch's ANOVA with Games-Howell post hoc is used; "
                            "otherwise Tukey post hoc.")])


def assumption_label(key, lang="es"):
    if key.endswith("_diff"):
        base = t(key[:-5], lang)
        return base + (" de las diferencias" if lang == "es" else " of differences")
    return t(key, lang)


def hypothesis_text(d: Design, lang="es") -> str:
    """Directional H1 in words, e.g. 'terapia < control' or 'relación positiva'. Empty for two-sided."""
    if not d.one_sided:
        return ""
    op = ">" if d.alternative == "greater" else "<"
    if d.analysis == "correlation":
        if lang == "es":
            return f"relación {'positiva' if d.alternative == 'greater' else 'negativa'} entre {d.iv_name} y {d.dv_name}"
        return f"{'positive' if d.alternative == 'greater' else 'negative'} relationship between {d.iv_name} and {d.dv_name}"
    a, b = d.levels[0], d.levels[1]
    return f"{d.dv_name}: {a} {op} {b}"
