"""Study design description and test recommendation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .i18n import t

ALTERNATIVES = ("two-sided", "greater", "less")
# tests for which a one-sided (directional) alternative is valid
DIRECTIONAL_TESTS = {"ttest_ind", "mannwhitney", "ttest_paired", "wilcoxon", "pearson", "spearman"}
ANALYSES = ("compare", "correlation", "regression", "association", "mediation", "moderation", "reliability")
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
    "path": {"small": 0.14, "medium": 0.39, "large": 0.59},  # Cohen-style standardized paths (Fritz & MacKinnon, 2007)
    # reliability: planning value of Cronbach's α (tested against `alpha_null`, .70 by default; Bonett, 2002)
    "rel": {"small": 0.80, "medium": 0.85, "large": 0.90},
}


@dataclass
class Design:
    title: str = ""
    question: str = ""
    analysis: str = "compare"          # compare / correlation / regression / association / mediation / moderation / reliability
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
    mediator: str = "mediador"       # simple mediation: X (iv_name) → M → Y (dv_name)
    moderator: str = "moderador"     # moderation: effect of X (iv_name) on Y (dv_name) depends on W
    moderator_type: str = "continuous"   # continuous / nominal (dichotomous W)
    moderator_levels: list = field(default_factory=list)   # 2 categories of a dichotomous W (0 / 1 order)
    center: bool = True              # moderation: mean-centre continuous X and W before forming X × W
    items: list = field(default_factory=list)          # reliability: item columns of the scale (dv_name = scale name)
    reverse_items: list = field(default_factory=list)  # reliability: reverse-keyed items, recoded as min + max − x
    scale_min: float = 1.0           # reliability: lowest response option (used to reverse items)
    scale_max: float = 5.0           # reliability: highest response option
    alpha_null: float = 0.70         # reliability: minimum acceptable α tested in H₀ (Feldt F test; Bonett N)
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
        if self.dv_type not in LEVELS or self.iv_type not in LEVELS or self.moderator_type not in LEVELS:
            raise ValueError(f"variable types must be one of {LEVELS}")
        self.moderator_levels = [str(x).strip() for x in self.moderator_levels]
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
        if self.analysis == "regression" and self.dv_type == "ordinal":
            # ordered categories of the DV, from lowest to highest (default: a 5-point Likert item)
            if not self.dv_levels or [str(x).strip().lower() for x in self.dv_levels] in (["si", "no"], ["sí", "no"]):
                self.dv_levels = ["1", "2", "3", "4", "5"]
        if not self.dv_levels:
            self.dv_levels = ["si", "no"]
        self.dv_levels = [str(x).strip() for x in self.dv_levels]
        if not self.predictors:
            self.predictors = ["x1"]
        self.items = [str(x).strip() for x in self.items if str(x).strip()]
        self.reverse_items = [str(x).strip() for x in self.reverse_items if str(x).strip()]
        if self.analysis == "reliability":
            if not self.items:
                self.items = [f"item{i + 1}" for i in range(5)]
            if len(self.items) < 3:
                raise ValueError("El análisis de fiabilidad necesita al menos 3 ítems / reliability analysis needs at least 3 items")
            if len(set(self.items)) != len(self.items):
                raise ValueError("Hay ítems repetidos / duplicated item names")
            bad = [x for x in self.reverse_items if x not in self.items]
            if bad:
                raise ValueError(f"Ítems inversos que no están en la escala / reverse-keyed items not in the scale: {bad}")
            self.scale_min, self.scale_max = float(self.scale_min), float(self.scale_max)
            if not self.scale_min < self.scale_max:
                raise ValueError("El mínimo de la escala de respuesta debe ser menor que el máximo / scale minimum must be below the maximum")
            self.alpha_null = float(self.alpha_null)
            if not 0 <= self.alpha_null < 1:
                raise ValueError("El α mínimo aceptable debe estar en [0, 1) / the minimum acceptable α must be in [0, 1)")
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
    if d.analysis == "mediation":
        notes = [("Mediación simple (X → M → Y): sendas a, b, c y c′ por MCO; efecto indirecto a×b con IC 95% bootstrap percentil "
                  "(5000 remuestreos). La VD y el mediador deben ser cuantitativos; la VI puede ser cuantitativa o dicotómica.",
                  "Simple mediation (X → M → Y): paths a, b, c and c′ by OLS; indirect effect a×b with percentile bootstrap 95% CI "
                  "(5000 resamples). The DV and mediator must be continuous; the IV may be continuous or dichotomous.")]
        if d.dv_type != "continuous":
            notes.append(("La VD debe ser cuantitativa para esta mediación lineal; con VD ordinal o dicotómica usa otros modelos (p. ej., PROCESS).",
                          "The DV must be continuous for this linear mediation; with an ordinal or dichotomous DV use other models (e.g., PROCESS)."))
        return Recommendation("mediation", None, "path", [indep, "a_linearity", N, "a_homoscedasticity"], notes)
    if d.analysis == "moderation":
        notes = [("Moderación (modelo 1 de Hayes): regresión MCO de Y sobre X, W y X × W"
                  + (" con los predictores cuantitativos centrados en la media" if d.center else "")
                  + "; ΔR² de la interacción (prueba F), pendientes simples de X en la media de W y ±1 DT (o en cada categoría "
                  "de un W dicotómico) y gráfico de la interacción. La VD debe ser cuantitativa; X y W pueden ser cuantitativas o dicotómicas.",
                  "Moderation (Hayes model 1): OLS regression of Y on X, W and X × W"
                  + (" with continuous predictors mean-centred" if d.center else "")
                  + "; ΔR² for the interaction (F test), simple slopes of X at the mean of W and ±1 SD (or at each category "
                  "of a dichotomous W) and an interaction plot. The DV must be continuous; X and W may be continuous or dichotomous."),
                 ("Los efectos de interacción suelen ser pequeños (Aguinis et al., 2005): planifica N con un f² pequeño salvo que tengas un estudio previo.",
                  "Interaction effects are usually small (Aguinis et al., 2005): plan N with a small f² unless you have prior evidence.")]
        if d.dv_type != "continuous":
            notes.append(("La VD debe ser cuantitativa para esta moderación lineal; con VD dicotómica u ordinal usa una regresión logística con el término X × W.",
                          "The DV must be continuous for this linear moderation; with a dichotomous or ordinal DV use logistic regression with the X × W term."))
        return Recommendation("moderation", None, "f2", [indep, "a_linearity", N, "a_homoscedasticity", "a_multicollinearity"], notes)
    if d.analysis == "reliability":
        notes = [("Fiabilidad (consistencia interna) de una escala de varios ítems: α de Cronbach con IC 95% exacto de Feldt et al. (1987) "
                  f"y prueba F de H₀: α = {_b(d.alpha_null)}; ω total de McDonald (1999) a partir de un modelo de un factor (máxima "
                  "verosimilitud) con IC 95% bootstrap; correlación ítem-resto corregida y α si se elimina cada ítem."
                  + (f" Los ítems inversos ({', '.join(d.reverse_items)}) se recodifican como {_g(d.scale_min)} + {_g(d.scale_max)} − x."
                     if d.reverse_items else ""),
                  "Internal-consistency reliability of a multi-item scale: Cronbach's α with Feldt et al. (1987) exact 95% CI "
                  f"and F test of H₀: α = {_b(d.alpha_null)}; McDonald's (1999) ω total from a one-factor model (maximum likelihood) "
                  "with bootstrap 95% CI; corrected item-rest correlation and α if each item is deleted."
                  + (f" Reverse-keyed items ({', '.join(d.reverse_items)}) are recoded as {_g(d.scale_min)} + {_g(d.scale_max)} − x."
                     if d.reverse_items else "")),
                 ("α asume que todos los ítems miden el constructo con la misma fuerza (tau-equivalencia); si no es así lo infraestima, "
                  "por eso se informa también ω (McNeish, 2018).",
                  "α assumes all items measure the construct equally strongly (tau-equivalence); otherwise it underestimates "
                  "reliability, so ω is reported too (McNeish, 2018).")]
        return Recommendation("reliability", None, "rel", ["a_independent_obs", "a_unidimensionality"], notes)
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
        if d.dv_type == "ordinal":
            return Recommendation("ordreg", None, "or", [indep, "a_multicollinearity", "a_proportional_odds"],
                                  [("VD ordinal (p. ej., un ítem Likert): modelo de odds proporcionales (logit acumulado); "
                                    "el supuesto de odds proporcionales se evalúa con la prueba de Brant (1990). "
                                    "Las escalas formadas por la suma de varios ítems suelen analizarse como cuantitativas.",
                                    "Ordinal DV (e.g., a single Likert item): proportional-odds (cumulative logit) model; "
                                    "the proportional-odds assumption is checked with the Brant (1990) test. "
                                    "Scales formed by summing several items are usually analysed as quantitative.")])
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


def _g(x):
    """Compact number for scale bounds (1.0 -> '1', 0.5 -> '0.5')."""
    return f"{float(x):g}"


def _b(x):
    """Bounded-coefficient APA style: 0.70 -> '.70'."""
    return f"{float(x):.2f}".replace("0.", ".", 1) if abs(float(x)) < 1 else f"{float(x):.2f}"


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
