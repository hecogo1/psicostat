"""PsicoStat web app (Streamlit). Run with:  psicostat app   (or: streamlit run streamlit_app.py)

Desktop-first research-tool layout: step navigation in the sidebar, one step per screen,
multi-column panels with a plain-language "concept" panel on the right.
"""
import base64
import html
import io
import os
import tempfile

import pandas as pd
import streamlit as st

from psicostat import __version__
from psicostat.analysis import DataError, analyze
from psicostat.design import Design, assumption_label, recommend
from psicostat.effects import fmt_num, fmt_p, magnitude
from psicostat.examples import EXAMPLES, example
from psicostat.i18n import t
from psicostat.power import explain, sample_size
from psicostat.report import build_report, stat_str, to_decimal_comma, to_docx, to_markdown
from psicostat.template import make_template

ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def _svg(name):
    with open(os.path.join(ASSETS, name), "rb") as fh:
        return "data:image/svg+xml;base64," + base64.b64encode(fh.read()).decode()


st.set_page_config(page_title="PsicoStat · Metodología y estadística", page_icon=os.path.join(ASSETS, "logo.svg"),
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- styles
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
:root { --teal:#0F766E; --teal-50:#F0FDFA; --teal-100:#CCFBF1; --indigo:#3730A3; --amber:#D97706;
        --ink:#1E293B; --muted:#64748B; --line:#E7E3DA; --card:#FFFFFF; --bg:#FAF8F4; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3, .ps-serif { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -0.01em; color: var(--ink); }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1480px; }
section[data-testid="stSidebar"] { background: #F3F0E9; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div { padding-top: 1rem; }
/* hero */
.ps-hero { display:flex; align-items:center; gap:1.2rem; padding:1rem 1.6rem; border-radius:18px; color:#fff;
           background: linear-gradient(120deg, #0F766E 0%, #115E59 45%, #312E81 100%); box-shadow: 0 10px 30px -18px rgba(15,23,42,.55); }
.ps-hero img.logo { width:58px; height:58px; flex:none; }
.ps-hero h1 { color:#fff !important; margin:0; font-size:1.8rem; line-height:1.1; padding:0; }
.ps-hero p { margin:.25rem 0 0; color:#CCFBF1; font-size:1.02rem; }
.ps-hero .art { margin-left:auto; height:92px; flex:none; }
.ps-hero .badges span { display:inline-block; margin:.55rem .4rem 0 0; padding:.15rem .6rem; border-radius:999px; font-size:.78rem;
           background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); }
/* stepper */
.ps-steps { display:flex; gap:.5rem; margin:1.1rem 0 .4rem; }
.ps-step { flex:1; display:flex; align-items:center; gap:.55rem; padding:.55rem .7rem; border-radius:12px; background:#fff;
           border:1px solid var(--line); color:var(--muted); font-size:.86rem; font-weight:500; min-width:0; }
.ps-step .n { flex:none; width:26px; height:26px; border-radius:50%; display:grid; place-items:center; background:#EEF2F1; font-weight:700; font-size:.8rem; }
.ps-step .lbl { line-height:1.15; font-size:.8rem; }
.ps-step.done { color:var(--teal); } .ps-step.done .n { background:var(--teal-100); color:var(--teal); }
.ps-step.cur { color:#fff; background:var(--teal); border-color:var(--teal); box-shadow:0 6px 18px -10px rgba(15,118,110,.9); }
.ps-step.cur .n { background:#fff; color:var(--teal); }
.ps-bar { height:6px; background:#E9E5DC; border-radius:99px; overflow:hidden; margin-bottom:1.2rem; }
.ps-bar > div { height:100%; background:linear-gradient(90deg,#14B8A6,#0F766E 60%,#3730A3); border-radius:99px; }
/* cards */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > div .ps-card-mark),
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px !important; }
div[data-testid="stVerticalBlockBorderWrapper"][data-test-border="true"], .st-key-main_card, .st-key-concept_card {
    background: var(--card); border:1px solid var(--line) !important; border-radius:16px; box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -18px rgba(15,23,42,.25); }
.st-key-main_card, .st-key-concept_card { padding: 1.3rem 1.5rem 1.1rem; }
.st-key-concept_card { background: linear-gradient(180deg,#F0FDFA, #FFFFFF 60%); border-color:#CDEDE7 !important; }
.ps-stephead { display:flex; align-items:center; gap:.8rem; margin-bottom:.35rem; }
.ps-stephead .ic { width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:1.35rem; background:var(--teal-50); border:1px solid #CDEDE7; }
.ps-stephead h2 { margin:0; padding:0; font-size:1.55rem; }
.ps-stephead .k { color:var(--muted); font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.ps-lead { color:#475569; margin:.2rem 0 1rem; font-size:.98rem; }
.ps-concept h4 { font-family:'Source Serif 4',Georgia,serif; margin:.1rem 0 .4rem; color:var(--teal); font-size:1.08rem; }
.ps-concept p, .ps-concept li { color:#334155; font-size:.92rem; line-height:1.5; }
.ps-concept .tip { background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:.55rem .75rem; font-size:.88rem; color:#78350F; margin-top:.6rem; }
/* metric cards */
.ps-cards { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:.8rem; margin:.4rem 0 1rem; }
.ps-mc { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.9rem 1rem; position:relative; overflow:hidden; }
.ps-mc:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--teal); }
.ps-mc.indigo:before { background:var(--indigo);} .ps-mc.amber:before { background:var(--amber);} .ps-mc.rose:before { background:#BE123C;}
.ps-mc .l { color:var(--muted); font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.ps-mc .v { font-family:'Source Serif 4',Georgia,serif; font-size:1.55rem; font-weight:700; color:var(--ink); line-height:1.25; margin-top:.15rem; }
.ps-mc .s { color:#475569; font-size:.84rem; margin-top:.15rem; }
.ps-chip { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.76rem; font-weight:600; }
.ps-chip.ok { background:#DCFCE7; color:#166534; } .ps-chip.no { background:#FEE2E2; color:#991B1B; } .ps-chip.neu { background:#E0E7FF; color:#3730A3; }
.ps-apa { font-family:'Source Serif 4',Georgia,serif; font-weight:400; font-size:1.04rem; line-height:1.65; color:#1E293B; background:#FFFDF8;
          border:1px solid var(--line); border-left:4px solid var(--indigo); border-radius:10px; padding:1rem 1.2rem; }
.ps-apa p { margin:0 0 .7rem; }
.ps-assump { width:100%; border-collapse:collapse; font-size:.9rem; }
.ps-assump td { padding:.45rem .4rem; border-bottom:1px solid #F1EEE7; vertical-align:top; }
/* buttons */
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; padding:.5rem 1.1rem; transition: all .15s ease; }
.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
.stButton > button[kind="primary"]:hover { background:#115E59; border-color:#115E59; }
.stDownloadButton > button { background:#fff; color:var(--indigo); border:1.5px solid #C7D2FE; }
.stDownloadButton > button:hover { background:#EEF2FF; border-color:var(--indigo); color:var(--indigo); }
section[data-testid="stSidebar"] .stButton > button { width:100%; justify-content:flex-start; text-align:left; border-radius:10px; font-weight:500;
           background:transparent; border:1px solid transparent; color:#334155; padding:.45rem .7rem; }
section[data-testid="stSidebar"] .stButton > button > div { justify-content:flex-start; width:100%; }
section[data-testid="stSidebar"] .stButton > button p { text-align:left; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] .stButton > button:hover { background:#fff; border-color:var(--line); }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:#fff; color:var(--teal); border-color:#99D8CE; font-weight:700;
           box-shadow: inset 3px 0 0 var(--teal); }
.ps-side-brand { display:flex; align-items:center; gap:.6rem; margin-bottom:.6rem; }
.ps-side-brand img { width:34px; } .ps-side-brand b { font-family:'Source Serif 4',Georgia,serif; font-size:1.25rem; color:var(--ink); }
.ps-side-brand small { display:block; color:var(--muted); font-size:.72rem; margin-top:-.1rem; }
.ps-side-h { color:var(--muted); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; margin:.9rem 0 .3rem; }
.ps-foot { color:var(--muted); font-size:.78rem; margin-top:1.4rem; }
div[data-testid="stTabs"] button p { font-weight:600; }
@media (max-width: 1100px) { .ps-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 900px) { .ps-hero .art { display:none; } .ps-steps .lbl { display:none; } .ps-hero h1 { font-size:1.5rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- state
FIELDS = ["title", "question", "analysis", "design", "dv_type", "iv_type", "dv_name", "iv_name", "levels", "n_factors",
          "factor_b_name", "levels_b", "dv_levels", "covariates", "predictors", "rho", "event_rate",
          "effect_label", "effect_custom", "alpha", "power", "dropout", "decimal_comma"]
LISTS = {"levels", "levels_b", "dv_levels", "covariates", "predictors"}


def _load_design(d: Design):
    for f in FIELDS:
        if f == "effect_custom":
            v = "" if d.effect_size is None else str(d.effect_size)
        else:
            v = getattr(d, f)
        if f in LISTS:
            v = ", ".join(v)
        st.session_state["k_" + f] = v


ss = st.session_state
for k in list(ss.keys()):          # keep widget values alive across steps (widgets not rendered on other screens)
    if isinstance(k, str) and k.startswith("k_"):
        ss[k] = ss[k]
if "step" not in ss:
    ss.step = 0
    ss.data = None
    ss.example = "—"
    _load_design(Design(levels=["control", "tratamiento"], iv_name="grupo", dv_name="puntuacion"))

lang = ss.get("k_lang", "es")


def L(es, en):
    return es if lang == "es" else en


def cap(s):
    return s[:1].upper() + s[1:]


def lst(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def current_design():
    g = {f: ss.get("k_" + f) for f in FIELDS}
    kw = {"title": g["title"], "question": g["question"], "analysis": g["analysis"]}
    a = g["analysis"]
    if a == "compare":
        kw.update(design=g["design"], dv_type=g["dv_type"], dv_name=g["dv_name"], iv_name=g["iv_name"], levels=lst(g["levels"]))
        two = g["design"] == "mixed" or int(g["n_factors"]) == 2
        if two and g["dv_type"] != "nominal":
            kw.update(n_factors=2, factor_b_name=g["factor_b_name"], levels_b=lst(g["levels_b"]))
        if g["dv_type"] == "nominal" and g["design"] == "between":
            kw["dv_levels"] = lst(g["dv_levels"])
        if g["design"] == "between" and not two and g["dv_type"] == "continuous":
            kw["covariates"] = lst(g["covariates"])
        if g["design"] in ("within", "mixed"):
            kw["rho"] = float(g["rho"])
    elif a == "association":
        kw.update(dv_type="nominal", iv_type="nominal", iv_name=g["iv_name"], levels=lst(g["levels"]),
                  dv_name=g["dv_name"], dv_levels=lst(g["dv_levels"]))
    else:
        kw.update(dv_type=g["dv_type"], iv_type=g["iv_type"], dv_name=g["dv_name"])
        if a == "correlation":
            kw["iv_name"] = g["iv_name"]
        else:
            kw["predictors"] = lst(g["predictors"])
            if g["dv_type"] == "nominal":
                kw["event_rate"] = float(g["event_rate"])
    kw["n_groups"] = max(len(kw.get("levels", [])), 2)
    kw["effect_label"] = g["effect_label"]
    c = str(g["effect_custom"] or "").replace(",", ".").strip()
    kw["effect_size"] = float(c) if c else None
    kw.update(alpha=float(g["alpha"]), power=float(g["power"]), dropout=float(g["dropout"]), decimal_comma=bool(g["decimal_comma"]))
    return Design(**kw)


def on_example():
    name = ss.k_example
    if name == "—":
        ss.data = None
        return
    d, df = example(name)
    _load_design(d)
    ss.data = df
    ss.do_run = False


STEPS = [
    ("🧭", L("Pregunta y diseño", "Question & design")),
    ("🧩", L("Variables", "Variables")),
    ("🎯", L("Efecto, α y potencia", "Effect, α & power")),
    ("🧪", L("Prueba y tamaño muestral", "Test & sample size")),
    ("📋", L("Plantilla de datos", "Data template")),
    ("📊", L("Análisis e informe APA", "Analysis & APA report")),
]


def go(i):
    ss.step = max(0, min(len(STEPS) - 1, i))


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"<div class='ps-side-brand'><img src='{_svg('logo.svg')}' alt='PsicoStat logo'/>"
                f"<div><b>PsicoStat</b><small>v{__version__} · open source</small></div></div>", unsafe_allow_html=True)
    st.segmented_control("Idioma / Language", ["es", "en"], key="k_lang", default="es",
                         format_func=lambda x: {"es": "🇪🇸 Español", "en": "🇬🇧 English"}[x], label_visibility="collapsed")
    if lang == "es":
        st.toggle("Coma decimal en el informe (3,25)", key="k_decimal_comma",
                  help="Algunas normas APA en español usan coma decimal. Afecta al texto, las tablas y el DOCX en español; el informe en inglés siempre usa punto.")
    st.markdown(f"<div class='ps-side-h'>{L('Pasos', 'Steps')}</div>", unsafe_allow_html=True)
    for i, (ic, name) in enumerate(STEPS):
        mark = "✓" if i < ss.step else ic
        st.button(f"{mark}  {i + 1}. {name}", key=f"nav_{i}", on_click=go, args=(i,),
                  type="primary" if i == ss.step else "secondary", use_container_width=True)
    st.markdown(f"<div class='ps-side-h'>{L('Datos de ejemplo', 'Example data')}</div>", unsafe_allow_html=True)
    st.selectbox(L("Cargar un estudio de ejemplo", "Load an example study"), ["—"] + EXAMPLES, key="k_example", on_change=on_example,
                 help=L("Rellena el asistente con un diseño y datos simulados para explorar la herramienta.",
                        "Fills the wizard with a simulated design and data to explore the tool."), label_visibility="collapsed")
    st.markdown(f"<div class='ps-foot'>{t('disclaimer', lang)}<br><br>"
                f"<a href='https://github.com/hecogo1/psicostat' target='_blank'>GitHub</a> · MIT</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- header + stepper
st.markdown(
    f"""<div class='ps-hero'><img class='logo' src='{_svg('logo.svg')}' alt=''/>
    <div><h1>PsicoStat</h1><p>{L('Del diseño del estudio al texto APA: metodología y estadística para psicología, paso a paso.',
                                  'From study design to APA text: methods and statistics for psychology, step by step.')}</p>
    <div class='badges'><span>{L('Tamaño muestral a priori', 'A priori sample size')}</span><span>{L('Supuestos y alternativas', 'Assumptions & fallbacks')}</span>
    <span>{L('Tamaños del efecto con IC 95%', 'Effect sizes with 95% CI')}</span><span>APA 7 · es / en</span></div></div>
    <img class='art' src='{_svg('hero.svg')}' alt=''/></div>""", unsafe_allow_html=True)

chips = "".join(
    f"<div class='ps-step {'cur' if i == ss.step else 'done' if i < ss.step else ''}'><span class='n'>{'✓' if i < ss.step else i + 1}</span>"
    f"<span class='lbl'>{html.escape(n)}</span></div>" for i, (_, n) in enumerate(STEPS))
st.markdown(f"<div class='ps-steps'>{chips}</div><div class='ps-bar'><div style='width:{(ss.step + 1) / len(STEPS) * 100:.0f}%'></div></div>",
            unsafe_allow_html=True)


def head(i, lead):
    ic, name = STEPS[i]
    st.markdown(f"<div class='ps-stephead'><div class='ic'>{ic}</div><div><div class='k'>{L('Paso', 'Step')} {i + 1} / {len(STEPS)}</div>"
                f"<h2>{html.escape(name)}</h2></div></div><p class='ps-lead'>{lead}</p>", unsafe_allow_html=True)


def concept(title, body_html, tip=None):
    with st.container(key="concept_card"):
        st.markdown(f"<div class='ps-concept'><h4>💡 {title}</h4>{body_html}" + (f"<div class='tip'>{tip}</div>" if tip else "") + "</div>",
                    unsafe_allow_html=True)


def cards(items):
    st.markdown("<div class='ps-cards'>" + "".join(
        f"<div class='ps-mc {c}'><div class='l'>{html.escape(lab)}</div><div class='v'>{html.escape(str(v))}</div>"
        f"<div class='s'>{s}</div></div>" for lab, v, s, c in items) + "</div>", unsafe_allow_html=True)


def nav_buttons(can_next=True):
    c1, _, c3 = st.columns([1, 3, 1])
    if ss.step > 0:
        c1.button("← " + L("Atrás", "Back"), key="back", on_click=go, args=(ss.step - 1,), use_container_width=True)
    if ss.step < len(STEPS) - 1:
        c3.button(L("Siguiente", "Next") + " →", key="next", type="primary", on_click=go, args=(ss.step + 1,),
                  disabled=not can_next, use_container_width=True)


TYPES = ["continuous", "ordinal", "nominal"]
step = ss.step
main, side = st.columns([3, 1] if step == 5 else [2.15, 1], gap="large")

try:
    design = current_design()
    design_error = None
except (ValueError, TypeError) as exc:
    design, design_error = None, str(exc)

# --------------------------------------------------------------------------- steps
with main:
    with st.container(key="main_card"):
        if step == 0:
            head(0, L("Cuéntanos qué quieres investigar. Con esto elegimos el tipo de análisis.",
                      "Tell us what you want to study. This determines the type of analysis."))
            c1, c2 = st.columns(2)
            c1.text_input(t("q_title", lang), key="k_title", placeholder=L("p. ej., Mindfulness y estrés académico", "e.g., Mindfulness and academic stress"))
            c2.text_input(t("q_question", lang), key="k_question", placeholder=L("¿Reduce X la variable Y?", "Does X reduce Y?"))
            st.radio(t("q_analysis", lang), ["compare", "correlation", "regression", "association"], key="k_analysis",
                     format_func=lambda a: t("an_" + a, lang),
                     help=L("Comparar: diferencias entre grupos o momentos. Correlación: si dos variables varían juntas. "
                            "Regresión: predecir una variable a partir de otras. Asociación: relación entre dos variables de categorías.",
                            "Compare: differences between groups or time points. Correlation: whether two variables vary together. "
                            "Regression: predict one variable from others. Association: relation between two categorical variables."))
            if ss.k_analysis == "compare":
                st.radio(t("q_design", lang), ["between", "within", "mixed"], key="k_design", format_func=lambda x: t("d_" + x, lang),
                         help=L("Entre sujetos: cada persona está en un solo grupo. Intra sujetos: las mismas personas se miden varias veces. "
                                "Mixto: grupos distintos medidos varias veces (p. ej., tratamiento vs control en pre y post).",
                                "Between: each person is in one group only. Within: the same people are measured several times. "
                                "Mixed: different groups measured several times (e.g., treatment vs control at pre and post)."))
        elif step == 1:
            head(1, L("Define las variables. Usa nombres cortos sin espacios: serán las columnas de tu CSV.",
                      "Define your variables. Use short names without spaces: they become your CSV columns."))
            a = ss.k_analysis
            tip_type = L("Nominal = categorías sin orden (grupo, sexo). Ordinal = orden pero sin distancias iguales (un ítem Likert). "
                         "Intervalo/razón = números con distancias iguales (puntuación total de un test, tiempo de reacción).",
                         "Nominal = unordered categories (group, sex). Ordinal = ordered but unequal distances (one Likert item). "
                         "Interval/ratio = numbers with equal distances (total test score, reaction time).")
            c1, c2, c3 = st.columns(3)
            if a == "compare":
                c1.text_input(t("q_iv_name", lang), key="k_iv_name", help=L("La variable que define los grupos o momentos (VI).", "The variable that defines groups or time points (IV)."))
                c2.text_input(t("q_levels", lang), key="k_levels", help=L("Nombres de los grupos o condiciones, separados por comas.", "Group or condition names, comma separated."))
                c3.selectbox(t("q_dv_type", lang), TYPES, key="k_dv_type", format_func=lambda x: t("t_" + x, lang), help=tip_type)
                c1.text_input(t("q_dv_name", lang), key="k_dv_name", help=L("Lo que mides como resultado (VD).", "What you measure as outcome (DV)."))
                if ss.k_design != "mixed" and ss.k_dv_type != "nominal":
                    c2.selectbox(t("q_n_factors", lang), [1, 2], key="k_n_factors",
                                 help=L("2 = diseño factorial (p. ej., terapia × sexo); permite estudiar la interacción.",
                                        "2 = factorial design (e.g., therapy × sex); lets you test the interaction."))
                if ss.k_design == "mixed" or (int(ss.k_n_factors) == 2 and ss.k_dv_type != "nominal"):
                    c3.text_input(t("q_factor_b_name", lang), key="k_factor_b_name")
                    c3.text_input(t("q_levels_b", lang), key="k_levels_b")
                if ss.k_dv_type == "nominal" and ss.k_design == "between":
                    c3.text_input(t("q_dv_levels", lang), key="k_dv_levels")
                if ss.k_design == "between" and int(ss.k_n_factors) == 1 and ss.k_dv_type == "continuous":
                    c2.text_input(t("q_covariates", lang), key="k_covariates",
                                  help=L("Variables que quieres controlar (p. ej., la puntuación pre). Convierte el análisis en un ANCOVA.",
                                         "Variables you want to control for (e.g., the pre score). Turns the analysis into an ANCOVA."))
                if ss.k_design in ("within", "mixed"):
                    st.slider(t("q_rho", lang), 0.0, 0.9, step=0.05, key="k_rho",
                              help=L("Cuánto se parecen las puntuaciones de una misma persona entre medidas. 0.5 es un valor prudente si no lo sabes.",
                                     "How similar a person's scores are across measurements. 0.5 is a sensible default if unknown."))
            elif a == "association":
                c1.text_input(t("q_iv_name", lang), key="k_iv_name")
                c1.text_input(t("q_levels", lang), key="k_levels")
                c2.text_input(t("q_dv_name", lang), key="k_dv_name")
                c2.text_input(t("q_dv_levels", lang), key="k_dv_levels")
            else:
                c1.selectbox(t("q_dv_type", lang), TYPES, key="k_dv_type", format_func=lambda x: t("t_" + x, lang), help=tip_type)
                c2.selectbox(t("q_iv_type", lang), TYPES, key="k_iv_type", format_func=lambda x: t("t_" + x, lang), help=tip_type)
                c3.text_input(t("q_dv_name", lang), key="k_dv_name")
                if a == "correlation":
                    c1.text_input(t("q_iv_name", lang), key="k_iv_name")
                else:
                    c1.text_input(t("q_predictors", lang), key="k_predictors")
                    if ss.k_dv_type == "nominal":
                        st.slider(t("q_event_rate", lang), 0.05, 0.95, step=0.05, key="k_event_rate")
        elif step == 2:
            head(2, L("Decide qué efecto quieres poder detectar y con qué seguridad.", "Decide which effect you want to be able to detect, and how reliably."))
            c1, c2 = st.columns(2)
            c1.select_slider(t("q_effect_label", lang), ["small", "medium", "large"], key="k_effect_label", format_func=lambda x: t(x, lang),
                             help=t("effect_help", lang))
            c2.text_input(t("q_effect_custom", lang), key="k_effect_custom", placeholder="0.40",
                          help=L("Si conoces el efecto de un estudio previo o metaanálisis, escríbelo aquí (en la métrica indicada a la derecha).",
                                 "If you know the effect from a prior study or meta-analysis, enter it here (in the metric shown on the right)."))
            c1, c2, c3 = st.columns(3)
            c1.number_input(t("q_alpha", lang), 0.001, 0.2, step=0.005, format="%.3f", key="k_alpha",
                            help=L("Riesgo aceptado de un falso positivo. Por convención, .05.", "Accepted risk of a false positive. Conventionally .05."))
            c2.number_input(t("q_power", lang), 0.5, 0.99, step=0.05, key="k_power",
                            help=L("Probabilidad de detectar el efecto si existe. Se recomienda ≥ .80 (mejor .90).",
                                   "Probability of detecting the effect if it exists. ≥ .80 recommended (.90 is better)."))
            c3.number_input(t("q_dropout", lang), 0.0, 0.5, step=0.05, key="k_dropout",
                            help=L("Proporción de participantes que esperas perder (abandono, datos inválidos).",
                                   "Share of participants you expect to lose (dropout, invalid data)."))
        elif step == 3:
            head(3, L("Según tu diseño, esta es la prueba adecuada y la muestra que necesitas.", "Based on your design, this is the right test and the sample you need."))
            if design is None:
                st.error(design_error)
            else:
                rec, ssz = recommend(design), sample_size(design)
                fam = {"d": "d", "f": "f", "r": "r", "w": "w", "f2": "f²", "or": "OR"}[ssz["family"]]
                cards([
                    (L("Prueba recomendada", "Recommended test"), cap(rec.label(lang).split(" (")[0]), L("Alternativa: ", "Fallback: ") + rec.fallback_label(lang), ""),
                    (L("Tamaño muestral", "Sample size"), f"N = {ssz['n_total']}",
                     (f"{ssz['n_per_group']} {t('per_group', lang)}" if ssz["n_per_group"] and ssz["groups"] > 1 else L("participantes", "participants")), "indigo"),
                    (L("Efecto a detectar", "Effect to detect"), f"{fam} = {ssz['effect']:.2f}", t(ssz["effect_label"], lang) if ssz["effect_label"] != "custom" else L("valor propio", "custom value"), "amber"),
                    ("α · " + L("potencia", "power"), f"{fmt_num(design.alpha, True)} · {fmt_num(design.power, True)}", L("bilateral", "two-sided"), "rose"),
                ])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{t('assumptions', lang)}**")
                    st.markdown("\n".join(f"- {assumption_label(a, lang)}" for a in rec.assumptions))
                with c2:
                    st.markdown(f"**{L('Cómo se ha calculado', 'How it was computed')}**")
                    st.write(explain(ssz, lang))
                    for n in rec.notes_text(lang):
                        st.caption(n)
        elif step == 4:
            head(4, L("Descarga la plantilla, rellénala con tus datos (una fila por participante) y vuelve para analizarla.",
                      "Download the template, fill it with your data (one row per participant) and come back to analyse it."))
            if design is None:
                st.error(design_error)
            else:
                ssz = sample_size(design)
                tpl, dic = make_template(design, ssz["n_total"])
                st.markdown(f"**{L('Diccionario de datos', 'Data dictionary')}**")
                st.dataframe(dic, hide_index=True, use_container_width=True)
                st.markdown(f"**{L('Vista previa de la plantilla', 'Template preview')}** · {len(tpl)} {L('filas', 'rows')}")
                st.dataframe(tpl.head(6), hide_index=True, use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.download_button("⬇ " + t("download_template", lang), tpl.to_csv(index=False).encode("utf-8"), "plantilla.csv", "text/csv", use_container_width=True)
                c2.download_button("⬇ " + t("download_dict", lang), dic.to_csv(index=False).encode("utf-8"), "diccionario.csv", "text/csv", use_container_width=True)
                c3.download_button("⬇ diseno.json", design.to_json().encode("utf-8"), "diseno.json", "application/json", use_container_width=True)
        elif step == 5:
            head(5, L("Sube tu CSV relleno (o usa los datos de ejemplo) y obtén resultados, gráficos y el texto APA.",
                      "Upload your filled CSV (or use the example data) and get results, plots and APA text."))
            up = st.file_uploader(t("upload", lang), type=["csv"], help=L("Usa punto decimal. Las filas incompletas se excluyen y se informa.",
                                                                          "Use a dot as decimal separator. Incomplete rows are excluded and reported."))
            if up is not None:
                ss.data = pd.read_csv(up)
            data = ss.data
            if data is None:
                st.info(L("Aún no hay datos. Sube un CSV o elige un estudio de ejemplo en la barra lateral.",
                          "No data yet. Upload a CSV or pick an example study in the sidebar."))
            elif design is None:
                st.error(design_error)
            else:
                with st.expander(L(f"Vista previa de los datos ({len(data)} filas)", f"Data preview ({len(data)} rows)")):
                    st.dataframe(data.head(10), hide_index=True, use_container_width=True)
                if st.button("▶ " + t("run", lang), type="primary", key="run"):
                    ss.do_run = True
                if ss.get("do_run"):
                    with tempfile.TemporaryDirectory() as tmp:
                        try:
                            res = analyze(design, data, outdir=tmp)
                        except (DataError, KeyError, ValueError) as exc:
                            st.error(str(exc))
                            st.stop()
                        rep = build_report(res, lang)
                        dc = to_decimal_comma if rep["decimal_comma"] else (lambda x: x)
                        r0 = res.rows[0]
                        mag = magnitude(r0["es_symbol"], r0["es"]) if r0["es_symbol"] else None
                        sig = r0["p"] is not None and r0["p"] < design.alpha
                        chip = (f"<span class='ps-chip {'ok' if sig else 'no'}'>{L('significativo', 'significant') if sig else L('no significativo', 'not significant')}</span>")
                        es_txt = dc(f"{r0['es_symbol']} = {fmt_num(r0['es'], r0['bounded'])}") if r0["es"] is not None else "—"
                        ci_txt = (dc(f"{L('IC 95%', '95% CI')} [{fmt_num(r0['ci'][0], r0['bounded'])}, {fmt_num(r0['ci'][1], r0['bounded'])}]")
                                  if r0["ci"] else "")
                        cards([
                            (L("Prueba aplicada", "Test used"), cap(t(res.test, lang).split(" (")[0]),
                             (L("alternativa por supuestos", "fallback due to assumptions") if res.fallback_reason else L("según lo planificado", "as planned")), ""),
                            ("N", res.n, (L(f"{res.n_excluded} casos excluidos", f"{res.n_excluded} cases excluded") if res.n_excluded else L("sin exclusiones", "no exclusions")), "indigo"),
                            (L("Tamaño del efecto", "Effect size"), es_txt, (ci_txt + (f" · {t(mag, lang)}" if mag else "")), "amber"),
                            (r0["label"], dc(fmt_p(r0["p"])), chip, "rose"),
                        ])
                        tabs = st.tabs([L("📝 Resultados APA", "📝 APA results"), L("📈 Gráficos", "📈 Plots"), L("✅ Supuestos", "✅ Assumptions"),
                                        L("🔢 Tablas", "🔢 Tables"), L("📄 Método", "📄 Method")])
                        with tabs[0]:
                            st.markdown("<div class='ps-apa'>" + "".join(f"<p>{html.escape(p)}</p>" for p in rep["results"]) + "</div>", unsafe_allow_html=True)
                            st.caption(" · ".join(stat_str(r, lang, rep["decimal_comma"]) for r in res.rows[:1]))
                        with tabs[1]:
                            cols = st.columns(min(2, max(1, len(rep["figures"]))))
                            for i, f in enumerate(rep["figures"]):
                                cols[i % len(cols)].image(f, use_container_width=True)
                        with tabs[2]:
                            rows = "".join(
                                f"<tr><td>{html.escape(assumption_label(a['key'], lang))}</td><td>{html.escape(a['detail'])}</td>"
                                f"<td><span class='ps-chip {'ok' if a['ok'] else 'no'}'>{'✓' if a['ok'] else '✗'}</span></td></tr>"
                                for a in res.assumptions)
                            st.markdown(f"<table class='ps-assump'>{rows}</table>" if rows else "—", unsafe_allow_html=True)
                            for n in res.notes:
                                st.caption(n[0 if lang == "es" else 1])
                        with tabs[3]:
                            for name, tab in rep["tables"]:
                                st.markdown(f"*{name}*")
                                st.dataframe(tab, hide_index=True, use_container_width=True)
                        with tabs[4]:
                            st.markdown("<div class='ps-apa'>" + "".join(f"<p><b>{html.escape(h)}.</b> {html.escape(p)}</p>" for h, p in rep["methods"]) + "</div>",
                                        unsafe_allow_html=True)
                        md = to_markdown(rep)
                        to_docx(rep, os.path.join(tmp, "informe.docx"))
                        buf = io.BytesIO(open(os.path.join(tmp, "informe.docx"), "rb").read())
                        st.markdown("")
                        c1, c2, _ = st.columns([1, 1, 1])
                        c1.download_button("⬇ " + L("Informe Word (DOCX)", "Word report (DOCX)"), buf.getvalue(), f"informe_{lang}.docx", use_container_width=True)
                        c2.download_button("⬇ " + L("Informe Markdown", "Markdown report"), md.encode("utf-8"), f"informe_{lang}.md", "text/markdown", use_container_width=True)
        if design_error and step in (0, 1, 2):
            st.warning(design_error)
        st.markdown("")
        nav_buttons(can_next=design is not None or step < 2)

# --------------------------------------------------------------------------- concept panel
CONCEPTS = [
    (L("Tipos de diseño", "Design types"),
     L("<p><b>Entre sujetos</b>: comparas personas distintas (terapia vs control).</p><p><b>Intra sujetos</b>: comparas a las mismas personas en varios momentos (pre vs post). Suele necesitar menos participantes porque cada persona es su propio control.</p><p><b>Mixto</b>: combina ambos; es el diseño típico de un ensayo con medida pre y post.</p>",
       "<p><b>Between-subjects</b>: you compare different people (therapy vs control).</p><p><b>Within-subjects</b>: you compare the same people at several times (pre vs post). Usually needs fewer participants because each person is their own control.</p><p><b>Mixed</b>: combines both; the typical design of a pre/post trial.</p>"),
     L("Escribe la pregunta de investigación en una frase: luego te servirá para el apartado de Método.", "Write the research question in one sentence: it will help you with the Method section.")),
    (L("Niveles de medida", "Measurement levels"),
     L("<p>El nivel de medida decide si usamos pruebas <b>paramétricas</b> (medias, como la t o el ANOVA) o <b>no paramétricas</b> (rangos, como Mann-Whitney).</p><ul><li><b>Nominal</b>: categorías (grupo, sexo).</li><li><b>Ordinal</b>: orden sin distancias iguales (un ítem Likert).</li><li><b>Intervalo/razón</b>: puntuaciones totales, tiempos.</li></ul>",
       "<p>The measurement level decides whether we use <b>parametric</b> tests (means, like t or ANOVA) or <b>nonparametric</b> ones (ranks, like Mann-Whitney).</p><ul><li><b>Nominal</b>: categories (group, sex).</li><li><b>Ordinal</b>: ordered, unequal distances (one Likert item).</li><li><b>Interval/ratio</b>: total scores, times.</li></ul>"),
     L("La suma de varios ítems Likert suele tratarse como intervalo.", "A sum of several Likert items is usually treated as interval.")),
    (L("¿Qué es la potencia?", "What is power?"),
     L("<p>El <b>tamaño del efecto</b> es cuánto importa la diferencia, independientemente de N. Cohen propuso referencias: pequeño, mediano y grande.</p><p><b>α</b> es el riesgo de ver un efecto que no existe; la <b>potencia</b>, la probabilidad de detectar un efecto que sí existe.</p><p>Efectos pequeños necesitan muestras grandes.</p>",
       "<p>The <b>effect size</b> is how much the difference matters, regardless of N. Cohen proposed benchmarks: small, medium and large.</p><p><b>α</b> is the risk of seeing an effect that does not exist; <b>power</b> is the probability of detecting an effect that does.</p><p>Small effects need large samples.</p>"),
     L("En psicología los efectos reales suelen ser pequeños o medianos: evita suponer «grande» sin justificarlo.", "In psychology real effects are often small to medium: avoid assuming “large” without justification.")),
    (L("Supuestos", "Assumptions"),
     L("<p>Cada prueba asume ciertas condiciones (p. ej., normalidad o varianzas parecidas). PsicoStat las comprueba con tus datos y, si no se cumplen, aplica la <b>alternativa</b> planificada y lo explica en el informe.</p>",
       "<p>Each test assumes certain conditions (e.g., normality or similar variances). PsicoStat checks them on your data and, if they fail, applies the planned <b>fallback</b> and explains it in the report.</p>"),
     L("Justifica el N en tu TFG/TFM con el texto de «Cómo se ha calculado».", "Justify N in your thesis with the “How it was computed” text.")),
    (L("Prepara tus datos", "Prepare your data"),
     L("<ul><li>Una fila por participante y una columna <code>id</code>.</li><li>Medidas repetidas: una columna por momento (<code>vd_pre</code>, <code>vd_post</code>).</li><li>Usa punto decimal y deja vacías las celdas sin dato.</li><li>No incluyas nombres ni datos identificativos.</li></ul>",
       "<ul><li>One row per participant and an <code>id</code> column.</li><li>Repeated measures: one column per time point (<code>dv_pre</code>, <code>dv_post</code>).</li><li>Use a dot as decimal separator and leave missing cells empty.</li><li>Do not include names or identifying data.</li></ul>"),
     L("Guarda también diseno.json: así podrás repetir el análisis exactamente.", "Keep diseno.json too: it lets you repeat the analysis exactly.")),
    (L("Cómo leer los resultados", "How to read the results"),
     L("<p>El <b>valor p</b> indica si el resultado es compatible con «no hay efecto»; el <b>tamaño del efecto</b> y su <b>intervalo de confianza</b> dicen cuánto efecto hay y con qué precisión.</p><p>Informa siempre ambos (APA 7).</p>",
       "<p>The <b>p value</b> tells whether the result is compatible with “no effect”; the <b>effect size</b> and its <b>confidence interval</b> tell how large the effect is and how precisely it is estimated.</p><p>Always report both (APA 7).</p>"),
     L("Un resultado no significativo no demuestra que no haya efecto.", "A non-significant result does not show there is no effect.")),
]
with side:
    title, body, tip = CONCEPTS[step]
    concept(title, body, tip)
    if design is not None and step < 3:
        rec = recommend(design)
        st.markdown(f"<div class='ps-mc indigo' style='margin-top:.8rem'><div class='l'>{L('Prueba prevista', 'Planned test')}</div>"
                    f"<div class='v' style='font-size:1.15rem'>{cap(html.escape(rec.label(lang)))}</div>"
                    f"<div class='s'>N ≈ {sample_size(design)['n_total']}</div></div>", unsafe_allow_html=True)
