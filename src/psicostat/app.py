"""Streamlit web UI. Run with:  psicostat app   (or: streamlit run src/psicostat/app.py)"""
import io
import os
import tempfile

import pandas as pd
import streamlit as st

from psicostat.analysis import DataError, analyze
from psicostat.design import Design, assumption_label, recommend
from psicostat.examples import EXAMPLES, example
from psicostat.i18n import t
from psicostat.power import explain, sample_size
from psicostat.report import build_report, to_docx, to_markdown
from psicostat.template import make_template

st.set_page_config(page_title="PsicoStat", page_icon="📊", layout="wide")
lang = st.sidebar.radio("Idioma / Language", ["es", "en"], format_func=lambda x: {"es": "Español", "en": "English"}[x])
st.title(t("app_title", lang))
st.caption(t("disclaimer", lang))


def lst(s):
    return [x.strip() for x in s.split(",") if x.strip()]


st.sidebar.markdown("---")
ex_name = st.sidebar.selectbox(t("use_example", lang), ["—"] + EXAMPLES)
if ex_name != "—":
    d_ex, df_ex = example(ex_name)
else:
    d_ex, df_ex = None, None
base = d_ex or Design()

st.header(t("step1", lang))
c1, c2 = st.columns(2)
title = c1.text_input(t("q_title", lang), base.title)
question = c2.text_input(t("q_question", lang), base.question)
analyses = ["compare", "correlation", "regression", "association"]
analysis = st.radio(t("q_analysis", lang), analyses, index=analyses.index(base.analysis),
                    format_func=lambda a: t("an_" + a, lang), horizontal=True)
kw = dict(title=title, question=question, analysis=analysis)

st.header(t("step2", lang))
types = ["continuous", "ordinal", "nominal"]
c1, c2, c3 = st.columns(3)
if analysis == "compare":
    designs = ["between", "within", "mixed"]
    kw["design"] = c1.selectbox(t("q_design", lang), designs, index=designs.index(base.design), format_func=lambda x: t("d_" + x, lang))
    kw["dv_type"] = c2.selectbox(t("q_dv_type", lang), types, index=types.index(base.dv_type), format_func=lambda x: t("t_" + x, lang))
    kw["dv_name"] = c3.text_input(t("q_dv_name", lang), base.dv_name)
    kw["iv_name"] = c1.text_input(t("q_iv_name", lang), base.iv_name)
    kw["levels"] = lst(c2.text_input(t("q_levels", lang), ", ".join(base.levels)))
    two = kw["design"] == "mixed" or c3.number_input(t("q_n_factors", lang), 1, 2, base.n_factors) == 2
    if two and kw["dv_type"] != "nominal":
        kw["n_factors"] = 2
        kw["factor_b_name"] = c1.text_input(t("q_factor_b_name", lang), base.factor_b_name)
        kw["levels_b"] = lst(c2.text_input(t("q_levels_b", lang), ", ".join(base.levels_b)))
    if kw["dv_type"] == "nominal" and kw["design"] == "between":
        kw["dv_levels"] = lst(c3.text_input(t("q_dv_levels", lang), ", ".join(base.dv_levels)))
    if kw["design"] == "between" and not two and kw["dv_type"] == "continuous":
        kw["covariates"] = lst(c3.text_input(t("q_covariates", lang), ", ".join(base.covariates)))
    if kw["design"] in ("within", "mixed"):
        kw["rho"] = c3.slider(t("q_rho", lang), 0.0, 0.9, float(base.rho), 0.05)
elif analysis == "association":
    kw.update(dv_type="nominal", iv_type="nominal")
    kw["iv_name"] = c1.text_input(t("q_iv_name", lang), base.iv_name)
    kw["levels"] = lst(c1.text_input(t("q_levels", lang), ", ".join(base.levels)))
    kw["dv_name"] = c2.text_input(t("q_dv_name", lang), base.dv_name)
    kw["dv_levels"] = lst(c2.text_input(t("q_dv_levels", lang), ", ".join(base.dv_levels)))
else:
    kw["dv_type"] = c1.selectbox(t("q_dv_type", lang), types, index=types.index(base.dv_type), format_func=lambda x: t("t_" + x, lang))
    kw["iv_type"] = c2.selectbox(t("q_iv_type", lang), types, index=types.index(base.iv_type), format_func=lambda x: t("t_" + x, lang))
    kw["dv_name"] = c3.text_input(t("q_dv_name", lang), base.dv_name)
    if analysis == "correlation":
        kw["iv_name"] = c1.text_input(t("q_iv_name", lang), base.iv_name)
    else:
        kw["predictors"] = lst(c1.text_input(t("q_predictors", lang), ", ".join(base.predictors)))
        if kw["dv_type"] == "nominal":
            kw["event_rate"] = c2.slider(t("q_event_rate", lang), 0.05, 0.95, float(base.event_rate), 0.05)
kw["n_groups"] = max(len(kw.get("levels", [])), 2)

st.header(t("step3", lang))
st.info(t("effect_help", lang))
c1, c2, c3, c4, c5 = st.columns(5)
labels = ["small", "medium", "large"]
kw["effect_label"] = c1.selectbox(t("q_effect_label", lang), labels, index=1, format_func=lambda x: t(x, lang))
custom = c2.text_input(t("q_effect_custom", lang), "")
kw["effect_size"] = float(custom.replace(",", ".")) if custom.strip() else None
kw["alpha"] = c3.number_input(t("q_alpha", lang), 0.001, 0.2, 0.05, 0.005, format="%.3f")
kw["power"] = c4.number_input(t("q_power", lang), 0.5, 0.99, 0.80, 0.05)
kw["dropout"] = c5.number_input(t("q_dropout", lang), 0.0, 0.5, 0.0, 0.05)

try:
    d = Design(**kw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.header(t("step4", lang))
rec = recommend(d)
ss = sample_size(d, rec)
c1, c2 = st.columns([2, 1])
c1.subheader(f"{t('recommended', lang)}: {rec.label(lang)}")
c1.write(f"**{t('fallback', lang)}:** {rec.fallback_label(lang)}")
c1.write(f"**{t('assumptions', lang)}:**\n" + "\n".join(f"- {assumption_label(a, lang)}" for a in rec.assumptions))
for n in rec.notes_text(lang):
    c1.caption(n)
c2.metric(t("sample_size", lang), f"N = {ss['n_total']}",
          f"{ss['n_per_group']} {t('per_group', lang)}" if ss["n_per_group"] and ss["groups"] > 1 else None, delta_color="off")
c2.write(explain(ss, lang))

st.header(t("step5", lang))
tpl, dic = make_template(d, ss["n_total"])
st.dataframe(dic, hide_index=True, use_container_width=True)
c1, c2, c3 = st.columns(3)
c1.download_button(t("download_template", lang), tpl.to_csv(index=False).encode("utf-8"), "plantilla.csv", "text/csv")
c2.download_button(t("download_dict", lang), dic.to_csv(index=False).encode("utf-8"), "diccionario.csv", "text/csv")
c3.download_button("diseno.json", d.to_json().encode("utf-8"), "diseno.json", "application/json")

st.header(t("step6", lang))
up = st.file_uploader(t("upload", lang), type=["csv"])
data = pd.read_csv(up) if up is not None else df_ex
if data is not None:
    st.dataframe(data.head(10), hide_index=True)
if data is not None and st.button(t("run", lang), type="primary"):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            res = analyze(d, data, outdir=tmp)
        except (DataError, KeyError, ValueError) as exc:
            st.error(str(exc))
            st.stop()
        rep = build_report(res, lang)
        md = to_markdown(rep)
        for p in rep["results"]:
            st.write(p)
        for name, tab in rep["tables"]:
            st.markdown(f"*{name}*")
            st.dataframe(tab, hide_index=True)
        for f in rep["figures"]:
            st.image(f, width=550)
        with st.expander("Método / Method"):
            for h, p in rep["methods"]:
                st.markdown(f"**{h}.** {p}")
        buf = io.BytesIO()
        to_docx(rep, os.path.join(tmp, "informe.docx"))
        with open(os.path.join(tmp, "informe.docx"), "rb") as fh:
            buf.write(fh.read())
        c1, c2 = st.columns(2)
        c1.download_button(t("download_md", lang), md.encode("utf-8"), f"informe_{lang}.md", "text/markdown")
        c2.download_button(t("download_docx", lang), buf.getvalue(), f"informe_{lang}.docx")
