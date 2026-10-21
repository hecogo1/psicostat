"""Command line interface: psicostat wizard | plan | template | analyze | example | app."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .analysis import analyze
from .design import Design, assumption_label, recommend
from .examples import EXAMPLES, example
from .i18n import t
from .power import explain, sample_size
from .report import build_report, to_docx, to_markdown
from .template import make_template


def _ask(prompt, default=None, choices=None, cast=str):
    while True:
        opts = ""
        if choices:
            opts = "\n" + "\n".join(f"  {i + 1}) {lab}" for i, (_, lab) in enumerate(choices)) + "\n"
        dflt = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{opts}{dflt}: ").strip()
        if not raw and default is not None:
            raw = str(default)
        if choices:
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                return choices[int(raw) - 1][0]
            keys = [c[0] for c in choices]
            if raw in keys:
                return raw
            print("  ?")
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  ?")


def _list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def wizard(lang="es"):
    """Interactive step-by-step questionnaire. Returns a Design."""
    print(f"\n=== {t('app_title', lang)} ===\n")
    print(t("step1", lang))
    title = _ask(t("q_title", lang), "")
    question = _ask(t("q_question", lang), "")
    analysis = _ask(t("q_analysis", lang), 1, [(a, t("an_" + a, lang)) for a in ("compare", "correlation", "regression", "association")])
    kw = dict(title=title, question=question, analysis=analysis)
    print("\n" + t("step2", lang))
    if analysis == "compare":
        kw["design"] = _ask(t("q_design", lang), 1, [(x, t("d_" + x, lang)) for x in ("between", "within", "mixed")])
        kw["dv_type"] = _ask(t("q_dv_type", lang), 1, [(x, t("t_" + x, lang)) for x in ("continuous", "ordinal", "nominal")])
        kw["dv_name"] = _ask(t("q_dv_name", lang), "puntuacion")
        kw["iv_name"] = _ask(t("q_iv_name", lang), "grupo")
        kw["levels"] = _list(_ask(t("q_levels", lang), "g1,g2" if kw["design"] != "within" else "pre,post"))
        if kw["design"] == "mixed" or (kw["dv_type"] != "nominal" and _ask(t("q_n_factors", lang), 1, cast=int) == 2):
            kw["n_factors"] = 2
            kw["factor_b_name"] = _ask(t("q_factor_b_name", lang), "tiempo" if kw["design"] == "mixed" else "factor_b")
            kw["levels_b"] = _list(_ask(t("q_levels_b", lang), "pre,post" if kw["design"] == "mixed" else "b1,b2"))
        if kw["dv_type"] == "nominal" and kw["design"] == "between":
            kw["dv_levels"] = _list(_ask(t("q_dv_levels", lang), "si,no"))
        if kw["design"] == "between" and kw.get("n_factors", 1) == 1 and kw["dv_type"] == "continuous":
            kw["covariates"] = _list(_ask(t("q_covariates", lang), ""))
        if kw["design"] in ("within", "mixed"):
            kw["rho"] = _ask(t("q_rho", lang), 0.5, cast=float)
    elif analysis == "association":
        kw.update(dv_type="nominal", iv_type="nominal")
        kw["iv_name"] = _ask(t("q_iv_name", lang), "var1")
        kw["levels"] = _list(_ask(t("q_levels", lang), "a,b"))
        kw["dv_name"] = _ask(t("q_dv_name", lang), "var2")
        kw["dv_levels"] = _list(_ask(t("q_dv_levels", lang), "si,no"))
    else:
        kw["dv_type"] = _ask(t("q_dv_type", lang), 1, [(x, t("t_" + x, lang)) for x in ("continuous", "ordinal", "nominal")])
        kw["iv_type"] = _ask(t("q_iv_type", lang), 1, [(x, t("t_" + x, lang)) for x in ("continuous", "ordinal", "nominal")])
        kw["dv_name"] = _ask(t("q_dv_name", lang), "y")
        if analysis == "correlation":
            kw["iv_name"] = _ask(t("q_iv_name", lang), "x")
        else:
            kw["predictors"] = _list(_ask(t("q_predictors", lang), "x1,x2"))
            if kw["dv_type"] == "nominal":
                kw["event_rate"] = _ask(t("q_event_rate", lang), 0.3, cast=float)
    kw["n_groups"] = max(len(kw.get("levels", [])), 2)
    print("\n" + t("step3", lang))
    print("  " + t("effect_help", lang))
    kw["effect_label"] = _ask(t("q_effect_label", lang), 2, [(x, t(x, lang)) for x in ("small", "medium", "large")])
    custom = _ask(t("q_effect_custom", lang), "")
    kw["effect_size"] = float(custom) if custom else None
    kw["alpha"] = _ask(t("q_alpha", lang), 0.05, cast=float)
    kw["power"] = _ask(t("q_power", lang), 0.80, cast=float)
    kw["dropout"] = _ask(t("q_dropout", lang), 0.0, cast=float)
    return Design(**kw)


def print_plan(d: Design, lang="es"):
    rec = recommend(d)
    ss = sample_size(d, rec)
    print(f"\n{t('recommended', lang)}: {rec.label(lang)}")
    print(f"{t('fallback', lang)}: {rec.fallback_label(lang)}")
    print(f"{t('assumptions', lang)}:")
    for a in rec.assumptions:
        print(f"  - {assumption_label(a, lang)}")
    for n in rec.notes_text(lang):
        print(f"  * {n}")
    print(f"\n{t('sample_size', lang)}: N = {ss['n_total']}" +
          (f" ({ss['n_per_group']} {t('per_group', lang)})" if ss["n_per_group"] and ss["groups"] > 1 else ""))
    print(explain(ss, lang))
    return rec, ss


def write_template(d: Design, n: int, out: str):
    tpl, dic = make_template(d, n)
    tpl.to_csv(out, index=False)
    dic_path = os.path.splitext(out)[0] + "_diccionario.csv"
    dic.to_csv(dic_path, index=False)
    return out, dic_path


def run_analysis(d: Design, csv, outdir, langs=("es", "en"), formats=("md", "docx")):
    os.makedirs(outdir, exist_ok=True)
    res = analyze(d, csv, outdir=os.path.join(outdir, "figuras"))
    paths = []
    for lang in langs:
        rep = build_report(res, lang)
        if "md" in formats:
            p = os.path.join(outdir, f"informe_{lang}.md")
            to_markdown(rep, p, rel_to=outdir)
            paths.append(p)
        if "docx" in formats:
            p = os.path.join(outdir, f"informe_{lang}.docx")
            to_docx(rep, p)
            paths.append(p)
    return res, paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="psicostat", description="PsicoStat: methods & statistics assistant for psychology")
    ap.add_argument("--lang", choices=("es", "en"), default="es")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wizard", help="asistente interactivo / interactive wizard")
    w.add_argument("--out", default="estudio", help="carpeta de salida / output folder")
    p = sub.add_parser("plan", help="prueba recomendada y N / recommended test and N")
    p.add_argument("design")
    tp = sub.add_parser("template", help="plantilla CSV / CSV template")
    tp.add_argument("design")
    tp.add_argument("--n", type=int, default=None)
    tp.add_argument("--out", default="plantilla.csv")
    a = sub.add_parser("analyze", help="analizar CSV / analyse CSV")
    a.add_argument("design")
    a.add_argument("csv")
    a.add_argument("--out", default="resultados")
    a.add_argument("--formats", default="md,docx")
    a.add_argument("--decimal-comma", action="store_true", help="coma decimal en el informe en español / decimal comma in the Spanish report")
    e = sub.add_parser("example", help="crear ejemplo / create example")
    e.add_argument("name", choices=EXAMPLES)
    e.add_argument("--out", default=None)
    sub.add_parser("app", help="abrir la interfaz web / launch web UI")
    args = ap.parse_args(argv)
    # Windows consoles often use a legacy code page (cp1252) that cannot print η², χ², →, etc.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass
    lang = args.lang

    if args.cmd == "wizard":
        d = wizard(lang)
        os.makedirs(args.out, exist_ok=True)
        d.to_json(os.path.join(args.out, "diseno.json"))
        _, ss = print_plan(d, lang)
        tpl, dic = write_template(d, ss["n_total"], os.path.join(args.out, "plantilla.csv"))
        print(f"\n→ {os.path.join(args.out, 'diseno.json')}\n→ {tpl}\n→ {dic}")
        print(("\nRellena la plantilla y ejecuta:" if lang == "es" else "\nFill in the template and run:") +
              f"\n  psicostat analyze {os.path.join(args.out, 'diseno.json')} {tpl} --out {os.path.join(args.out, 'resultados')}")
    elif args.cmd == "plan":
        print_plan(Design.from_json(args.design), lang)
    elif args.cmd == "template":
        d = Design.from_json(args.design)
        n = args.n or sample_size(d)["n_total"]
        for x in write_template(d, n, args.out):
            print(f"→ {x}")
    elif args.cmd == "analyze":
        d = Design.from_json(args.design)
        if args.decimal_comma:
            d.decimal_comma = True
        res, paths = run_analysis(d, args.csv, args.out, formats=tuple(args.formats.split(",")))
        rep = build_report(res, lang)
        print("\n".join(rep["results"]))
        print()
        for x in paths:
            print(f"→ {x}")
    elif args.cmd == "example":
        d, df = example(args.name)
        out = args.out or f"ejemplo_{args.name}"
        os.makedirs(out, exist_ok=True)
        d.to_json(os.path.join(out, "diseno.json"))
        df.to_csv(os.path.join(out, "datos.csv"), index=False)
        print(f"→ {out}/diseno.json\n→ {out}/datos.csv")
    elif args.cmd == "app":
        app = os.path.join(os.path.dirname(__file__), "app.py")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
