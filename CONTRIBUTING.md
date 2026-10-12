# Contribuir a PsicoStat / Contributing

¡Gracias por tu interés! / Thanks for your interest! Issues and pull requests are welcome in Spanish or English.

## Formas de contribuir / Ways to contribute

- 🐞 Informar de errores (con el `diseno.json` y un CSV mínimo que lo reproduzca) / Report bugs with a minimal design + CSV.
- 📐 Revisar la exactitud estadística (comparaciones con R, JASP, jamovi, SPSS o G*Power) / Check statistical accuracy against reference software.
- ✍️ Mejorar la redacción APA en español o inglés / Improve APA wording.
- 🌍 Traducciones (`src/psicostat/i18n.py`) / Translations.
- 🧪 Nuevas pruebas estadísticas o diseños / New tests or designs.

## Desarrollo / Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
pytest            # tests
ruff check .      # lint
psicostat app     # web UI
```

## Normas / Guidelines

1. Abre un issue antes de cambios grandes / Open an issue before large changes.
2. Toda función estadística nueva necesita un test que la compare con un valor de referencia (SciPy, statsmodels, R o G*Power) / Every new statistical routine needs a test against a reference value.
3. Todo texto visible debe existir en español e inglés / All user-facing text must be bilingual.
4. Mantén los commits pequeños y descriptivos; actualiza `CHANGELOG.md` / Keep commits small; update the changelog.
5. Al contribuir aceptas licenciar tu aportación bajo MIT / Contributions are licensed under MIT.

## Estructura / Layout

```
src/psicostat/
  design.py     diseño + recomendación de prueba / design + test recommendation
  power.py      tamaño muestral a priori / a priori sample size
  template.py   plantilla CSV + diccionario / CSV template + data dictionary
  analysis.py   análisis, supuestos, post hoc, gráficos / analysis, assumptions, post hoc, plots
  effects.py    tamaños del efecto e IC / effect sizes and CIs
  report.py     texto APA, Markdown, DOCX / APA text, Markdown, DOCX
  i18n.py       textos es/en / strings
  cli.py, app.py
tests/
```
