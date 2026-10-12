# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/); versionado [SemVer](https://semver.org/lang/es/).

## [0.1.0] - 2026-10-12

### Añadido / Added
- Asistente paso a paso (CLI `psicostat wizard` e interfaz web Streamlit `psicostat app`), bilingüe es/en.
- Recomendación de prueba para 19 situaciones con supuestos y alternativas no paramétricas.
- Tamaño muestral a priori (t, ANOVA, factorial, ANCOVA, medidas repetidas, mixto, χ², correlación, regresión lineal y logística), validado frente a G*Power.
- Plantilla CSV y diccionario de datos según el diseño.
- Análisis con comprobación de supuestos, cambio automático a la alternativa, tamaños del efecto con IC 95%, post hoc y gráficos.
- Informes APA 7 en español e inglés (Markdown y DOCX).
- 13 conjuntos de datos de ejemplo simulados, tests con pytest y CI en GitHub Actions.
