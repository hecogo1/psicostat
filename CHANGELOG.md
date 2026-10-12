# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/); versionado [SemVer](https://semver.org/lang/es/).

## [Unreleased]

### Cambiado / Changed
- Rediseño completo de la interfaz web: tema propio (verde azulado/índigo con acento ámbar), cabecera con logotipo, navegación por pasos en la barra lateral con indicador de progreso, una pantalla por paso con Atrás/Siguiente, panel de conceptos en lenguaje sencillo, tarjetas de resultados y pestañas (APA, gráficos, supuestos, tablas, método) / Full web UI redesign.
- Gráficos con paleta coherente / Consistent plot palette.

### Añadido / Added
- Logotipo e ilustraciones SVG originales, imagen OG y capturas de pantalla en el README / Original SVG logo, illustrations, OG image and README screenshots.

## [0.1.0] - 2026-10-12

### Añadido / Added
- Asistente paso a paso (CLI `psicostat wizard` e interfaz web Streamlit `psicostat app`), bilingüe es/en.
- Recomendación de prueba para 19 situaciones con supuestos y alternativas no paramétricas.
- Tamaño muestral a priori (t, ANOVA, factorial, ANCOVA, medidas repetidas, mixto, χ², correlación, regresión lineal y logística), validado frente a G*Power.
- Plantilla CSV y diccionario de datos según el diseño.
- Análisis con comprobación de supuestos, cambio automático a la alternativa, tamaños del efecto con IC 95%, post hoc y gráficos.
- Informes APA 7 en español e inglés (Markdown y DOCX).
- 13 conjuntos de datos de ejemplo simulados, tests con pytest y CI en GitHub Actions.
- Archivos para desplegar la interfaz en Streamlit Community Cloud (`streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`).

### Corregido / Fixed
- La CLI ya no falla en consolas de Windows con codificación cp1252 al imprimir símbolos como η² o → / The CLI no longer crashes on Windows cp1252 consoles.
