# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/); versionado [SemVer](https://semver.org/lang/es/).

## [Sin publicar] / Unreleased

### Corregido / Fixed
- ANOVA mixto con grupos de distinto tamaño: el efecto intra sujetos usa sumas de cuadrados de tipo III (media no ponderada de los perfiles de cada grupo, como SPSS y R `afex`), en lugar de SC ponderadas; el aviso «sumas de cuadrados aproximadas» se sustituye por una nota que indica el tipo III. El efecto entre grupos, la interacción y los términos de error no cambian. Validado frente a R `car::Anova(type = 3)`/`afex::aov_ez` con grupos 6/5/4 y 6/3/2 (#4) / Mixed ANOVA with unequal group sizes: the within-subjects effect now uses Type III sums of squares (as SPSS and R afex) instead of weighted SS.
- ANOVA mixto: la prueba de Mauchly y las ε de Greenhouse-Geisser y Huynh-Feldt usan la matriz de covarianzas intragrupo combinada con N − G gl (antes N − 1), lo que daba una p de Mauchly demasiado pequeña y sesgaba ε_HF; el informe muestra ahora también ε_GG y ε_HF. Validado frente a R `car::Anova` (equivalente a afex/SPSS) con grupos equilibrados y no equilibrados (#18) / Mixed ANOVA: Mauchly's test and the GG/HF epsilons now use the pooled within-group covariance with N − G df instead of N − 1; ε_GG and ε_HF are now reported.

## [0.3.0] - 2026-10-25

### Añadido / Added
- IC 95% para ω² en el ANOVA de Welch (inversión de la F no central con los gl de Welch; Steiger, 2004), IC 95% exacto (Clopper-Pearson) para la g de Cohen en McNemar, y tamaño del efecto η²_Q = Q / [N(k − 1)] (Serlin et al., 1982) con IC 95% bootstrap para la Q de Cochran; el informe indica cómo se calculó cada intervalo (#7) / 95% CIs for Welch's ANOVA ω² and McNemar's Cohen's g, and η²_Q with bootstrap CI for Cochran's Q.
- Corrección de esfericidad en el ANOVA con dos factores intra sujetos: prueba de Mauchly, ε de Greenhouse-Geisser y de Huynh-Feldt para cada efecto (A, B y la interacción A × B, mediante contrastes ortonormales), y corrección de Greenhouse-Geisser de los gl y la p en los efectos que violan la esfericidad; validado frente a pingouin (equivalente a R `car`/`afex`) (#3) / Per-effect sphericity (Mauchly, ε_GG, ε_HF) and Greenhouse-Geisser correction in two-way within-subjects ANOVA.

### Cambiado / Changed
- La p de la prueba de Mauchly incluye el término de segundo orden de R `mauchly.test`, de modo que coincide con R/afex/JASP también con 4 o más condiciones / Mauchly's p now includes R's second-order term.

## [0.2.0] - 2026-10-21

### Añadido / Added
- Contrastes unilaterales (direccionales) para t de Welch, t pareada, Mann-Whitney, Wilcoxon, Pearson y Spearman: opción en el asistente, la web y la CLI (`--alternative`), tamaño muestral a priori con α unilateral (validado con G*Power), redacción APA «(unilateral)/(one-tailed)» con H₁ en el Método y aviso si el efecto va en dirección contraria (#6) / One-sided tests.
- Coma decimal opcional en la salida en español (texto APA, tablas y DOCX), con `;` como separador en intervalos y grados de libertad; opción `--decimal-comma`, interruptor en la web y campo `decimal_comma` del diseño. El inglés mantiene el punto (#8) / Optional decimal comma for Spanish output.
- Logotipo e ilustraciones SVG originales, imagen OG y capturas de pantalla en el README / Original SVG logo, illustrations, OG image and README screenshots.

### Cambiado / Changed
- Rediseño completo de la interfaz web: tema propio (verde azulado/índigo con acento ámbar), cabecera con logotipo, navegación por pasos en la barra lateral con indicador de progreso, una pantalla por paso con Atrás/Siguiente, panel de conceptos en lenguaje sencillo, tarjetas de resultados y pestañas (APA, gráficos, supuestos, tablas, método) / Full web UI redesign.
- Gráficos con paleta coherente / Consistent plot palette.

### Corregido / Fixed
- Concordancia de género en el texto en español («La diferencia … fue estadísticamente significativa»).

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
