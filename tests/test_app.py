import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = "../src/psicostat/app.py"


def _app():
    return st_testing.AppTest.from_file(APP, default_timeout=90).run()


def test_app_renders_first_step():
    at = _app()
    assert not at.exception
    assert any("PsicoStat" in m.value for m in at.markdown)
    assert at.session_state.step == 0


def test_wizard_navigation_and_analysis():
    at = _app()
    at.selectbox(key="k_example").set_value("anova").run()
    assert at.session_state.k_dv_name == "rendimiento"
    for expected in range(1, 6):
        at.button(key="next").click().run()
        assert not at.exception
        assert at.session_state.step == expected
    # values from earlier steps survive navigation
    assert at.session_state.k_levels == "lectura, subrayado, autoevaluacion"
    at.button(key="run").click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "ANOVA" in html and "η²p" in html
    at.button(key="back").click().run()
    assert at.session_state.step == 4


def test_english_and_sidebar_nav():
    at = _app()
    at.session_state.k_lang = "en"
    at.button(key="nav_3").click().run()
    assert not at.exception
    assert any("Recommended test" in m.value for m in at.markdown)


def test_decimal_comma_toggle():
    at = _app()
    at.selectbox(key="k_example").set_value("ttest").run()
    at.toggle(key="k_decimal_comma").set_value(True).run()
    at.button(key="nav_5").click().run()
    at.button(key="run").click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "IC 95% [-1," in html or "IC 95% [-" in html
    assert "p &lt; ,001" in html or "p < ,001" in html


def test_one_sided_option_in_ui():
    at = _app()
    at.selectbox(key="k_example").set_value("ttest").run()
    at.button(key="nav_2").click().run()
    at.radio(key="k_alternative").set_value("less").run()
    at.button(key="next").click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "N = 102" in html and "unilateral" in html
    at.button(key="nav_5").click().run()
    at.button(key="run").click().run()
    assert not at.exception
    assert "(unilateral)" in " ".join(m.value for m in at.markdown)
