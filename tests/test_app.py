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
