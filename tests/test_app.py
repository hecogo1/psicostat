import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_renders():
    at = st_testing.AppTest.from_file("../src/psicostat/app.py", default_timeout=60).run()
    assert not at.exception
    assert any("PsicoStat" in t.value for t in at.title)
