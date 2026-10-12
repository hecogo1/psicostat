import os

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_cloud_entrypoint_renders():
    path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
    at = st_testing.AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception
    assert any("PsicoStat" in m.value for m in at.markdown)
