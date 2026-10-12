"""Entry point for Streamlit Community Cloud (main file path: streamlit_app.py)."""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))  # use the package from the repo without installing it

runpy.run_path(os.path.join(ROOT, "src", "psicostat", "app.py"), run_name="__main__")
