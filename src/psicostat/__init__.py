"""PsicoStat: asistente open source de metodología y análisis estadístico para psicología."""
__version__ = "0.3.1"
from .analysis import analyze  # noqa: F401
from .design import Design, recommend  # noqa: F401
from .power import sample_size  # noqa: F401
from .report import build_report, to_docx, to_markdown  # noqa: F401
from .template import make_template  # noqa: F401
