"""Sphinx configuration for Systrophe."""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

project = "Systrophe"
author = "Christopher Knopp"
copyright = f"{date.today().year}, {author}"
release = "0.5.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "alabaster"
html_static_path: list[str] = []
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Allow imports without optional deps to fail silently in autodoc
autodoc_mock_imports = ["qiskit", "qiskit_ibm_runtime", "matplotlib"]
