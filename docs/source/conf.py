"""Sphinx configuration for MDBench."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "MDBench"
copyright = "2026, YuMeow"
author = "YuMeow"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
templates_path = ["_templates"]
exclude_patterns = []
language = "en"
suppress_warnings = ["ref.python", "ref.term", "ref.ref"]

html_theme = "sphinx_book_theme"
html_title = "MDBench"
html_theme_options = {
    "show_toc_level": 2,
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
