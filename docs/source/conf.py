# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib import metadata

# -- Path setup --------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(__file__, "../../.."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# -- Project information -----------------------------------------------------

project = "deepaudio-x"

try:
    release = metadata.version("deepaudio-x")
except metadata.PackageNotFoundError:
    # Fallback when package metadata is not available (e.g., local builds)
    release = "0.0.0"

version = release

author = "Christos Nikou, Stefanos Vlachos, Ellie Vakalaki"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx.ext.mathjax", 
    "sphinx.ext.coverage",
    "nbsphinx",
    "nbsphinx_link"
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}

autoclass_content = "both"

intersphinx_disabled_domains = ["std"]

# -- Options for HTML output -------------------------------------------------

html_logo = "_static/DeepAudioX_logo.png"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 2,
}
html_css_files = ["custom.css"]
