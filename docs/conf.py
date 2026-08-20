from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath("..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

project = "noisefloat"
author = "Xinye Chen"
copyright = "2026, Xinye Chen"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"show-inheritance": True}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "noisefloat documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "navigation_with_keys": True,
    "source_repository": "https://github.com/chenxinye/noisefloat2/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#1f5f8b",
        "color-brand-content": "#1f5f8b",
        "font-stack": "'Hiragino Sans', 'Yu Gothic', 'Noto Sans CJK JP', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    "dark_css_variables": {
        "color-brand-primary": "#78c4ff",
        "color-brand-content": "#78c4ff",
    },
}
