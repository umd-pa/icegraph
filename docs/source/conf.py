import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

project = 'IceGraph'
author = 'Taylor St Jean'
release = '0.11.2'
copyright = '2026, University of Maryland and the IceCube Collaboration'

html_theme = "renku"
root_doc = 'index'

html_logo = "../../img/logo-dark.png"

html_static_path = ["../../img"]

html_theme_options = {
    'collapse_navigation': False,
    'titles_only': True,
    'navigation_depth': 6,
    'logo_only': True,
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # Google/Numpy style docstrings
    'sphinx_autodoc_typehints',  # Type hint support
    'sphinx.ext.autosummary'
]

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

exclude_patterns = [
    '**/__pycache__',
    '**/*.egg-info'
]
