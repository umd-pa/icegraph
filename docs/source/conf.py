import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

project = 'IceGraph'
author = 'Taylor St Jean'
release = '0.1.0'

html_theme = "sphinx_rtd_theme"
master_doc = 'index'

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