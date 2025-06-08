import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust path to your source

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # Google/Numpy style docstrings
    'sphinx_autodoc_typehints',  # Type hint support
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

html_theme = "furo"
html_baseurl = 'https://taylornstjean.github.io/icegraph/'
