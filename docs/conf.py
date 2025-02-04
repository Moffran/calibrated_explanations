# pylint: disable=invalid-name, redefined-builtin, missing-module-docstring
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add project directories to sys.path
# sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../src'))
# sys.path.insert(0, os.path.abspath('../notebooks'))
# sys.path.insert(0, os.path.abspath('../src/calibrated_explanations'))
# sys.path.insert(0, os.path.abspath('../src/calibrated_explanations/utils'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'calibrated_explanations'
copyright = '2023, Helena Löfström, Tuwe Löfström'
author = 'Helena Löfström, Tuwe Löfström'

# The short X.Y version
version = '0.5.1'

# The full version, including alpha/beta/rc tags
release = '0.5.1'

# -- General configuration ---------------------------------------------------

# Sphinx extension module names
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'numpydoc',
    'nbsphinx',
    'myst_parser',
]

# The master toctree document
master_doc = 'index'

# Intersphinx mapping
intersphinx_mapping = {
    'crepes': ('https://crepes.readthedocs.io/en/latest/', None),
}

# Templates path
templates_path = ['_templates']

# Language for autogenerated content
language = 'en'

# Patterns to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Pygments style for syntax highlighting
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# HTML theme
html_theme = 'pydata_sphinx_theme'

# HTML theme options
html_theme_options = {
    'navbar_end': ['navbar-icon-links'],
    'sidebarwidth': 270,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'show_toc_level': 2,
    'github_url': 'https://github.com/Moffran/calibrated_explanations',
}

# HTML sidebars
html_sidebars = {}

# HTML context
html_context = {
    'default_mode': 'light',
}

# HTML title
html_title = f'{project} v{version}'

# Last updated format
html_last_updated_fmt = '%b %d, %Y'

# -- Extension configuration -------------------------------------------------

# Source suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc settings
autoclass_content = 'init'
autodoc_member_order = 'bysource'
autoclass_member_order = 'bysource'

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# The todo extension settings
todo_include_todos = True
