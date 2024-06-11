# Configuration file for the Sphinx documentation builder.

import os
import sys
# Location of Sphinx files
sys.path.insert(0, os.path.abspath('./../..'))


# -- Project information

project = 'Meent'
copyright = '2024, The Meent Authors'
# author = 'Graziella'

release = ''
version = ''

# -- General configuration

# extensions = [
#     'sphinx.ext.duration',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.intersphinx',
#     'myst_parser',
#     'sphinx_design'
# ]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    'myst_nb',
    "sphinx_remove_toctrees",
    'sphinx_copybutton',
    # 'jax_extensions',
    'sphinx_design',
    'sphinxext.rediraffe',
]

myst_enable_extensions = ["colon_fence"]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/kc-ml2/meent',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = ['.rst', '.ipynb', '.md']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'images/meent_logo.png'
html_favicon = 'images/meent_logo2.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'style.css',
]

##################

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'meent'
copyright = '2024, yongha'
author = 'yongha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/kc-ml2/meent',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}


import os
import sys
sys.path.insert(0, os.path.abspath('../../meent'))
