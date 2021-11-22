# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_gallery
from sphinx_gallery.sorting import ExampleTitleSortKey



sys.path.insert(0, os.path.join(os.path.abspath(".."), 'src'))
sys.path.append(os.path.abspath('extensions'))


# -- Project information -----------------------------------------------------

project = 'cr-sparse'
copyright = '2021, CR-Sparse Development Team'
author = 'CR-Sparse Development Team'

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    'sphinx.ext.graphviz',
    "sphinx.ext.autodoc",
    'sphinx.ext.autosummary',
    "sphinx.ext.doctest",
    'sphinx.ext.ifconfig',
    'sphinx.ext.extlinks',
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    "sphinx.ext.githubpages",
    'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex',
    'sphinx_gallery.gen_gallery',
    'sphinx_panels',
    # markdown support
    'myst_parser',
    # project specific extensions
    'environments',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    "sklearn": ("http://scikit-learn.org/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pyfftw": ("https://pyfftw.readthedocs.io/en/latest/", None),
    "spgl1": ("https://spgl1.readthedocs.io/en/latest/", None),
    "cr.sparse": ("https://cr-sparse.readthedocs.io/en/latest/", None),
}

suppress_warnings = [
    'ref.citation',  # Many duplicated citations in numpy/scipy docstrings.
    'ref.footnote',  # Many unreferenced footnotes in numpy/scipy docstrings
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = "index"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 
    'Thumbs.db',
    '.DS_Store', 
    "**/.DS_Store", 
    ".ipynb_checkpoints", 
    "**/.ipynb_checkpoints"
    ]

autosummary_generate = True
napolean_use_rtype = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "CRSparse"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "CRSparse.tex", "CR-Sparse", "CR-Sparse Development Team", "manual"),
]


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# change the matplotlib backend to avoid tkinter dependency
import matplotlib

matplotlib.use("agg")

# Tell sphinx-autodoc-typehints to generate stub parameter annotations including
# types, even if the parameters aren't explicitly documented.
always_document_param_types = True

# BIBTEX Settings
bibtex_bibfiles = ['references.bib']


sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'gallery',  # path to where to save gallery generated output
     # pattern of files to match
    'filename_pattern': '\.py',
    # Remove the "Download all examples" button from the top level gallery
    'download_all_examples': False,
    # Sort gallery example by file name instead of number of lines (default)
    'within_subsection_order': ExampleTitleSortKey,
    # Modules for which function level galleries are created.
    'doc_module': ('cr.sparse', 'cr.sparse.data', 'cr.sparse.dict', 'cr.sparse.lop', 'cr.sparse.wt'),
    'reference_url' : {
        # The module you locally document uses None
        'cr.sparse': None,
        'cr.sparse.data': None,
        'cr.sparse.dict': None,
        'cr.sparse.lop': None,
        'cr.sparse.wt': None,
    }
}

math_number_all = True


### Theme customization ###



html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]
html_js_files = [
    'js/custom.js',
    'js/mathconf.js',
]

html_theme_options = {
    "fixed_sidebar": True,
    "github_banner": True,
    "github_button": True,
    "github_repo": 'cr-sparse',
    "github_user": 'carnotresearch',
    "show_powered_by": False,
    "sidebar_width": "300px",
    "page_width": "1200px",
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}