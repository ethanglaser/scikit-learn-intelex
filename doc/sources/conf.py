#!/usr/bin/env python
# ==============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

# sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "Extension for Scikit-learn*"
copyright = "oneDAL project"
author = "oneDAL project"

# The short X.Y version
# Note: it should not have more than  two parts (year.month), otherwise the
# version switcher will not be able to pick it.
version = os.environ.get("SHORT_DOC_VERSION", "latest")
release = version

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx_tabs.tabs",
    "notfound.extension",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "dpctl": ("https://intelpython.github.io/dpctl/latest", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/stable/", None),
    # from scikit-learn, in case some object in sklearnex points to them:
    # https://github.com/scikit-learn/scikit-learn/blob/main/doc/conf.py
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "skops": ("https://skops.readthedocs.io/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "usage.rst",
    "patching/*",
    "kaggle/note-about-tps.rst",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# substitutions

rst_prolog = """
.. |sklearnex| replace:: Extension for Scikit-learn*
.. |onedal| replace:: oneAPI Data Analytics Library
"""

# Note: sklearn oftentimes uses single backticks to render code.
# Some docstrings here are inherited from theirs, so this setting
# is needed to render them the same way they do.
default_role = "literal"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "_static/uxl-foundation-logo-horizontal-color.png"
html_favicon = "_static/favicon.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#

html_theme_options = {
    "logo_only": False,
    "version_selector": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": -1,
    "includehidden": True,
    "titles_only": False,
}


html_context = {
    "current_version": version,
    "project_name": "scikit-learn-intelex",
    "switcher_url": "/scikit-learn-intelex/versions.json",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("custom.css")
    app.add_js_file("version_switcher.js")


# -- Options for HTMLHelp output ---------------------------------------------


# Output file base name for HTML help builder.
htmlhelp_basename = "sklearnexdoc"


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
    (
        master_doc,
        "scikit-learn-intelex.tex",
        "Extension for Scikit-learn* Documentation",
        "Intel",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "scikit-learn-intelex",
        "Extension for Scikit-learn* Documentation",
        [author],
        1,
    )
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "scikit-learn-intelex",
        "Extension for Scikit-learn* Documentation",
        author,
        "scikit-learn-intelex",
        "Extension for Scikit-learn speeds up scikit-learn "
        "beyond by providing drop-in patching.",
        "Miscellaneous",
    ),
]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# not found 404 page

notfound_urls_prefix = "/scikit-learn-intelex/"

# Any link matching these regexes will be skipped by linkcheck
linkcheck_ignore = [
    re.compile(r"https://github\.com/.+"),  # Avoid rate error
    re.compile(r"https://.*intel\.com/.+"),  # Avoid permission error
]

# Speed up link-check and fail faster
linkcheck_timeout = 10
linkcheck_workers = 8
