#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# opty documentation build configuration file, created by
# sphinx-quickstart on Thu Jun  1 08:16:55 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import opty

DOCS_CONF_PATH = os.path.realpath(__file__)
DOCS_DIR = os.path.dirname(DOCS_CONF_PATH)
REPO_DIR = os.path.realpath(os.path.join(DOCS_DIR, '..'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '8.1'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'sphinx_reredirects',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {'.rst': 'restructuredtext'}

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'opty'
copyright = '2014-2025, opty authors'
author = 'Jason K. Moore'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = opty.__version__
# The full version, including alpha/beta/rc tags.
release = opty.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
if ("READTHEDOCS" in os.environ) or ("ONGITHUB" in os.environ):
    todo_include_todos = False
else:
    todo_include_todos = True

# Show the __init__ docstring on classes.
autoclass_content = 'both'

# Display long function signatures better.
maximum_signature_line_length = 50

# sphinx-gallery settings
sphinx_gallery_conf = {
    'examples_dirs': os.path.join(REPO_DIR, 'examples-gallery'),
    'gallery_dirs': 'examples',
    'matplotlib_animations': True,
    'copyfile_regex': r'.*\.(svg|npy|csv|yml)',
    'remove_config_comments': True,
    'parallel': True,
}

# NOTE : The subsections are only sorted online due to it preventing caching.
# See https://github.com/sphinx-doc/sphinx/issues/12300 for more info.
if ("READTHEDOCS" in os.environ) or ("ONGITHUB" in os.environ):
    suppress_warnings = ["config.cache"]

    def sort_subsections(path):
        if 'beginner' in path:
            return '101'
        elif 'intermediate' in path:
            return '102'
        elif 'advanced' in path:
            return '103'
        else:
            return path

    sphinx_gallery_conf['subsection_order'] = sort_subsections

# sphinx-reredirects
redirects = {
    'examples/plot_betts2003':
    'beginner/plot_betts2003.html',

    'examples/plot_drone':
    'intermediate/plot_drone.html',

    'examples/plot_one_legged_time_trial':
    'advanced/plot_one_legged_time_trial.html',

    'examples/plot_parallel_park':
    'intermediate/plot_parallel_park.html',

    'examples/plot_pendulum_swing_up_fixed_duration':
    'beginner/plot_pendulum_swing_up_fixed_duration.html',

    'examples/plot_pendulum_swing_up_variable_duration':
    'beginner/plot_pendulum_swing_up_variable_duration.html',

    'examples/plot_sliding_block':
    'beginner/plot_sliding_block.html',

    'examples/plot_two_link_pendulum_on_a_cart':
    'intermediate/plot_two_link_pendulum_on_a_cart.html',

    'examples/plot_vyasarayani':
    'beginner/plot_vyasarayani.html',
}

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'optydoc'


# -- Options for LaTeX output ---------------------------------------------

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
    (master_doc, 'opty.tex', 'opty Documentation',
     'Jason K. Moore', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'opty', 'opty Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'opty', 'opty Documentation',
     author, 'opty', 'One line description of project.',
     'Miscellaneous'),
]
