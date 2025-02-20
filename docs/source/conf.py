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

sys.path.insert(0, os.path.abspath("../../src/"))
import icon_registration
import matplotlib.sphinxext.plot_directive


# -- Project information -----------------------------------------------------

project = "ICON"
copyright = "2022, Hastings Greer"
author = "Hastings Greer"

# The full version, including alpha/beta/rc tags
release = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
]

plot_html_show_source_link = False
autodoc_inherit_docstrings = False

import inspect


def line_number(info):
    # linkcode gets angry about namedtuple lol
    if "ICONLoss" in info["fullname"]:
        return 22
    if "BendingLoss" in info["fullname"]:
        return 235
    mod = icon_registration
    for elem in info["module"].split(".")[1:]:
        mod = getattr(mod, elem)
    thing = getattr(mod, info["fullname"].split(".")[0])
    return inspect.getsourcelines(thing)[1]


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if "." in info["fullname"]:
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    number = line_number(info)
    return "https://github.com/uncbiag/ICON/blob/master/src/%s.py#L%d" % (
        filename,
        number,
    )

def setup(app):
    if not ("READTHEDOCS" in os.environ):
        app.add_js_file("live.js")


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
