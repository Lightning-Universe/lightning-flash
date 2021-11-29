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
import glob
import os
import shutil
import sys
import warnings
from importlib.util import module_from_spec, spec_from_file_location

import pt_lightning_sphinx_theme

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.join(_PATH_HERE, "..", "..")
_PATH_RAW_NB = os.path.join(_PATH_ROOT, "_notebooks")
sys.path.insert(0, os.path.abspath(_PATH_ROOT))
sys.path.insert(0, os.path.abspath(os.path.join(_PATH_HERE, "..", "extensions")))
sys.path.append(os.path.join(_PATH_RAW_NB, ".actions"))

_SHOULD_COPY_NOTEBOOKS = True

try:
    from helpers import HelperCLI
except Exception:
    _SHOULD_COPY_NOTEBOOKS = False
    warnings.warn("To build the code, please run: `git submodule update --init --recursive`", stacklevel=2)


def _load_py_module(fname, pkg="flash"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


try:
    from flash import __about__ as about
    from flash.core.utilities import providers

except ModuleNotFoundError:

    about = _load_py_module("__about__.py")
    providers = _load_py_module("core/utilities/providers.py")

SPHINX_MOCK_REQUIREMENTS = int(os.environ.get("SPHINX_MOCK_REQUIREMENTS", True))

html_favicon = "_static/images/icon.svg"

# -- Project information -----------------------------------------------------

project = "Flash"
copyright = "2020-2021, PyTorch Lightning"
author = "PyTorch Lightning"

# -- Project documents -------------------------------------------------------
if _SHOULD_COPY_NOTEBOOKS:
    HelperCLI.copy_notebooks(os.path.join(_PATH_RAW_NB, "flash_tutorials"), _PATH_HERE, "notebooks")


def _transform_changelog(path_in: str, path_out: str) -> None:
    with open(path_in) as fp:
        chlog_lines = fp.readlines()
    # enrich short subsub-titles to be unique
    chlog_ver = ""
    for i, ln in enumerate(chlog_lines):
        if ln.startswith("## "):
            chlog_ver = ln[2:].split("-")[0].strip()
        elif ln.startswith("### "):
            ln = ln.replace("###", f"### {chlog_ver} -")
            chlog_lines[i] = ln
    with open(path_out, "w") as fp:
        fp.writelines(chlog_lines)


generated_dir = os.path.join(_PATH_HERE, "generated")

os.makedirs(generated_dir, exist_ok=True)
# copy all documents from GH templates like contribution guide
for md in glob.glob(os.path.join(_PATH_ROOT, ".github", "*.md")):
    shutil.copy(md, os.path.join(generated_dir, os.path.basename(md)))
# copy also the changelog
_transform_changelog(os.path.join(_PATH_ROOT, "CHANGELOG.md"), os.path.join(generated_dir, "CHANGELOG.md"))

# -- Generate providers ------------------------------------------------------

lines = []
for provider in providers.PROVIDERS:
    lines.append(f"- {str(provider)}\n")

generated_dir = os.path.join("integrations", "generated")
os.makedirs(generated_dir, exist_ok=True)

with open(os.path.join(generated_dir, "providers.rst"), "w") as f:
    f.writelines(sorted(lines, key=str.casefold))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    # 'sphinx.ext.coverage',
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.imgmath",
    "recommonmark",
    # 'sphinx.ext.autosectionlabel',
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx_togglebutton",
    "pt_lightning_sphinx_theme.extensions.lightning_tutorials",
    "autodatasources",
]

# autodoc: Default to members and undoc-members
autodoc_default_options = {"members": True}

# autodoc: Don't inherit docstrings (e.g. for nn.Module.forward)
autodoc_inherit_docstrings = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["generated/PULL_REQUEST_TEMPLATE.md"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

needs_sphinx = "4.0"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "pytorchvideo": ("https://pytorchvideo.readthedocs.io/en/latest/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
    "fiftyone": ("https://voxel51.com/docs/fiftyone/", "fiftyone_objects.inv"),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pt_lightning_sphinx_theme"
html_theme_path = [pt_lightning_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": "https://pytorchlightning.ai",
    "canonical_url": about.__docs_url__,
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = []


def setup(app):
    # this is for hiding doctest decoration,
    # see: http://z4r.github.io/python/2011/12/02/hides-the-prompts-and-output/
    app.add_js_file("copybutton.js")
    app.add_css_file("main.css")


# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule
def _package_list_from_file(pfile):
    assert os.path.isfile(pfile)
    with open(pfile) as fp:
        lines = fp.readlines()
    list_pkgs = []
    for ln in lines:
        found = [ln.index(ch) for ch in list(",=<>#@") if ch in ln]
        pkg = ln[: min(found)] if found else ln
        if pkg.strip():
            list_pkgs.append(pkg.strip())
    return list_pkgs


# define mapping from PyPI names to python imports
PACKAGE_MAPPING = {
    "pytorch-lightning": "pytorch_lightning",
    "scikit-learn": "sklearn",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "rouge-score": "rouge_score",
    "lightning-bolts": "pl_bolts",
    "pytorch-tabnet": "pytorch_tabnet",
    "pyDeprecate": "deprecate",
}
MOCK_PACKAGES = ["numpy", "PyYAML", "tqdm"]
if SPHINX_MOCK_REQUIREMENTS:
    # mock also base packages when we are on RTD since we don't install them there
    MOCK_PACKAGES += _package_list_from_file(os.path.join(_PATH_ROOT, "requirements.txt"))
# replace PyPI packages by importing ones
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]

autodoc_mock_imports = MOCK_PACKAGES

# only run doctests marked with a ".. doctest::" directive
doctest_test_doctest_blocks = ""
doctest_global_setup = """
import torch
import pytorch_lightning as pl
import flash
"""
