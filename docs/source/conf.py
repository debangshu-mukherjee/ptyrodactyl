import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("./_ext"))

project = "ptyrodactyl"
copyright = "2025"
author = "Debangshu Mukherjee"

release = "2025.05.10"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
    "param_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True
napoleon_custom_sections = ["Description", "Parameters", "Returns", "Flow"]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

html_css_files = [
    "custom.css",
]

autodoc_default_options = {"exclude-members": "Float, Array, Int, Num, beartype"}

nitpicky = True

napoleon_type_aliases = {
    'Float[Array, ""]': "array",
    'Float[Array, "3"]': "array",
    'Float[Array, "3 3"]': "array",
    'Int[Array, ""]': "array",
    'Num[Array, "*"]': "array",
}

nitpick_ignore = [
    ("py:class", "Float"),
    ("py:class", "Array"),
    ("py:class", "Int"),
    ("py:class", "Num"),
    ("py:class", "Bool"),
    ("py:class", "jaxtyping.Float"),
    ("py:class", "jaxtyping.Array"),
    ("py:class", "jaxtyping.Int"),
    ("py:class", "jaxtyping.Num"),
    ("py:class", "jaxtyping.Bool"),
    ("py:class", "beartype.typing.NamedTuple"),
]