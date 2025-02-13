import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("./_ext"))

project = "ptyrodactyl"
copyright = "2024"
author = "Debangshu Mukherjee"

# The full version, including alpha/beta/rc tags
release = "2025.01.28"

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

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Napoleon settings for custom docstring format
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

# Add nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue building even if there are errors

# Specify the path to the tutorials directory relative to conf.py
nbsphinx_notebooks_dir = "../../tutorials"

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Intersphinx mapping for external references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Custom CSS to improve type hint rendering
html_css_files = [
    "custom.css",
]

html_theme = "sphinx_rtd_theme"

# Add these configurations to exclude imported items and type hints
autodoc_default_options = {"exclude-members": "Float, Array, Int, Num, beartype"}

# Add nitpicky mode to catch any broken references
nitpicky = True

# Add type aliases to handle jaxtyping types cleanly
napoleon_type_aliases = {
    # Add common type hints you want to simplify
    'Float[Array, ""]': "array",
    'Float[Array, "3"]': "array",
    'Float[Array, "3 3"]': "array",
    'Int[Array, ""]': "array",
    'Num[Array, "*"]': "array",
}

# Add any modules to ignore when warning about missing references
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


# Define what to skip during documentation
def skip_member(app, what, name, obj, skip, options):
    # Skip all imported jaxtyping and beartype items
    if name in ["Float", "Array", "Int", "Num", "Bool", "beartype"]:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
