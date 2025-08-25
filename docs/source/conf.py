import os
import sys
from datetime import datetime

import tomllib
from sphinx.application import Sphinx

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["BUILDING_DOCS"] = "1"

project_root = os.path.abspath("../..")
src_path = os.path.join(project_root, "src")

sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

print(f"Added to sys.path: {src_path}")

pyproject_path = os.path.join(project_root, "pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]

authors_data = pyproject_data["project"]["authors"]
author = (
    authors_data[0]["name"] if isinstance(authors_data[0], dict) else authors_data[0]
)

project_copyright = f"{datetime.now().year}, {author}"
release = pyproject_data["project"]["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": None,
    ".md": None,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3399ff",
        "color-brand-content": "#3399ff",
    },
    "sidebar_hide_name": False,
}

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

napoleon_custom_sections = [
    ("Algorithm", "notes_style"),
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

autodoc_mock_imports = [
    "pandas",
    "scipy",
    "scipy.interpolate",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "gemmi",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "ignore-module-all": False,
}

autodoc_typehints = (
    "none"
)
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
python_use_unqualified_type_names = True
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = False
typehints_use_signature = False
typehints_use_signature_return = False
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True

nitpicky = False

napoleon_type_aliases = {
    'Float[Array, " "]': "scalar float",
    'Float[Array, " 3"]': "3D float array",
    'Float[Array, " 3 3"]': "3x3 float array",
    'Float[Array, " M 3"]': "Mx3 float array",
    'Float[Array, " N 3"]': "Nx3 float array",
    'Float[Array, " N 4"]': "Nx4 float array",
    'Int[Array, " "]': "scalar int",
    'Int[Array, " N"]': "N int array",
    'Num[Array, " N"]': "N numeric array",
    'Bool[Array, " "]': "scalar bool",
    "scalar_float": "float",
    "scalar_int": "int",
    "scalar_num": "numeric",
}

typehints_defaults = "comma"
typehints_type_aliases = napoleon_type_aliases

nitpick_ignore = [
    ("py:class", "Float"),
    ("py:class", "Array"),
    ("py:class", "Int"),
    ("py:class", "Num"),
    ("py:class", "Bool"),
    ("py:class", "jaxtyping.Float"),
    ("py:class", "jaxtyping.Array"),
    ("py:class", "beartype"),
    ("py:class", "jaxtyped"),
    ("py:obj", "beartype"),
    ("py:obj", "jaxtyped"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

html_css_files = ["custom.css"]
html_js_files = ["custom.js"]


def skip_member(app, what, name, obj, skip, options):
    """Skip specific members in documentation."""
    skip_names = [
        "Float",
        "Array",
        "Int",
        "Num",
        "Bool",
        "beartype",
        "jaxtyped",
        "tree_flatten",
        "tree_unflatten",
    ]
    if name in skip_names:
        return True
    if name.startswith("_") and not name.startswith("__"):
        return True
    return skip


def process_signature(app, what, name, obj, options, signature, return_annotation):
    """Process signatures to handle jaxtyping annotations."""
    if signature:
        signature = signature.replace('Float[Array, " ', "FloatArray[")
        signature = signature.replace('Int[Array, " ', "IntArray[")
        signature = signature.replace('Bool[Array, " ', "BoolArray[")
        signature = signature.replace('Num[Array, " ', "NumArray[")
        signature = signature.replace('"]', "]")
    return signature, return_annotation


def setup(app: Sphinx) -> None:
    app.connect("autodoc-skip-member", skip_member)
    app.connect("autodoc-process-signature", process_signature)
