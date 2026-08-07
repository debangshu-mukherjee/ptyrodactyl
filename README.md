[![PyPI version](https://badge.fury.io/py/ptyrodactyl.svg)](https://badge.fury.io/py/ptyrodactyl)
[![Documentation Status](https://readthedocs.org/projects/ptyrodactyl/badge/?version=latest)](https://ptyrodactyl.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/debangshu-mukherjee/ptyrodactyl/actions/workflows/tests.yml/badge.svg)](https://github.com/debangshu-mukherjee/ptyrodactyl/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/debangshu-mukherjee/ptyrodactyl/blob/main/LICENSE.md)
[![DOI](https://zenodo.org/badge/905915185.svg)](https://doi.org/10.5281/zenodo.14861992)

# ptyrodactyl

Ptyrodactyl composes differentiable electron-scattering forward models in
[JAX](https://github.com/jax-ml/jax). It keeps physical reductions explicit so
inverse methods can optimize declared microscope and specimen parameters.

The package supports Python 3.12 through 3.14. CPU execution is the reference
development path. A CUDA-capable GPU is optional.

## Installation

Install the published package from PyPI:

```bash
python -m pip install ptyrodactyl
```

For development, clone the repository and synchronize the locked environment:

```bash
git clone https://github.com/debangshu-mukherjee/ptyrodactyl.git
cd ptyrodactyl
uv sync --extra dev
```

Use `uv sync --extra dev_cuda` only on a supported Linux CUDA system.

## Documentation and contributions

Read the [documentation](https://ptyrodactyl.readthedocs.io/) for package
organization, API details, and tutorials. Read
[CONTRIBUTING.md](https://github.com/debangshu-mukherjee/ptyrodactyl/blob/main/CONTRIBUTING.md)
before changing code, tests, documentation, or packaging.

The project uses the MIT License. See
[LICENSE.md](https://github.com/debangshu-mukherjee/ptyrodactyl/blob/main/LICENSE.md).
