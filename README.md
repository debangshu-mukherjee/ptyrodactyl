[![PyPI Downloads](https://static.pepy.tech/badge/ptyrodactyl)](https://pepy.tech/projects/ptyrodactyl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ptyrodactyl.svg)](https://badge.fury.io/py/ptyrodactyl)
[![Documentation Status](https://readthedocs.org/projects/ptyrodactyl/badge/?version=latest)](https://ptyrodactyl.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/905915185.svg)](https://doi.org/10.5281/zenodo.14861992)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/ptyrodactyl/branch/main/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/ptyrodactyl)
[![Tests](https://github.com/debangshu-mukherjee/ptyrodactyl/workflows/Tests/badge.svg)](https://github.com/debangshu-mukherjee/ptyrodactyl/actions)

# Ptychography through Differentiable Programming

The aim of this project is to write the _forward_ problem: aka writing the microscope data generation, both for electron and optical microscopes in [JAX](https://github.com/google/jax) so that it's end to end differentiable and using this differentiability to run modern optimizers such as [Adam](
https://doi.org/10.48550/arXiv.1412.6980
) and [Adagrad](https://arxiv.org/abs/2003.02395) to solve for the inverse problem - which is ptychography in our case.

All the work here is in Python, performed on a x64 based processor workstation, running Ubuntu Linux 22.04. However, none of the packages here have Linux as a dependency, so this should run in Windows/Mac environments too -- just the path commands may be a bit different.

This will install the package as `ptyrodactyl`, which is the package that all the codes are.


The codes themselves are in the _src_ directory, following the modern toml convention as the _ptyrodactyl_ folder.