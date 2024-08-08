# Ptychography through Differentiable Programming

The aim of this project is to write the _forward_ problem: aka writing the microscope data generation, both for electron and optical microscopes in [JAX](https://github.com/google/jax) so that it's end to end differentiable and using this differentiability to run modern optimizers such as [Adam](
https://doi.org/10.48550/arXiv.1412.6980
) and [Adagrad](https://arxiv.org/abs/2003.02395) to solve for the inverse problem - which is ptychography in our case.

All the work here is in Python, performed on a x64 based processor workstation, running Ubuntu Linux 22.04. However, none of the packages here have Linux as a dependency, so this should run in Windows/Mac environments too -- just the path commands may be a bit different.

This will install the package as `ptyrodactyl`, which is the package that all the codes are.


The codes themselves are in the _src_ directory, following the modern toml convention as the _ptyrodactyl_ folder.