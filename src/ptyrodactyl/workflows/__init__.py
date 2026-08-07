"""Combined workflows for electron microscopy simulations.

Extended Summary
----------------
This package implements combined workflows, which takes in
multiple functions together and gives you a big global function.

The submodules are organized as follows:

- :mod:`stem_4d`
    High-level workflows for electron microscopy simulations.

Routine Listings
----------------
:func:`crystal2stem4d`
    4D-STEM simulation from crystal data with automatic sharding.
:func:`crystal2stem4d_tiled`
    Tiled 4D-STEM simulation for arbitrarily large samples.

Notes
-----
Workflows are convenience functions that chain together
lower-level simulation functions from the ``simulations``
and ``atom_potentials`` modules. See the ``stem_4d``
submodule for implementation details.
"""

from .stem_4d import crystal2stem4d, crystal2stem4d_tiled

__all__: list[str] = [
    "crystal2stem4d",
    "crystal2stem4d_tiled",
]
