"""Combined workflows for electron microscopy simulations.

Extended Summary
----------------
This package implements combined workflows, which takes in multiple
functions together and gives you a big global function.

Submodules
----------
stem_4d
    High-level workflows combining simulation steps for common use cases
    such as simulating 4D-STEM data from crystal structure files.

Routine Listings
----------------
crystal2stem4d : function
    Smart dispatcher for 4D-STEM simulation, auto-selects implementation.
crystal2stem4d_parallel : function
    Parallel sharded 4D-STEM simulation for large-scale computations.
crystal2stem4d_single : function
    Single-device 4D-STEM simulation from crystal structure file.

"""

from .stem_4d import (
    crystal2stem4d,
    crystal2stem4d_parallel,
    crystal2stem4d_single,
)

__all__: list[str] = [
    "crystal2stem4d",
    "crystal2stem4d_parallel",
    "crystal2stem4d_single",
]
