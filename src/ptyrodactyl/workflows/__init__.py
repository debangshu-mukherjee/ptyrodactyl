"""Combined workflows for electron microscopy simulations.

Extended Summary
----------------
This package implements combined workflows, which takes in multiple
functions together and gives you a big global function.

Submodules
----------
stem_4d
    High-level workflows combining simulation steps for common use cases
    such as simulating 4D-STEM data from XYZ structure files.

Routine Listings
----------------
xyz_4dstem : function
    Smart dispatcher for 4D-STEM simulation, auto-selects implementation.
xyz_4dstem_parallel : function
    Parallel sharded 4D-STEM simulation for large-scale computations.
xyz_4dstem_single : function
    Single-device 4D-STEM simulation from XYZ structure file.

"""

from .stem_4d import xyz_4dstem, xyz_4dstem_parallel, xyz_4dstem_single

__all__: list[str] = [
    "xyz_4dstem",
    "xyz_4dstem_parallel",
    "xyz_4dstem_single",
]
