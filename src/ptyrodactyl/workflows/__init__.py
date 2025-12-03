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
xyz_to_4d_stem : function
    Simulates 4D-STEM data from an XYZ structure file.

"""

from .stem_4d import xyz_to_4d_stem

__all__: list[str] = [
    "xyz_to_4d_stem",
]
