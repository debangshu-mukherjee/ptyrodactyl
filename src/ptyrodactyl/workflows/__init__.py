"""Combined workflows for electron microscopy simulations.

Extended Summary
----------------
This package implements combined workflows, which takes in multiple
functions together and gives you a big global function.

Submodules
----------
stem_4d
    High-level workflows combining simulation steps for common use cases
    such as simulating 4D-STEM data from CrystalData.

Routine Listings
----------------
crystal2stem4d : function
    4D-STEM simulation from CrystalData with automatic sharding.
crystal2stem4d_tiled : function
    Tiled 4D-STEM simulation for large samples with fixed memory per tile.

"""

from .stem_4d import crystal2stem4d, crystal2stem4d_tiled

__all__: list[str] = [
    "crystal2stem4d",
    "crystal2stem4d_tiled",
]
