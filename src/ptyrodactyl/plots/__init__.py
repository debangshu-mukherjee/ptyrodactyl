"""Plotting and visualization helper exports.

Extended Summary
----------------
This package owns presentation transforms and visualization colormaps.

The submodules are organized as follows:

- :mod:`figuring`
    Plotting-oriented transforms and colormap helpers.

Routine Listings
----------------
:func:`clip_cbed`
    Clip CBED pattern to mrad extent and resize.
:func:`contrast_stretch`
    Rescale image intensity between specified percentiles.
:func:`create_phosphor_colormap`
    Create a custom colormap that simulates a phosphor screen appearance.

"""

from .figuring import (
    clip_cbed,
    contrast_stretch,
    create_phosphor_colormap,
)

__all__: list[str] = [
    "clip_cbed",
    "contrast_stretch",
    "create_phosphor_colormap",
]
