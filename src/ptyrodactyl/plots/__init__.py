"""Plotting and visualization helper exports.

Extended Summary
----------------
This package owns presentation transforms and visualization colormaps.

Routine Listings
----------------
:func:`clip_cbed`
    Clip CBED patterns to mrad extent and resize to target
    shape.
:func:`contrast_stretch`
    Rescale image intensity between specified percentiles.
:func:`create_phosphor_colormap`
    Create custom colormap simulating phosphor screen appearance.
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
