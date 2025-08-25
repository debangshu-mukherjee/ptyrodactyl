"""Utility module for handling type checking and JAX decorators.

This module provides conditional decorators that can be disabled during
documentation builds to allow Sphinx autodoc to properly introspect functions.

Notes
-----
The decorators in this module check the BUILDING_DOCS environment variable
to determine whether to apply type checking and JAX transformations. During
documentation builds, the decorators become no-ops to allow Sphinx to
introspect the functions.
"""

import os
from collections.abc import Callable

from beartype.typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

BUILDING_DOCS = os.environ.get("BUILDING_DOCS", "").lower() in ("1", "true", "yes")

if BUILDING_DOCS:

    def jaxtyped(typechecker: Any = None) -> Callable[[F], F]:
        """No-op decorator for documentation builds.

        Parameters
        ----------
        typechecker : Any, optional
            Type checker to use (ignored in documentation builds)

        Returns
        -------
        Callable[[F], F]
            Decorator function that returns the input function unchanged
        """

        def decorator(func: F) -> F:
            return func

        return decorator

    def beartype(func: F) -> F:
        """No-op decorator for documentation builds.

        Parameters
        ----------
        func : F
            Function to decorate

        Returns
        -------
        F
            The input function unchanged
        """
        return func

else:
    # Normal runtime - use actual decorators
    try:
        from beartype import beartype
        from jaxtyping import jaxtyped
    except ImportError:

        def jaxtyped(typechecker: Any = None) -> Callable[[F], F]:
            """Fallback no-op decorator when jaxtyping is not installed.

            Parameters
            ----------
            typechecker : Any, optional
                Type checker to use (ignored when package not installed)

            Returns
            -------
            Callable[[F], F]
                Decorator function that returns the input function unchanged
            """

            def decorator(func: F) -> F:
                return func

            return decorator

        def beartype(func: F) -> F:
            """Fallback no-op decorator when beartype is not installed.

            Parameters
            ----------
            func : F
                Function to decorate

            Returns
            -------
            F
                The input function unchanged
            """
            return func


__all__ = ["jaxtyped", "beartype", "BUILDING_DOCS"]
