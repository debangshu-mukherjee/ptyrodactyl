r"""Private PyTree algebra helpers for Jacobian modules.

Extended Summary
----------------
Provides small algebraic operations over matching JAX PyTrees so
solvers, gauge analysis, block updates, and Fisher utilities use one
shared implementation.

Routine Listings
----------------
:func:`_tree_add`
    Element-wise addition of two PyTrees.
:func:`_tree_dot`
    Inner product between two PyTrees.
:func:`_tree_norm`
    L2 norm of a PyTree.
:func:`_tree_scalar_mul`
    Scalar multiplication of a PyTree.
:func:`_tree_sub`
    Element-wise subtraction of two PyTrees.
:func:`_tree_zeros_like`
    Zero-valued PyTree matching an input structure.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PyTree, jaxtyped


@jaxtyped(typechecker=beartype)
def _tree_add(
    tree_a: PyTree,
    tree_b: PyTree,
) -> PyTree:
    """Element-wise addition of two PyTrees.

    Parameters
    ----------
    tree_a : PyTree
        First PyTree operand.
    tree_b : PyTree
        Second PyTree operand with same structure as
        *tree_a*.

    Returns
    -------
    result : PyTree
        PyTree with element-wise sum of leaves.
    """
    result: PyTree = jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)
    return result


@jaxtyped(typechecker=beartype)
def _tree_dot(
    tree_a: PyTree,
    tree_b: PyTree,
) -> Float[Array, ""]:
    """Compute inner product between two PyTrees.

    Parameters
    ----------
    tree_a : PyTree
        First PyTree operand.
    tree_b : PyTree
        Second PyTree operand with same structure as
        *tree_a*.

    Returns
    -------
    result : Float[Array, ""]
        Sum of element-wise products across all leaves.
    """
    leaves_a, _ = jax.tree_util.tree_flatten(tree_a)
    leaves_b, _ = jax.tree_util.tree_flatten(tree_b)
    products: list = [
        jnp.sum(a * b) for a, b in zip(leaves_a, leaves_b, strict=False)
    ]
    result: Float[Array, ""] = jnp.sum(jnp.array(products))
    return result


@jaxtyped(typechecker=beartype)
def _tree_norm(
    tree: PyTree,
) -> Float[Array, ""]:
    """Compute L2 norm of a PyTree.

    Parameters
    ----------
    tree : PyTree
        Input PyTree.

    Returns
    -------
    result : Float[Array, ""]
        Square root of sum of squared elements across all leaves.
    """
    dot_product: Float[Array, ""] = _tree_dot(tree, tree)
    result: Float[Array, ""] = jnp.sqrt(dot_product)
    return result


@jaxtyped(typechecker=beartype)
def _tree_scalar_mul(
    scalar: Float[Array, ""],
    tree: PyTree,
) -> PyTree:
    """Multiply all leaves of a PyTree by a scalar.

    Parameters
    ----------
    scalar : Float[Array, ""]
        Scalar multiplier.
    tree : PyTree
        PyTree to scale.

    Returns
    -------
    result : PyTree
        Scaled PyTree.
    """
    result: PyTree = jax.tree_util.tree_map(lambda x: scalar * x, tree)
    return result


@jaxtyped(typechecker=beartype)
def _tree_sub(
    tree_a: PyTree,
    tree_b: PyTree,
) -> PyTree:
    """Element-wise subtraction of two PyTrees.

    Parameters
    ----------
    tree_a : PyTree
        First PyTree operand.
    tree_b : PyTree
        Second PyTree operand to subtract from *tree_a*.

    Returns
    -------
    result : PyTree
        PyTree with element-wise difference of leaves.
    """
    result: PyTree = jax.tree_util.tree_map(lambda a, b: a - b, tree_a, tree_b)
    return result


@jaxtyped(typechecker=beartype)
def _tree_zeros_like(
    tree: PyTree,
) -> PyTree:
    """Create a PyTree of zeros matching the input structure.

    Parameters
    ----------
    tree : PyTree
        Template PyTree whose leaf shapes and dtypes are copied.

    Returns
    -------
    zeros : PyTree
        PyTree of zeros with the same structure as *tree*.
    """
    zeros: PyTree = jax.tree_util.tree_map(jnp.zeros_like, tree)
    return zeros


__all__: list[str] = [
    "_tree_add",
    "_tree_dot",
    "_tree_norm",
    "_tree_scalar_mul",
    "_tree_sub",
    "_tree_zeros_like",
]
