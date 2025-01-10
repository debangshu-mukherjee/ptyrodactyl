from functools import partial
from typing import NamedTuple, Tuple
from typing_extensions import TypeAlias

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, Num, jaxtyped

jax.config.update("jax_enable_x64", True)

num_type: TypeAlias = int | float  # Non-JAX scalar number
scalar_number: TypeAlias = (
    int | float | Num[Array, ""]
)  # Scalar number that is outputted from a JAX function

class LensParams(NamedTuple):
    """PyTree structure for lens parameters"""
    focal_length: Float[Array, ""]  # meters
    diameter: Float[Array, ""]      # meters
    n: Float[Array, ""]            # refractive index
    wavelength: Float[Array, ""]   # meters
    num_points: Int[Array, ""]  # grid points

class GridParams(NamedTuple):
    """PyTree structure for computational grid parameters"""
    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    phase_profile: Float[Array, "H W"]
    transmission: Float[Array, "H W"]

@jaxtyped(typechecker=typechecker)
@jax.jit
def create_spatial_grid(
    diameter: scalar_number, 
    num_points: int | Int[Array, ""]
) -> Tuple[Float[Array, "a a"] , Float[Array, "a a"]]:
    """Pure function to create spatial grid coordinates"""
    x: Float[Array, "a"] = jnp.linspace(-diameter / 2, diameter / 2, num_points)
    y: Float[Array, "a"] = jnp.linspace(-diameter / 2, diameter / 2, num_points)
    return jnp.meshgrid(x, y)


@jax.jit
def calculate_radius_curvature(
    focal_length: Float[Array, ""], 
    n: Float[Array, ""]
    ) -> Float[Array, ""]:
    """Pure function to calculate radius of curvature"""
    return (n - 1) * focal_length


@jax.jit
def calculate_thickness_profile(
    r: jnp.ndarray, R: float, diameter: float
) -> jnp.ndarray:
    """Pure function to calculate lens thickness profile"""
    return jnp.where(
        r <= diameter / 2, -(R - jnp.sqrt(jnp.maximum(R**2 - r**2, 0.0))), 0.0
    )


@jax.jit
def calculate_transmission_mask(r: jnp.ndarray, diameter: float) -> jnp.ndarray:
    """Pure function to calculate transmission mask"""
    return (r <= diameter / 2).astype(float)


@jax.jit
def calculate_phase_profile(
    X: jnp.ndarray, Y: jnp.ndarray, params: LensParams
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pure function to calculate phase profile and transmission mask"""
    r = jnp.sqrt(X**2 + Y**2)
    R = calculate_radius_curvature(params.focal_length, params.n)

    thickness = calculate_thickness_profile(r, R, params.diameter)
    transmission = calculate_transmission_mask(r, params.diameter)

    k = 2 * jnp.pi / params.wavelength
    phase_profile = k * (params.n - 1) * thickness

    return phase_profile, transmission


@partial(jax.jit, static_argnums=(1,))
def initialize_grid(params: LensParams, device: str = "cpu") -> GridParams:
    """Pure function to initialize computational grid and lens properties"""
    X, Y = create_spatial_grid(params.diameter, params.num_points)
    phase_profile, transmission = calculate_phase_profile(X, Y, params)

    return GridParams(X=X, Y=Y, phase_profile=phase_profile, transmission=transmission)


@jax.jit
def propagate_wavefront(incident_wave: jnp.ndarray, grid: GridParams) -> jnp.ndarray:
    """Pure function to propagate wavefront through lens"""
    return incident_wave * grid.transmission * jnp.exp(1j * grid.phase_profile)


@jax.jit
def calculate_focal_spot(wavefront: jnp.ndarray) -> jnp.ndarray:
    """Pure function to calculate focal spot intensity"""
    return jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(wavefront))) ** 2


# Composition functions for common operations
@jax.jit
def wavefront_to_focal_spot(
    incident_wave: jnp.ndarray, grid: GridParams
) -> jnp.ndarray:
    """Function composition of propagation and focal spot calculation"""
    return calculate_focal_spot(propagate_wavefront(incident_wave, grid))


# Vectorized operations for batch processing
batch_propagate = jax.vmap(propagate_wavefront, in_axes=(0, None))
batch_focal_spot = jax.vmap(calculate_focal_spot, in_axes=0)


# Example analysis functions
def analyze_wavefront_sensitivity(
    incident_wave: jnp.ndarray,
    grid: GridParams,
    param_range: jnp.ndarray,
    param_update_fn,
) -> jnp.ndarray:
    """Analyze focal spot sensitivity to parameter variations"""
    return jax.vmap(
        lambda p: wavefront_to_focal_spot(incident_wave, param_update_fn(grid, p))
    )(param_range)
