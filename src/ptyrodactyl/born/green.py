"""
Module: lippmann.green
----------------------

Green's function of the homogeneous Helmholtz equation in Fourier space.
Precomputed once per simulation and reused across all Born iterations.
Central to the convergent Born series of Osnabrugge et al. 2016.

Functions
---------
- `wavenumber_background`:
    Background wavenumber k₀ centered between min and max of specimen k².
- `convergence_parameter`:
    Convergence parameter ε ≥ max|k²(r) − k₀²| guaranteeing ρ(M) < 1.
- `reciprocal_coords`:
    Reciprocal space coordinate arrays for an isotropic 3D grid.
- `green_function_fourier`:
    Green's function g̃₀(p) = 1/(|p|² − k₀² − iε) on the 3D Fourier grid.
"""

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def wavenumber_background(
        k_squared: Float[Array, "Nx Ny Nz"],
) -> Float[Array, ""]:
    """
    Description
    ------------
    Compute the optimal background wavenumber k₀² as the midpoint between
    the minimum and maximum of the real part of k²(r) over the simulation
    volume. This choice minimises the convergence parameter ε and thereby
    maximises the pseudo-propagation speed 2k₀/ε per iteration.

    k₀² = (min_r Re{k²(r)} + max_r Re{k²(r)}) / 2

    Parameters
    ------------
    - `k_squared` (Float[Array, "Nx Ny Nz"]):
        Squared wavenumber field k²(r) in Å⁻² including specimen potential
        and absorbing boundary contributions. Real-valued; complex k² is
        handled after adding the absorbing boundary imaginary part.

    Returns
    --------
    - `k0_squared` (Float[Array, ""]):
        Background wavenumber squared k₀² in Å⁻², scalar.

    Flow
    ----
    1. Take real part to handle any residual imaginary contributions
    2. Find global minimum and maximum over all voxels
    3. Return arithmetic mean as optimal background
    """
    k_squared_real: Float[Array, "Nx Ny Nz"] = jnp.real(k_squared)

    k_min: Float[Array, ""] = jnp.min(k_squared_real)
    k_max: Float[Array, ""] = jnp.max(k_squared_real)

    k0_squared: Float[Array, ""] = (k_min + k_max) / 2.0

    return k0_squared


def convergence_parameter(
        scattering_potential: Complex[Array, "Nx Ny Nz"],
        safety_factor: float = 1.01,
) -> Float[Array, ""]:
    """
    Description
    ------------
    Compute the convergence parameter ε from the scattering potential
    U(r) = k²(r) − k₀². The sufficient condition for convergence of the
    Born series is ε ≥ max_r |U(r)| (Osnabrugge et al. 2016, Eq. 11).

    A safety factor slightly above unity is applied to ensure strict
    inequality and numerical stability at voxels where |U| = ε exactly.

    Parameters
    ------------
    - `scattering_potential` (Complex[Array, "Nx Ny Nz"]):
        Scattering potential U(r) = k²(r) − k₀² in Å⁻². Complex-valued
        when absorbing boundary layers are present; the convergence
        condition applies to the complex modulus |U(r)|.
    - `safety_factor` (float):
        Multiplicative factor applied to max|U(r)|. Must be > 1.0.
        Default 1.01 provides a 1% margin above the strict bound.

    Returns
    --------
    - `epsilon` (Float[Array, ""]):
        Convergence parameter ε in Å⁻², scalar. Guaranteed ≥ max|U(r)|.

    Flow
    ----
    1. Compute pointwise complex modulus |U(r)|
    2. Find global maximum over all voxels
    3. Apply safety factor to ensure strict convergence condition
    """
    modulus: Float[Array, "Nx Ny Nz"] = jnp.abs(scattering_potential)

    max_modulus: Float[Array, ""] = jnp.max(modulus)

    epsilon: Float[Array, ""] = safety_factor * max_modulus

    return epsilon


def reciprocal_coords(
        grid_shape: tuple[int, int, int],
        grid_spacing_ang: float,
) -> tuple[
    Float[Array, "Nx Ny Nz"],
    Float[Array, "Nx Ny Nz"],
    Float[Array, "Nx Ny Nz"],
]:
    """
    Description
    ------------
    Construct 3D reciprocal space coordinate arrays for an isotropic
    simulation grid. Coordinates are in units of Å⁻¹ and follow the
    FFT frequency convention (zero at index 0, negative frequencies
    in the upper half of the array), consistent with jnp.fft.fftn.

    For an isotropic grid with spacing Δx, the reciprocal coordinate
    along each axis n at index j is:
        pₙ(j) = 2π · fftfreq(Nₙ, d=Δx)

    Parameters
    ------------
    - `grid_shape` (tuple[int, int, int]):
        Number of voxels (Nx, Ny, Nz) along each axis.
    - `grid_spacing_ang` (float):
        Isotropic voxel size in Ångström. At 10 pm this is 0.1 Å.

    Returns
    --------
    - `px` (Float[Array, "Nx Ny Nz"]):
        Reciprocal coordinate along x axis in Å⁻¹.
    - `py` (Float[Array, "Nx Ny Nz"]):
        Reciprocal coordinate along y axis in Å⁻¹.
    - `pz` (Float[Array, "Nx Ny Nz"]):
        Reciprocal coordinate along z axis in Å⁻¹.

    Flow
    ----
    1. Compute 1D fftfreq arrays along each axis
    2. Scale by 2π to convert cycles/Å to radians/Å
    3. Broadcast to 3D grids via meshgrid with 'ij' indexing
    """
    px_1d: Float[Array, "Nx"] = (
        jnp.fft.fftfreq(grid_shape[0], d=grid_spacing_ang) * 2.0 * jnp.pi
    )
    py_1d: Float[Array, "Ny"] = (
        jnp.fft.fftfreq(grid_shape[1], d=grid_spacing_ang) * 2.0 * jnp.pi
    )
    pz_1d: Float[Array, "Nz"] = (
        jnp.fft.fftfreq(grid_shape[2], d=grid_spacing_ang) * 2.0 * jnp.pi
    )

    px: Float[Array, "Nx Ny Nz"]
    py: Float[Array, "Nx Ny Nz"]
    pz: Float[Array, "Nx Ny Nz"]
    px, py, pz = jnp.meshgrid(px_1d, py_1d, pz_1d, indexing="ij")

    return px, py, pz


def green_function_fourier(
        grid_shape: tuple[int, int, int],
        grid_spacing_ang: float,
        k0_squared: Float[Array, ""],
        epsilon: Float[Array, ""],
) -> Complex[Array, "Nx Ny Nz"]:
    """
    Description
    ------------
    Construct the Fourier-space Green's function of the homogeneous
    Helmholtz equation with complex wavenumber:

        g̃₀(p) = 1 / (|p|² − k₀² − iε)

    The imaginary shift −iε moves the pole off the real axis, ensuring
    the Green's function decays exponentially in real space with decay
    length λ_decay = k₀/ε. This localisation is the mathematical basis
    for the convergence guarantee of the Born series: the Green's function
    operator G has finite operator norm, enabling the preconditioned
    iteration M = γGV − γ + 1 to satisfy ρ(M) < 1.

    The Green's function is precomputed once and reused for every Born
    iteration step via Fourier-space convolution: G[f](r) = IFFT[g̃₀ · FFT[f]].

    No singularity handling is required: the iε term ensures the
    denominator is never zero for real p.

    Parameters
    ------------
    - `grid_shape` (tuple[int, int, int]):
        Number of voxels (Nx, Ny, Nz) along each axis.
    - `grid_spacing_ang` (float):
        Isotropic voxel size in Ångström.
    - `k0_squared` (Float[Array, ""]):
        Background wavenumber squared k₀² in Å⁻².
    - `epsilon` (Float[Array, ""]):
        Convergence parameter ε in Å⁻². Must satisfy ε ≥ max|U(r)|.

    Returns
    --------
    - `g0_tilde` (Complex[Array, "Nx Ny Nz"]):
        Green's function in Fourier space in Å². Complex128.
        The real part captures propagation; imaginary part captures
        the exponential decay enforced by ε.

    Flow
    ----
    1. Construct reciprocal coordinate grids
    2. Compute |p|² = px² + py² + pz²
    3. Evaluate 1 / (|p|² − k₀² − iε) pointwise
    """
    px: Float[Array, "Nx Ny Nz"]
    py: Float[Array, "Nx Ny Nz"]
    pz: Float[Array, "Nx Ny Nz"]
    px, py, pz = reciprocal_coords(grid_shape, grid_spacing_ang)

    p_squared: Float[Array, "Nx Ny Nz"] = px**2 + py**2 + pz**2

    denominator: Complex[Array, "Nx Ny Nz"] = (
        p_squared - k0_squared - 1j * epsilon
    )

    g0_tilde: Complex[Array, "Nx Ny Nz"] = 1.0 / denominator

    return g0_tilde