r"""Fourier-space Green's function for the homogeneous Helmholtz equation.

Extended Summary
----------------
Provides the free-space Green's function of the Helmholtz equation
in reciprocal space, precomputed once per simulation and reused
across all Born iterations. The Green's function is central to the
convergent Born series of Osnabrugge et al. (2016):

.. math::

    \tilde{g}_0(\mathbf{p})
        = \frac{1}{|\mathbf{p}|^2 - k_0^2 - i\varepsilon}

The imaginary shift :math:`-i\varepsilon` moves the pole off the
real axis, ensuring exponential decay in real space with decay
length :math:`\lambda = k_0 / \varepsilon`.

Routine Listings
----------------
:func:`wavenumber_background`
    Background wavenumber :math:`k_0^2` centred between min and
    max of the specimen :math:`k^2`.
:func:`convergence_parameter`
    Convergence parameter :math:`\varepsilon \geq \max|U(\mathbf{r})|`
    guaranteeing :math:`\rho(M) < 1`.
:func:`reciprocal_coords`
    Reciprocal-space coordinate arrays for an isotropic 3-D grid.
:func:`green_function_fourier`
    Green's function
    :math:`\tilde{g}_0(\mathbf{p})` on the 3-D Fourier grid.

Notes
-----
All functions are JIT-compatible and operate on JAX arrays.
Units are Angstroms and inverse Angstroms throughout.
"""

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def wavenumber_background(
        k_squared: Float[Array, "Nx Ny Nz"],
) -> Float[Array, ""]:
    r"""Compute the optimal background wavenumber squared.

    Extended Summary
    ----------------
    Computes :math:`k_0^2` as the midpoint between the minimum
    and maximum of :math:`\operatorname{Re}\{k^2(\mathbf{r})\}`
    over the simulation volume. This choice minimises the
    convergence parameter :math:`\varepsilon` and thereby
    maximises the pseudo-propagation speed
    :math:`2 k_0 / \varepsilon` per iteration:

    .. math::

        k_0^2
            = \frac{\min_{\mathbf{r}}
              \operatorname{Re}\{k^2(\mathbf{r})\}
              + \max_{\mathbf{r}}
              \operatorname{Re}\{k^2(\mathbf{r})\}}{2}

    Implementation Logic
    --------------------
    1. **Extract real part** --
       Take the real part of ``k_squared`` to handle any
       residual imaginary contributions.
    2. **Find extrema** --
       Compute the global minimum and maximum over all voxels.
    3. **Compute midpoint** --
       Return the arithmetic mean as the optimal background
       wavenumber squared.

    Parameters
    ----------
    k_squared : Float[Array, "Nx Ny Nz"]
        Squared wavenumber field :math:`k^2(\mathbf{r})` in
        Angstrom :sup:`-2`, including specimen potential and
        absorbing boundary contributions. Real-valued; complex
        :math:`k^2` is handled after adding the absorbing
        boundary imaginary part.

    Returns
    -------
    k0_squared : Float[Array, ""]
        Background wavenumber squared :math:`k_0^2` in
        Angstrom :sup:`-2`, scalar.

    See Also
    --------
    :func:`convergence_parameter` :
        Uses :math:`k_0^2` to compute the convergence bound.
    :func:`green_function_fourier` :
        Consumes :math:`k_0^2` to build the Green's function.
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
    r"""Compute the convergence parameter from the scattering potential.

    Extended Summary
    ----------------
    Derives :math:`\varepsilon` from the scattering potential
    :math:`U(\mathbf{r}) = k^2(\mathbf{r}) - k_0^2`. The
    sufficient condition for convergence of the Born series is
    (Osnabrugge et al. 2016, Eq. 11):

    .. math::

        \varepsilon \geq \max_{\mathbf{r}} |U(\mathbf{r})|

    A safety factor slightly above unity is applied to ensure
    strict inequality and numerical stability at voxels where
    :math:`|U| = \varepsilon` exactly.

    Implementation Logic
    --------------------
    1. **Compute complex modulus** --
       Evaluate :math:`|U(\mathbf{r})|` pointwise.
    2. **Find global maximum** --
       Take the maximum over all voxels.
    3. **Apply safety factor** --
       Multiply by ``safety_factor`` to guarantee strict
       convergence.

    Parameters
    ----------
    scattering_potential : Complex[Array, "Nx Ny Nz"]
        Scattering potential
        :math:`U(\mathbf{r}) = k^2(\mathbf{r}) - k_0^2` in
        Angstrom :sup:`-2`. Complex-valued when absorbing
        boundary layers are present; the convergence condition
        applies to the complex modulus :math:`|U(\mathbf{r})|`.
    safety_factor : float
        Multiplicative factor applied to
        :math:`\max|U(\mathbf{r})|`. Must be > 1.0. Default
        1.01 provides a 1 % margin above the strict bound.

    Returns
    -------
    epsilon : Float[Array, ""]
        Convergence parameter :math:`\varepsilon` in
        Angstrom :sup:`-2`, scalar. Guaranteed
        :math:`\geq \max|U(\mathbf{r})|`.

    See Also
    --------
    :func:`wavenumber_background` :
        Computes :math:`k_0^2` used to form :math:`U`.
    :func:`green_function_fourier` :
        Consumes :math:`\varepsilon` to build the Green's
        function.
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
    r"""Construct 3-D reciprocal-space coordinate arrays.

    Extended Summary
    ----------------
    Builds coordinate arrays for an isotropic simulation grid in
    units of Angstrom :sup:`-1`, following the FFT frequency
    convention (zero at index 0, negative frequencies in the
    upper half), consistent with ``jnp.fft.fftn``.

    For isotropic spacing :math:`\Delta x`, the reciprocal
    coordinate along axis *n* at index *j* is:

    .. math::

        p_n(j) = 2\pi \cdot \texttt{fftfreq}(N_n,\;
                 d = \Delta x)

    Implementation Logic
    --------------------
    1. **Compute 1-D frequencies** --
       Call ``jnp.fft.fftfreq`` along each axis.
    2. **Scale to radians** --
       Multiply by :math:`2\pi` to convert from
       cycles/Angstrom to radians/Angstrom.
    3. **Broadcast to 3-D** --
       Use ``jnp.meshgrid`` with ``indexing="ij"`` to
       produce full-volume arrays.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Number of voxels ``(Nx, Ny, Nz)`` along each axis.
    grid_spacing_ang : float
        Isotropic voxel size in Angstrom. At 10 pm this is
        0.1 Angstrom.

    Returns
    -------
    px : Float[Array, "Nx Ny Nz"]
        Reciprocal coordinate along *x* in Angstrom :sup:`-1`.
    py : Float[Array, "Nx Ny Nz"]
        Reciprocal coordinate along *y* in Angstrom :sup:`-1`.
    pz : Float[Array, "Nx Ny Nz"]
        Reciprocal coordinate along *z* in Angstrom :sup:`-1`.

    See Also
    --------
    :func:`green_function_fourier` :
        Consumes these coordinates to evaluate :math:`|p|^2`.
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
    r"""Construct the Fourier-space Green's function.

    Extended Summary
    ----------------
    Builds the Green's function of the homogeneous Helmholtz
    equation with a complex wavenumber shift:

    .. math::

        \tilde{g}_0(\mathbf{p})
            = \frac{1}{|\mathbf{p}|^2 - k_0^2 - i\varepsilon}

    The imaginary shift :math:`-i\varepsilon` moves the pole off
    the real axis, ensuring the Green's function decays
    exponentially in real space with decay length
    :math:`\lambda = k_0 / \varepsilon`. This localisation is the
    mathematical basis for the convergence guarantee of the Born
    series: the operator :math:`G` has finite norm, enabling the
    preconditioned iteration
    :math:`M = \gamma G V - \gamma + 1` to satisfy
    :math:`\rho(M) < 1`.

    The Green's function is precomputed once and reused for every
    Born iteration via Fourier-space convolution:

    .. math::

        G[f](\mathbf{r})
            = \mathrm{IFFT}\!\bigl[
              \tilde{g}_0 \cdot \mathrm{FFT}[f]
              \bigr]

    No singularity handling is required: the
    :math:`i\varepsilon` term ensures the denominator is never
    zero for real :math:`\mathbf{p}`.

    Implementation Logic
    --------------------
    1. **Build reciprocal grids** --
       Call :func:`reciprocal_coords` for
       :math:`p_x, p_y, p_z`.
    2. **Compute squared magnitude** --
       :math:`|\mathbf{p}|^2 = p_x^2 + p_y^2 + p_z^2`.
    3. **Evaluate Green's function** --
       Pointwise division
       :math:`1 / (|\mathbf{p}|^2 - k_0^2 - i\varepsilon)`.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Number of voxels ``(Nx, Ny, Nz)`` along each axis.
    grid_spacing_ang : float
        Isotropic voxel size in Angstrom.
    k0_squared : Float[Array, ""]
        Background wavenumber squared :math:`k_0^2` in
        Angstrom :sup:`-2`.
    epsilon : Float[Array, ""]
        Convergence parameter :math:`\varepsilon` in
        Angstrom :sup:`-2`. Must satisfy
        :math:`\varepsilon \geq \max|U(\mathbf{r})|`.

    Returns
    -------
    g0_tilde : Complex[Array, "Nx Ny Nz"]
        Green's function in Fourier space in
        Angstrom :sup:`2`. Complex128. The real part captures
        propagation; the imaginary part captures exponential
        decay enforced by :math:`\varepsilon`.

    See Also
    --------
    :func:`wavenumber_background` :
        Computes :math:`k_0^2`.
    :func:`convergence_parameter` :
        Computes :math:`\varepsilon`.
    :func:`reciprocal_coords` :
        Builds the reciprocal-space grids consumed here.
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

__all__: list[str] = [
    "convergence_parameter",
    "green_function_fourier",
    "reciprocal_coords",
    "wavenumber_background",
]
