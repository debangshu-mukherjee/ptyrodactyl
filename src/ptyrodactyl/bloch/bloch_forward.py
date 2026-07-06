"""Bloch-wave forward solver for dynamical electron diffraction.

Extended Summary
----------------
This module provides a differentiable Bloch-wave cross-check for the
Convergent Born Series forward model. The scattering matrix is evaluated by a
matrix exponential of the structure matrix rather than by eigendecomposition,
which keeps propagation JIT-compatible and robust at degenerate zone-axis
symmetries.

Routine Listings
----------------
``structure_matrix``
    Assemble the dynamical matrix from Fourier potentials and excitation
    errors.
``excitation_errors``
    Compute Ewald-sphere excitation errors for reflected beams.
``scattering_matrix``
    Propagate beam amplitudes through a slab by matrix exponential.
``bloch_beam_amplitudes``
    Compute amplitudes at one thickness from an incident-beam condition.
``bloch_thickness_series``
    Compute amplitudes over uniform thickness steps with ``lax.scan``.
``two_beam_pendellosung``
    Evaluate the analytic two-beam Pendellosung solution.
``extinction_distance``
    Compute the two-beam extinction distance.
``fourier_potential_from_grid``
    Sample Fourier potential coefficients from a real-space grid.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, Int, jaxtyped

MOTT_BETHE_VOLT_ANGSTROM_SQ: float = 47.87801


@jaxtyped(typechecker=beartype)
def excitation_errors(
    reflection_vectors: Float[Array, "n_beams 3"],
    tilt_vector: Float[Array, "3"],
    wavenumber: Float[Array, ""],
) -> Float[Array, "n_beams"]:
    """Compute Ewald-sphere excitation errors.

    Parameters
    ----------
    reflection_vectors
        Reciprocal-lattice vectors ``g`` in inverse Angstroms.
    tilt_vector
        Incident-beam propagation direction, dimensionless.
    wavenumber
        Incident wavenumber magnitude ``k0 = 2 pi / lambda`` in radians per
        Angstrom.

    Returns
    -------
    Float[Array, "n_beams"]
        Excitation error ``s_g`` per beam in inverse Angstroms.

    Notes
    -----
    Uses ``s_g = -(2 k_vec . g + |g|^2) / (2 |k_vec|)`` with the
    normalized tilted incident direction. The origin reflection therefore has
    zero excitation error.

    :see: :class:`~.test_bloch_forward.TestExcitationErrors`
    """
    incident_wavevector: Float[Array, "3"] = (
        wavenumber * tilt_vector / jnp.linalg.norm(tilt_vector)
    )
    dot_terms: Float[Array, "n_beams"] = (
        reflection_vectors @ incident_wavevector
    )
    squared_lengths: Float[Array, "n_beams"] = jnp.sum(
        reflection_vectors * reflection_vectors, axis=1
    )
    excitation: Float[Array, "n_beams"] = -(
        2.0 * dot_terms + squared_lengths
    ) / (2.0 * wavenumber)
    return excitation


@jaxtyped(typechecker=beartype)
def structure_matrix(
    fourier_potential: Complex[Array, "n_beams n_beams"],
    excitation: Float[Array, "n_beams"],
    wavelength: Float[Array, ""],
) -> Complex[Array, "n_beams n_beams"]:
    """Assemble the Bloch dynamical structure matrix.

    Parameters
    ----------
    fourier_potential
        Matrix of Fourier potential coefficients ``U_{g-h}`` in inverse
        Angstroms squared.
    excitation
        Excitation error ``s_g`` per beam in inverse Angstroms.
    wavelength
        Electron wavelength ``lambda`` in Angstroms.

    Returns
    -------
    Complex[Array, "n_beams n_beams"]
        Structure matrix ``A`` in inverse Angstroms.

    Notes
    -----
    Off-diagonal entries use ``A_gh = lambda U_{g-h} / (4 pi)`` and the
    diagonal is replaced by the excitation errors.

    :see: :class:`~.test_bloch_forward.TestStructureMatrix`
    """
    off_diagonal: Complex[Array, "n_beams n_beams"] = (
        fourier_potential * wavelength / (4.0 * jnp.pi)
    )
    diagonal_removed: Complex[Array, "n_beams n_beams"] = (
        off_diagonal - jnp.diag(jnp.diag(off_diagonal))
    )
    matrix: Complex[Array, "n_beams n_beams"] = diagonal_removed + jnp.diag(
        excitation.astype(off_diagonal.dtype)
    )
    return matrix


@jaxtyped(typechecker=beartype)
def scattering_matrix(
    matrix: Complex[Array, "n_beams n_beams"],
    thickness: Float[Array, ""],
    wavelength: Float[Array, ""],
) -> Complex[Array, "n_beams n_beams"]:
    """Propagate beam amplitudes with a matrix exponential.

    Parameters
    ----------
    matrix
        Structure matrix ``A`` in inverse Angstroms.
    thickness
        Slab thickness ``t`` in Angstroms.
    wavelength
        Electron wavelength in Angstroms, retained for interface symmetry.

    Returns
    -------
    Complex[Array, "n_beams n_beams"]
        Scattering matrix ``S(t)``, dimensionless.

    Notes
    -----
    Uses ``S(t) = expm(2 pi i A t)`` rather than an eigendecomposition so the
    propagator remains differentiable at repeated Bloch eigenvalues.

    :see: :class:`~.test_bloch_forward.TestScatteringMatrix`
    """
    del wavelength
    exponent: Complex[Array, "n_beams n_beams"] = (
        2.0j * jnp.pi * matrix * thickness
    )
    propagator: Complex[Array, "n_beams n_beams"] = jax.scipy.linalg.expm(
        exponent
    )
    return propagator


@jaxtyped(typechecker=beartype)
def bloch_beam_amplitudes(
    fourier_potential: Complex[Array, "n_beams n_beams"],
    reflection_vectors: Float[Array, "n_beams 3"],
    tilt_vector: Float[Array, "3"],
    wavenumber: Float[Array, ""],
    wavelength: Float[Array, ""],
    thickness: Float[Array, ""],
    incident_index: Int[Array, ""],
) -> Complex[Array, "n_beams"]:
    """Compute beam amplitudes at one thickness.

    Parameters
    ----------
    fourier_potential
        Fourier potential matrix ``U_{g-h}`` in inverse Angstroms squared.
    reflection_vectors
        Reciprocal-lattice vectors ``g`` in inverse Angstroms.
    tilt_vector
        Incident-beam propagation direction, dimensionless.
    wavenumber
        Incident wavenumber ``k0`` in radians per Angstrom.
    wavelength
        Electron wavelength ``lambda`` in Angstroms.
    thickness
        Slab thickness ``t`` in Angstroms.
    incident_index
        Index of the reflection carrying the incident unit amplitude.

    Returns
    -------
    Complex[Array, "n_beams"]
        Complex beam amplitudes ``psi_g`` at thickness ``t``.

    Notes
    -----
    This is the selected incident-beam column of the scattering matrix after
    assembling excitation errors and the structure matrix.

    :see: :class:`~.test_bloch_forward.TestBlochBeamAmplitudes`
    """
    excitation: Float[Array, "n_beams"] = excitation_errors(
        reflection_vectors, tilt_vector, wavenumber
    )
    matrix: Complex[Array, "n_beams n_beams"] = structure_matrix(
        fourier_potential, excitation, wavelength
    )
    propagator: Complex[Array, "n_beams n_beams"] = scattering_matrix(
        matrix, thickness, wavelength
    )
    amplitudes: Complex[Array, "n_beams"] = propagator[:, incident_index]
    return amplitudes


@jaxtyped(typechecker=beartype)
def bloch_thickness_series(
    fourier_potential: Complex[Array, "n_beams n_beams"],
    reflection_vectors: Float[Array, "n_beams 3"],
    tilt_vector: Float[Array, "3"],
    wavenumber: Float[Array, ""],
    wavelength: Float[Array, ""],
    thickness_step: Float[Array, ""],
    n_steps: int,
    incident_index: Int[Array, ""],
) -> Complex[Array, "n_steps n_beams"]:
    """Compute amplitudes across uniform thickness steps.

    Parameters
    ----------
    fourier_potential
        Fourier potential matrix ``U_{g-h}`` in inverse Angstroms squared.
    reflection_vectors
        Reciprocal-lattice vectors ``g`` in inverse Angstroms.
    tilt_vector
        Incident-beam propagation direction, dimensionless.
    wavenumber
        Incident wavenumber ``k0`` in radians per Angstrom.
    wavelength
        Electron wavelength ``lambda`` in Angstroms.
    thickness_step
        Thickness increment per step in Angstroms.
    n_steps
        Number of thickness steps; this is static for JAX tracing.
    incident_index
        Index of the reflection carrying the incident unit amplitude.

    Returns
    -------
    Complex[Array, "n_steps n_beams"]
        Beam amplitudes at each accumulated thickness.

    Notes
    -----
    Builds the single-step scattering matrix once, initializes the incident
    beam, and records repeated applications with ``jax.lax.scan``.

    :see: :class:`~.test_bloch_forward.TestBlochThicknessSeries`
    """
    excitation: Float[Array, "n_beams"] = excitation_errors(
        reflection_vectors, tilt_vector, wavenumber
    )
    matrix: Complex[Array, "n_beams n_beams"] = structure_matrix(
        fourier_potential, excitation, wavelength
    )
    step_propagator: Complex[Array, "n_beams n_beams"] = scattering_matrix(
        matrix, thickness_step, wavelength
    )
    n_beams: int = reflection_vectors.shape[0]
    initial_state: Complex[Array, "n_beams"] = (
        jnp.zeros((n_beams,), dtype=step_propagator.dtype)
        .at[incident_index]
        .set(1.0 + 0.0j)
    )

    def scan_step(
        state: Complex[Array, "n_beams"],
        _: None,
    ) -> Tuple[Complex[Array, "n_beams"], Complex[Array, "n_beams"]]:
        advanced: Complex[Array, "n_beams"] = step_propagator @ state
        return advanced, advanced

    _: Complex[Array, "n_beams"]
    series: Complex[Array, "n_steps n_beams"]
    _, series = jax.lax.scan(scan_step, initial_state, None, length=n_steps)
    return series


@jaxtyped(typechecker=beartype)
def extinction_distance(
    fourier_coefficient: Complex[Array, ""],
    wavelength: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute the two-beam extinction distance.

    Parameters
    ----------
    fourier_coefficient
        Fourier potential coefficient ``U_g`` in inverse Angstroms squared.
    wavelength
        Electron wavelength ``lambda`` in Angstroms.

    Returns
    -------
    Float[Array, ""]
        Extinction distance ``xi_g`` in Angstroms.

    Notes
    -----
    Uses ``xi_g = 2 pi / (lambda |U_g|)``, the exact-Bragg thickness scale
    for complete two-beam intensity transfer.

    :see: :class:`~.test_bloch_forward.TestExtinctionDistance`
    """
    modulus: Float[Array, ""] = jnp.abs(fourier_coefficient)
    distance: Float[Array, ""] = 2.0 * jnp.pi / (wavelength * modulus)
    return distance


@jaxtyped(typechecker=beartype)
def two_beam_pendellosung(
    fourier_coefficient: Complex[Array, ""],
    excitation_error: Float[Array, ""],
    wavelength: Float[Array, ""],
    thickness: Float[Array, ""],
) -> Complex[Array, "2"]:
    """Evaluate the two-beam Pendellosung amplitudes.

    Parameters
    ----------
    fourier_coefficient
        Fourier potential coefficient ``U_g`` in inverse Angstroms squared.
    excitation_error
        Excitation error ``s_g`` in inverse Angstroms.
    wavelength
        Electron wavelength ``lambda`` in Angstroms.
    thickness
        Slab thickness ``t`` in Angstroms.

    Returns
    -------
    Complex[Array, "2"]
        Transmitted amplitude followed by diffracted amplitude.

    Notes
    -----
    At exact Bragg condition, the diffracted intensity follows
    ``sin^2(pi t / xi_g)``. Nonzero excitation error enters through the
    effective two-beam wavevector.

    :see: :class:`~.test_bloch_forward.TestTwoBeamPendellosung`
    """
    coupling: Float[Array, ""] = (
        wavelength * jnp.abs(fourier_coefficient) / (4.0 * jnp.pi)
    )
    half_excitation: Float[Array, ""] = excitation_error / 2.0
    effective_wavevector: Float[Array, ""] = jnp.sqrt(
        coupling * coupling + half_excitation * half_excitation
    )
    phase_argument: Float[Array, ""] = (
        2.0 * jnp.pi * effective_wavevector * thickness
    )
    common_phase: Complex[Array, ""] = jnp.exp(
        1.0j * jnp.pi * excitation_error * thickness
    )
    transmitted: Complex[Array, ""] = common_phase * (
        jnp.cos(phase_argument)
        - 1.0j
        * (half_excitation / effective_wavevector)
        * jnp.sin(phase_argument)
    )
    diffracted: Complex[Array, ""] = common_phase * (
        1.0j * (coupling / effective_wavevector) * jnp.sin(phase_argument)
    )
    amplitudes: Complex[Array, "2"] = jnp.stack([transmitted, diffracted])
    return amplitudes


@jaxtyped(typechecker=beartype)
def fourier_potential_from_grid(
    potential_grid: Float[Array, "nx ny nz"],
    miller_indices: Int[Array, "n_beams 3"],
    grid_shape: Int[Array, "3"],
) -> Complex[Array, "n_beams"]:
    """Sample Fourier potential coefficients from a grid.

    Parameters
    ----------
    potential_grid
        Real-space scattering potential in inverse Angstroms squared.
    miller_indices
        Integer voxel-frequency indices of the requested reflections.
    grid_shape
        Voxel counts along each axis, used to wrap negative indices.

    Returns
    -------
    Complex[Array, "n_beams"]
        Fourier potential coefficients ``U_g`` in inverse Angstroms squared.

    Notes
    -----
    The discrete Fourier transform is normalized by the voxel count before
    coefficients are gathered at modulo-wrapped integer frequency indices.

    :see: :class:`~.test_bloch_forward.TestFourierPotentialFromGrid`
    """
    transform: Complex[Array, "nx ny nz"] = (
        jnp.fft.fftn(potential_grid) / potential_grid.size
    )
    wrapped_indices: Int[Array, "n_beams 3"] = jnp.mod(
        miller_indices, grid_shape
    )
    coefficients: Complex[Array, "n_beams"] = transform[
        wrapped_indices[:, 0], wrapped_indices[:, 1], wrapped_indices[:, 2]
    ]
    return coefficients
