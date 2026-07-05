"""Validate the Bloch-wave forward solver against analytic invariants.

Extended Summary
----------------
This module checks every public symbol in
``ptyrodactyl.bloch.bloch_forward`` against two-beam analytic diffraction,
discrete Fourier coefficient extraction, unitary propagation, and JAX
transformation requirements. The tolerances are set for 64-bit JAX execution.

Notes
-----
The tests use small deterministic fixtures, ``chex`` array assertions,
``absl.testing.parameterized`` tables, and ``chex`` eager/JIT variants where
the public function accepts fully dynamic JAX-array inputs.
"""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from beartype import beartype
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from ptyrodactyl.bloch.bloch_forward import (
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
    bloch_beam_amplitudes,
    bloch_thickness_series,
    excitation_errors,
    extinction_distance,
    fourier_potential_from_grid,
    scattering_matrix,
    structure_matrix,
    two_beam_pendellosung,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def _wavelength_100kv() -> Float[Array, ""]:
    """Return the relativistic 100 kV electron wavelength in Angstrom."""
    wavelength: Float[Array, ""] = jnp.asarray(0.0370143, dtype=jnp.float64)
    return wavelength


@jaxtyped(typechecker=beartype)
def _si_220_coefficient() -> Complex[Array, ""]:
    """Return a silicon 220 Fourier potential coefficient."""
    coeff_a: Float[Array, "5"] = jnp.asarray(
        [0.0688157, 0.383575, 0.808651, 1.19559, 0.877985],
        dtype=jnp.float64,
    )
    coeff_b: Float[Array, "5"] = jnp.asarray(
        [0.0801515, 0.729672, 3.36331, 12.7135, 43.1093],
        dtype=jnp.float64,
    )
    g_magnitude: Float[Array, ""] = 2.0 * jnp.sqrt(2.0) / 5.4307
    x_argument: Float[Array, "5"] = coeff_b * g_magnitude * g_magnitude
    atomic_scattering: Float[Array, ""] = jnp.sum(
        coeff_a * (2.0 + x_argument) / ((1.0 + x_argument) ** 2)
    )
    structure_factor: Float[Array, ""] = 4.0 * atomic_scattering
    volume: Float[Array, ""] = jnp.asarray(5.4307**3, dtype=jnp.float64)
    potential: Float[Array, ""] = (
        MOTT_BETHE_VOLT_ANGSTROM_SQ * structure_factor / volume
    )
    coefficient: Complex[Array, ""] = potential.astype(jnp.complex128)
    return coefficient


@jaxtyped(typechecker=beartype)
def _two_beam_reflections() -> Float[Array, "2 3"]:
    """Return the origin and a silicon 220-like reciprocal vector."""
    g_magnitude: Float[Array, ""] = 2.0 * jnp.sqrt(2.0) / 5.4307
    reflections: Float[Array, "2 3"] = jnp.asarray(
        [[0.0, 0.0, 0.0], [g_magnitude, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    return reflections


@jaxtyped(typechecker=beartype)
def _two_beam_fourier_matrix(
    fourier_coefficient: Complex[Array, ""],
) -> Complex[Array, "2 2"]:
    """Build a Hermitian two-beam Fourier coupling matrix."""
    matrix: Complex[Array, "2 2"] = jnp.zeros((2, 2), dtype=jnp.complex128)
    matrix = matrix.at[0, 1].set(jnp.conj(fourier_coefficient))
    matrix = matrix.at[1, 0].set(fourier_coefficient)
    return matrix


class TestExcitationErrors(chex.TestCase):
    """Validate :func:`~ptyrodactyl.bloch.bloch_forward.excitation_errors`.

    Covers the Ewald-sphere excitation-error formula for multiple reflections,
    including the origin beam whose excitation error must be exactly zero.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.excitation_errors`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_matches_ewald_sphere_formula(self) -> None:
        """Compute excitation errors from the explicit Ewald expression.

        Extended Summary
        ----------------
        Verifies that the returned ``s_g`` values have shape ``(3,)``, are
        finite, and match ``-(2 k . g + |g|^2) / (2 |k|)`` to ``1e-12`` inverse
        Angstrom for a tilted 100 kV beam.

        Notes
        -----
        Evaluates the public function under eager and JIT variants, computes
        the reference expression independently in the test body, and checks the
        origin reflection separately.
        """
        reflections: Float[Array, "3 3"] = jnp.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, -0.25, 0.1]],
            dtype=jnp.float64,
        )
        tilt: Float[Array, "3"] = jnp.asarray(
            [0.01, -0.02, 1.0], dtype=jnp.float64
        )
        wavenumber: Float[Array, ""] = 2.0 * jnp.pi / _wavelength_100kv()

        excitation: Float[Array, "3"] = self.variant(excitation_errors)(
            reflections,
            tilt,
            wavenumber,
        )
        incident_wavevector: Float[Array, "3"] = (
            wavenumber * tilt / jnp.linalg.norm(tilt)
        )
        expected: Float[Array, "3"] = -(
            2.0 * (reflections @ incident_wavevector)
            + jnp.sum(reflections * reflections, axis=1)
        ) / (2.0 * wavenumber)

        chex.assert_shape(excitation, (3,))
        chex.assert_tree_all_finite(excitation)
        chex.assert_trees_all_close(excitation, expected, atol=1e-12)
        chex.assert_trees_all_close(excitation[0], 0.0, atol=1e-12)


class TestStructureMatrix(chex.TestCase):
    """Validate :func:`~ptyrodactyl.bloch.bloch_forward.structure_matrix`.

    Covers assembly of the Bloch dynamical matrix: diagonal excitation errors,
    off-diagonal Fourier coupling scaled by ``lambda / (4 pi)``, and Hermitian
    symmetry for Hermitian Fourier inputs.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.structure_matrix`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_places_excitation_on_diagonal(self) -> None:
        """Assemble a structure matrix with explicit diagonal replacement.

        Extended Summary
        ----------------
        Verifies that the diagonal is exactly the supplied excitation vector
        and that the off-diagonal entries equal the Fourier matrix scaled by
        ``lambda / (4 pi)`` to ``1e-12``.

        Notes
        -----
        Runs the public function through eager and JIT variants using a
        three-beam Hermitian input, then compares diagonal and off-diagonal
        pieces separately with ``chex`` assertions.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        fourier: Complex[Array, "3 3"] = jnp.asarray(
            [
                [1.0 + 0.0j, 0.1 - 0.2j, -0.05 + 0.0j],
                [0.1 + 0.2j, 2.0 + 0.0j, 0.03 + 0.04j],
                [-0.05 + 0.0j, 0.03 - 0.04j, 3.0 + 0.0j],
            ],
            dtype=jnp.complex128,
        )
        excitation: Float[Array, "3"] = jnp.asarray(
            [0.0, 0.002, -0.003], dtype=jnp.float64
        )

        matrix: Complex[Array, "3 3"] = self.variant(structure_matrix)(
            fourier,
            excitation,
            wavelength,
        )
        scaled: Complex[Array, "3 3"] = fourier * wavelength / (4.0 * jnp.pi)
        expected: Complex[Array, "3 3"] = (
            scaled
            - jnp.diag(jnp.diag(scaled))
            + jnp.diag(excitation.astype(jnp.complex128))
        )

        chex.assert_shape(matrix, (3, 3))
        chex.assert_tree_all_finite(matrix)
        chex.assert_trees_all_close(matrix, expected, atol=1e-12)
        chex.assert_trees_all_close(matrix, matrix.conj().T, atol=1e-12)


class TestScatteringMatrix(chex.TestCase):
    """Validate :func:`~ptyrodactyl.bloch.bloch_forward.scattering_matrix`.

    Covers matrix-exponential propagation against the analytic two-beam
    Pendellosung intensity and verifies finite differentiation at a degenerate
    zone-axis-like matrix.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.scattering_matrix`
    """

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("thin", 20.0),
        ("moderate", 80.0),
        ("thick", 200.0),
        ("very_thick", 500.0),
    )
    def test_matches_two_beam_pendellosung_intensity(
        self,
        thickness_angstrom: float,
    ) -> None:
        """Compare propagated two-beam intensities to the closed form.

        Extended Summary
        ----------------
        Verifies that the expm-generated scattering matrix reproduces
        transmitted and diffracted two-beam intensities to ``1e-6`` for
        thicknesses from 20 to 500 Angstrom with a nonzero excitation error.

        Notes
        -----
        Builds the same structure matrix used by the analytic solution,
        extracts the incident-beam column after eager/JIT propagation, and
        compares intensities rather than arbitrary global phase.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        wavenumber: Float[Array, ""] = 2.0 * jnp.pi / wavelength
        fourier_coefficient: Complex[Array, ""] = _si_220_coefficient()
        reflections: Float[Array, "2 3"] = _two_beam_reflections()
        fourier_matrix: Complex[Array, "2 2"] = _two_beam_fourier_matrix(
            fourier_coefficient
        )
        target_excitation: Float[Array, ""] = jnp.asarray(
            0.0015, dtype=jnp.float64
        )
        tilt: Float[Array, "3"] = jnp.asarray(
            [target_excitation / wavenumber, 0.0, 1.0],
            dtype=jnp.float64,
        )
        excitation: Float[Array, "2"] = excitation_errors(
            reflections,
            tilt,
            wavenumber,
        )
        matrix: Complex[Array, "2 2"] = structure_matrix(
            fourier_matrix,
            excitation,
            wavelength,
        )
        thickness: Float[Array, ""] = jnp.asarray(
            thickness_angstrom, dtype=jnp.float64
        )

        propagator: Complex[Array, "2 2"] = self.variant(scattering_matrix)(
            matrix,
            thickness,
            wavelength,
        )
        actual_intensity: Float[Array, "2"] = jnp.abs(propagator[:, 0]) ** 2
        analytic: Complex[Array, "2"] = two_beam_pendellosung(
            fourier_coefficient,
            excitation[1],
            wavelength,
            thickness,
        )
        expected_intensity: Float[Array, "2"] = jnp.abs(analytic) ** 2

        chex.assert_shape(propagator, (2, 2))
        chex.assert_tree_all_finite(propagator)
        chex.assert_trees_all_close(
            actual_intensity,
            expected_intensity,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_degenerate_matrix_has_finite_gradient(self) -> None:
        """Differentiate a repeated-eigenvalue propagator without NaNs.

        Extended Summary
        ----------------
        Verifies that a symmetric five-beam matrix with repeated eigenvalues
        propagates unitarily to ``1e-9`` and has a finite gradient of total
        scattered intensity with respect to a scalar coupling scale.

        Notes
        -----
        Uses ``jax.grad`` through ``scattering_matrix`` at 300 Angstrom and
        checks both gradient finiteness and ``S^H S = I``.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        n_beams: int = 5
        coefficient: Complex[Array, ""] = jnp.asarray(
            0.02 + 0.0j, dtype=jnp.complex128
        )
        fourier_matrix: Complex[Array, "5 5"] = coefficient * (
            jnp.ones((n_beams, n_beams), dtype=jnp.complex128)
            - jnp.eye(n_beams, dtype=jnp.complex128)
        )
        excitation: Float[Array, "5"] = jnp.zeros(
            (n_beams,), dtype=jnp.float64
        )
        matrix: Complex[Array, "5 5"] = structure_matrix(
            fourier_matrix,
            excitation,
            wavelength,
        )

        def total_scattered(scale: Float[Array, ""]) -> Float[Array, ""]:
            scaled_matrix: Complex[Array, "5 5"] = matrix * scale
            propagator: Complex[Array, "5 5"] = scattering_matrix(
                scaled_matrix,
                jnp.asarray(300.0, dtype=jnp.float64),
                wavelength,
            )
            intensity: Float[Array, ""] = jnp.sum(
                jnp.abs(propagator[1:, 0]) ** 2
            )
            return intensity

        grad_value: Float[Array, ""] = jax.grad(total_scattered)(
            jnp.asarray(1.0, dtype=jnp.float64)
        )
        propagator: Complex[Array, "5 5"] = scattering_matrix(
            matrix,
            jnp.asarray(300.0, dtype=jnp.float64),
            wavelength,
        )
        unitarity_error: Float[Array, ""] = jnp.max(
            jnp.abs(
                propagator.conj().T @ propagator
                - jnp.eye(n_beams, dtype=jnp.complex128)
            )
        )

        chex.assert_tree_all_finite(grad_value)
        chex.assert_trees_all_close(unitarity_error, 0.0, atol=1e-9)


class TestBlochBeamAmplitudes(chex.TestCase):
    """Validate :func:`~ptyrodactyl.bloch.bloch_forward.bloch_beam_amplitudes`.

    Covers single-thickness beam amplitudes as scattering-matrix columns and
    verifies that gradients flow from diffracted intensity to Fourier coupling.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.bloch_beam_amplitudes`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_returns_incident_column(self) -> None:
        """Match the selected incident-beam column of the propagator.

        Extended Summary
        ----------------
        Verifies that beam amplitudes at 150 Angstrom are identical to column
        zero of the assembled scattering matrix to ``1e-12``.

        Notes
        -----
        Runs the public function under eager and JIT variants, constructs the
        reference matrix explicitly, and compares the complex amplitude vector.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        wavenumber: Float[Array, ""] = 2.0 * jnp.pi / wavelength
        fourier_coefficient: Complex[Array, ""] = _si_220_coefficient()
        fourier_matrix: Complex[Array, "2 2"] = _two_beam_fourier_matrix(
            fourier_coefficient
        )
        reflections: Float[Array, "2 3"] = _two_beam_reflections()
        tilt: Float[Array, "3"] = jnp.asarray(
            [0.0, 0.0, 1.0], dtype=jnp.float64
        )
        thickness: Float[Array, ""] = jnp.asarray(150.0, dtype=jnp.float64)
        incident_index: Int[Array, ""] = jnp.asarray(0, dtype=jnp.int64)

        amplitudes: Complex[Array, "2"] = self.variant(bloch_beam_amplitudes)(
            fourier_matrix,
            reflections,
            tilt,
            wavenumber,
            wavelength,
            thickness,
            incident_index,
        )
        excitation: Float[Array, "2"] = excitation_errors(
            reflections,
            tilt,
            wavenumber,
        )
        matrix: Complex[Array, "2 2"] = structure_matrix(
            fourier_matrix,
            excitation,
            wavelength,
        )
        expected: Complex[Array, "2"] = scattering_matrix(
            matrix,
            thickness,
            wavelength,
        )[:, 0]

        chex.assert_shape(amplitudes, (2,))
        chex.assert_tree_all_finite(amplitudes)
        chex.assert_trees_all_close(amplitudes, expected, atol=1e-12)

    @chex.variants(with_jit=True, without_jit=True)
    def test_diffracted_intensity_has_nonzero_gradient(self) -> None:
        """Differentiate diffracted intensity with respect to coupling.

        Extended Summary
        ----------------
        Verifies that the diffracted two-beam intensity at 150 Angstrom has a
        finite, nonzero gradient with respect to the real part of ``U_g``.

        Notes
        -----
        Defines a real scalar loss using the eager/JIT variant of
        ``bloch_beam_amplitudes`` and differentiates it with ``jax.grad``.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        wavenumber: Float[Array, ""] = 2.0 * jnp.pi / wavelength
        reflections: Float[Array, "2 3"] = _two_beam_reflections()
        tilt: Float[Array, "3"] = jnp.asarray(
            [0.0, 0.0, 1.0], dtype=jnp.float64
        )

        def diffracted_intensity(
            coupling_real: Float[Array, ""],
        ) -> Float[Array, ""]:
            fourier_coefficient: Complex[Array, ""] = coupling_real.astype(
                jnp.complex128
            )
            fourier_matrix: Complex[Array, "2 2"] = _two_beam_fourier_matrix(
                fourier_coefficient
            )
            amplitudes: Complex[Array, "2"] = self.variant(
                bloch_beam_amplitudes
            )(
                fourier_matrix,
                reflections,
                tilt,
                wavenumber,
                wavelength,
                jnp.asarray(150.0, dtype=jnp.float64),
                jnp.asarray(0, dtype=jnp.int64),
            )
            intensity: Float[Array, ""] = jnp.abs(amplitudes[1]) ** 2
            return intensity

        initial_coupling: Float[Array, ""] = jnp.real(_si_220_coefficient())
        grad_value: Float[Array, ""] = jax.grad(diffracted_intensity)(
            initial_coupling
        )

        chex.assert_tree_all_finite(grad_value)
        assert bool(jnp.abs(grad_value) > 0.0)


class TestBlochThicknessSeries(chex.TestCase):
    """Validate Bloch thickness-series propagation.

    Covers uniform-thickness propagation by ``lax.scan`` against direct matrix
    exponentials at selected accumulated thicknesses.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.bloch_thickness_series`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_scan_series_matches_direct_propagation(self) -> None:
        """Compare scanned propagation to direct accumulated propagation.

        Extended Summary
        ----------------
        Verifies that a 20-step, 10 Angstrom step-size series matches direct
        scattering-matrix propagation at steps 1, 6, 11, and 20 to ``1e-9``.

        Notes
        -----
        Keeps ``n_steps`` lexical for JIT compatibility, evaluates the public
        function under eager/JIT variants, and uses ``vmap`` to build the
        direct reference amplitudes.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        wavenumber: Float[Array, ""] = 2.0 * jnp.pi / wavelength
        fourier_coefficient: Complex[Array, ""] = _si_220_coefficient()
        fourier_matrix: Complex[Array, "2 2"] = _two_beam_fourier_matrix(
            fourier_coefficient
        )
        reflections: Float[Array, "2 3"] = _two_beam_reflections()
        tilt: Float[Array, "3"] = jnp.asarray(
            [0.0, 0.0, 1.0], dtype=jnp.float64
        )
        step: Float[Array, ""] = jnp.asarray(10.0, dtype=jnp.float64)
        n_steps: int = 20
        incident_index: Int[Array, ""] = jnp.asarray(0, dtype=jnp.int64)

        def series_fn(
            dynamic_fourier_matrix: Complex[Array, "2 2"],
            dynamic_reflections: Float[Array, "2 3"],
            dynamic_tilt: Float[Array, "3"],
            dynamic_wavenumber: Float[Array, ""],
            dynamic_wavelength: Float[Array, ""],
            dynamic_step: Float[Array, ""],
            dynamic_incident_index: Int[Array, ""],
        ) -> Complex[Array, "20 2"]:
            series: Complex[Array, "20 2"] = bloch_thickness_series(
                dynamic_fourier_matrix,
                dynamic_reflections,
                dynamic_tilt,
                dynamic_wavenumber,
                dynamic_wavelength,
                dynamic_step,
                n_steps,
                dynamic_incident_index,
            )
            return series

        series: Complex[Array, "20 2"] = self.variant(series_fn)(
            fourier_matrix,
            reflections,
            tilt,
            wavenumber,
            wavelength,
            step,
            incident_index,
        )
        excitation: Float[Array, "2"] = excitation_errors(
            reflections,
            tilt,
            wavenumber,
        )
        matrix: Complex[Array, "2 2"] = structure_matrix(
            fourier_matrix,
            excitation,
            wavelength,
        )
        sample_indices: Int[Array, "4"] = jnp.asarray(
            [0, 5, 10, 19], dtype=jnp.int64
        )
        accumulated_thicknesses: Float[Array, "4"] = step * (
            sample_indices.astype(jnp.float64) + 1.0
        )
        direct: Complex[Array, "4 2"] = jax.vmap(
            lambda thickness: scattering_matrix(
                matrix,
                thickness,
                wavelength,
            )[:, 0]
        )(accumulated_thicknesses)

        chex.assert_shape(series, (20, 2))
        chex.assert_tree_all_finite(series)
        chex.assert_trees_all_close(series[sample_indices], direct, atol=1e-9)


class TestExtinctionDistance(chex.TestCase):
    """Validate :func:`~ptyrodactyl.bloch.bloch_forward.extinction_distance`.

    Covers the two-beam extinction-distance formula and positivity for a
    silicon 220-like Fourier coefficient at 100 kV.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.extinction_distance`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_matches_closed_form_distance(self) -> None:
        """Compute extinction distance from ``2 pi / (lambda |U_g|)``.

        Extended Summary
        ----------------
        Verifies that the returned distance is a finite positive scalar and
        matches the analytic formula to ``1e-12`` relative tolerance.

        Notes
        -----
        Runs the public function under eager and JIT variants using the local
        silicon coefficient fixture, then checks shape, finiteness, positivity,
        and formula agreement.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        fourier_coefficient: Complex[Array, ""] = _si_220_coefficient()

        distance: Float[Array, ""] = self.variant(extinction_distance)(
            fourier_coefficient,
            wavelength,
        )
        expected: Float[Array, ""] = (
            2.0 * jnp.pi / (wavelength * jnp.abs(fourier_coefficient))
        )

        chex.assert_shape(distance, ())
        chex.assert_tree_all_finite(distance)
        assert bool(distance > 0.0)
        chex.assert_trees_all_close(distance, expected, rtol=1e-12)


class TestTwoBeamPendellosung(chex.TestCase):
    """Validate :func:`~ptyrodactyl.bloch.bloch_forward.two_beam_pendellosung`.

    Covers the exact-Bragg Pendellosung limit where diffracted intensity
    follows ``sin^2(pi t / xi_g)`` and reaches unity at half an extinction
    distance.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.two_beam_pendellosung`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_bragg_intensity_matches_sin_squared(self) -> None:
        """Recover the exact-Bragg two-beam intensity transfer.

        Extended Summary
        ----------------
        Verifies that at ``t = xi_g / 2`` the diffracted intensity equals both
        ``sin^2(pi t / xi_g)`` and unity to ``1e-9``.

        Notes
        -----
        Computes ``xi_g`` with the public extinction-distance helper, evaluates
        the eager/JIT Pendellosung variant at zero excitation error, and checks
        the diffracted beam intensity.
        """
        wavelength: Float[Array, ""] = _wavelength_100kv()
        fourier_coefficient: Complex[Array, ""] = _si_220_coefficient()
        xi_g: Float[Array, ""] = extinction_distance(
            fourier_coefficient,
            wavelength,
        )
        thickness: Float[Array, ""] = 0.5 * xi_g

        amplitudes: Complex[Array, "2"] = self.variant(two_beam_pendellosung)(
            fourier_coefficient,
            jnp.asarray(0.0, dtype=jnp.float64),
            wavelength,
            thickness,
        )
        diffracted_intensity: Float[Array, ""] = jnp.abs(amplitudes[1]) ** 2
        expected: Float[Array, ""] = jnp.sin(jnp.pi * thickness / xi_g) ** 2

        chex.assert_shape(amplitudes, (2,))
        chex.assert_tree_all_finite(amplitudes)
        chex.assert_trees_all_close(diffracted_intensity, expected, atol=1e-9)
        chex.assert_trees_all_close(diffracted_intensity, 1.0, atol=1e-9)


class TestFourierPotentialFromGrid(chex.TestCase):
    """Validate Fourier-potential grid sampling.

    Covers normalized DFT sampling of requested integer Miller/voxel indices,
    including negative-index wrapping through the supplied grid shape.

    :see: :func:`~ptyrodactyl.bloch.bloch_forward.fourier_potential_from_grid`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_extracts_wrapped_normalized_fft_coefficients(self) -> None:
        """Sample known Fourier coefficients from a cosine potential grid.

        Extended Summary
        ----------------
        Verifies that a ``2 cos(2 pi x / 4)`` grid has normalized Fourier
        coefficients equal to one at ``+1`` and ``-1`` along ``x`` and zero at
        the origin to ``1e-12``.

        Notes
        -----
        Broadcasts a deterministic real-space grid to shape ``(4, 4, 4)``,
        gathers positive, negative, and origin indices under eager/JIT
        variants, and compares the complex coefficients directly.
        """
        grid_shape: Int[Array, "3"] = jnp.asarray([4, 4, 4], dtype=jnp.int64)
        x_indices: Float[Array, "4 1 1"] = jnp.arange(
            4, dtype=jnp.float64
        ).reshape((4, 1, 1))
        one_dimensional: Float[Array, "4 1 1"] = 2.0 * jnp.cos(
            2.0 * jnp.pi * x_indices / 4.0
        )
        potential_grid: Float[Array, "4 4 4"] = jnp.broadcast_to(
            one_dimensional,
            (4, 4, 4),
        )
        miller_indices: Int[Array, "3 3"] = jnp.asarray(
            [[1, 0, 0], [-1, 0, 0], [0, 0, 0]],
            dtype=jnp.int64,
        )

        coefficients: Complex[Array, "3"] = self.variant(
            fourier_potential_from_grid
        )(
            potential_grid,
            miller_indices,
            grid_shape,
        )
        expected: Complex[Array, "3"] = jnp.asarray(
            [1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            dtype=jnp.complex128,
        )

        chex.assert_shape(coefficients, (3,))
        chex.assert_tree_all_finite(coefficients)
        chex.assert_trees_all_close(coefficients, expected, atol=1e-12)
