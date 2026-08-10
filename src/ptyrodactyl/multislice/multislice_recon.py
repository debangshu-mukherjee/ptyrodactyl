"""Multislice reconstruction algorithms for electron ptychography.

Extended Summary
----------------
Provides gradient-based optimization routines for reconstructing
sample electrostatic potentials and electron probe functions from
experimental 4D-STEM ptychographic datasets. Each public function
constructs a differentiable forward model via
:func:`ptyrodactyl.multislice.simulations.stem_4d`, computes the loss
and its gradients with ``jax.value_and_grad``, and iteratively
updates the reconstruction variables using a first-order optimizer
from :mod:`optax`.

Routine Listings
----------------
:func:`multi_slice_multi_modal`
    Reconstruct potential, beam, and positions with multi-slice.
:func:`single_slice_multi_modal`
    Reconstruct potential, multi-modal beam, and positions.
:func:`single_slice_poscorrected`
    Reconstruct potential, beam, and positions from 4D-STEM data.
:func:`single_slice_ptychography`
    Reconstruct potential and beam from 4D-STEM data.
:obj:`OPTIMIZERS`
    Registry mapping optimizer name strings to
    Optax gradient-transformation factories.

Notes
-----
All reconstruction functions use Optax optimizers and
support automatic differentiation. The functions are designed to
work with experimental data and can handle various noise levels
and experimental conditions. Input data should be properly
preprocessed and validated using the factory functions from
:mod:`ptyrodactyl.types`.

JAX gradients of real losses with respect to complex reconstruction
parameters are conjugated before they are passed to Optax. Scan positions
remain real-valued and use their JAX gradients directly.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from beartype.typing import Any, Callable, Dict, Tuple, TypeAlias, Union, cast
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from ptyrodactyl.types import (
    STEM4D,
    CalibratedArray,
    LossType,
    ProbeModes,
    create_calibrated_array,
    create_detector_config,
    create_microscope_config,
    create_potential_slices,
    create_probe_modes,
    scalar_float,
    scalar_num,
)

from .simulations import stem_4d

_OptimizerFactory: TypeAlias = Callable[
    [scalar_float], optax.GradientTransformation
]

OPTIMIZERS: Dict[str, _OptimizerFactory] = {
    "adam": lambda learning_rate: optax.adam(learning_rate),
    "adagrad": lambda learning_rate: optax.adagrad(
        learning_rate,
        initial_accumulator_value=0.0,
        eps=1e-8,
    ),
    "rmsprop": lambda learning_rate: optax.rmsprop(
        learning_rate,
        decay=0.9,
        eps=1e-8,
        initial_scale=0.0,
    ),
}
"""Registry mapping optimizer names to Optax transformation factories.

:see: :mod:`~.test_multislice_recon`

Notes
-----
Complex gradients are conjugated by :func:`_apply_optimizer_step`
before the configured transformation receives them.
"""


@beartype
def _get_optimizer(
    optimizer_name: str,
    learning_rate: scalar_float,
) -> optax.GradientTransformation:
    """PRIVATE: Build a named Optax optimizer at one learning rate.

    Parameters
    ----------
    optimizer_name : str
        Key into :data:`OPTIMIZERS` (e.g. ``"adam"``).
    learning_rate : scalar_float
        Step size supplied to the Optax transformation factory.

    Returns
    -------
    result : optax.GradientTransformation
        Configured Optax gradient transformation.

    Raises
    ------
    ValueError
        If *optimizer_name* is not a key in :data:`OPTIMIZERS`.
    """
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    result: optax.GradientTransformation = OPTIMIZERS[optimizer_name](
        learning_rate
    )
    return result


@jaxtyped(typechecker=beartype)
def _reduce_loss(
    model_output: Float[Array, " ..."],
    experimental_data: Float[Array, " ..."],
    loss_mode: LossType,
) -> Float[Array, " "]:
    """PRIVATE: Reduce elementwise model errors to a scalar loss.

    Parameters
    ----------
    model_output : Float[Array, " ..."]
        Simulated detector data.
    experimental_data : Float[Array, " ..."]
        Experimental detector data with the same shape as *model_output*.
    loss_mode : LossType
        Static reduction selection.

    Returns
    -------
    result : Float[Array, " "]
        Mean absolute, mean squared, or root-mean-squared error.

    Notes
    -----
    MSE and RMSE use :func:`optax.squared_error`; MAE uses the elementwise
    absolute residual because Optax does not expose a dedicated MAE loss.
    """
    if loss_mode is LossType.MAE:
        loss = jnp.mean(jnp.abs(model_output - experimental_data))
    elif loss_mode is LossType.MSE:
        loss = jnp.mean(optax.squared_error(model_output, experimental_data))
    else:
        loss = jnp.sqrt(
            jnp.mean(optax.squared_error(model_output, experimental_data))
        )
    result: Float[Array, " "] = loss
    return result


@jaxtyped(typechecker=beartype)
def _apply_optimizer_step(
    params: Num[Array, " ..."],
    grads: Num[Array, " ..."],
    optimizer: optax.GradientTransformation,
    optimizer_state: Any,
) -> Tuple[Num[Array, " ..."], Any]:
    """PRIVATE: Apply one Optax step with the JAX gradient convention.

    Parameters
    ----------
    params : Num[Array, " ..."]
        Current real- or complex-valued parameters.
    grads : Num[Array, " ..."]
        Gradient returned by JAX for a real-valued loss.
    optimizer : optax.GradientTransformation
        Configured Optax gradient transformation.
    optimizer_state : Any
        Current Optax state for *params*.

    Returns
    -------
    new_params : Num[Array, " ..."]
        Parameters after one descent step.
    new_state : Any
        Updated Optax state.

    Notes
    -----
    JAX returns the conjugate covector convention for a real loss with
    complex inputs. Optax expects that gradient to be conjugated before it
    constructs a complex descent update. Conjugation is a no-op for real
    parameters.
    """
    updates, new_state = optimizer.update(
        jnp.conj(grads),
        optimizer_state,
        params,
    )
    new_params = cast(
        Num[Array, " ..."],
        optax.apply_updates(params, updates),
    )
    result: Tuple[Num[Array, " ..."], Any] = new_params, new_state
    return result


@jaxtyped(typechecker=beartype)
def _promote_to_complex(
    values: Num[Array, " ..."],
) -> Complex[Array, " ..."]:
    """PRIVATE: Promote numeric values to a width-preserving complex dtype.

    Parameters
    ----------
    values : Num[Array, " ..."]
        Real- or complex-valued input array.

    Returns
    -------
    result : Complex[Array, " ..."]
        Complex array with the input's floating-point precision.

    Notes
    -----
    Promotion uses ``result_type(values.dtype, complex64)`` so float32 and
    complex64 inputs produce complex64, while float64 and complex128 inputs
    produce complex128.
    """
    result: Complex[Array, " ..."] = jnp.asarray(
        values,
        dtype=jnp.result_type(values.dtype, jnp.complex64),
    )
    return result


@jaxtyped(typechecker=beartype)
def single_slice_ptychography(  # noqa: PLR0915
    experimental_data: STEM4D,
    initial_potential: CalibratedArray,
    initial_beam: CalibratedArray,
    slice_thickness: scalar_num,
    save_every: int = 10,
    num_iterations: int = 1000,
    learning_rate: scalar_float = 0.001,
    loss_type: str = "mse",
    optimizer_name: str = "adam",
) -> Tuple[
    CalibratedArray,
    CalibratedArray,
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
]:
    r"""Reconstruct potential and beam from 4D-STEM data.

    Extended Summary
    ----------------
    Performs single-slice ptychography where the electrostatic
    potential slice and the beam guess share the same spatial
    dimensions.  The reconstruction minimises a pixel-wise loss
    between experimental and simulated diffraction patterns:

    .. math::

        \mathcal{L}
        = \sum_{p}
          \bigl\lVert
            I_p^{\mathrm{exp}}
            - I_p^{\mathrm{sim}}(V, \psi)
          \bigr\rVert^2

    where :math:`V` is the potential slice and :math:`\psi` is the
    probe wavefunction.

    :see: :func:`~.test_single_slice_ptychography_runs_and_regresses`

    Implementation Logic
    --------------------
    1. **Build forward model** --
       Wraps :func:`~ptyrodactyl.multislice.simulations.stem_4d` to
       map ``(pot_slice, beam)`` to simulated 4D-STEM data.
    2. **Construct loss** --
       Reduces detector errors with the selected loss mode.
    3. **Iterate** --
       At each step compute gradients with
       ``jax.value_and_grad`` and update potential and beam with
       the chosen optimizer.
    4. **Snapshot** --
       Every *save_every* iterations, store the current
       potential and beam into intermediate arrays.

    Parameters
    ----------
    experimental_data : :class:`~ptyrodactyl.types.STEM4D`
        Experimental 4D-STEM data PyTree containing diffraction
        patterns, scan positions, and calibration information.
    initial_potential : :class:`~ptyrodactyl.types.CalibratedArray`
        Initial guess for the electrostatic potential slice.
    initial_beam : :class:`~ptyrodactyl.types.CalibratedArray`
        Initial guess for the electron beam.  If stored in
        reciprocal space (``real_space=False``), an inverse FFT
        is applied before optimisation.
    slice_thickness : scalar_num
        Thickness of the potential slice, in Angstroms.
    save_every : int, optional
        Store intermediate results every *save_every* iterations.
        Default is ``10``.
    num_iterations : int, optional
        Total number of optimisation iterations.
        Default is ``1000``.
    learning_rate : scalar_float, optional
        Step size for the optimizer.  Default is ``0.001``.
    loss_type : str, optional
        Loss reduction identifier.
        Default is ``"mse"``.
    optimizer_name : str, optional
        Key into :data:`OPTIMIZERS`.  Default is ``"adam"``.

    Returns
    -------
    final_potential : :class:`~ptyrodactyl.types.CalibratedArray`
        Optimised electrostatic potential slice.
    final_beam : :class:`~ptyrodactyl.types.CalibratedArray`
        Optimised electron beam in real space.
    intermediate_potslice : Complex[Array, "H W S"]
        Potential snapshots at saved iterations.
    intermediate_beam : Complex[Array, "H W S"]
        Beam snapshots at saved iterations.

    See Also
    --------
    :func:`single_slice_poscorrected` :
        Adds scan-position refinement.
    :func:`single_slice_multi_modal` :
        Supports multi-modal probe modes.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    pos_list: Float[Array, "P 2"] = experimental_data.scan_positions
    voltage_kv: Float[Array, " "] = jnp.asarray(
        experimental_data.voltage_kv,
        dtype=jnp.float64,
    )
    calib_ang: Float[Array, " "] = jnp.asarray(
        experimental_data.real_space_calib,
        dtype=jnp.float64,
    )
    microscope = create_microscope_config(
        voltage_kv=voltage_kv,
        aperture_mrad=1.0,
    )

    def _forward_fn(
        pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W"]
    ) -> Float[Array, "P H W"]:
        """PRIVATE: Simulate 4D-STEM patterns from potential and beam.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Electrostatic potential slice.
        beam : Complex[Array, "H W"]
            Probe wavefunction in real space.

        Returns
        -------
        result : Float[Array, "P H W"]
            Simulated diffraction patterns.
        """
        potential_slices = create_potential_slices(
            slices=jnp.real(pot_slice)[..., jnp.newaxis],
            slice_thickness=slice_thickness,
            calib=calib_ang,
        )
        probe_modes = create_probe_modes(
            modes=beam[..., jnp.newaxis],
            weights=jnp.ones((1,), dtype=experimental_4dstem.dtype),
            calib=calib_ang,
        )
        detector = create_detector_config(
            real_space_calib_ang=calib_ang,
            scan_positions_px=pos_list,
        )
        stem4d_result = stem_4d(
            potential_slices,
            probe_modes,
            microscope,
            detector,
        )
        result: Float[Array, "P H W"] = stem4d_result.data
        return result

    loss_mode: LossType = LossType(loss_type)
    loss_func: Callable[..., Float[Array, " "]] = jax.jit(
        lambda potential, probe: _reduce_loss(
            _forward_fn(potential, probe),
            experimental_4dstem,
            loss_mode,
        )
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W"]
    ) -> Tuple[Float[Array, " "], Dict[str, Complex[Array, "H W"]]]:
        """PRIVATE: Compute loss and gradients for potential and beam.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice estimate.
        beam : Complex[Array, "H W"]
            Current beam estimate.

        Returns
        -------
        loss : Float[Array, " "]
            Scalar loss value.
        grads : Dict[str, Complex[Array, "H W"]]
            Gradient dictionary with keys ``"pot_slice"``
            and ``"beam"``.
        """
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1))(
            pot_slice, beam
        )
        result: Tuple[Float[Array, " "], Dict[str, Complex[Array, "H W"]]] = (
            loss,
            {"pot_slice": grads[0], "beam": grads[1]},
        )
        return result

    pot_slice: Complex[Array, "H W"] = _promote_to_complex(
        initial_potential.data_array
    )
    initial_beam_values: Complex[Array, "H W"] = _promote_to_complex(
        initial_beam.data_array
    )
    beam: Complex[Array, "H W"] = jax.lax.cond(
        initial_beam.real_space,
        lambda beam_data: beam_data,
        lambda beam_data: jnp.fft.ifft2(beam_data),
        initial_beam_values,
    )
    optimizer: optax.GradientTransformation = _get_optimizer(
        optimizer_name,
        learning_rate,
    )
    pot_slice_state: Any = optimizer.init(pot_slice)
    beam_state: Any = optimizer.init(beam)

    snapshot_count: int = num_iterations // save_every

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pot_slice_state: Any,
        beam_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"],
        Complex[Array, "H W"],
        Any,
        Any,
        Float[Array, " "],
    ]:
        """PRIVATE: Perform one optimisation step for potential and beam.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice.
        beam : Complex[Array, "H W"]
            Current beam.
        pot_slice_state : Any
            Optimizer state for the potential.
        beam_state : Any
            Optimizer state for the beam.

        Returns
        -------
        pot_slice : Complex[Array, "H W"]
            Updated potential slice.
        beam : Complex[Array, "H W"]
            Updated beam.
        pot_slice_state : Any
            Updated optimizer state for the potential.
        beam_state : Any
            Updated optimizer state for the beam.
        loss : Float[Array, " "]
            Scalar loss after this step.
        """
        loss: Float[Array, " "]
        grads: Dict[str, Complex[Array, "H W"]]
        loss, grads = _loss_and_grad(pot_slice, beam)
        pot_slice_next, pot_slice_state = _apply_optimizer_step(
            pot_slice,
            grads["pot_slice"],
            optimizer,
            pot_slice_state,
        )
        pot_slice = cast(Complex[Array, "H W"], pot_slice_next)
        beam_next, beam_state = _apply_optimizer_step(
            beam,
            grads["beam"],
            optimizer,
            beam_state,
        )
        beam = cast(Complex[Array, "H W"], beam_next)
        result: Tuple[
            Complex[Array, "H W"],
            Complex[Array, "H W"],
            Any,
            Any,
            Float[Array, " "],
        ] = pot_slice, beam, pot_slice_state, beam_state, loss
        return result

    intermediate_potslice: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            pot_slice.shape[0],
            pot_slice.shape[1],
            snapshot_count,
        ),
        dtype=pot_slice.dtype,
    )
    intermediate_beam: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            beam.shape[0],
            beam.shape[1],
            snapshot_count,
        ),
        dtype=beam.dtype,
    )

    def _scan_step(
        carry: Tuple[Any, ...], ii: Int[Array, ""]
    ) -> Tuple[Tuple[Any, ...], Float[Array, " "]]:
        """PRIVATE: Advance one potential-and-beam reconstruction step.

        Parameters
        ----------
        carry : Tuple[Any, ...]
            Current potential, beam, optimizer, and snapshot state.
        ii : Int[Array, ""]
            Zero-based optimization iteration.

        Returns
        -------
        carry : Tuple[Any, ...]
            Updated reconstruction and snapshot state.
        loss : Float[Array, " "]
            Scalar loss for the completed iteration.
        """
        (
            pot_slice,
            beam,
            pot_slice_state,
            beam_state,
            intermediate_potslice,
            intermediate_beam,
        ) = carry
        pot_slice, beam, pot_slice_state, beam_state, loss = _update_step(
            pot_slice, beam, pot_slice_state, beam_state
        )

        def _save_snapshot(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
            """PRIVATE: Save the current potential and beam snapshots.

            Parameters
            ----------
            args : Tuple[Any, ...]
                Potential and beam snapshot arrays.

            Returns
            -------
            result : Tuple[Any, ...]
                Snapshot arrays with the current estimates stored.
            """
            pots, beams = args
            saver: Int[Array, ""] = (ii // save_every).astype(jnp.int32)
            pots = pots.at[:, :, saver].set(pot_slice)
            beams = beams.at[:, :, saver].set(beam)
            result: Tuple[Any, ...] = pots, beams
            return result

        intermediate_potslice, intermediate_beam = jax.lax.cond(
            ii % save_every == 0,
            _save_snapshot,
            lambda args: args,
            (intermediate_potslice, intermediate_beam),
        )
        result: Tuple[Tuple[Any, ...], Float[Array, " "]] = (
            (
                pot_slice,
                beam,
                pot_slice_state,
                beam_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            loss,
        )
        return result

    if num_iterations > 0:
        pot_slice, beam, pot_slice_state, beam_state, _ = _update_step(
            pot_slice, beam, pot_slice_state, beam_state
        )
        if snapshot_count > 0:
            intermediate_potslice = intermediate_potslice.at[:, :, 0].set(
                pot_slice
            )
            intermediate_beam = intermediate_beam.at[:, :, 0].set(beam)

        (
            (
                pot_slice,
                beam,
                pot_slice_state,
                beam_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            _,
        ) = jax.lax.scan(
            _scan_step,
            (
                pot_slice,
                beam,
                pot_slice_state,
                beam_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            jnp.arange(1, num_iterations),
            unroll=True,
        )

    final_potential: CalibratedArray = create_calibrated_array(
        data_array=pot_slice,
        calib_y=initial_potential.calib_y,
        calib_x=initial_potential.calib_x,
        real_space=True,
    )
    final_beam: CalibratedArray = create_calibrated_array(
        data_array=beam,
        calib_y=initial_beam.calib_y,
        calib_x=initial_beam.calib_x,
        real_space=True,
    )

    reconstruction_result: Tuple[
        CalibratedArray,
        CalibratedArray,
        Complex[Array, "H W S"],
        Complex[Array, "H W S"],
    ] = (
        final_potential,
        final_beam,
        intermediate_potslice,
        intermediate_beam,
    )
    return reconstruction_result


@jaxtyped(typechecker=beartype)
def single_slice_poscorrected(  # noqa: PLR0915
    experimental_data: STEM4D,
    initial_potential: CalibratedArray,
    initial_beam: CalibratedArray,
    slice_thickness: scalar_num,
    save_every: int = 10,
    num_iterations: int = 1000,
    learning_rate: Union[scalar_float, Float[Array, "2"]] = 0.01,
    loss_type: str = "mse",
    optimizer_name: str = "adam",
) -> Tuple[
    CalibratedArray,
    CalibratedArray,
    Float[Array, "P 2"],
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
    Float[Array, "P 2 S"],
]:
    r"""Reconstruct potential, beam, and positions from 4D-STEM data.

    Extended Summary
    ----------------
    Single-slice ptychographic reconstruction that simultaneously
    refines the electrostatic potential, the probe wavefunction,
    and the scan positions.  Position correction compensates for
    drift and scan distortions by treating the probe coordinates
    as differentiable variables:

    .. math::

        \mathcal{L}
        = \sum_{p}
          \bigl\lVert
            I_p^{\mathrm{exp}}
            - I_p^{\mathrm{sim}}(V, \psi, \mathbf{r}_p)
          \bigr\rVert^2

    where :math:`\mathbf{r}_p` are the corrected scan positions.

    :see: :func:`~.test_single_slice_poscorrected_regresses`

    Implementation Logic
    --------------------
    1. **Build forward model** --
       Wraps :func:`~ptyrodactyl.multislice.simulations.stem_4d` to
       map ``(pot_slice, beam, pos_list)`` to simulated 4D-STEM.
    2. **Construct loss** --
       Reduces detector errors with the selected loss mode.
    3. **Parse learning rate** --
       If scalar, reuse for both potential/beam and positions;
       if length-2 array, element 0 is for potential/beam and
       element 1 is for positions.
    4. **Iterate** --
       At each step compute gradients with
       ``jax.value_and_grad`` over all three variable groups
       and apply the chosen optimizer.
    5. **Snapshot** --
       Every *save_every* iterations, store the current
       potential, beam, and positions into intermediate arrays.

    Parameters
    ----------
    experimental_data : :class:`~ptyrodactyl.types.STEM4D`
        Experimental 4D-STEM data PyTree containing diffraction
        patterns, scan positions, and calibration information.
    initial_potential : :class:`~ptyrodactyl.types.CalibratedArray`
        Initial guess for the electrostatic potential slice.
    initial_beam : :class:`~ptyrodactyl.types.CalibratedArray`
        Initial guess for the electron beam.
    slice_thickness : scalar_num
        Thickness of the potential slice, in Angstroms.
    save_every : int, optional
        Store intermediate results every *save_every* iterations.
        Default is ``10``.
    num_iterations : int, optional
        Total number of optimisation iterations.
        Default is ``1000``.
    learning_rate : scalar_float or Float[Array, "2"], optional
        Step size(s) for the optimizer.  If scalar, the same
        rate is used for potential/beam and positions.  If a
        length-2 array, element 0 controls potential/beam and
        element 1 controls positions.  Default is ``0.01``.
    loss_type : str, optional
        Loss reduction identifier.
        Default is ``"mse"``.
    optimizer_name : str, optional
        Key into :data:`OPTIMIZERS`.  Default is ``"adam"``.

    Returns
    -------
    final_potential : :class:`~ptyrodactyl.types.CalibratedArray`
        Optimised electrostatic potential slice.
    final_beam : :class:`~ptyrodactyl.types.CalibratedArray`
        Optimised electron beam in real space.
    pos_guess : Float[Array, "P 2"]
        Refined scan positions, in Angstroms.
    intermediate_potslices : Complex[Array, "H W S"]
        Potential snapshots at saved iterations.
    intermediate_beams : Complex[Array, "H W S"]
        Beam snapshots at saved iterations.
    intermediate_positions : Float[Array, "P 2 S"]
        Position snapshots at saved iterations.

    See Also
    --------
    :func:`single_slice_ptychography` :
        Variant without position correction.
    :func:`single_slice_multi_modal` :
        Adds multi-modal probe support.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    voltage_kv: Float[Array, " "] = jnp.asarray(
        experimental_data.voltage_kv,
        dtype=jnp.float64,
    )
    calib_ang: Float[Array, " "] = jnp.asarray(
        experimental_data.real_space_calib,
        dtype=jnp.float64,
    )
    initial_pos_list: Float[Array, "P 2"] = experimental_data.scan_positions
    microscope = create_microscope_config(
        voltage_kv=voltage_kv,
        aperture_mrad=1.0,
    )

    def _forward_fn(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Float[Array, "P H W"]:
        """PRIVATE: Simulate 4D-STEM with position-corrected scan.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Electrostatic potential slice.
        beam : Complex[Array, "H W"]
            Probe wavefunction in real space.
        pos_list : Float[Array, "P 2"]
            Scan positions, in Angstroms.

        Returns
        -------
        result : Float[Array, "P H W"]
            Simulated diffraction patterns.
        """
        potential_slices = create_potential_slices(
            slices=jnp.real(pot_slice)[..., jnp.newaxis],
            slice_thickness=slice_thickness,
            calib=calib_ang,
        )
        probe_modes = create_probe_modes(
            modes=beam[..., jnp.newaxis],
            weights=jnp.ones((1,), dtype=experimental_4dstem.dtype),
            calib=calib_ang,
        )
        detector = create_detector_config(
            real_space_calib_ang=calib_ang,
            scan_positions_px=pos_list,
        )
        stem4d_result = stem_4d(
            potential_slices,
            probe_modes,
            microscope,
            detector,
        )
        result: Float[Array, "P H W"] = stem4d_result.data
        return result

    loss_mode: LossType = LossType(loss_type)
    loss_func: Callable[..., Float[Array, " "]] = jax.jit(
        lambda potential, probe, positions: _reduce_loss(
            _forward_fn(potential, probe, positions),
            experimental_4dstem,
            loss_mode,
        )
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Tuple[Float[Array, " "], Dict[str, Array]]:
        """PRIVATE: Compute loss and all position-corrected gradients.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice estimate.
        beam : Complex[Array, "H W"]
            Current beam estimate.
        pos_list : Float[Array, "P 2"]
            Current scan positions, in Angstroms.

        Returns
        -------
        loss : Float[Array, " "]
            Scalar loss value.
        grads : Dict[str, Array]
            Gradient dictionary with keys ``"pot_slice"``,
            ``"beam"``, and ``"pos_list"``.
        """
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slice, beam, pos_list
        )
        result: Tuple[Float[Array, " "], Dict[str, Array]] = (
            loss,
            {
                "pot_slice": grads[0],
                "beam": grads[1],
                "pos_list": grads[2],
            },
        )
        return result

    learning_rates: Float[Array, ...] = jnp.array(learning_rate)
    if len(learning_rates.shape) == 0:
        parameter_learning_rate: float = float(learning_rates)
        position_learning_rate: float = float(learning_rates)
    else:
        parameter_learning_rate = float(learning_rates[0])
        position_learning_rate = float(learning_rates[1])
    parameter_optimizer: optax.GradientTransformation = _get_optimizer(
        optimizer_name,
        parameter_learning_rate,
    )
    position_optimizer: optax.GradientTransformation = _get_optimizer(
        optimizer_name,
        position_learning_rate,
    )
    pot_guess: Complex[Array, "H W"] = _promote_to_complex(
        initial_potential.data_array
    )
    beam_guess: Complex[Array, "H W"] = _promote_to_complex(
        initial_beam.data_array
    )
    pos_guess: Float[Array, "P 2"] = initial_pos_list
    pot_slice_state: Any = parameter_optimizer.init(pot_guess)
    beam_state: Any = parameter_optimizer.init(beam_guess)
    pos_state: Any = position_optimizer.init(initial_pos_list)

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
        pot_slice_state: Any,
        beam_state: Any,
        pos_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"],
        Complex[Array, "H W"],
        Float[Array, "P 2"],
        Any,
        Any,
        Any,
        Float[Array, " "],
    ]:
        """PRIVATE: Update potential, beam, and positions by one step.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice.
        beam : Complex[Array, "H W"]
            Current beam.
        pos_list : Float[Array, "P 2"]
            Current scan positions, in Angstroms.
        pot_slice_state : Any
            Optimizer state for the potential.
        beam_state : Any
            Optimizer state for the beam.
        pos_state : Any
            Optimizer state for the positions.

        Returns
        -------
        pot_slice : Complex[Array, "H W"]
            Updated potential slice.
        beam : Complex[Array, "H W"]
            Updated beam.
        pos_list : Float[Array, "P 2"]
            Updated scan positions.
        pot_slice_state : Any
            Updated optimizer state for the potential.
        beam_state : Any
            Updated optimizer state for the beam.
        pos_state : Any
            Updated optimizer state for positions.
        loss : Float[Array, " "]
            Scalar loss after this step.
        """
        loss: Float[Array, " "]
        grads: Dict[str, Array]
        loss, grads = _loss_and_grad(pot_slice, beam, pos_list)
        pot_slice_next, pot_slice_state = _apply_optimizer_step(
            pot_slice,
            grads["pot_slice"],
            parameter_optimizer,
            pot_slice_state,
        )
        pot_slice = cast(Complex[Array, "H W"], pot_slice_next)
        beam_next, beam_state = _apply_optimizer_step(
            beam,
            grads["beam"],
            parameter_optimizer,
            beam_state,
        )
        beam = cast(Complex[Array, "H W"], beam_next)
        pos_list_updates, pos_state = position_optimizer.update(
            grads["pos_list"],
            pos_state,
            pos_list,
        )
        pos_list = cast(
            Float[Array, "P 2"],
            optax.apply_updates(
                pos_list,
                pos_list_updates,
            ),
        )
        result: Tuple[
            Complex[Array, "H W"],
            Complex[Array, "H W"],
            Float[Array, "P 2"],
            Any,
            Any,
            Any,
            Float[Array, " "],
        ] = (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            loss,
        )
        return result

    snapshot_count: int = num_iterations // save_every

    intermediate_potslices: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            pot_guess.shape[0],
            pot_guess.shape[1],
            snapshot_count,
        ),
        dtype=pot_guess.dtype,
    )
    intermediate_beams: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            beam_guess.shape[0],
            beam_guess.shape[1],
            snapshot_count,
        ),
        dtype=beam_guess.dtype,
    )
    intermediate_positions: Float[Array, "P 2 S"] = jnp.zeros(
        shape=(
            pos_guess.shape[0],
            pos_guess.shape[1],
            snapshot_count,
        ),
        dtype=pos_guess.dtype,
    )

    def _scan_step(
        carry: Tuple[Any, ...], ii: Int[Array, ""]
    ) -> Tuple[Tuple[Any, ...], Float[Array, " "]]:
        """PRIVATE: Advance one position-corrected reconstruction step.

        Parameters
        ----------
        carry : Tuple[Any, ...]
            Current potential, beam, positions, optimizers, and snapshots.
        ii : Int[Array, ""]
            Zero-based optimization iteration.

        Returns
        -------
        carry : Tuple[Any, ...]
            Updated reconstruction and snapshot state.
        loss : Float[Array, " "]
            Scalar loss for the completed iteration.
        """
        (
            pot_guess,
            beam_guess,
            pos_guess,
            pot_slice_state,
            beam_state,
            pos_state,
            intermediate_potslices,
            intermediate_beams,
            intermediate_positions,
        ) = carry
        (
            pot_guess,
            beam_guess,
            pos_guess,
            pot_slice_state,
            beam_state,
            pos_state,
            loss,
        ) = _update_step(
            pot_guess,
            beam_guess,
            pos_guess,
            pot_slice_state,
            beam_state,
            pos_state,
        )

        def _save_snapshot(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
            """PRIVATE: Save potential, beam, and position snapshots.

            Parameters
            ----------
            args : Tuple[Any, ...]
                Potential, beam, and position snapshot arrays.

            Returns
            -------
            result : Tuple[Any, ...]
                Snapshot arrays with the current estimates stored.
            """
            pots, beams, positions = args
            saver: Int[Array, ""] = (ii // save_every).astype(jnp.int32)
            pots = pots.at[:, :, saver].set(pot_guess)
            beams = beams.at[:, :, saver].set(beam_guess)
            positions = positions.at[:, :, saver].set(pos_guess)
            result: Tuple[Any, ...] = pots, beams, positions
            return result

        (
            intermediate_potslices,
            intermediate_beams,
            intermediate_positions,
        ) = jax.lax.cond(
            ii % save_every == 0,
            _save_snapshot,
            lambda args: args,
            (
                intermediate_potslices,
                intermediate_beams,
                intermediate_positions,
            ),
        )
        result: Tuple[Tuple[Any, ...], Float[Array, " "]] = (
            (
                pot_guess,
                beam_guess,
                pos_guess,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslices,
                intermediate_beams,
                intermediate_positions,
            ),
            loss,
        )
        return result

    if num_iterations > 0:
        (
            pot_guess,
            beam_guess,
            pos_guess,
            pot_slice_state,
            beam_state,
            pos_state,
            _,
        ) = _update_step(
            pot_guess,
            beam_guess,
            pos_guess,
            pot_slice_state,
            beam_state,
            pos_state,
        )
        if snapshot_count > 0:
            intermediate_potslices = intermediate_potslices.at[:, :, 0].set(
                pot_guess
            )
            intermediate_beams = intermediate_beams.at[:, :, 0].set(beam_guess)
            intermediate_positions = intermediate_positions.at[:, :, 0].set(
                pos_guess
            )

        (
            (
                pot_guess,
                beam_guess,
                pos_guess,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslices,
                intermediate_beams,
                intermediate_positions,
            ),
            _,
        ) = jax.lax.scan(
            _scan_step,
            (
                pot_guess,
                beam_guess,
                pos_guess,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslices,
                intermediate_beams,
                intermediate_positions,
            ),
            jnp.arange(1, num_iterations),
            unroll=True,
        )

    final_potential: CalibratedArray = create_calibrated_array(
        data_array=pot_guess,
        calib_y=initial_potential.calib_y,
        calib_x=initial_potential.calib_x,
        real_space=True,
    )
    final_beam: CalibratedArray = create_calibrated_array(
        data_array=beam_guess,
        calib_y=initial_beam.calib_y,
        calib_x=initial_beam.calib_x,
        real_space=True,
    )
    reconstruction_result: Tuple[
        CalibratedArray,
        CalibratedArray,
        Float[Array, "P 2"],
        Complex[Array, "H W S"],
        Complex[Array, "H W S"],
        Float[Array, "P 2 S"],
    ] = (
        final_potential,
        final_beam,
        pos_guess,
        intermediate_potslices,
        intermediate_beams,
        intermediate_positions,
    )
    return reconstruction_result


@jaxtyped(typechecker=beartype)
def single_slice_multi_modal(  # noqa: PLR0915
    experimental_data: STEM4D,
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: ProbeModes,
    slice_thickness: scalar_num,
    save_every: int = 10,
    num_iterations: int = 1000,
    learning_rate: Union[scalar_float, Float[Array, "2"]] = 0.01,
    loss_type: str = "mse",
    optimizer_name: str = "adam",
) -> Tuple[
    Complex[Array, "H W"],
    ProbeModes,
    Float[Array, "P 2"],
    Complex[Array, "H W S"],
    Complex[Array, "H W M S"],
]:
    r"""Reconstruct potential, multi-modal beam, and positions.

    Extended Summary
    ----------------
    Single-slice ptychographic reconstruction that models the
    probe as a superposition of coherent modes stored in a
    :class:`~ptyrodactyl.types.ProbeModes` PyTree.  The
    optimiser simultaneously refines the potential, all probe
    modes, and the scan positions:

    .. math::

        \mathcal{L}
        = \sum_{p}
          \bigl\lVert
            I_p^{\mathrm{exp}}
            - \sum_{m} w_m \,
              \lvert \mathcal{F}\{
                \psi_m \cdot t(V, \mathbf{r}_p)
              \} \rvert^2
          \bigr\rVert^2

    where :math:`\psi_m` are the probe modes with weights
    :math:`w_m` and :math:`t` is the transmission function.

    :see: :func:`~.test_single_slice_multi_modal_regresses`

    Implementation Logic
    --------------------
    1. **Build forward model** --
       Wraps :func:`~ptyrodactyl.multislice.simulations.stem_4d`
       accepting ``(pot_slice, beam, pos_list)`` where *beam*
       is a :class:`~ptyrodactyl.types.ProbeModes` instance.
    2. **Construct loss** --
       Reduces detector errors with the selected loss mode.
    3. **Parse learning rate** --
       Scalar is broadcast to both groups; length-2 array
       splits into potential/beam (index 0) and positions
       (index 1).
    4. **Iterate** --
       Gradients are computed for the potential array, the
       ``modes`` field of :class:`~ptyrodactyl.types.ProbeModes`,
       and positions, then applied with the chosen optimizer.
    5. **Snapshot** --
       Every *save_every* iterations, store the current
       potential and beam modes into intermediate arrays.

    Parameters
    ----------
    experimental_data : :class:`~ptyrodactyl.types.STEM4D`
        Experimental 4D-STEM data PyTree containing diffraction
        patterns, scan positions, and calibration information.
    initial_pot_slice : Complex[Array, "H W"]
        Initial guess for the electrostatic potential slice.
    initial_beam : :class:`~ptyrodactyl.types.ProbeModes`
        Initial multi-modal probe containing mode arrays,
        weights, and calibration.
    slice_thickness : scalar_num
        Thickness of the potential slice, in Angstroms.
    save_every : int, optional
        Store intermediate results every *save_every* iterations.
        Default is ``10``.
    num_iterations : int, optional
        Total number of optimisation iterations.
        Default is ``1000``.
    learning_rate : scalar_float or Float[Array, "2"], optional
        Step size(s) for the optimizer.  If scalar, the same
        rate is used for potential/beam and positions.  If a
        length-2 array, element 0 controls potential/beam and
        element 1 controls positions.  Default is ``0.01``.
    loss_type : str, optional
        Loss reduction identifier.
        Default is ``"mse"``.
    optimizer_name : str, optional
        Key into :data:`OPTIMIZERS`.  Default is ``"adam"``.

    Returns
    -------
    pot_slice : Complex[Array, "H W"]
        Optimised electrostatic potential slice.
    beam : :class:`~ptyrodactyl.types.ProbeModes`
        Optimised multi-modal probe.
    pos_list : Float[Array, "P 2"]
        Refined scan positions, in Angstroms.
    intermediate_potslice : Complex[Array, "H W S"]
        Potential snapshots at saved iterations.
    intermediate_beam : Complex[Array, "H W M S"]
        Beam-mode snapshots at saved iterations.

    See Also
    --------
    :func:`single_slice_ptychography` :
        Single-mode, fixed-position variant.
    :func:`multi_slice_multi_modal` :
        Multi-slice variant with position correction.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    voltage_kv: Float[Array, " "] = jnp.asarray(
        experimental_data.voltage_kv,
        dtype=jnp.float64,
    )
    calib_ang: Float[Array, " "] = jnp.asarray(
        experimental_data.real_space_calib,
        dtype=jnp.float64,
    )
    initial_pos_list: Float[Array, "P 2"] = experimental_data.scan_positions
    microscope = create_microscope_config(
        voltage_kv=voltage_kv,
        aperture_mrad=1.0,
    )

    def _forward_fn(
        pot_slice: Complex[Array, "H W"],
        beam: ProbeModes,
        pos_list: Float[Array, "P 2"],
    ) -> Float[Array, "P H W"]:
        """PRIVATE: Simulate 4D-STEM with multi-modal probe.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Electrostatic potential slice.
        beam : ProbeModes
            Multi-modal probe.
        pos_list : Float[Array, "P 2"]
            Scan positions, in Angstroms.

        Returns
        -------
        result : Float[Array, "P H W"]
            Simulated diffraction patterns.
        """
        potential_slices = create_potential_slices(
            slices=jnp.real(pot_slice)[..., jnp.newaxis],
            slice_thickness=slice_thickness,
            calib=calib_ang,
        )
        detector = create_detector_config(
            real_space_calib_ang=calib_ang,
            scan_positions_px=pos_list,
        )
        stem4d_result = stem_4d(
            potential_slices,
            beam,
            microscope,
            detector,
        )
        result: Float[Array, "P H W"] = stem4d_result.data
        return result

    loss_mode: LossType = LossType(loss_type)
    loss_func: Callable[..., Float[Array, " "]] = jax.jit(
        lambda potential, probe, positions: _reduce_loss(
            _forward_fn(potential, probe, positions),
            experimental_4dstem,
            loss_mode,
        )
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"],
        beam: ProbeModes,
        pos_list: Float[Array, "P 2"],
    ) -> Tuple[Float[Array, " "], Dict[str, Any]]:
        """PRIVATE: Compute loss and gradients for potential, modes, positions.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice estimate.
        beam : ProbeModes
            Current multi-modal probe estimate.
        pos_list : Float[Array, "P 2"]
            Current scan positions, in Angstroms.

        Returns
        -------
        loss : Float[Array, " "]
            Scalar loss value.
        grads : Dict[str, Any]
            Gradient dictionary with keys ``"pot_slice"``,
            ``"beam"``, and ``"pos_list"``.
        """
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slice, beam, pos_list
        )
        result: Tuple[Float[Array, " "], Dict[str, Any]] = (
            loss,
            {
                "pot_slice": grads[0],
                "beam": grads[1],
                "pos_list": grads[2],
            },
        )
        return result

    learning_rates: Float[Array, ...] = jnp.array(learning_rate)
    if len(learning_rates.shape) == 0:
        parameter_learning_rate: float = float(learning_rates)
        position_learning_rate: float = float(learning_rates)
    else:
        parameter_learning_rate = float(learning_rates[0])
        position_learning_rate = float(learning_rates[1])
    parameter_optimizer: optax.GradientTransformation = _get_optimizer(
        optimizer_name,
        parameter_learning_rate,
    )
    position_optimizer: optax.GradientTransformation = _get_optimizer(
        optimizer_name,
        position_learning_rate,
    )
    pot_slice: Complex[Array, "H W"] = _promote_to_complex(initial_pot_slice)
    initial_beam_modes: Complex[Array, "H W M"] = _promote_to_complex(
        initial_beam.modes
    )
    beam: ProbeModes = eqx.tree_at(
        lambda probe: probe.modes,
        initial_beam,
        initial_beam_modes,
    )
    pos_list: Float[Array, "P 2"] = initial_pos_list
    pot_slice_state: Any = parameter_optimizer.init(pot_slice)
    beam_state: Any = parameter_optimizer.init(beam.modes)
    pos_state: Any = position_optimizer.init(initial_pos_list)

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: ProbeModes,
        pos_list: Float[Array, "P 2"],
        pot_slice_state: Any,
        beam_state: Any,
        pos_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"],
        ProbeModes,
        Float[Array, "P 2"],
        Any,
        Any,
        Any,
        Float[Array, " "],
    ]:
        """PRIVATE: Update potential, multi-modal beam, and positions.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice.
        beam : ProbeModes
            Current multi-modal probe.
        pos_list : Float[Array, "P 2"]
            Current scan positions, in Angstroms.
        pot_slice_state : Any
            Optimizer state for the potential.
        beam_state : Any
            Optimizer state for the beam modes.
        pos_state : Any
            Optimizer state for the positions.

        Returns
        -------
        pot_slice : Complex[Array, "H W"]
            Updated potential slice.
        beam : ProbeModes
            Updated multi-modal probe.
        pos_list : Float[Array, "P 2"]
            Updated scan positions.
        pot_slice_state : Any
            Updated optimizer state for the potential.
        beam_state : Any
            Updated optimizer state for beam modes.
        pos_state : Any
            Updated optimizer state for positions.
        loss : Float[Array, " "]
            Scalar loss after this step.
        """
        loss: Float[Array, " "]
        grads: Dict[str, Any]
        loss, grads = _loss_and_grad(pot_slice, beam, pos_list)
        pot_slice_next, pot_slice_state = _apply_optimizer_step(
            pot_slice,
            grads["pot_slice"],
            parameter_optimizer,
            pot_slice_state,
        )
        pot_slice = cast(Complex[Array, "H W"], pot_slice_next)
        beam_modes: Complex[Array, "H W M"]
        beam_modes_next, beam_state = _apply_optimizer_step(
            beam.modes,
            grads["beam"].modes,
            parameter_optimizer,
            beam_state,
        )
        beam_modes = cast(Complex[Array, "H W M"], beam_modes_next)
        beam = eqx.tree_at(
            lambda probe: probe.modes,
            beam,
            beam_modes,
        )
        pos_list_updates, pos_state = position_optimizer.update(
            grads["pos_list"],
            pos_state,
            pos_list,
        )
        pos_list = cast(
            Float[Array, "P 2"],
            optax.apply_updates(
                pos_list,
                pos_list_updates,
            ),
        )
        result: Tuple[
            Complex[Array, "H W"],
            ProbeModes,
            Float[Array, "P 2"],
            Any,
            Any,
            Any,
            Float[Array, " "],
        ] = (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            loss,
        )
        return result

    snapshot_count: int = num_iterations // save_every

    intermediate_potslice: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            initial_pot_slice.shape[0],
            initial_pot_slice.shape[1],
            snapshot_count,
        ),
        dtype=initial_pot_slice.dtype,
    )
    intermediate_beam: Complex[Array, "H W M S"] = jnp.zeros(
        shape=(
            initial_beam.modes.shape[0],
            initial_beam.modes.shape[1],
            initial_beam.modes.shape[2],
            snapshot_count,
        ),
        dtype=initial_beam.modes.dtype,
    )

    def _scan_step(
        carry: Tuple[Any, ...], ii: Int[Array, ""]
    ) -> Tuple[Tuple[Any, ...], Float[Array, " "]]:
        """PRIVATE: Advance one multi-modal reconstruction step.

        Parameters
        ----------
        carry : Tuple[Any, ...]
            Current potential, probe modes, positions, optimizers, and
            snapshots.
        ii : Int[Array, ""]
            Zero-based optimization iteration.

        Returns
        -------
        carry : Tuple[Any, ...]
            Updated reconstruction and snapshot state.
        loss : Float[Array, " "]
            Scalar loss for the completed iteration.
        """
        (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            intermediate_potslice,
            intermediate_beam,
        ) = carry
        (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            loss,
        ) = _update_step(
            pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
        )

        def _save_snapshot(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
            """PRIVATE: Save the current potential and probe-mode snapshots.

            Parameters
            ----------
            args : Tuple[Any, ...]
                Potential and multi-modal beam snapshot arrays.

            Returns
            -------
            result : Tuple[Any, ...]
                Snapshot arrays with the current estimates stored.
            """
            pots, beams = args
            saver: Int[Array, ""] = (ii // save_every).astype(jnp.int32)
            pots = pots.at[:, :, saver].set(pot_slice)
            beams = beams.at[:, :, :, saver].set(beam.modes)
            result: Tuple[Any, ...] = pots, beams
            return result

        intermediate_potslice, intermediate_beam = jax.lax.cond(
            ii % save_every == 0,
            _save_snapshot,
            lambda args: args,
            (intermediate_potslice, intermediate_beam),
        )
        result: Tuple[Tuple[Any, ...], Float[Array, " "]] = (
            (
                pot_slice,
                beam,
                pos_list,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            loss,
        )
        return result

    if num_iterations > 0:
        (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            _,
        ) = _update_step(
            pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
        )
        if snapshot_count > 0:
            intermediate_potslice = intermediate_potslice.at[:, :, 0].set(
                pot_slice
            )
            intermediate_beam = intermediate_beam.at[:, :, :, 0].set(
                beam.modes
            )

        (
            (
                pot_slice,
                beam,
                pos_list,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            _,
        ) = jax.lax.scan(
            _scan_step,
            (
                pot_slice,
                beam,
                pos_list,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            jnp.arange(1, num_iterations),
            unroll=True,
        )

    reconstruction_result: Tuple[
        Complex[Array, "H W"],
        ProbeModes,
        Float[Array, "P 2"],
        Complex[Array, "H W S"],
        Complex[Array, "H W M S"],
    ] = (
        pot_slice,
        beam,
        pos_list,
        intermediate_potslice,
        intermediate_beam,
    )
    return reconstruction_result


@jaxtyped(typechecker=beartype)
def multi_slice_multi_modal(  # noqa: PLR0915
    experimental_data: STEM4D,
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: Complex[Array, "H W"],
    slice_thickness: scalar_num,
    save_every: int = 10,
    num_iterations: int = 1000,
    learning_rate: scalar_float = 0.001,
    pos_learning_rate: scalar_float = 0.01,
    loss_type: str = "mse",
    optimizer_name: str = "adam",
) -> Tuple[
    Complex[Array, "H W"],
    Complex[Array, "H W"],
    Float[Array, "P 2"],
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
]:
    r"""Reconstruct potential, beam, and positions with multi-slice.

    Extended Summary
    ----------------
    Multi-slice ptychographic reconstruction that propagates the
    probe through multiple identical potential slices while
    simultaneously refining the potential, the probe
    wavefunction, and the scan positions.  Separate learning
    rates are used for the potential/beam group and the position
    group:

    .. math::

        \mathcal{L}
        = \sum_{p}
          \bigl\lVert
            I_p^{\mathrm{exp}}
            - I_p^{\mathrm{sim}}(V, \psi, \mathbf{r}_p)
          \bigr\rVert^2

    where the forward model applies the multislice algorithm
    through repeated transmission and propagation steps.

    :see: :func:`~.test_multi_slice_multi_modal_regresses`

    Implementation Logic
    --------------------
    1. **Build forward model** --
       Wraps :func:`~ptyrodactyl.multislice.simulations.stem_4d`
       accepting ``(pot_slice, beam, pos_list)``.
    2. **Construct loss** --
       Reduces detector errors with the selected loss mode.
    3. **Iterate** --
       Gradients are computed for all three variable groups;
       potential and beam use *learning_rate* while positions
       use *pos_learning_rate*.
    4. **Snapshot** --
       Every *save_every* iterations, store the current
       potential and beam into intermediate arrays.

    Parameters
    ----------
    experimental_data : :class:`~ptyrodactyl.types.STEM4D`
        Experimental 4D-STEM data PyTree containing diffraction
        patterns, scan positions, and calibration information.
    initial_pot_slice : Complex[Array, "H W"]
        Initial guess for the electrostatic potential slice.
    initial_beam : Complex[Array, "H W"]
        Initial guess for the electron beam.
    slice_thickness : scalar_num
        Thickness of each potential slice, in Angstroms.
    save_every : int, optional
        Store intermediate results every *save_every* iterations.
        Default is ``10``.
    num_iterations : int, optional
        Total number of optimisation iterations.
        Default is ``1000``.
    learning_rate : scalar_float, optional
        Step size for potential and beam updates.
        Default is ``0.001``.
    pos_learning_rate : scalar_float, optional
        Step size for position updates.
        Default is ``0.01``.
    loss_type : str, optional
        Loss reduction identifier.
        Default is ``"mse"``.
    optimizer_name : str, optional
        Key into :data:`OPTIMIZERS`.  Default is ``"adam"``.

    Returns
    -------
    pot_slice : Complex[Array, "H W"]
        Optimised electrostatic potential slice.
    beam : Complex[Array, "H W"]
        Optimised electron beam.
    pos_list : Float[Array, "P 2"]
        Refined scan positions, in Angstroms.
    intermediate_potslice : Complex[Array, "H W S"]
        Potential snapshots at saved iterations.
    intermediate_beam : Complex[Array, "H W S"]
        Beam snapshots at saved iterations.

    See Also
    --------
    :func:`single_slice_ptychography` :
        Single-slice, single-mode variant.
    :func:`single_slice_multi_modal` :
        Single-slice with multi-modal probe.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    voltage_kv: Float[Array, " "] = jnp.asarray(
        experimental_data.voltage_kv,
        dtype=jnp.float64,
    )
    calib_ang: Float[Array, " "] = jnp.asarray(
        experimental_data.real_space_calib,
        dtype=jnp.float64,
    )
    initial_pos_list: Float[Array, "P 2"] = experimental_data.scan_positions
    microscope = create_microscope_config(
        voltage_kv=voltage_kv,
        aperture_mrad=1.0,
    )

    def _forward_fn(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Float[Array, "P H W"]:
        """PRIVATE: Simulate multi-slice 4D-STEM from potential and beam.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Electrostatic potential slice.
        beam : Complex[Array, "H W"]
            Probe wavefunction in real space.
        pos_list : Float[Array, "P 2"]
            Scan positions, in Angstroms.

        Returns
        -------
        result : Float[Array, "P H W"]
            Simulated diffraction patterns.
        """
        potential_slices = create_potential_slices(
            slices=jnp.real(pot_slice)[..., jnp.newaxis],
            slice_thickness=slice_thickness,
            calib=calib_ang,
        )
        probe_modes = create_probe_modes(
            modes=beam[..., jnp.newaxis],
            weights=jnp.ones((1,), dtype=experimental_4dstem.dtype),
            calib=calib_ang,
        )
        detector = create_detector_config(
            real_space_calib_ang=calib_ang,
            scan_positions_px=pos_list,
        )
        stem4d_result = stem_4d(
            potential_slices,
            probe_modes,
            microscope,
            detector,
        )
        result: Float[Array, "P H W"] = stem4d_result.data
        return result

    loss_mode: LossType = LossType(loss_type)
    loss_func: Callable[..., Float[Array, " "]] = jax.jit(
        lambda potential, probe, positions: _reduce_loss(
            _forward_fn(potential, probe, positions),
            experimental_4dstem,
            loss_mode,
        )
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Tuple[Float[Array, " "], Dict[str, Array]]:
        """PRIVATE: Compute loss and gradients for multi-slice reconstruction.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice estimate.
        beam : Complex[Array, "H W"]
            Current beam estimate.
        pos_list : Float[Array, "P 2"]
            Current scan positions, in Angstroms.

        Returns
        -------
        loss : Float[Array, " "]
            Scalar loss value.
        grads : Dict[str, Array]
            Gradient dictionary with keys ``"pot_slice"``,
            ``"beam"``, and ``"pos_list"``.
        """
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slice, beam, pos_list
        )
        result: Tuple[Float[Array, " "], Dict[str, Array]] = (
            loss,
            {
                "pot_slice": grads[0],
                "beam": grads[1],
                "pos_list": grads[2],
            },
        )
        return result

    parameter_optimizer: optax.GradientTransformation = _get_optimizer(
        optimizer_name,
        learning_rate,
    )
    position_optimizer: optax.GradientTransformation = _get_optimizer(
        optimizer_name,
        pos_learning_rate,
    )
    pot_slice: Complex[Array, "H W"] = _promote_to_complex(initial_pot_slice)
    beam: Complex[Array, "H W"] = _promote_to_complex(initial_beam)
    pos_list: Float[Array, "P 2"] = initial_pos_list
    pot_slice_state: Any = parameter_optimizer.init(pot_slice)
    beam_state: Any = parameter_optimizer.init(beam)
    pos_state: Any = position_optimizer.init(initial_pos_list)

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
        pot_slice_state: Any,
        beam_state: Any,
        pos_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"],
        Complex[Array, "H W"],
        Float[Array, "P 2"],
        Any,
        Any,
        Any,
        Float[Array, " "],
    ]:
        """PRIVATE: Update potential, beam, and positions for multi-slice.

        Parameters
        ----------
        pot_slice : Complex[Array, "H W"]
            Current potential slice.
        beam : Complex[Array, "H W"]
            Current beam.
        pos_list : Float[Array, "P 2"]
            Current scan positions, in Angstroms.
        pot_slice_state : Any
            Optimizer state for the potential.
        beam_state : Any
            Optimizer state for the beam.
        pos_state : Any
            Optimizer state for the positions.

        Returns
        -------
        pot_slice : Complex[Array, "H W"]
            Updated potential slice.
        beam : Complex[Array, "H W"]
            Updated beam.
        pos_list : Float[Array, "P 2"]
            Updated scan positions.
        pot_slice_state : Any
            Updated optimizer state for the potential.
        beam_state : Any
            Updated optimizer state for the beam.
        pos_state : Any
            Updated optimizer state for positions.
        loss : Float[Array, " "]
            Scalar loss after this step.
        """
        loss: Float[Array, " "]
        grads: Dict[str, Array]
        loss, grads = _loss_and_grad(pot_slice, beam, pos_list)
        pot_slice_next, pot_slice_state = _apply_optimizer_step(
            pot_slice,
            grads["pot_slice"],
            parameter_optimizer,
            pot_slice_state,
        )
        pot_slice = cast(Complex[Array, "H W"], pot_slice_next)
        beam_next, beam_state = _apply_optimizer_step(
            beam,
            grads["beam"],
            parameter_optimizer,
            beam_state,
        )
        beam = cast(Complex[Array, "H W"], beam_next)
        pos_list_updates, pos_state = position_optimizer.update(
            grads["pos_list"],
            pos_state,
            pos_list,
        )
        pos_list = cast(
            Float[Array, "P 2"],
            optax.apply_updates(
                pos_list,
                pos_list_updates,
            ),
        )
        result: Tuple[
            Complex[Array, "H W"],
            Complex[Array, "H W"],
            Float[Array, "P 2"],
            Any,
            Any,
            Any,
            Float[Array, " "],
        ] = (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            loss,
        )
        return result

    snapshot_count: int = num_iterations // save_every

    intermediate_potslice: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            initial_pot_slice.shape[0],
            initial_pot_slice.shape[1],
            snapshot_count,
        ),
        dtype=initial_pot_slice.dtype,
    )
    intermediate_beam: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            initial_beam.shape[0],
            initial_beam.shape[1],
            snapshot_count,
        ),
        dtype=initial_beam.dtype,
    )

    def _scan_step(
        carry: Tuple[Any, ...], ii: Int[Array, ""]
    ) -> Tuple[Tuple[Any, ...], Float[Array, " "]]:
        """PRIVATE: Advance one multi-slice reconstruction step.

        Parameters
        ----------
        carry : Tuple[Any, ...]
            Current potential, beam, positions, optimizers, and snapshots.
        ii : Int[Array, ""]
            Zero-based optimization iteration.

        Returns
        -------
        carry : Tuple[Any, ...]
            Updated reconstruction and snapshot state.
        loss : Float[Array, " "]
            Scalar loss for the completed iteration.
        """
        (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            intermediate_potslice,
            intermediate_beam,
        ) = carry
        (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            loss,
        ) = _update_step(
            pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
        )

        def _save_snapshot(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
            """PRIVATE: Save the current multi-slice reconstruction snapshots.

            Parameters
            ----------
            args : Tuple[Any, ...]
                Potential and beam snapshot arrays.

            Returns
            -------
            result : Tuple[Any, ...]
                Snapshot arrays with the current estimates stored.
            """
            pots, beams = args
            saver: Int[Array, ""] = (ii // save_every).astype(jnp.int32)
            pots = pots.at[:, :, saver].set(pot_slice)
            beams = beams.at[:, :, saver].set(beam)
            result: Tuple[Any, ...] = pots, beams
            return result

        intermediate_potslice, intermediate_beam = jax.lax.cond(
            ii % save_every == 0,
            _save_snapshot,
            lambda args: args,
            (intermediate_potslice, intermediate_beam),
        )
        result: Tuple[Tuple[Any, ...], Float[Array, " "]] = (
            (
                pot_slice,
                beam,
                pos_list,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            loss,
        )
        return result

    if num_iterations > 0:
        (
            pot_slice,
            beam,
            pos_list,
            pot_slice_state,
            beam_state,
            pos_state,
            _,
        ) = _update_step(
            pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
        )
        if snapshot_count > 0:
            intermediate_potslice = intermediate_potslice.at[:, :, 0].set(
                pot_slice
            )
            intermediate_beam = intermediate_beam.at[:, :, 0].set(beam)

        (
            (
                pot_slice,
                beam,
                pos_list,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            _,
        ) = jax.lax.scan(
            _scan_step,
            (
                pot_slice,
                beam,
                pos_list,
                pot_slice_state,
                beam_state,
                pos_state,
                intermediate_potslice,
                intermediate_beam,
            ),
            jnp.arange(1, num_iterations),
            unroll=True,
        )

    reconstruction_result: Tuple[
        Complex[Array, "H W"],
        Complex[Array, "H W"],
        Float[Array, "P 2"],
        Complex[Array, "H W S"],
        Complex[Array, "H W S"],
    ] = (
        pot_slice,
        beam,
        pos_list,
        intermediate_potslice,
        intermediate_beam,
    )
    return reconstruction_result


__all__: list[str] = [
    "OPTIMIZERS",
    "multi_slice_multi_modal",
    "single_slice_multi_modal",
    "single_slice_poscorrected",
    "single_slice_ptychography",
]
