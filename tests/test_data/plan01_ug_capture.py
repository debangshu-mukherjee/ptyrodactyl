"""Capture / verify the Plan-01 UG regression reference (not a pytest module).

Usage:
    python tests/plan01_ug_capture.py capture
    python tests/plan01_ug_capture.py verify

The reference is a fixed-seed CBED + 4D-STEM + sharded 4D-STEM + gradient
capture. It was re-captured at Plan-03 IM7 after carrier bundling of the
forward-simulation arguments. The IM7 pre-vs-post carrier re-capture had
EXACTLY zero numeric delta by np.array_equal for CBED intensity, 4D-STEM,
sharded 4D-STEM, and potential-gradient outputs. Every phase must keep
`verify` passing bit-for-bit with exact array equality — the UG regression
wall.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

import ptyrodactyl.multislice as ps
from ptyrodactyl.multislice.parallelized import stem4d_sharded
from ptyrodactyl.multislice.simulations import stem_4d
from ptyrodactyl.types import (
    create_atomic_slice_data,
    create_detector_config,
    create_microscope_config,
    create_potential_slices,
    create_probe_modes,
)

REF_PATH = "tests/test_data/plan01_ug_reference.npz"
H = W = 32
N_SLICES = 2
VOLTAGE_KV = 100.0
CALIB_ANG = 0.2
SLICE_THICKNESS_ANG = 2.0


def _build():
    key = jax.random.PRNGKey(42)
    pot = jax.random.uniform(key, (H, W, N_SLICES), dtype=jnp.float64) * 10.0
    slices = create_potential_slices(pot, SLICE_THICKNESS_ANG, CALIB_ANG)
    microscope = create_microscope_config(
        voltage_kv=VOLTAGE_KV,
        aperture_mrad=20.0,
        probe_shape=jnp.array([H, W]),
    )
    detector = create_detector_config(
        real_space_calib_ang=CALIB_ANG,
        probe_calibration_pm=CALIB_ANG * 100.0,
    )
    probe = ps.make_probe(
        microscope=microscope,
        detector=detector,
    )
    pm = create_probe_modes(probe[..., None], jnp.array([1.0]), CALIB_ANG)

    cbed_out = ps.cbed_image(slices, pm, microscope).data_array
    positions = jnp.array(
        [[12.0, 12.0], [12.0, 20.0], [20.0, 12.0], [20.0, 20.0]]
    )
    stem_detector = create_detector_config(
        real_space_calib_ang=CALIB_ANG,
        probe_calibration_pm=CALIB_ANG * 100.0,
        scan_positions_px=positions,
    )
    s4d = stem_4d(slices, pm, microscope, stem_detector)
    atom_coords = jnp.array(
        [[2.0, 2.5, 0.25], [4.5, 5.0, 2.25], [3.0, 4.0, 1.25]],
        dtype=jnp.float64,
    )
    atom_types = jnp.array([0, 0, 0], dtype=jnp.int32)
    slice_z_bounds = jnp.array([[0.0, 2.0], [2.0, 4.0]], dtype=jnp.float64)
    atom_potentials = pot[..., 0][jnp.newaxis, ...]
    sample = create_atomic_slice_data(
        atom_coords=atom_coords,
        atom_types=atom_types,
        slice_z_bounds=slice_z_bounds,
        atom_potentials=atom_potentials,
    )
    sharded_detector = create_detector_config(
        real_space_calib_ang=CALIB_ANG,
        probe_calibration_pm=CALIB_ANG * 100.0,
        scan_positions_ang=positions * CALIB_ANG,
    )
    s4d_sharded = stem4d_sharded(pm, sample, microscope, sharded_detector)

    def loss(p):
        sl = create_potential_slices(p, SLICE_THICKNESS_ANG, CALIB_ANG)
        return jnp.sum(ps.cbed_image(sl, pm, microscope).data_array)

    grad_pot = jax.grad(loss)(pot)

    # NOTE: the short-recon leg specified by Plan 01 UG is EXCLUDED: at E0,
    # single_slice_ptychography is broken for any num_iterations/save_every
    # (jnp.floor(...) used as an array shape, phase_recon.py:316 — the
    # pre-existing plumbing bug documented in Plan 04 §A.6). The reference
    # therefore pins the forward models + gradient; recon regression is owned
    # by Plan 04, which fixes that path.
    return {
        "cbed": np.asarray(cbed_out),
        "stem4d": np.asarray(s4d.data),
        "stem4d_sharded": np.asarray(s4d_sharded.data),
        "grad_pot": np.asarray(grad_pot),
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    out = _build()
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise SystemExit(
                f"NON-FINITE values in {k} — invalid reference/state"
            )
    if mode == "capture":
        np.savez(REF_PATH, **out)
        shapes = ", ".join(f"{k}{v.shape}" for k, v in out.items())
        print(f"captured -> {REF_PATH}: {shapes}")
    else:
        ref = np.load(REF_PATH)
        failed = []
        for k, v in out.items():
            if not np.array_equal(ref[k], v):
                failed.append((k, float(np.max(np.abs(ref[k] - v)))))
        if failed:
            raise SystemExit(f"UG REGRESSION FAILURE: {failed}")
        print("UG verify: all outputs match the IM7 reference exactly")


if __name__ == "__main__":
    main()
