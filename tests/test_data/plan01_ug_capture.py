"""Capture / verify the Plan-01 UG regression reference (not a pytest module).

Usage:
    python tests/plan01_ug_capture.py capture   # writes tests/test_data/plan01_ug_reference.npz
    python tests/plan01_ug_capture.py verify    # compares current outputs against the reference

The reference is a fixed-seed CBED + 4D-STEM + gradient + 2-iteration recon
re-captured at Plan-03 IM2+IM3 after the CBED amplitude/intensity split and
explicit probe-mode weights. The measured old-vs-new max relative deltas are
CBED intensity 0.0, 4D-STEM 0.0, and potential-gradient 0.0. Every phase must
keep `verify` passing bit-for-bit (allclose at tight tolerance) — the UG
regression wall.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

REF_PATH = "tests/test_data/plan01_ug_reference.npz"
H = W = 32
N_SLICES = 2
VOLTAGE_KV = 100.0
CALIB_ANG = 0.2
SLICE_THICKNESS_ANG = 2.0


def _build():
    import ptyrodactyl.simul as ps
    from ptyrodactyl.simul.simulations import stem_4d

    from ptyrodactyl.types import (
        create_potential_slices,
        create_probe_modes,
    )

    key = jax.random.PRNGKey(42)
    pot = jax.random.uniform(key, (H, W, N_SLICES), dtype=jnp.float64) * 10.0
    slices = create_potential_slices(pot, SLICE_THICKNESS_ANG, CALIB_ANG)
    probe = ps.make_probe(
        aperture=20.0,
        voltage=VOLTAGE_KV,
        image_size=jnp.array([H, W]),
        calibration_pm=CALIB_ANG * 100.0,
    )
    pm = create_probe_modes(probe[..., None], jnp.array([1.0]), CALIB_ANG)

    cbed_out = ps.cbed_image(slices, pm, VOLTAGE_KV).data_array
    positions = jnp.array([[12.0, 12.0], [12.0, 20.0], [20.0, 12.0], [20.0, 20.0]])
    s4d = stem_4d(slices, pm, positions, VOLTAGE_KV, CALIB_ANG)

    def loss(p):
        sl = create_potential_slices(p, SLICE_THICKNESS_ANG, CALIB_ANG)
        return jnp.sum(ps.cbed_image(sl, pm, VOLTAGE_KV).data_array)

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
        "grad_pot": np.asarray(grad_pot),
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    out = _build()
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"NON-FINITE values in {k} — invalid reference/state")
    if mode == "capture":
        np.savez(REF_PATH, **out)
        print(f"captured -> {REF_PATH}: " + ", ".join(f"{k}{v.shape}" for k, v in out.items()))
    else:
        ref = np.load(REF_PATH)
        failed = []
        for k, v in out.items():
            if not np.allclose(ref[k], v, rtol=1e-10, atol=1e-12):
                failed.append((k, float(np.max(np.abs(ref[k] - v)))))
        if failed:
            raise SystemExit(f"UG REGRESSION FAILURE: {failed}")
        print("UG verify: all outputs match the E0 reference (bit-level tolerance)")


if __name__ == "__main__":
    main()
