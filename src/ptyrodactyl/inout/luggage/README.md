# Atomic form-factor tables

`Lobato_van_Dyck.csv` contains the five `a_i` and five `b_i`
coefficients for neutral atoms H--Lr (Z = 1--103) from I. Lobato and
D. Van Dyck, “An accurate parameterization for scattering factors,
electron densities and electrostatic potentials for neutral atoms that
obey all physical constraints,” *Acta Crystallographica A* **70**
(2014), 636--649, DOI: 10.1107/S205327331401643X.

The Lobato asset is copied from Rheedium's reference table and is
numerically identical to the program's canonical
`ptyrodactyl-plans/physics/lobato.json`. Its source columns are
`Z, element, a1..a5, b1..b5`; the package loader returns the ten
coefficients interleaved as `a1, b1, ..., a5, b5`.

`Kirkland_Potentials.csv` contains the three Lorentzian and three
Gaussian amplitude/scale pairs for H--Lr from E. J. Kirkland,
*Advanced Computing in Electron Microscopy*. The package loader returns
its existing interleaved order unchanged.

Both files are validated for their exact element count, coefficient
shape, finite entries, and positive scale coefficients at package import.
