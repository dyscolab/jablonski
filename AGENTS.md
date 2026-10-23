# AGENTS.md — jablonski

Instructions for coding agents working **on** this repository. User-facing
examples are the notebooks in [examples/](examples/). Record user-visible
changes in [CHANGELOG.md](CHANGELOG.md).

## Project overview

jablonski describes photophysical systems (Jablonski diagrams) as ODE models
and simulates them. A user declares electronic states with energies and the
transitions between them. Radiative transitions are absorption,
fluorescence and phosphorescence. Non-radiative ones are vibrational
relaxation, internal conversion, intersystem crossing and energy-transfer
upconversion. Each transition becomes a poincare `MassAction` reaction.
jablonski then provides spectroscopic workflows on top of
`poincare.Simulator`:

- time-resolved and steady-state emission/absorption,
- spectra and widened spectra,
- quantum yields and excitation–emission matrices,
- piecewise laser excitation,
- sweeps, fits, stochastic (rebop) runs,
- Jablonski-diagram plots.

It has no solver or compiler of its own. It depends on poincare (and through
it on symbolite) and **not** on simbio. Typical tasks are adding a transition
type, fixing spectral or units logic, and extending simulation, sweep or fit
helpers.

## Repository map

| Path | Responsibility | Touch when |
| --- | --- | --- |
| `src/jablonski/states.py` | `SpinState`/`SingletState`/`TripletState` (poincare `Variable` + `energy`), `initial(energy, multiplicity, default=)`, `SpectroscopicSystem` (time `Independent` in seconds, `_check` hook) | States, energy units |
| `src/jablonski/transitions.py` | Transition classes, each a `SpectroscopicSystem` wrapping a `MassAction`, with `_check` physical validation | Adding or fixing a transition |
| `src/jablonski/_typing.py` | Runtime protocols `Pumper`, `RadiativeDecay`, `Drawable`, and the types `Excitation`, `Time` | Changing how transitions are discovered |
| `src/jablonski/util.py` | `yield_transitions`, `lines_and_transform` (spectral line → rate expression), `SpectraKind` | Which transitions count as emission or absorption |
| `src/jablonski/simulation.py` | `piecewise` excitation, `step`/`pulse`/`delta_excitation`, `spectral_*`, `time_resolved`, `steady_state`, `quantum_yield`, `spectra`, `widened_spectra`, `excitation_*`, and the `*_emission` compatibility wrappers | Simulation workflows |
| `src/jablonski/sweeps.py`, `fits.py` | Parameter sweeps, fit wrappers around `poincare.analysis.fits.Fitter` | Sweeps and fitting |
| `src/jablonski/rebop/` | Stochastic counterparts using `poincare.reactions.rebop.RebopSimulator` | Stochastic runs |
| `src/jablonski/plots/` | `graph_spectra`, `jablonski_diagram`, `model_report` (extends poincare LaTeX) | Plots and reports |
| `src/jablonski/helpers.py`, `_units.py` | `pump_from_laser`, shared pint registry and dimension constants | Laser excitation, units |
| Tests | `src/jablonski/tests/`, `src/jablonski/rebop/test_rebop.py` | Always |

## Architecture and core abstractions

- **States** are poincare `Variable`s (populations) with a pint `energy`
  attribute stored in eV. `initial()` reads a plain float as eV and converts
  wavelength, wavenumber or frequency through pint's `"spectroscopy"`
  context. Any other dimensionality raises `ValueError`.
- **Transitions** are small `System`s. Their states are declared **from
  source to target** (module docstring of `transitions.py`), and they set
  `self._source`/`self._target` in `__init__`. They expose
  `energy_difference` and validate it in `_check`, which also checks
  multiplicity rules and runs on instantiation. The attribute *names* are
  part of the contract, because they are discovered by protocol:
  - `Pumper`: `pump` (photon flux parameter) plus `absorption` (the
    `MassAction`). Example: `Absorption`.
  - `RadiativeDecay`: `radiative_decay` (the `MassAction`). Examples:
    `Fluorescence`, `Phosphorescence`.
  - `Drawable`: `_source`/`_target`, used by `plots/`.
- **Spectral observables.** `util.lines_and_transform` maps each discovered
  transition to `"emission_<name>"` or `"absorption_<name>"`, with the
  `MassAction.rate_law` as the transform expression. Signals are therefore
  photon *rates*, and `join_by_energy` sums lines with the same
  `energy_difference`.
- **Excitation** is a mapping `{pumper: photon_flux}`. Time-resolved runs
  use `piecewise(sim, events={t: excitation}, save_at=...)`, which merges
  events at the same time and integrates segment by segment with
  `Simulator.with_values`.

## Invariants and constraints

Verified:

- **Units.** Absorption `rate` is a cross-section (cm²) and `pump` is a
  photon flux (1/(cm² s)) (CHANGELOG 0.2.0). Event times and `save_at` in
  `piecewise` must be pint quantities with time dimensionality, otherwise
  `PintError` is raised. Keep quantities in pint. Do not strip them to bare
  floats inside the API.
- **Physical validation.** Transitions raise when `energy_difference < 0`
  and when multiplicities break the transition's spin rule (for example
  `VibrationalRelaxation` and `InternalConversion` need equal
  multiplicities). Keep these checks when you add transitions.
- **Consistency between workflows**, all tested in `tests/test_simulation.py`:
  - step and pulse excitations of equal dose give the same result,
  - `quantum_yield` equals emission / absorption from `steady_state`,
  - the `*_emission` wrappers are **identical** to the generic functions
    (`test_backward_compatibility_wrappers`).
- **Simulation helpers take a `Simulator`, not a `System`** (CHANGELOG 0.4.0).

Established conventions:

- Each module that imports `pint_xarray` resets
  `pint.get_application_registry().force_ndarray_like = False`
  (see the poincare AGENTS.md).
- Expected rate ranges are documented in comments and only checked as
  warnings with `check_all()` / `_check(check_range=True)`.

Unresolved (report, do not decide):

- Some `_check_range` limits are inverted (for example `Absorption` passes
  `1e-14, 1e-16` as min and max) and are marked `TODO`. The correct ranges
  are not established.
- `util.convert()` refers to undefined names (`unit`, `system`, `include`).
  It is dead or unfinished code.
- The `Pumper` protocol declares `absorption: Parameter`, but transitions
  define `absorption` as a `MassAction`.

## Development environment and commands

The pixi manifest is in [pyproject.toml](pyproject.toml). These environments
install the **published** poincare and symbolite. If your change relies on
unreleased poincare or symbolite behaviour, test with source installs of
them and say so in your report.

```bash
pixi run test                                   # pytest --doctest-modules (default env: py312, test)
pixi run -e test-py314 test                     # CI matrix: test-py312, test-py313, test-py314
pixi run test src/jablonski/tests/test_simulation.py
pixi run -e lint lint                           # `pre-commit run`: only *staged* files
pixi run -e lint pre-commit run --files <f>     # lint specific files
pixi run -e build _build                        # uv build
```

- The tree currently has about 47 ruff findings (including `F821`
  undefined names) and 14 unformatted files. Fix only what you touch, and
  report pre-existing `F821`s near your change.
- Pyright is configured, but its pre-commit hook is commented out. No type
  check is enforced.
- The example notebooks are not tested.

Before calling a change done, run `pixi run test` and run ruff on the
changed files.

## Making changes

- **New transition:** copy the closest existing class in `transitions.py`.
  - Declare states source → target with `initial(0.0, spin_multiplicity=...)`.
  - Declare `rate: Parameter = assign()` and wrap the `MassAction` in an
    attribute named `radiative_decay` (emissive), `absorption` plus `pump`
    (pumped), or a descriptive name for non-radiative ones.
  - Set `_source`/`_target` and define `energy_difference` and `_check`.
  - If it is a new spectral kind, register it in `util.TYPE_MAP`.
  - Add a test in `tests/test_transitions.py` that simulates a correct
    model **and** asserts with `pytest.raises` that invalid ones fail.
- **New simulation or sweep helper:** take a `Simulator` and derive new
  ones with `with_values`/`with_transform` rather than rebuilding them. Keep
  the outputs as pint-aware `xr.Dataset`/`DataArray`.
- **Changes to excitation or `piecewise`:** keep equal-dose equivalence and
  event merging, and run `test_piecewise*` and `test_excitations`.
- **Plots:** `plots/` relies on `Drawable` and on the `_source`/`_target`
  shapes. Single variables and sequences are both supported, for ETU.

## Testing and scientific correctness

- Check photophysics against closed forms where they exist:
  - single-exponential decay with lifetime 1/k,
  - steady-state populations of two- and three-level systems,
  - energy ↔ wavelength via `h c / ΔE`,
  - a quantum yield between 0 and 1 for pure radiative/non-radiative
    branching.
- Use `np.testing.assert_allclose` with explicit `rtol`/`atol` (the existing
  tests use rtol `1e-3` for widened spectra). Compare dequantified data only
  after the units have been checked.
- Tests that "expect" an error must use `pytest.raises`. Some existing
  `test_transitions.py` cases use `try/except: pass`, which cannot fail. Do
  not copy that pattern.
- Do not weaken assertions, delete tests or increase tolerances to hide
  failures.

## Ecosystem compatibility

- Consumes from poincare: `System`, `Variable`, `Independent`, `Parameter`,
  `assign`, `Simulator`, `SteadyState`, `solvers` (`LSODA`, `Solver`),
  `reactions.MassAction`/`Reactant`, `reactions.rebop.RebopSimulator`,
  `analysis.fits` (`Fitter`, `FitResult`, `UnitsHandler`),
  `printing.latex` (`ToLatex`, `Latex`, `model_report`, `default_sections`, `default_packages`),
  `simulator.Components`, `types.Initial`, and the private `_node.Node`.
- Fix bugs that live in these poincare APIs in poincare. Its AGENTS.md then
  requires running the jablonski and simbio suites against that change. No
  package depends on jablonski.

## Documentation and public API

- The public API is `jablonski.__all__` (`transitions`, `simulation`,
  `sweeps`, `util`, states, `pump_from_laser`, re-exported poincare names)
  plus `jablonski.plots`, `jablonski.fits` and `jablonski.rebop`.
- Give every transition a docstring with its physical meaning and the units
  of `rate`. The comment with the typical timescale should be kept.
- Keep the `*_emission` wrappers until a maintainer approves removing them.
  Record API changes in CHANGELOG.md.

## Completion checklist

- [ ] Transition attribute names still satisfy `Pumper`, `RadiativeDecay` and `Drawable`, and states are declared source → target.
- [ ] Energy and rate units are preserved, and physical `_check`s are kept or added.
- [ ] Tests check physics against closed forms, and error cases use `pytest.raises`.
- [ ] `pixi run test` passes, and ruff is clean on the changed files.
- [ ] CHANGELOG.md updated, and unverified assumptions reported to the user.
