# Changelog

## 0.5.0

- Requires poincare 1.3.0.
- Breaking: the wavelength coordinate is named `wavelength` instead of `wavelenght`, and `excitation_spectra` names its result `excitation spectra`. Migration: replace `wavelenght` with `wavelength`.
- Breaking: `EnergyTransferUpconversion4` is renamed `EnergyTransferUpconversion`, replacing the three-state version. Its energy check is removed.
- Breaking: spectral line outputs are named `emission_<transition>` (or `absorption_<transition>`) instead of `line_<transition>`.
- Breaking: `util.emission_transitions` and `util.excitation_transitions` are replaced by `util.yield_transitions(system, kind=)`. Added `util.lines_and_transform`.
- Simulation functions have generic names (`spectral_time_resolved`, `time_resolved`, `spectral_steady_state`, `steady_state`, `spectra`, `widened_spectra`, and the sweeps), since they also handle absorption. The `*_emission` names remain as wrappers. `kind` accepts `"absorption"` and lists of kinds.
- Simulation times are optional and fall back to the simulator's `with_times`, which is now the recommended way. `piecewise` starts at the simulator's `t_span[0]` when set.
- Added `quantum_yield`, `spectral_quantum_yield` and `sweeps.sweep_quantum_yield`.
- Added `jablonski.fits` for fitting time-resolved data and spectra, built on poincare's `Fitter`.
- Added `rebop_spectral_time_resolved` and `rebop_time_resolved` for stochastic photon counts, and `n_runs` to all rebop functions.
- Added `TripletTripletAnnihilation` (T + T -> S + S).
- Added `ConstantGroundState`, a state with constant population usable in any transition, and the `SingletStateLike`, `TripletStateLike` and `SpinStateLike` type aliases.
- Added `plots.notebook_report` with a Jablonski diagram. The diagram is now the first section of `model_report`.
- `jablonski_diagram` draws each subsystem in its own columns and draws energy transfer upconversions.
- Ships its part of the `dyscolab-modeling` agent skill. `python -m jablonski.skill` is an alias of `python -m poincare.skill`.
- fix: spectral lines and spectra are returned in photon-rate units (1/s), instead of each transition's rate unit or the wavelength unit. `excitation_spectra` picks the nearest emission wavelength.
- fix: excitation events at the start of `save_at`, very short pulses, and events at the end of `save_at` are handled correctly by `piecewise`.
- fix: rebop simulations share one random generator across piecewise segments instead of reusing the seed.
- fix: `transitions.check_all` works again, with unit-aware rate comparison.
- fix: `model_report` no longer leaves its diagram open or switches the matplotlib backend.

## 0.4.0

- Updated poincare dependency to 1.2.0, brings breaking changes in values, transform and simulator API (with_values/transform/simulator instead of as argument) and in switching warning on wrong simulation units to error.

- All simulation functions now take simulators instead of system's.

- fix: model report now properly passes on transform to the oncare base version.

## 0.3.0

- Added pump_from_laser helper to make excitations.
- All simulation functions now take a dict{pumper: height} instead of separate excitation_transition and height parameters.
- Added stochastic simulations with rebop.
- Simulation and sweep functions now take a solver as an argument.
- New EnergyTransferUpconversion4 transition with high and low levels for sensitizer and activator.
- `graph_spectra` now takes figsize as argument.

## 0.2.0

- Absorption rates are now cross sections ($cm^2$ units), pumps a are now photon flux (1/($cm^2$ s) units).
- Added generic SpinState to represent SingletState or TripletState.

## 0.1.0

- First Jablonski release.
