# Changelog

## Unreleased

- Added `TripletTripletAnnihilation` (`triplet_1`, `triplet_2`, `singlet_1`, `singlet_2`, `rate`): the mass-action reaction T + T -> S + S, with rate `rate * triplet_1 * triplet_2`. The four states can be different levels (for example T1 + T1 -> S1 + S0), and passing the same state twice gives 2 T -> 2 S. It raises `ValueError` when the singlets' total energy is higher than the triplets'.
- fix: `model_report` no longer leaves its Jablonski diagram open, so notebooks don't display it under the report, and no longer switches the matplotlib backend, which closed the user's open figures.
- Added `jablonski.plots.notebook_report` to show a model report with a Jablonski diagram (an embedded PNG image) in marimo or Jupyter notebooks.
- The Jablonski diagram is now the first section of `model_report` (and of `notebook_report`), before the equations. Requires the modular `notebook_report` in poincare (unreleased).
- Added `ConstantGroundState`, a `Parameter` with an `energy` (default 0 eV) whose population is held constant. It can replace a singlet or triplet state in any transition and passes both the same- and different-multiplicity checks. Added the `SingletStateLike`, `TripletStateLike` and `SpinStateLike` type aliases for transition states. Requires constant species support in poincare (unreleased).
- Added `n_runs` to all rebop simulation functions to run the stochastic simulation several times and return the average.
- fix: `rng` in rebop simulations is turned into one generator shared by all piecewise segments, instead of reusing the same seed in each segment.
- fix: `transitions.check_all` works again. It raised `AttributeError` because transition rates are `Parameter`s. Rates are now resolved through linked parameters and converted to the unit of their expected range (1/s, or cm² for absorption cross-sections) before comparing. Rates given as expressions are skipped. The warning message is formatted, and inverted limits (`Absorption`, `TripletTripletAbsorption`) are ordered.
- Breaking: `EnergyTransferUpconversion4` is renamed `EnergyTransferUpconversion` (same keywords: `sensitizer_high`, `sensitizer_low`, `activator_low`, `activator_high`). The previous three-state `EnergyTransferUpconversion` is removed. The energy check of the transfer is removed, so endothermic transfers are accepted. Migration: replace `EnergyTransferUpconversion4` with `EnergyTransferUpconversion`, and write three-state transfers as a custom transition.
- Breaking: spectral line outputs are named `emission_<transition>` (and `absorption_<transition>`) instead of `line_<transition>`. Migration: rename the keys, or use the totals from `time_resolved`/`steady_state`.
- Breaking: `util.emission_transitions` and `util.excitation_transitions` are replaced by `util.yield_transitions(system, kind=)` (`kind="absorption"` for pumpers). Added `util.lines_and_transform`.
- Simulation functions have generic names, because they also handle absorption: `spectral_time_resolved`, `time_resolved`, `spectral_steady_state`, `steady_state`, `spectra`, `widened_spectra`, `sweeps.sweep_spectra` and `sweeps.sweep_spectral_steady_state`. The `*_emission` names are kept as wrappers.
- `kind` accepts `"absorption"` (absorbed photon rate of each pumper) and lists of kinds such as `["absorption", "fluorescence"]`.
- Added `quantum_yield`, `spectral_quantum_yield` and `sweeps.sweep_quantum_yield` (steady state, emitted / absorbed photons).
- Added `jablonski.fits`: `fit_spectral_time_resolved`, `make_time_resolved_target`, `fit_spectra`, `make_spectra_target`, and the `*_emission` wrappers, built on poincare's `Fitter`.
- Added `rebop_spectral_time_resolved` and `rebop_time_resolved` (with `*_emission` wrappers) for stochastic photon counts. The time of rebop results is in the unit of `upto_t`.
- `jablonski_diagram` draws each subsystem in its own columns and draws energy transfer upconversions, colour-coded per transfer (`etu_colors=`). Transitions may set lists of states in `_source`/`_target`.
- fix: an excitation event at the start of `save_at` (for example at t = 0) is applied from the start instead of one segment late, and very short pulses and events at the end of `save_at` no longer break `piecewise`.
- fix: spectra (`spectra`, `excitation_emission_matrix`, `excitation_spectra`, sweeps) carry photon-rate units instead of the wavelength unit. `excitation_spectra` picks the nearest emission wavelength instead of requiring an exact match.

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
