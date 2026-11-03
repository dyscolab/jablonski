# Changelog

## Unreleased

- fix: `model_report` no longer leaves its Jablonski diagram open, so notebooks don't display it under the report, and no longer switches the matplotlib backend, which closed the user's open figures.
- Added `jablonski.plots.notebook_report` to show a model report with a Jablonski diagram (an embedded PNG image) in marimo or Jupyter notebooks.
- The Jablonski diagram is now the first section of `model_report` (and of `notebook_report`), before the equations. Requires the modular `notebook_report` in poincare (unreleased).
- Added `ConstantGroundState`, a `Parameter` with an `energy` (default 0 eV) whose population is held constant. It can replace a singlet or triplet state in any transition and passes both the same- and different-multiplicity checks. Added the `SingletStateLike`, `TripletStateLike` and `SpinStateLike` type aliases for transition states. Requires constant species support in poincare (unreleased).
- Added `n_runs` to all rebop simulation functions to run the stochastic simulation several times and return the average.
- fix: `rng` in rebop simulations is turned into one generator shared by all piecewise segments, instead of reusing the same seed in each segment.

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
