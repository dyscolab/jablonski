# Changelog

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
