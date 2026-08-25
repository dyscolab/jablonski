"""
jablonski.simulation
~~~~~~~~~~~~~~~~~~~~

Simulation functions.

:copyright: 2024 by jablonski Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from itertools import chain, pairwise
from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt
import pint
import pint_xarray
import scipy.constants as constants
import xarray as xr

from poincare import Simulator, SteadyState
from poincare.simulator import Components, Initial
from symbolite import Real

from . import util
from ._typing import Pumper, Time, Excitation
from ._units import DEFAULT_DELTA, ureg
from .states import SpectroscopicSystem
from .util import SpectraKind

pint.get_application_registry().force_ndarray_like = False
# When pint-xarray is imported it sets it to true, it can break poincare compilation


def piecewise(
    sim: Simulator,
    *,
    events: dict[Time, Mapping[Components, Initial | Real | None]],
    save_at: npt.NDArray[np.float64],
) -> xr.Dataset:
    try:
        event_keys = np.array([key.to(ureg.s).magnitude for key in events.keys()])
    except (AttributeError, pint.DimensionalityError):
        raise pint.PintError(
            "events keys must be pint Quantities and have time dimensionality."
        )
    
    try:
        adimensional_save_at = save_at.to(ureg.s).magnitude
    except (AttributeError, pint.DimensionalityError):
        raise pint.PintError(
            "save_at must be pint Quantity and have time dimensionality."
        )
    t_events = np.sort(event_keys)
    adimensional_save_at = np.union1d(adimensional_save_at, t_events)
    pos = np.searchsorted(adimensional_save_at, t_events)
    adimensional_save_ats = np.split(adimensional_save_at, pos + 1)
    adimensional_t_spans = pairwise(chain((0,), t_events, (adimensional_save_at[-1],)))
    t_spans = [span * ureg.s for span in adimensional_t_spans]
    save_ats = [save * ureg.s for save in adimensional_save_ats]
    dss = []
    state = {}
    for t_span, save_at in zip(t_spans, save_ats):
        ds = sim.with_values(state).solve(t_span=t_span, save_at=save_at)
        for k, v in events.get(save_at[-1], {}).items():
            if v is None and k in state:
                del state[k]
            else:
                state[k] = v
            # str(k) porque en el output no usamos el objeto Variable aun
            as_str = str(k)
            if as_str in ds:
                ds[as_str][-1] = v

        state.update({k: ds[str(k)][-1].item() for k in sim.compiled.variables})
        dss.append(ds.pint.dequantify())

    ds = xr.concat(dss, dim="time")

    pint_xarray.setup_registry(ureg)
    ds = ds.pint.quantify()
    ureg.force_ndarray_like = False
    return ds


def step_excitation(
    excitation: Excitation, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Real | None]]:
    return {
        start: {pumper.pump: height for pumper, height in excitation.items()},
    }


def pulse_excitation(
    excitation: Excitation, width: Time, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Real | None]]:
    return {
        start: {pumper.pump: height for pumper, height in excitation.items()},
        (start + width): {pumper.pump: None for pumper in excitation.keys()},
    }


def delta_excitation(
    excitation_transition: Pumper, area: Time, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Real | None]]:
    width = DEFAULT_DELTA
    height = area / width
    return pulse_excitation({excitation_transition: height}, width, start)


def spectral_time_resolved_emission(
    sim: Simulator,
    excitation: dict[Time, Mapping[Components, Initial | Real | None]],
    save_at: npt.NDArray[np.float64],
    kind: util.SpectraKind = "emission",
    join_by_energy: bool = False,
) -> xr.Dataset:
    """Single transition square excitation."""
    lines = {
        f"line_{transition}": transition
        for transition in util.emission_transitions(sim.model, kind=kind)
    }

    transform = {k: v.radiative_decay.rate_law for k, v in lines.items()}

    sim = sim.with_transform(transform, append=True)
    ds = piecewise(sim, events=excitation, save_at=save_at)
    if not join_by_energy:
        for line in lines:
            ds.attrs[line] = lines[line].energy_difference
        return ds[list(lines.keys())]
    else:
        return lines_to_energies(lines, ds)


def spectral_steady_state_emission(
    sim: Simulator,
    excitation: Excitation,
    kind: util.SpectraKind = "emission",
    join_by_energy: bool = False,
) -> xr.Dataset:

    lines = {
        f"line_{transition}": transition
        for transition in util.emission_transitions(sim.model, kind=kind)
    }

    transform = {k: v.radiative_decay.rate_law for k, v in lines.items()}
    sim = sim.with_transform(transform)
    steady = SteadyState()
    sim = sim.with_values(
        {excitation.pump: height for excitation, height in excitation.items()}
    )
    ds = steady.solve(sim)
    if not join_by_energy:
        for line in lines:
            ds.attrs[line] = lines[line].energy_difference
        return ds
    else:
        return lines_to_energies(lines, ds)


def time_resolved_emission(
    sim: Simulator,
    excitation: dict[Time, Mapping[Components, Initial | Real | None]],
    save_at: npt.NDArray[np.float64],
    kind: util.SpectraKind = "emission",
):
    spectral = spectral_time_resolved_emission(sim, excitation, save_at, kind)
    summed = spectral.to_array().sum(dim="variable")
    return summed.to_dataset(name="emission")


def steady_state_emission(
    sim: Simulator,
    excitation: Excitation,
    kind: util.SpectraKind = "emission",
):
    spectral = spectral_steady_state_emission(sim, excitation, kind)
    summed = spectral.to_array().sum(dim="variable")
    return summed.to_dataset(name="emission")


# TODO: how to excite multiple pumpers? Can't be a dictionary because they are not hashable.
def emission_spectra(
    sim: Simulator,
    excitation: Excitation,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
):
    """CW emission spectra."""
    if isinstance(unit, str):
        unit = ureg[unit]
    spectral = spectral_steady_state_emission(
        sim, excitation, kind, join_by_energy=True
    )
    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = np.array(
        [
            (c * h / spectral.attrs[energy]).to(unit).magnitude
            for energy in spectral.data_vars.keys()
        ]
    )
    import pint_xarray

    da = xr.DataArray(
        data=np.array(
            [
                spectral[energy].pint.dequantify().values.item()
                for energy in spectral.data_vars.keys()
            ]
        ),
        dims="wavelenght",
        coords={"wavelenght": wavelenghts},
    ).pint.quantify(unit, pint_xarray.setup_registry(unit._REGISTRY))
    unit._REGISTRY.force_ndarray_like = False
    da.name = "spectrum"
    return da


def excitation_emission_matrix(
    sim: Simulator,
    height: pint.Quantity,
    unit: str | pint.Unit = ureg.nm,
):
    results = {}
    for pumper in sim.model._yield(Pumper):
        results[str(pumper)] = emission_spectra(
            sim,
            excitation={pumper: height},
            unit=unit,
        )
    return xr.Dataset(results)


def excitation_spectra(
    sim: Simulator,
    emission: float | int | pint.Quantity,
    height: pint.Quantity,
    unit: str | pint.Unit = ureg.nm,
):
    """CW excitation spectra."""
    if isinstance(unit, str):
        unit = ureg[unit]
    if not isinstance(emission, pint.Quantity):
        emission = emission * unit
    emission = emission.to(unit).magnitude
    matrix = excitation_emission_matrix(
        sim=sim, height=height, unit=unit
    )
    ds = matrix.sel(wavelenght=emission).drop_vars("wavelenght")
    return ds.to_dataarray(dim="pumper", name="exitation spectra")


def widened_emission_spectra(
    sim: Simulator,
    excitation: Excitation,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
    samples: Iterable[float] = np.linspace(380, 700, 1000),
    width: float = 5,
):
    """CW emission spectra."""
    if isinstance(unit, str):
        unit = ureg[unit]
    spectral = spectral_steady_state_emission(sim, excitation, kind)
    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = {
        line: (c * h / energy).to(unit).magnitude
        for line, energy in spectral.attrs.items()
    }

    def gaussian(x, mu, A, sigma):
        return A * np.exp(-(((x - mu) / sigma) ** 2))

    if isinstance(samples, pint.Quantity):
        samples = samples.to(unit).magnitude

    spectrum = np.zeros_like(samples, dtype=float)

    for line, wavelenght in wavelenghts.items():
        spectrum += gaussian(
            samples,
            wavelenght,
            spectral[line].values.item(),
            width,
        )

    import pint_xarray

    da = xr.DataArray(
        data=spectrum,
        dims="wavelenght",
        coords={"wavelenght": samples},
    ).pint.quantify(unit, pint_xarray.setup_registry(unit._REGISTRY))
    unit._REGISTRY.force_ndarray_like = False
    da.name = "spectrum"
    return da


def lines_to_energies(lines: Mapping[str, Pumper], ds: xr.Dataset) -> xr.Dataset:
    energies = {}
    for line, transition in lines.items():
        if transition.energy_difference in energies:
            energies[transition.energy_difference].append(line)
        else:
            energies[transition.energy_difference] = [line]
    energies_ds = xr.Dataset(
        {
            # sum over all lines with the same energy
            str(energy): xr.concat([ds[line] for line in e_lines], dim="temp").sum(
                dim="temp"
            )
            for energy, e_lines in energies.items()
        }
    )
    for energy in energies.keys():
        energies_ds.attrs[str(energy)] = energy
    return energies_ds