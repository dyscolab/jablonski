"""
jablonski.simulation
~~~~~~~~~~~~~~~~~~~~

Simulation functions.

:copyright: 2024 by jablonski Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from itertools import chain, pairwise
from typing import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd  # delete once all references to pandas once they are converted to arrays
import pint
import xarray as xr
from poincare import Simulator, SteadyState
from poincare.simulator import Components, Initial
from symbolite import Real

from . import util
from ._typing import Pumper, Time
from ._units import DEFAULT_DELTA, ureg
from .states import SpectroscopicSystem
from .util import SpectraKind


def piecewise(
    sim: Simulator,
    *,
    events: dict[Time, Mapping[Components, Initial | Real | None]],
    save_at: npt.NDArray[np.float64],
) -> xr.Dataset:
    event_keys = [
        k.m_as("s") if isinstance(k, pint.Quantity) else k for k in events.keys()
    ]
    t_events = np.sort(event_keys)
    save_at = np.union1d(save_at, t_events)
    pos = np.searchsorted(save_at, t_events)
    save_ats = np.split(save_at, pos + 1)
    t_spans = pairwise(chain((0,), t_events, (save_at[-1],)))
    # TODO: support save_at with units?

    dss = []
    state = {}
    for t_span, save_at in zip(t_spans, save_ats):
        ds = sim.solve(t_span=t_span, save_at=save_at, values=state)
        for k, v in events.get(save_at[-1], {}).items():
            if v is None and k in state:
                del state[k]
            else:
                state[k] = v
            # str(k) porque en el output no usamos el objeto Variable aun
            as_str = str(k)
            if as_str in ds:
                ds[as_str].values[-1] = v

        state.update({k: ds[str(k)].values[-1] for k in sim.compiled.variables})
        dss.append(ds)

    ds = xr.concat(dss, dim="time")
    # df = df.d
    return ds


def step_excitation(
    excitation_transition: Pumper, height: float, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Real | None]]:
    return {
        start: {excitation_transition.pump: height},
    }


def pulse_excitation(
    excitation_transition: Pumper, height: float, width: Time, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Real | None]]:
    return {
        start: {excitation_transition.pump: height},
        (start + width): {excitation_transition.pump: None},
    }


def delta_excitation(
    excitation_transition: Pumper, area: Time, start: Time = 0 * ureg.s
) -> dict[Time, Mapping[Components, Initial | Real | None]]:
    width = DEFAULT_DELTA
    height = area / width
    return pulse_excitation(excitation_transition, height, width, start)


def spectral_time_resolved_emission(
    system: SpectroscopicSystem,
    excitation: dict[Time, Mapping[Components, Initial | Real | None]],
    save_at: npt.NDArray[np.float64],
    kind: util.SpectraKind = "emission",
) -> pd.DataFrame:
    """Single transition square excitation."""

    lines = {
        f"line{ndx}": transition
        for ndx, transition in enumerate(util.emission_transitions(system, kind=kind))
    }

    transform = {k: v.radiative_decay.rate_law for k, v in lines.items()}

    sim = Simulator(system, transform=transform, append_transform=True)
    ds = piecewise(sim, events=excitation, save_at=save_at)
    for line in lines:
        ds.attrs[line] = lines[line].energy_difference
    return ds[list(lines.keys())]
    # TODO: return lines or energies? Add parameter to toggle it?
    # return lines_to_energies(lines, ds)


def spectral_steady_state_emission(
    system: SpectroscopicSystem,
    excitation_transition: Pumper,
    height: float,
    kind: util.SpectraKind = "emission",
) -> pd.DataFrame:

    lines = {
        f"line{ndx}": transition
        for ndx, transition in enumerate(util.emission_transitions(system, kind=kind))
    }

    transform = {k: v.radiative_decay.rate_law for k, v in lines.items()}

    sim = Simulator(system, transform=transform)

    steady = SteadyState()
    ds = steady.solve(sim, values={excitation_transition.pump: height})

    for line in lines:
        ds.attrs[line] = lines[line].energy_difference
    return ds[list(lines.keys())]
    # return lines_to_energies(lines, ds)


def time_resolved_emission():
    pass


def excitation_spectra(
    excitation: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
    emission: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
):
    """CW excitation spectra."""


def emission_spectra(
    system: SpectroscopicSystem,
    excitation_transition: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
    height: float,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
):
    """CW emission spectra."""
    # steady_state_emission(system, excitation_transition, height, kind)
    # for k, v in util.emission_transitions(system, unit, kind).items()


def lines_to_energies(lines: Mapping[str, Pumper], ds: xr.Dataset) -> xr.Dataset:
    energies = {}
    for line, transition in lines.items():
        if transition.energy_difference in energies:
            energies[transition.energy_difference].append(line)
        else:
            energies[transition.energy_difference] = [line]
    energies_ds = xr.Dataset(
        {
            # sum over all lines with th same energy
            energy: xr.concat([ds[line] for line in e_lines], dim="temp").sum(
                dim="temp"
            )
            for energy, e_lines in energies.items()
        }
    )
    return energies_ds
