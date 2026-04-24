"""
jablonski.simulation
~~~~~~~~~~~~~~~~~~~~

Simulation functions.

:copyright: 2024 by jablonski Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from itertools import chain, pairwise
from typing import Iterable, Mapping

import numpy as np
import numpy.typing as npt
import pint
import scipy.constants as constants
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
    join_by_energy: bool = False,
) -> xr.Dataset:
    """Single transition square excitation."""

    lines = {
        f"line_{transition}": transition
        for transition in util.emission_transitions(system, kind=kind)
    }

    transform = {k: v.radiative_decay.rate_law for k, v in lines.items()}

    sim = Simulator(system, transform=transform, append_transform=True)
    ds = piecewise(sim, events=excitation, save_at=save_at)
    if not join_by_energy:
        for line in lines:
            ds.attrs[line] = lines[line].energy_difference
        return ds[list(lines.keys())]
    else:
        return lines_to_energies(lines, ds)


def spectral_steady_state_emission(
    system: SpectroscopicSystem,
    excitation_transition: Pumper | Iterable[Pumper],
    height: float | Iterable[float],
    kind: util.SpectraKind = "emission",
    join_by_energy: bool = False,
) -> xr.Dataset:

    lines = {
        f"line_{transition}": transition
        for transition in util.emission_transitions(system, kind=kind)
    }

    transform = {k: v.radiative_decay.rate_law for k, v in lines.items()}

    sim = Simulator(system, transform=transform)
    steady = SteadyState()

    if not isinstance(excitation_transition, Iterable):
        excitation_transition = [excitation_transition]
    if not isinstance(height, Iterable):
        height = [height] * len(excitation_transition)
    ds = steady.solve(
        sim,
        values={
            excitation.pump: value
            for excitation, value in zip(excitation_transition, height)
        },
    )
    # ds = steady.solve(sim, values={excitation_transition.pump: height})
    if not join_by_energy:
        for line in lines:
            ds.attrs[line] = lines[line].energy_difference
        return ds
    else:
        return lines_to_energies(lines, ds)


def time_resolved_emission(
    system: SpectroscopicSystem,
    excitation: dict[Time, Mapping[Components, Initial | Real | None]],
    save_at: npt.NDArray[np.float64],
    kind: util.SpectraKind = "emission",
):
    spectral = spectral_time_resolved_emission(system, excitation, save_at, kind)

    summed = spectral.to_array().sum(dim="variable")
    return summed.to_dataset(name="emission")


def steady_state_emission(
    system: SpectroscopicSystem,
    excitation_transition: Pumper,
    height: float,
    kind: util.SpectraKind = "emission",
):
    spectral = spectral_steady_state_emission(
        system, excitation_transition, height, kind
    )

    summed = spectral.to_array().sum(dim="variable")
    return summed.to_dataset(name="emission")


def excitation_spectra(
    excitation: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
    emission: pint.Quantity | tuple[pint.Quantity, pint.Quantity],
):
    """CW excitation spectra."""


# TODO: how to excite multiple pumpers? Can't be a dictionary because they are not hashable.
def emission_spectra(
    system: SpectroscopicSystem,
    excitation_transition: Pumper | Iterable[Pumper],
    height: float | Iterable[float],
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
):
    """CW emission spectra."""
    if isinstance(unit, str):
        unit = ureg[unit]
    spectral = spectral_steady_state_emission(
        system, excitation_transition, height, kind, join_by_energy=True
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
            [spectral[energy].values.item() for energy in spectral.data_vars.keys()]
        ),
        dims="wavelenght",
        coords={"wavelenght": wavelenghts},
    ).pint.quantify(unit, pint_xarray.setup_registry(unit._REGISTRY))
    unit._REGISTRY.force_ndarray_like = False
    da.name = "spectrum"
    return da


def widened_emission_spectra(
    system: SpectroscopicSystem,
    excitation_transition: Pumper | Iterable[Pumper],
    height: float | Iterable[float],
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
    samples: Iterable[float] = np.linspace(380, 700, 1000),
    width: float = 5,
):
    """CW emission spectra."""
    if isinstance(unit, str):
        unit = ureg[unit]

    spectral = spectral_steady_state_emission(
        system, excitation_transition, height, kind
    )
    h = constants.h * ureg.J * ureg.s
    c = constants.c * ureg.m / ureg.s
    wavelenghts = {
        line: (c * h / energy).to(unit).magnitude
        for line, energy in spectral.attrs.items()
    }

    # TODO: What profile should lines have? Should it be an argument with doppler/natural?
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


def graph_spectra(
    system: SpectroscopicSystem,
    excitation_transition: Pumper | Iterable[Pumper],
    height: float,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
    samples: Iterable[float] = np.linspace(380, 700, 1000),
    width: float = 5,  # TODO: what is the right width?
):
    spectra = widened_emission_spectra(
        system, excitation_transition, height, unit, kind, samples
    )

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    points = spectra["wavelenght"].values
    spectrum = spectra.values
    plot_points = np.array([points, spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([plot_points[:-1], plot_points[1:]], axis=1)
    lc = LineCollection(segments, cmap="nipy_spectral")
    lc.set_array(points)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_xlim(points.min(), points.max())
    ax.set_ylim(spectrum.min(), spectrum.max())
    ax.set_xlabel(f"Wavelenght [ {unit} ]")
    ax.set_ylabel("Emission [ photons/s ]")
    return fig, ax


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
