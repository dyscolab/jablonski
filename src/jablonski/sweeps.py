from typing import Iterable

import pint
import xarray as xr
from poincare.solvers import Solver, LSODA

from . import util
from ._typing import Pumper
from ._units import ureg
from .simulation import emission_spectra, spectral_steady_state_emission
from .states import SpectroscopicSystem
from .util import SpectraKind


def sweep_spectral_steady_state_emission(
    system: SpectroscopicSystem,
    excitation_transition: Pumper | Iterable[Pumper],
    heights: Iterable[float],
    kind: util.SpectraKind = "emission",
    join_by_energy: bool = False,
    solver: Solver = LSODA(),
):
    ds = xr.Dataset()
    for height in heights:
        ds[height] = (
            spectral_steady_state_emission(
                system=system,
                excitation_transition=excitation_transition,
                height=height,
                kind=kind,
                join_by_energy=join_by_energy,
                solver=solver,
            )
            .to_dataarray(dim="energy" if join_by_energy else "line")
            .drop_vars("time")
            .squeeze()
        )
    return ds


def sweep_emission_spectra(
    system: SpectroscopicSystem,
    excitation_transition: Pumper | Iterable[Pumper],
    heights: Iterable[float],
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
    solver: Solver = LSODA(),
):
    ds = xr.Dataset()
    for height in heights:
        ds[height] = emission_spectra(
            system=system,
            excitation_transition=excitation_transition,
            height=height,
            kind=kind,
            solver=solver,
        )
    return ds
