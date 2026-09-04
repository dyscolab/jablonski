from typing import Hashable, Iterable

import numpy as np
import pint
from poincare import Simulator
import xarray as xr
from poincare.solvers import LSODA, Solver

from . import util
from ._typing import Excitation
from ._units import ureg
from .simulation import emission_spectra, spectral_steady_state_emission
from .states import SpectroscopicSystem
from .util import SpectraKind


def sweep_spectral_steady_state_emission(
    sim: Simulator,
    excitations: Iterable[Excitation],
    keys: Iterable[Hashable] | None = None,
    kind: util.SpectraKind = "emission",
    join_by_energy: bool = False,
):
    ds = xr.Dataset()
    if keys is None:
        if np.all([len(excitation) == 1 for excitation in excitations]):
            keys = [next(iter(excitation.values())) for excitation in excitations]
        else:
            raise (
                ValueError(
                    "sweep_spectral_steady_state_emission must pass an explicit keys argument if any excitation in excitations has more than one argument"
                )
            )
    elif len(keys) != len(excitations):
        raise (ValueError("excitation and keys are different lengths"))
    for key, excitation in zip(keys, excitations):
        ds[key] = (
            spectral_steady_state_emission(
                sim=sim,
                excitation=excitation,
                kind=kind,
                join_by_energy=join_by_energy,
            )
            .to_dataarray(dim="energy" if join_by_energy else "line")
            .drop_vars("time")
            .squeeze()
        )
    return ds


def sweep_emission_spectra(
    sim: Simulator,
    excitations: Iterable[Excitation],
    keys: Iterable[Hashable] | None = None,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
):
    ds = xr.Dataset()
    if keys is None:
        if np.all([len(excitation) == 1 for excitation in excitations]):
            keys = [next(iter(excitation.values())) for excitation in excitations]
        else:
            raise (
                ValueError(
                    "sweep_emission_spectra must pass an explicit keys argument if any excitation in excitations has more than one item"
                )
            )
    elif len(keys) != len(excitations):
        raise (ValueError("excitation and keys are different lenghts"))
    for key, excitation in zip(keys, excitations):
        ds[key] = emission_spectra(
            sim=sim,
            excitation=excitation,
            unit = unit,
            kind=kind,
        )
    return ds
