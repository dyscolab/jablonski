from typing import Hashable, Iterable

import numpy as np
import pint
import pint_xarray
from poincare import Simulator
import xarray as xr
from poincare.solvers import LSODA, Solver

from . import util
from ._typing import Excitation
from ._units import ureg
from .simulation import (
    quantum_yield,
    spectra,
    spectral_steady_state,
)
from .states import SpectroscopicSystem
from .util import SpectraKind


def sweep_spectral_steady_state(
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
                    "sweep_spectral_steady_state must pass an explicit keys argument if any excitation in excitations has more than one argument"
                )
            )
    elif len(keys) != len(excitations):
        raise (ValueError("excitation and keys are different lengths"))
    for key, excitation in zip(keys, excitations):
        ds[key] = (
            spectral_steady_state(
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


def sweep_spectra(
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
                    "sweep_spectra must pass an explicit keys argument if any excitation in excitations has more than one item"
                )
            )
    elif len(keys) != len(excitations):
        raise (ValueError("excitation and keys are different lengths"))
    for key, excitation in zip(keys, excitations):
        ds[key] = spectra(
            sim=sim,
            excitation=excitation,
            unit=unit,
            kind=kind,
        )
    return ds


def sweep_quantum_yield(
    sim: Simulator,
    excitations: Iterable[Excitation],
    keys: Iterable[Hashable] | None = None,
    kind: util.SpectraKind = "emission",
) -> xr.DataArray:
    excitations = list(excitations)
    if keys is None:
        if np.all([len(excitation) == 1 for excitation in excitations]):
            keys = [next(iter(excitation.values())) for excitation in excitations]
        else:
            raise ValueError(
                "sweep_quantum_yield must pass an explicit keys argument if any excitation in excitations has more than one item"
            )
    else:
        keys = list(keys)
    if len(keys) != len(excitations):
        raise ValueError("excitation and keys are different lengths")

    values = [
        quantum_yield(sim=sim, excitation=excitation, kind=kind)
        .drop_vars("time", errors="ignore")
        .squeeze()
        .pint.dequantify()
        .values.item()
        for excitation in excitations
    ]

    if all(isinstance(k, pint.Quantity) for k in keys):
        coord_unit = keys[0].units
        reg = keys[0]._REGISTRY
        magnitudes = np.array([k.to(coord_unit).magnitude for k in keys])
        da = xr.DataArray(
            data=np.array(values),
            dims=["excitation"],
            coords={"excitation": magnitudes},
            name="quantum_yield",
        ).pint.quantify(
            units="dimensionless",
            excitation=coord_unit,
            unit_registry=pint_xarray.setup_registry(reg),
        )
        reg.force_ndarray_like = False
    else:
        reg = pint.get_application_registry()
        da = xr.DataArray(
            data=np.array(values),
            dims=["excitation"],
            coords={"excitation": list(keys)},
            name="quantum_yield",
        ).pint.quantify(
            units="dimensionless",
            unit_registry=pint_xarray.setup_registry(reg),
        )
        reg.force_ndarray_like = False
    return da


# =============================================================================
# Backward Compatibility Wrappers
# =============================================================================


def sweep_spectral_steady_state_emission(
    sim: Simulator,
    excitations: Iterable[Excitation],
    keys: Iterable[Hashable] | None = None,
    kind: util.SpectraKind = "emission",
    join_by_energy: bool = False,
):
    return sweep_spectral_steady_state(
        sim=sim,
        excitations=excitations,
        keys=keys,
        kind=kind,
        join_by_energy=join_by_energy,
    )


def sweep_emission_spectra(
    sim: Simulator,
    excitations: Iterable[Excitation],
    keys: Iterable[Hashable] | None = None,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
):
    return sweep_spectra(
        sim=sim,
        excitations=excitations,
        keys=keys,
        unit=unit,
        kind=kind,
    )
