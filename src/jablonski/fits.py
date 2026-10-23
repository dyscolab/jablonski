from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pint
from poincare import Simulator
from poincare.analysis.fits import Fitter, FitResult, UnitsHandler
from poincare.simulator import Components
from poincare.types import Initial, Number
import xarray as xr

from . import util
from ._typing import Excitation, RadiativeDecay, Pumper
from ._units import ureg
from .simulation import (
    Time,
    spectra,
    spectral_time_resolved,
)


def fit_spectral_time_resolved(
    sim: Simulator,
    results: xr.Dataset,
    excitation: Mapping[Time, Excitation],
    kind: util.SpectraKind | Iterable[util.SpectraKind] = "emission",
    join_by_energy: bool = False,
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    sigma: xr.Dataset | None = None,
    **kwargs,
) -> FitResult:
    """Fit time resolved spectra (see make_time_resolved_target).

    sigma is the experimental error, of the same form as results. Extra
    keyword arguments are passed to scipy.optimize.least_squares.
    """
    # TODO: as is it will fit align the elements of the spectra associating larger to smaller wavelengths
    # regardless of whether they match. Should this be the behavior/ Or should it check with a
    # certain tolerance?
    save_at = UnitsHandler.get_save_at(results)

    def simulation_function(sim: Simulator) -> xr.Dataset:
        ds = spectral_time_resolved(
            sim=sim,
            excitation=excitation,
            save_at=save_at,
            kind=kind,
            join_by_energy=join_by_energy,
        )
        return _select_times(ds, save_at)

    fitter = Fitter(
        sim,
        [simulation_function],
        [results],
        p0=p0,
        scales=1 if scale is None else scale,
        sigma=None if sigma is None else [sigma],
    )
    return fitter.fit(**kwargs)


def _select_times(ds: xr.Dataset, save_at: pint.Quantity) -> xr.Dataset:
    """Keep only the times in save_at: piecewise also returns the times of the
    excitation events."""
    times = UnitsHandler.get_save_at(ds)
    if isinstance(times, pint.Quantity):
        times = times.to(ureg.s).magnitude
    return ds.isel(time=np.isin(times, save_at.to(ureg.s).magnitude))


def make_time_resolved_target(
    results: Mapping[Any, Sequence | pint.Quantity],
    save_at: Sequence | pint.Quantity,
    join_by_energy: bool = False,
) -> xr.Dataset:
    ureg = pint.get_application_registry()

    ureg.force_ndarray_like = True

    try:
        data_vars = {}

        if isinstance(save_at, pint.Quantity):
            units_map = {"time": save_at.units}
            dequantified_save_at = save_at.magnitude

        else:
            units_map = {}
            dequantified_save_at = save_at

        for key, value in results.items():
            if join_by_energy:
                key = str(key.energy_difference)
            else:
                prefix = (
                    "emission_"
                    if isinstance(key, RadiativeDecay)
                    else ("absorption_" if isinstance(key, Pumper) else "")
                )
                # Same names as the simulation output (see util.lines_and_transform)
                key = prefix + str(key) if prefix else str(key)
            if isinstance(value, pint.Quantity):
                units_map[key] = value.units
                dequantified_value = value.magnitude
            else:
                dequantified_value = value

            data_vars[key] = xr.DataArray(
                dequantified_value,
                dims=["time"],
            )

        ds = xr.Dataset(data_vars=data_vars, coords={"time": dequantified_save_at})

        ds = ds.pint.quantify(units_map)

    finally:
        ureg.force_ndarray_like = False

    return ds


def fit_spectra(
    sim: Simulator,
    results: xr.DataArray,
    excitation: Excitation,
    kind: util.SpectraKind | Iterable[util.SpectraKind] = "emission",
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    sigma: xr.DataArray | None = None,
    **kwargs,
) -> FitResult:
    """Fit CW spectra (see make_spectra_target).

    sigma is the experimental error, of the same form as results. Extra
    keyword arguments are passed to scipy.optimize.least_squares.
    """
    # TODO: as is it will fit align the elements of the spectra associating larger to smaller wavelengths
    # regardless of whether they match. Should this be the be? haviour/ Or should it check with a
    # certain tolerance?

    if results.xindexes:
        (index_name,) = results.xindexes.keys()
        index = getattr(results, index_name)
        unit = index.pint.units
    else:
        unit = None

    # Named so that scale can refer to it
    results = results.rename(results.name or "spectra")

    fitter = Fitter(
        sim,
        [lambda sim: spectra(sim=sim, excitation=excitation, unit=unit, kind=kind)],
        [results],
        p0=p0,
        scales=1 if scale is None else scale,
        sigma=None if sigma is None else [sigma],
    )
    return fitter.fit(**kwargs)


def make_spectra_target(wavelengths: Sequence | pint.Quantity, intensities: Sequence):
    if isinstance(wavelengths, pint.Quantity):
        import pint_xarray

        unit = wavelengths.units
        da = xr.DataArray(
            data=intensities,
            dims="wavelength",
            coords={"wavelength": wavelengths.magnitude},
        ).pint.quantify(
            {"wavelength": unit},
            pint_xarray.setup_registry(unit._REGISTRY),
        )
        unit._REGISTRY.force_ndarray_like = False
        return da
    else:
        return xr.DataArray(
            data=intensities,
            dims="wavelength",
            coords={"wavelength": wavelengths},
        )


# =============================================================================
# Backward Compatibility Wrappers
# =============================================================================


def fit_spectral_time_resolved_emission(
    sim: Simulator,
    results: xr.Dataset,
    excitation: Mapping[Time, Excitation],
    join_by_energy: bool = False,
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    sigma: xr.Dataset | None = None,
    **kwargs,
) -> FitResult:
    return fit_spectral_time_resolved(
        sim=sim,
        results=results,
        excitation=excitation,
        kind="emission",
        join_by_energy=join_by_energy,
        p0=p0,
        scale=scale,
        sigma=sigma,
        **kwargs,
    )


def make_time_resolved_emission_target(
    results: Mapping[RadiativeDecay, Sequence | pint.Quantity],
    save_at: Sequence | pint.Quantity,
    join_by_energy: bool = False,
) -> xr.Dataset:
    return make_time_resolved_target(
        results=results, save_at=save_at, join_by_energy=join_by_energy
    )


def fit_emission_spectra(
    sim: Simulator,
    results: xr.DataArray,
    excitation: Excitation,
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    sigma: xr.DataArray | None = None,
    **kwargs,
) -> FitResult:
    return fit_spectra(
        sim=sim,
        results=results,
        excitation=excitation,
        kind="emission",
        p0=p0,
        scale=scale,
        sigma=sigma,
        **kwargs,
    )