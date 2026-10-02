from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pint
from poincare import Simulator
from poincare.analysis.fits import Fitter, UnitsHandler
from poincare.simulator import Components
from poincare.types import Initial, Number
import xarray as xr

from . import util
from ._typing import Excitation, RadiativeDecay
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
    kind: util.SpectraKind = "emission",
    join_by_energy: bool = False,
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    **kwargs,
):
    # TODO: as is it will fit align the elements of the spectra associating larger to smaller wavelengths
    # regardless of whether they match. Should this be the be? haviour/ Or should it check with a
    # certain tolerance?

    def simulation_function_generator(fitter: Fitter) -> Callable[[list], xr.Dataset]:
        save_at = fitter.units.get_save_at(results)

        def simulation_function(x):
            return spectral_time_resolved(
                sim=sim.with_values(
                    {fitter.fit_parameters[i]: val for i, val in enumerate(x)}
                ),
                excitation=excitation,
                save_at=save_at,
                kind=kind,
                join_by_energy=join_by_energy,
            )

        return simulation_function

    fitter = Fitter(
        sim=sim,
        results=results,
        simulation_function_generator=simulation_function_generator,
        units_handler=UnitsHandler,
        p0=p0,
        scale=scale,
    )
    return fitter.solve(**kwargs)


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
                key = "line_" + str(key)
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
    kind: util.SpectraKind = "emission",
    p0: Mapping[
        Components, Initial | tuple[Initial | None, Initial, Initial] | None
    ] = {},  # read only
    scale: Mapping[Components | str, Number] | None = None,
    **kwargs,
):
    # TODO: as is it will fit align the elements of the spectra associating larger to smaller wavelengths
    # regardless of whether they match. Should this be the be? haviour/ Or should it check with a
    # certain tolerance?

    if results.xindexes:
        (index_name,) = results.xindexes.keys()
        index = getattr(results, index_name)
        unit = index.pint.units
    else:
        unit = None

    name = results.name or "spectra"
    results_ds = xr.Dataset({name: results})

    def simulation_function_generator(fitter: Fitter) -> Callable[[list], xr.Dataset]:
        def simulation_function(x):
            res_spectra = spectra(
                sim=sim.with_values(
                    {fitter.fit_parameters[i]: val for i, val in enumerate(x)}
                ),
                excitation=excitation,
                unit=unit,
                kind=kind,
            )
            return xr.Dataset({name: res_spectra})

        return simulation_function

    fitter = Fitter(
        sim=sim,
        results=results_ds,
        simulation_function_generator=simulation_function_generator,
        units_handler=UnitsHandler,
        p0=p0,
        scale=scale,
    )
    return fitter.solve(**kwargs)


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
    **kwargs,
):
    return fit_spectral_time_resolved(
        sim=sim,
        results=results,
        excitation=excitation,
        kind="emission",
        join_by_energy=join_by_energy,
        p0=p0,
        scale=scale,
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
    **kwargs,
):
    return fit_spectra(
        sim=sim,
        results=results,
        excitation=excitation,
        kind="emission",
        p0=p0,
        scale=scale,
        **kwargs,
    )