"""
    jablonski.util
    ~~~~~~~~~~~~~~

    Molecular states.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from types import UnionType
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import pint
from pint.facets.plain import PlainQuantity
from poincare import Parameter, Simulator

from .states import (
    DIM_ENERGY,
    DIM_FREQUENCY,
    DIM_WAVELENGTH,
    DIM_WAVENUMBER,
    SpectroscopicSystem,
)
from .transitions import FluorescenceTransition, PhosphorescenceTransition

_ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]

ureg = pint.get_application_registry()


def _spectra(
    system: SpectroscopicSystem, unit: str | pint.Unit, include: _ClassInfo
) -> dict[pint.Quantity | PlainQuantity[Any], Parameter]:
    dim = ureg.get_dimensionality(unit)

    if dim == DIM_ENERGY:
        # energy
        return {
            transition.energy_difference.to(unit): transition.val
            for transition in system._yield(include)
        }

    elif dim == DIM_WAVENUMBER:
        # wavenumber
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition.val
                for transition in system._yield(include)
            }

    elif dim == DIM_WAVELENGTH:
        # wavelength
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition.val
                for transition in system._yield(include)
            }

    elif dim == DIM_FREQUENCY:
        # frequency
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition.val
                for transition in system._yield(include)
            }

    else:
        raise ValueError(f"Cannot provide the spectra in {unit} ({dim})")


def fluorescence_spectra(
    system: SpectroscopicSystem, unit: str | pint.Unit
) -> dict[pint.Quantity | PlainQuantity[Any], Parameter]:
    return _spectra(system, unit, FluorescenceTransition)


def phosphorescence_spectra(
    system: SpectroscopicSystem, unit: str | pint.Unit
) -> dict[pint.Quantity | PlainQuantity[Any], Parameter]:
    return _spectra(system, unit, PhosphorescenceTransition)


def emission_spectra(
    system: SpectroscopicSystem, unit: str | pint.Unit
) -> dict[pint.Quantity | PlainQuantity[Any], Parameter]:
    return _spectra(system, unit, (FluorescenceTransition, PhosphorescenceTransition))


Power: TypeAlias = float | int
Time: TypeAlias = float | int


def _simple_time_resolved_emission(
    system: SpectroscopicSystem,
    step: Time,
    window: Time,
    excitation: tuple[Parameter, Power, Time],
    emission: dict[str, float],
):
    """Single transition square excitation."""

    excitation_state, excitation_power, excitation_width = excitation
    assert excitation_width < window

    sim = Simulator(system)

    out = []

    sol = sim.solve(
        values={excitation_state: excitation_power},
        save_at=np.arange(step),
        t_span=(0, excitation_width),
    )
    out.append(sol)

    sol = sim.solve = sol.iloc[-1]

    out.append(sol)

    for transition, factor in emission.items():
        sol["full_emission"] += factor * sol[transition]

    return pd.concat(out)


def simple_time_resolved_emission(
    system: SpectroscopicSystem,
    step: Time,
    window: Time,
    excitation: Parameter,
    emission: Parameter,
):
    out = _simple_time_resolved_emission(
        system, step, window, (excitation, 1, 1e-10), {emission: 1}
    )

    normalization_factor = out["full_emission"].sum()

    out["full_emission"] /= normalization_factor
    out[emission] /= normalization_factor

    return out
