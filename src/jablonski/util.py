"""
    jablonski.util
    ~~~~~~~~~~~~~~

    Molecular states.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from types import UnionType
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import pint
from pint.facets.plain import PlainQuantity
from poincare import Parameter, Simulator

from ._typing import Time
from .states import (
    DIM_ENERGY,
    DIM_FREQUENCY,
    DIM_WAVELENGTH,
    DIM_WAVENUMBER,
    SpectroscopicSystem,
)
from .transitions import Fluorescence, Phosphorescence

_ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]

ureg = pint.get_application_registry()

SpectraKind = Literal["emission", "fluorescence", "phosphorescence"]


def spectra(
    system: SpectroscopicSystem,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
) -> dict[pint.Quantity | PlainQuantity[Any], Parameter]:
    if kind == "emission":
        include = (Fluorescence, Phosphorescence)
    elif kind == "fluorescence":
        include = Fluorescence
    elif kind == "phosphorescence":
        include = Phosphorescence
    else:
        raise ValueError(f"kind must be {SpectraKind}")

    dim = ureg.get_dimensionality(unit)

    if dim == DIM_ENERGY:
        # energy
        return {
            transition.energy_difference.to(unit): transition
            for transition in system._yield(include)
        }

    elif dim == DIM_WAVENUMBER:
        # wavenumber
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition
                for transition in system._yield(include)
            }

    elif dim == DIM_WAVELENGTH:
        # wavelength
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition
                for transition in system._yield(include)
            }

    elif dim == DIM_FREQUENCY:
        # frequency
        with ureg.context("spectroscopy"):
            return {
                transition.energy_difference.to(unit): transition
                for transition in system._yield(include)
            }

    else:
        raise ValueError(f"Cannot provide the spectra in {unit} ({dim})")


def _simple_time_resolved_emission(
    system: SpectroscopicSystem,
    step: Time,
    window: Time,
    excitation: tuple[Parameter, Time],
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
