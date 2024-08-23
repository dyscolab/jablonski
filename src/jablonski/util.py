"""
    jablonski.util
    ~~~~~~~~~~~~~~

    Molecular states.

    :copyright: 2024 by jablonski Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from types import UnionType
from typing import Any, Literal, TypeAlias

import pint
from pint.facets.plain import PlainQuantity

from ._typing import RadiativeDecay
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


def emission_transitions(
    system: SpectroscopicSystem,
    unit: str | pint.Unit = ureg.nm,
    kind: SpectraKind = "emission",
) -> dict[pint.Quantity | PlainQuantity[Any], RadiativeDecay]:
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
